// AX650 axengine wrapper (named multi-IO, cached buffers + cache maintenance).
// Model is loaded into host memory.
//
// Vendored from https://github.com/ZY-2012/pyannote-speaker-diarization.axera
// (MIT License, see ../LICENSE).  Local changes for the ASR/LLM/TTS chain:
//   - AX Engine follows the device's current VNPU mode instead of forcing a
//     fixed one: the audio pipeline runs BIG_LITTLE (2+1), standalone runs
//     use the boot default.  The community-1 NPU3 models only load in a
//     VNPU-disabled device; NPU2 builds load under BIG_LITTLE.  Every handle
//     is still bound to affinity 0x1 when the mode supports it.
//   - Diagnostics go to stderr so the worker protocol can own stdout/fd 3.
#include "EngineWrapper.hpp"
#include "ax_sys_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#if !defined(AX630C)
static int ax_engine_init_once() {
    static bool inited = false;
    if (inited) return 0;
    // The device VNPU mode is global: every engine process must request the
    // mode already in use.  Community-1 NPU3 models require mode 0, while the
    // audio pipeline owns the device in BIG_LITTLE (2), so try the standalone
    // mode first and fall back to BIG_LITTLE when that mode is occupied.
    // DIAR_NPU_MODE is a board-debug override.
    const char* override_mode = getenv("DIAR_NPU_MODE");
    const int modes[2] = {
        override_mode != nullptr ? atoi(override_mode) : AX_ENGINE_VIRTUAL_NPU_DISABLE,
        override_mode != nullptr ? atoi(override_mode) : AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE};
    const int attempts = override_mode != nullptr ? 1 : 2;
    int last_error = -1;
    for (int i = 0; i < attempts; ++i) {
        AX_ENGINE_NPU_ATTR_T attr;
        memset(&attr, 0, sizeof(attr));
        attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(modes[i]);
        const int ret = AX_ENGINE_Init(&attr);
        if (ret == 0) {
            inited = true;
            fprintf(stderr, "[diar] AX Engine mode=%d\n", static_cast<int>(attr.eHardMode));
            return 0;
        }
        last_error = ret;
        fprintf(stderr, "[diar] AX_ENGINE_Init(mode=%d) ret=0x%x\n",
                modes[i], static_cast<unsigned>(ret));
    }
    return last_error;
}
#endif

EngineWrapper::EngineWrapper()
    : m_hasInit(false), m_handle(nullptr), m_io_info(nullptr),
      m_input_num(0), m_output_num(0) {
    memset(&m_io, 0, sizeof(m_io));
}

EngineWrapper::~EngineWrapper() { Release(); }

int EngineWrapper::Init(const char* strModelPath, uint32_t nNpuType, const char* axclConfig) {
    (void)nNpuType;
    (void)axclConfig;
    std::ifstream file(strModelPath, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "[diar] read model(%s) fail\n", strModelPath);
        return -1;
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<char> model((size_t)size);
    file.read(model.data(), size);
    file.close();

    if (ax_engine_init_once() != 0) {
        fprintf(stderr, "[diar] AX_ENGINE_Init fail\n");
        return -1;
    }
    const AX_S32 create_status =
        AX_ENGINE_CreateHandle(&m_handle, model.data(), (AX_U32)model.size());
    if (create_status != 0 || !m_handle) {
        fprintf(stderr, "[diar] create model(%s) handle fail ret=0x%x\n",
                strModelPath, static_cast<unsigned>(create_status));
        return -1;
    }
    if (AX_ENGINE_CreateContext(m_handle) != 0) {
        fprintf(stderr, "[diar] CreateContext fail\n");
        return -1;
    }
    if (AX_ENGINE_GetIOInfo(m_handle, &m_io_info) != 0) {
        fprintf(stderr, "[diar] GetIOInfo fail\n");
        return -1;
    }
    // Bind the diarization models to the BIG two-core partition (0x1).  The
    // pipeline holds the NPU2 Cell lock while a request is in flight.
    if (AX_ENGINE_SetAffinity(m_handle, 0x1U) != 0)
        fprintf(stderr, "[diar] warning: cannot bind %s to affinity 0x1\n",
                strModelPath);

    m_input_num = m_io_info->nInputSize;
    m_output_num = m_io_info->nOutputSize;
    m_input_name_to_idx.clear();
    m_output_name_to_idx.clear();
    for (int i = 0; i < m_input_num; ++i)
        m_input_name_to_idx[std::string(m_io_info->pInputs[i].pName)] = i;
    for (int i = 0; i < m_output_num; ++i)
        m_output_name_to_idx[std::string(m_io_info->pOutputs[i].pName)] = i;

    m_io.nInputSize = m_input_num;
    m_io.nOutputSize = m_output_num;
    m_io.pInputs = new AX_ENGINE_IO_BUFFER_T[m_input_num]();
    m_io.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_output_num]();
    for (int i = 0; i < m_input_num; ++i) {
        AX_U32 bytes = m_io_info->pInputs[i].nSize;
        m_io.pInputs[i].nSize = bytes;
        if (AX_SYS_MemAllocCached(&m_io.pInputs[i].phyAddr, &m_io.pInputs[i].pVirAddr,
                                  bytes, 128, (const AX_S8*)"diar-in") != 0) {
            fprintf(stderr, "[diar] input alloc fail\n");
            return -1;
        }
        memset(m_io.pInputs[i].pVirAddr, 0, bytes);
    }
    for (int i = 0; i < m_output_num; ++i) {
        AX_U32 bytes = m_io_info->pOutputs[i].nSize;
        m_io.pOutputs[i].nSize = bytes;
        if (AX_SYS_MemAllocCached(&m_io.pOutputs[i].phyAddr, &m_io.pOutputs[i].pVirAddr,
                                  bytes, 128, (const AX_S8*)"diar-out") != 0) {
            fprintf(stderr, "[diar] output alloc fail\n");
            return -1;
        }
    }
    m_hasInit = true;
    return 0;
}

int EngineWrapper::SetInput(void* pInput, int index) {
    if (!m_hasInit || index < 0 || index >= m_input_num) return -1;
    memcpy(m_io.pInputs[index].pVirAddr, pInput, m_io.pInputs[index].nSize);
    return 0;
}

int EngineWrapper::RunSync() {
    if (!m_hasInit) return -1;
    for (int i = 0; i < m_input_num; ++i)
        AX_SYS_MflushCache(m_io.pInputs[i].phyAddr, m_io.pInputs[i].pVirAddr,
                           m_io.pInputs[i].nSize);
    int ret = AX_ENGINE_RunSync(m_handle, &m_io);
    if (ret != 0) fprintf(stderr, "[diar] AX_ENGINE_RunSync failed. ret=0x%x\n", ret);
    for (int i = 0; i < m_output_num; ++i)
        AX_SYS_MinvalidateCache(m_io.pOutputs[i].phyAddr, m_io.pOutputs[i].pVirAddr,
                                m_io.pOutputs[i].nSize);
    return ret;
}

int EngineWrapper::GetOutput(void* pOutput, int index) {
    if (!m_hasInit || index < 0 || index >= m_output_num) return -1;
    memcpy(pOutput, m_io.pOutputs[index].pVirAddr, m_io.pOutputs[index].nSize);
    return 0;
}

int EngineWrapper::SetInputByName(const char* name, void* pInput) {
    auto it = m_input_name_to_idx.find(std::string(name));
    if (it == m_input_name_to_idx.end()) {
        fprintf(stderr, "[diar] input '%s' not found in model\n", name);
        return -1;
    }
    return SetInput(pInput, it->second);
}

int EngineWrapper::GetOutputByName(const char* name, void* pOutput) {
    auto it = m_output_name_to_idx.find(std::string(name));
    if (it == m_output_name_to_idx.end()) {
        fprintf(stderr, "[diar] output '%s' not found in model\n", name);
        return -1;
    }
    return GetOutput(pOutput, it->second);
}

int EngineWrapper::GetInputSize(int index) {
    if (index < 0 || index >= m_input_num) return -1;
    return m_io.pInputs[index].nSize;
}
int EngineWrapper::GetOutputSize(int index) {
    if (index < 0 || index >= m_output_num) return -1;
    return m_io.pOutputs[index].nSize;
}
int EngineWrapper::GetInputSizeByName(const char* name) { return GetInputSize(GetInputIndex(name)); }
int EngineWrapper::GetOutputSizeByName(const char* name) { return GetOutputSize(GetOutputIndex(name)); }

int EngineWrapper::GetInputIndex(const char* name) {
    auto it = m_input_name_to_idx.find(std::string(name));
    return it == m_input_name_to_idx.end() ? -1 : it->second;
}
int EngineWrapper::GetOutputIndex(const char* name) {
    auto it = m_output_name_to_idx.find(std::string(name));
    return it == m_output_name_to_idx.end() ? -1 : it->second;
}
const char* EngineWrapper::GetInputName(int index) const {
    return (index >= 0 && index < m_input_num) ? m_io_info->pInputs[index].pName : "";
}
const char* EngineWrapper::GetOutputName(int index) const {
    return (index >= 0 && index < m_output_num) ? m_io_info->pOutputs[index].pName : "";
}
const char* EngineWrapper::GetInputDtypeStr(int index) const { (void)index; return "float32"; }
const char* EngineWrapper::GetOutputDtypeStr(int index) const { (void)index; return "float32"; }

int EngineWrapper::Release() {
    if (!m_hasInit) return 0;
    for (int i = 0; i < m_input_num; ++i)
        if (m_io.pInputs[i].pVirAddr)
            AX_SYS_MemFree(m_io.pInputs[i].phyAddr, m_io.pInputs[i].pVirAddr);
    for (int i = 0; i < m_output_num; ++i)
        if (m_io.pOutputs[i].pVirAddr)
            AX_SYS_MemFree(m_io.pOutputs[i].phyAddr, m_io.pOutputs[i].pVirAddr);
    delete[] m_io.pInputs;
    delete[] m_io.pOutputs;
    memset(&m_io, 0, sizeof(m_io));
    if (m_handle) AX_ENGINE_DestroyHandle(m_handle);
    m_handle = nullptr;
    m_hasInit = false;
    return 0;
}

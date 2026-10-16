#include "ax_engine_runner.hpp"

#include "ax_engine_api.h"
#include "ax_engine_type.h"
#include "ax_sys_api.h"

#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace {
void check_ax(AX_S32 ret, const char* what) {
    if (ret != 0) throw std::runtime_error(std::string(what) + " failed");
}

// AX_SYS_Init/AX_ENGINE_Init are process-global; two runners (preencode +
// encoder) must share them, so reference-count the init/deinit pair.
//
// The VNPU mode is device-global and mutually exclusive across processes:
// when the transcription pipeline (BIG_LITTLE) is already running, a
// DISABLE-mode init fails with 0x80060087.  Try DISABLE first (upstream
// standalone behaviour, required by NPU3-compiled models) and fall back to
// BIG_LITTLE so NPU2-compiled models coexist with the pipeline — the same
// fallback the pyannote EngineWrapper uses.
std::mutex g_init_mutex;
int g_init_count = 0;

void engine_init_shared() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_init_count == 0) {
        AX_ENGINE_NPU_ATTR_T attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        check_ax(AX_SYS_Init(), "AX_SYS_Init");
        AX_S32 ret = AX_ENGINE_Init(&attr);
        if (ret != 0) {
            std::fprintf(stderr,
                         "ax_engine_runner: VNPU DISABLE init failed (0x%x); "
                         "retrying with BIG_LITTLE\n",
                         static_cast<unsigned>(ret));
            std::memset(&attr, 0, sizeof(attr));
            attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;
            ret = AX_ENGINE_Init(&attr);
            if (ret != 0) {
                AX_SYS_Deinit();
                throw std::runtime_error("AX_ENGINE_Init failed with both VNPU "
                                         "DISABLE and BIG_LITTLE modes: 0x" +
                                         std::to_string(static_cast<unsigned>(ret)));
            }
            std::fprintf(stderr, "ax_engine_runner: AX Engine mode=BIG_LITTLE (2+1)\n");
        }
    }
    g_init_count++;
}

void engine_deinit_shared() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_init_count > 0 && --g_init_count == 0) {
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
    }
}
}  // namespace

struct AxEngineRunner::Impl {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_IO_INFO_T* info = nullptr;
    AX_ENGINE_IO_T io{};
    std::vector<char> model;
    std::vector<std::string> in_names, out_names;
    std::unordered_map<std::string, size_t> in_idx, out_idx;
    bool engine_ready = false;
};

AxEngineRunner::AxEngineRunner(const std::string& model_path) : impl_(new Impl) {
    std::ifstream file(model_path.c_str(), std::ios::binary | std::ios::ate);
    if (!file) {
        delete impl_;
        throw std::runtime_error("cannot open axmodel: " + model_path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0);
    impl_->model.resize(static_cast<size_t>(size));
    file.read(impl_->model.data(), size);

    try {
        // AX_SYS_Init must precede AX_ENGINE_Init or the latter returns an error.
        engine_init_shared();
        impl_->engine_ready = true;

        check_ax(AX_ENGINE_CreateHandle(&impl_->handle, impl_->model.data(),
                                         static_cast<AX_U32>(impl_->model.size())),
                 "AX_ENGINE_CreateHandle");
        check_ax(AX_ENGINE_CreateContext(impl_->handle), "AX_ENGINE_CreateContext");
        check_ax(AX_ENGINE_GetIOInfo(impl_->handle, &impl_->info), "AX_ENGINE_GetIOInfo");
        // NPU2-compiled models live on the BIG partition under BIG_LITTLE
        // (affinity 0x1, the same slot SenseVoice binds); best-effort, a
        // DISABLE-mode device keeps all cores and ignores this.
        if (!set_affinity(0x1))
            std::fprintf(stderr, "ax_engine_runner: affinity 0x1 binding failed for %s\n",
                         model_path.c_str());
        std::fprintf(stderr, "ax_engine_runner: %s model_type=%d affinity=0x%x\n",
                     model_path.c_str(), model_type(), affinity());

        const AX_U32 n_in = impl_->info->nInputSize;
        const AX_U32 n_out = impl_->info->nOutputSize;
        impl_->io.nInputSize = n_in;
        impl_->io.nOutputSize = n_out;
        impl_->io.pInputs = new AX_ENGINE_IO_BUFFER_T[n_in]();
        impl_->io.pOutputs = new AX_ENGINE_IO_BUFFER_T[n_out]();

        for (AX_U32 i = 0; i < n_in; ++i) {
            const AX_U32 bytes = impl_->info->pInputs[i].nSize;
            check_ax(AX_SYS_MemAllocCached(&impl_->io.pInputs[i].phyAddr,
                                           &impl_->io.pInputs[i].pVirAddr, bytes, 128,
                                           reinterpret_cast<const AX_S8*>("npu")),
                     "input alloc");
            impl_->io.pInputs[i].nSize = bytes;
            std::memset(impl_->io.pInputs[i].pVirAddr, 0, bytes);
            const std::string name = impl_->info->pInputs[i].pName;
            impl_->in_names.push_back(name);
            impl_->in_idx[name] = i;
        }
        for (AX_U32 i = 0; i < n_out; ++i) {
            const AX_U32 bytes = impl_->info->pOutputs[i].nSize;
            check_ax(AX_SYS_MemAllocCached(&impl_->io.pOutputs[i].phyAddr,
                                           &impl_->io.pOutputs[i].pVirAddr, bytes, 128,
                                           reinterpret_cast<const AX_S8*>("npu")),
                     "output alloc");
            impl_->io.pOutputs[i].nSize = bytes;
            const std::string name = impl_->info->pOutputs[i].pName;
            impl_->out_names.push_back(name);
            impl_->out_idx[name] = i;
        }
    } catch (...) {
        AxEngineRunner::Impl* dying = impl_;
        impl_ = nullptr;
        if (dying->handle) AX_ENGINE_DestroyHandle(dying->handle);
        if (dying->engine_ready) {
            engine_deinit_shared();
        }
        delete dying;
        throw;
    }
}

AxEngineRunner::~AxEngineRunner() {
    if (!impl_) return;
    if (impl_->io.pInputs) {
        for (AX_U32 i = 0; i < impl_->io.nInputSize; ++i)
            AX_SYS_MemFree(impl_->io.pInputs[i].phyAddr, impl_->io.pInputs[i].pVirAddr);
        delete[] impl_->io.pInputs;
    }
    if (impl_->io.pOutputs) {
        for (AX_U32 i = 0; i < impl_->io.nOutputSize; ++i)
            AX_SYS_MemFree(impl_->io.pOutputs[i].phyAddr, impl_->io.pOutputs[i].pVirAddr);
        delete[] impl_->io.pOutputs;
    }
    if (impl_->handle) AX_ENGINE_DestroyHandle(impl_->handle);
    if (impl_->engine_ready) {
        engine_deinit_shared();
    }
    delete impl_;
}

float* AxEngineRunner::input(const std::string& name) {
    auto it = impl_->in_idx.find(name);
    if (it == impl_->in_idx.end()) throw std::runtime_error("no such input: " + name);
    return static_cast<float*>(impl_->io.pInputs[it->second].pVirAddr);
}

const float* AxEngineRunner::output(const std::string& name) const {
    auto it = impl_->out_idx.find(name);
    if (it == impl_->out_idx.end()) throw std::runtime_error("no such output: " + name);
    return static_cast<const float*>(impl_->io.pOutputs[it->second].pVirAddr);
}

size_t AxEngineRunner::input_elems(const std::string& name) const {
    auto it = impl_->in_idx.find(name);
    if (it == impl_->in_idx.end()) throw std::runtime_error("no such input: " + name);
    return impl_->info->pInputs[it->second].nSize / sizeof(float);
}

size_t AxEngineRunner::output_elems(const std::string& name) const {
    auto it = impl_->out_idx.find(name);
    if (it == impl_->out_idx.end()) throw std::runtime_error("no such output: " + name);
    return impl_->info->pOutputs[it->second].nSize / sizeof(float);
}

bool AxEngineRunner::has_input(const std::string& name) const {
    return impl_->in_idx.count(name) != 0;
}

const std::vector<std::string>& AxEngineRunner::input_names() const { return impl_->in_names; }
const std::vector<std::string>& AxEngineRunner::output_names() const { return impl_->out_names; }

void AxEngineRunner::run() {
    // Buffers come from AX_SYS_MemAllocCached, so the CPU's writes may still sit
    // in cache when the NPU reads DRAM. Without these two calls the run is
    // non-deterministic: identical input produced RTTMs differing by one frame
    // on ~1% of frames.
    for (AX_U32 i = 0; i < impl_->io.nInputSize; ++i) {
        check_ax(AX_SYS_MflushCache(impl_->io.pInputs[i].phyAddr, impl_->io.pInputs[i].pVirAddr,
                                    impl_->io.pInputs[i].nSize),
                 "AX_SYS_MflushCache(input)");
    }
    check_ax(AX_ENGINE_RunSync(impl_->handle, &impl_->io), "AX_ENGINE_RunSync");
    for (AX_U32 i = 0; i < impl_->io.nOutputSize; ++i) {
        check_ax(AX_SYS_MinvalidateCache(impl_->io.pOutputs[i].phyAddr,
                                         impl_->io.pOutputs[i].pVirAddr,
                                         impl_->io.pOutputs[i].nSize),
                 "AX_SYS_MinvalidateCache(output)");
    }
}

void AxEngineRunner::flush_inputs() {
    for (AX_U32 i = 0; i < impl_->io.nInputSize; ++i) {
        check_ax(AX_SYS_MflushCache(impl_->io.pInputs[i].phyAddr, impl_->io.pInputs[i].pVirAddr,
                                    impl_->io.pInputs[i].nSize),
                 "AX_SYS_MflushCache(input)");
    }
}

void AxEngineRunner::run_without_flush() {
    check_ax(AX_ENGINE_RunSync(impl_->handle, &impl_->io), "AX_ENGINE_RunSync");
}

unsigned AxEngineRunner::affinity() const {
    AX_ENGINE_NPU_SET_T npu_set = 0;
    if (AX_ENGINE_GetAffinity(impl_->handle, &npu_set) != 0) return 0;
    return static_cast<unsigned>(npu_set);
}

bool AxEngineRunner::set_affinity(unsigned mask) {
    return AX_ENGINE_SetAffinity(impl_->handle, static_cast<AX_ENGINE_NPU_SET_T>(mask)) == 0;
}

int AxEngineRunner::model_type() const {
    AX_ENGINE_MODEL_TYPE_T type = AX_ENGINE_MODEL_TYPE0;
    if (AX_ENGINE_GetHandleModelType(impl_->handle, &type) != 0) return -1;
    return static_cast<int>(type);
}

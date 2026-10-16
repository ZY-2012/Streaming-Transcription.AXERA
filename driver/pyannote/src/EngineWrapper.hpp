/**************************************************************************************************
 * ZipVoice C++ — Unified EngineWrapper (AXERA & AXCL)
 *
 * Supports both AXERA (demo board) and AXCL (compute card) via conditional compilation.
 *   AX650 / AX630C / AX620Q → AXERA  path (ax_engine_api)
 *   AXCL                           → AXCL   path (middleware::runner)
 *
 * Interface is identical — callers don't need to know which backend is active.
 **************************************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdint>
#include <memory>

#if defined(AX650) || defined(AX630C) || defined(AX620Q)
    #include "ax_engine_api.h"
#elif defined(AXCL)
    #include "middleware/axcl_runtime_runner.hpp"
#endif

class EngineWrapper {
public:
    EngineWrapper();
    ~EngineWrapper();

    /**
     * Initialize and load a model.
     * @param strModelPath  Path to .axmodel file
     * @param nNpuType      AXERA: VNPU type (0=default); AXCL: device_index (0=first card)
     * @param axclConfig    AXCL: path to axcl.json (nullptr → /usr/local/axcl/axcl.json). Ignored on AXERA.
     */
    int Init(const char* strModelPath, uint32_t nNpuType = 0, const char* axclConfig = nullptr);

    // Index-based I/O
    int SetInput(void* pInput, int index);
    int RunSync();
    int GetOutput(void* pOutput, int index);

    // Name-based I/O
    int SetInputByName(const char* name, void* pInput);
    int GetOutputByName(const char* name, void* pOutput);

    int GetInputSize(int index);
    int GetOutputSize(int index);
    int GetInputSizeByName(const char* name);
    int GetOutputSizeByName(const char* name);

    int GetInputIndex(const char* name);
    int GetOutputIndex(const char* name);
    int GetInputCount() const { return m_input_num; }
    int GetOutputCount() const { return m_output_num; }
    const char* GetInputName(int index) const;
    const char* GetOutputName(int index) const;
#if defined(AX650) || defined(AX630C) || defined(AX620Q)
    const char* GetInputDtypeStr(int index) const;
    const char* GetOutputDtypeStr(int index) const;
#endif

    bool HasInit() const { return m_hasInit; }
    int Release();

private:
    bool m_hasInit;
    int m_input_num, m_output_num;

    std::unordered_map<std::string, int> m_input_name_to_idx;
    std::unordered_map<std::string, int> m_output_name_to_idx;

#if defined(AX650) || defined(AX630C) || defined(AX620Q)
    AX_ENGINE_HANDLE m_handle;
    AX_ENGINE_IO_INFO_T *m_io_info;
    AX_ENGINE_IO_T m_io;
#elif defined(AXCL)
    std::unique_ptr<middleware::runner> m_runner;
#endif
};

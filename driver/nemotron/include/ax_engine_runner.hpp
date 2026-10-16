#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Minimal named multi-IO wrapper over AX Engine.
//
// The LS-EEND streaming graph has 14 inputs and 12 outputs, all FP32, so the
// fixed single-in/single-out helper used by the windowed model does not apply.
// Buffers are allocated once and reused for every frame; callers write into the
// input buffer returned by `input()` and read from `output()` after `run()`.
class AxEngineRunner {
public:
    explicit AxEngineRunner(const std::string& model_path);
    ~AxEngineRunner();

    AxEngineRunner(const AxEngineRunner&) = delete;
    AxEngineRunner& operator=(const AxEngineRunner&) = delete;

    // Writable FP32 view of a named input, length = element count.
    float* input(const std::string& name);
    // Read-only FP32 view of a named output.
    const float* output(const std::string& name) const;

    size_t input_elems(const std::string& name) const;
    size_t output_elems(const std::string& name) const;
    bool has_input(const std::string& name) const;

    const std::vector<std::string>& input_names() const;
    const std::vector<std::string>& output_names() const;

    void run();  // one synchronous NPU invocation

    // Benchmark/analysis helpers.
    void run_without_flush();          // AX_ENGINE_RunSync only (no cache maintenance)
    void flush_inputs();               // AX_SYS_MflushCache on every input
    unsigned affinity() const;         // current NPU core mask
    bool set_affinity(unsigned mask);  // e.g. 0b111 for all three cores
    int model_type() const;            // AX_ENGINE_MODEL_TYPE_T

private:
    struct Impl;
    Impl* impl_;
};

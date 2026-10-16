// Board-side encoder latency breakdown: flush / copy / RunSync / affinity.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ax_engine_runner.hpp"

namespace {
using Clock = std::chrono::steady_clock;

double ms_per_call(Clock::time_point start, int iterations) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / iterations;
}
}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string input_name = "seq";
    std::string lengths_name = "total_lengths";
    int iterations = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--encoder" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_name = argv[++i];
        else if (arg == "--lengths" && i + 1 < argc) lengths_name = argv[++i];
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: bench_encoder --encoder encoder.axmodel [--iterations N]\n");
        return 1;
    }

    AxEngineRunner runner(model_path);
    std::printf("model_type=%d affinity=0x%x\n", runner.model_type(), runner.affinity());
    std::fflush(stdout);

    const std::size_t seq_elems = runner.input_elems(input_name);
    const std::size_t lengths_elems = runner.has_input(lengths_name) ? runner.input_elems(lengths_name) : 0;
    std::vector<float> seq(seq_elems);
    for (std::size_t i = 0; i < seq_elems; ++i) seq[i] = 0.05f * static_cast<float>((i % 17) - 8);
    std::memcpy(runner.input(input_name), seq.data(), seq.size() * sizeof(float));
    if (lengths_elems) *runner.input(lengths_name) = static_cast<float>(seq_elems);

    // Warm-up (includes any first-run lazy compilation).
    for (int i = 0; i < 5; ++i) runner.run();
    volatile float sink = 0;
    for (const std::string& name : runner.output_names()) sink += runner.output(name)[0];
    (void)sink;

    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) runner.run();
    const double full_ms = ms_per_call(start, iterations);

    start = Clock::now();
    for (int i = 0; i < iterations; ++i) runner.flush_inputs();
    const double flush_ms = ms_per_call(start, iterations);

    start = Clock::now();
    for (int i = 0; i < iterations; ++i) runner.run_without_flush();
    const double sync_ms = ms_per_call(start, iterations);

    start = Clock::now();
    for (int i = 0; i < iterations; ++i) std::memcpy(runner.input(input_name), seq.data(), seq.size() * sizeof(float));
    const double copy_ms = ms_per_call(start, iterations);

    std::printf(
        "{\"full_ms\":%.3f,\"flush_ms\":%.3f,\"run_sync_ms\":%.3f,\"copy_ms\":%.3f,"
        "\"seq_bytes\":%zu,\"iterations\":%d}\n",
        full_ms, flush_ms, sync_ms, copy_ms, seq.size() * sizeof(float), iterations);

    const unsigned masks[] = {0x7u, 0x3u, 0x1u};
    for (unsigned mask : masks) {
        if (!runner.set_affinity(mask)) {
            std::printf("{\"affinity\":\"0x%x\",\"error\":\"set failed\"}\n", mask);
            std::fflush(stdout);
            continue;
        }
        for (int i = 0; i < 3; ++i) runner.run();
        start = Clock::now();
        for (int i = 0; i < iterations; ++i) runner.run();
        const double per_call = ms_per_call(start, iterations);
        std::printf("{\"affinity\":\"0x%x\",\"full_ms\":%.3f,\"reported\":\"0x%x\"}\n", mask, per_call,
                    runner.affinity());
    }
    return 0;
}

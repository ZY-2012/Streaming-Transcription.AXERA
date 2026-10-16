// Host-side parity test (no NPU): StreamingMel - the rolling pre-emphasized
// sample buffer plus rolling mel frame buffer used by the streaming worker -
// must reproduce MelFrontend::prepare + compute_range bit-exactly under
// arbitrary PCM arrival patterns, with trimming active like the real worker.
//
//   test_stream_mel [in.wav]
//
// Without a wav the test synthesizes a deterministic pseudo-random signal.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "audio_reader.hpp"
#include "chunk_layout.hpp"
#include "mel_frontend.hpp"
#include "streaming_frontend.hpp"

namespace {

int failures = 0;

std::vector<float> load_samples(int argc, char** argv, int* rate_out) {
    if (argc > 1) {
        AudioData audio = read_wav_mono(argv[1]);
        if (audio.sample_rate != MelFrontend::kSampleRate)
            audio = resample_audio(audio, MelFrontend::kSampleRate);
        *rate_out = audio.sample_rate;
        return audio.samples;
    }
    // 30 s of deterministic noise+sine so the mel sees non-trivial spectra.
    std::vector<float> samples(30 * MelFrontend::kSampleRate);
    std::mt19937 rng(42);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const double t = static_cast<double>(i) / MelFrontend::kSampleRate;
        samples[i] = static_cast<float>(0.3 * std::sin(2 * 3.14159265358979 * 220.0 * t) +
                                       0.6 * (static_cast<double>(rng() % 20001 - 10000) / 10000.0));
    }
    *rate_out = MelFrontend::kSampleRate;
    return samples;
}

// Feed the rolling front-end like the streaming worker does and verify every
// frame it exposes against the file-mode reference.
void run_pass(const char* name, const std::vector<float>& samples, const SortformerConfig& config,
              std::uint32_t seed) {
    const int raw = MelFrontend::raw_frames(samples.size());
    if (raw <= 0) return;

    MelFrontend fe;
    fe.set_threads(2);
    fe.prepare(samples);
    std::vector<float> ref(static_cast<std::size_t>(raw) * MelFrontend::kMels);
    fe.compute_range(0, raw, ref.data());

    MelFrontend sfe;  // same filterbank state (default or loaded assets)
    sfe.set_threads(2);
    StreamingMel mel(sfe);

    long long nulls = 0;
    long long mismatches = 0;
    auto check_row = [&](int frame) {
        const float* row = mel.row(frame);
        if (row == nullptr) {
            ++nulls;
            return;
        }
        if (std::memcmp(row, ref.data() + static_cast<std::size_t>(frame) * MelFrontend::kMels,
                        MelFrontend::kMels * sizeof(float)) != 0)
            ++mismatches;
    };

    std::mt19937 rng(seed);
    std::size_t fed = 0;
    int cursor = 0;
    const int left_needed = config.chunk_left_context * config.subsampling_factor;
    while (fed < samples.size()) {
        const std::size_t n = 1 + rng() % 4000;  // 0.0625 ms .. 250 ms blocks
        const std::size_t take = std::min(n, samples.size() - fed);
        mel.append(samples.data() + fed, take);
        fed += take;
        mel.advance(mel.computable_frames(), false);
        for (;;) {
            DiarChunk chunk;
            if (!NextChunk(cursor, mel.ready(), config, false, &chunk)) break;
            for (int f = chunk.start - chunk.left; f < chunk.end + chunk.right; ++f) check_row(f);
            cursor = chunk.end;
        }
        mel.trim(std::max(0, cursor - left_needed));
    }
    mel.advance(mel.total_frames(), true);
    for (;;) {
        DiarChunk chunk;
        if (!NextChunk(cursor, mel.total_frames(), config, true, &chunk)) break;
        for (int f = chunk.start - chunk.left; f < chunk.end + chunk.right; ++f) check_row(f);
        cursor = chunk.end;
    }
    mel.trim(std::max(0, cursor - left_needed));
    // Everything the worker needed was checked; also confirm the trim never
    // dropped a frame still referenced: rows in [cursor - left, total).
    for (int f = std::max(0, cursor - left_needed); f < mel.total_frames(); ++f) check_row(f);

    if (nulls != 0 || mismatches != 0) {
        ++failures;
        std::printf("FAIL %s seed=%u: nulls=%lld mismatches=%lld (raw=%d)\n", name, seed, nulls, mismatches, raw);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int rate = 0;
    const std::vector<float> samples = load_samples(argc, argv, &rate);
    std::printf("test signal: %zu samples @ %d Hz\n", samples.size(), rate);

    SortformerConfig lowlat;
    lowlat.chunk_len = 6; lowlat.chunk_left_context = 0; lowlat.chunk_right_context = 4;
    lowlat.fifo_len = 264; lowlat.spkcache_len = 264; lowlat.spkcache_update_period = 222;

    SortformerConfig lefty;  // non-zero left context exercises trim keep-zone
    lefty.chunk_len = 10; lefty.chunk_left_context = 2; lefty.chunk_right_context = 4;
    lefty.fifo_len = 40; lefty.spkcache_len = 264; lefty.spkcache_update_period = 300;

    for (std::uint32_t seed = 0; seed < 8; ++seed) {
        run_pass("lowlat", samples, lowlat, seed);
        run_pass("lefty", samples, lefty, seed * 31 + 7);
    }

    if (failures == 0) {
        std::printf("streaming mel parity: OK\n");
        return 0;
    }
    std::printf("streaming mel parity: %d FAILURES\n", failures);
    return 1;
}

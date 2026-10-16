// Host-side mel front-end check: wav -> raw float32 (frames x 128) dump.
// Compiled without the BSP so it runs on the development host.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "audio_reader.hpp"
#include "mel_frontend.hpp"
#include "npy_reader.hpp"

int main(int argc, char** argv) {
    std::string wav_path, out_path;
    std::string dump_mel_path, assets_dir;
    int threads = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--wav" && i + 1 < argc) wav_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (arg == "--dump-mel" && i + 1 < argc) dump_mel_path = argv[++i];
        else if (arg == "--assets" && i + 1 < argc) assets_dir = argv[++i];
    }
    if (wav_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: check_mel --wav in.wav --out feats.bin [--threads N]\n");
        return 1;
    }

    AudioData audio = read_wav_mono(wav_path);
    if (audio.sample_rate != MelFrontend::kSampleRate) {
        audio = resample_audio(audio, MelFrontend::kSampleRate);
    }

    MelFrontend frontend;
    frontend.set_threads(threads);
    if (!assets_dir.empty()) frontend.load_assets(assets_dir);
    int padded_frames = 0;
    const std::vector<float> features = frontend.compute(audio.samples, padded_frames);

    std::ofstream out(out_path.c_str(), std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    out.write(reinterpret_cast<const char*>(features.data()), static_cast<std::streamsize>(features.size() * sizeof(float)));
    std::printf("{\"frames\":%d,\"mels\":%d,\"raw_frames\":%d}\n", padded_frames, MelFrontend::kMels,
                MelFrontend::raw_frames(audio.samples.size()));
    if (!dump_mel_path.empty()) {
        // Expose the filterbank through a tiny accessor for host parity checks.
        std::ofstream mel_out(dump_mel_path.c_str(), std::ios::binary);
        const std::vector<float> matrix = MelFrontend::debug_filterbank();
        mel_out.write(reinterpret_cast<const char*>(matrix.data()),
                      static_cast<std::streamsize>(matrix.size() * sizeof(float)));
        std::printf("mel_matrix: %zu\n", matrix.size());
    }
    return 0;
}

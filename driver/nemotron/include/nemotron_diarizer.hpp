#pragma once

#include <string>
#include <vector>

#include "ax_engine_runner.hpp"
#include "nemotron_state.hpp"

struct DiarizerReport {
    int steps = 0;
    int frames = 0;
    double audio_seconds = 0.0;
    double mel_seconds = 0.0;
    double pre_seconds = 0.0;
    double enc_seconds = 0.0;
    double total_seconds = 0.0;
    int threads = 1;
};

class SortformerDiarizer {
public:
    // assets_dir: directory holding feature_window.npy / mel_filterbank.npy from
    // the checkpoint (empty -> analytically recomputed window / filterbank).
    SortformerDiarizer(const std::string& preencode_model, const std::string& encoder_model,
                       const SortformerConfig& config, int threads, const std::string& assets_dir = std::string());

    // samples: mono 16 kHz waveform. Returns frame probabilities (frames x speakers)
    // on the 10 ms high-resolution grid (truncated to floor(num_samples / 160) frames).
    std::vector<float> run(const std::vector<float>& samples, DiarizerReport& report);

private:
    AxEngineRunner preencode_;
    AxEngineRunner encoder_;
    SortformerConfig config_;
    int threads_;
    std::string assets_dir_;
};

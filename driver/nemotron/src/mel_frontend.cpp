#include "mel_frontend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "npy_reader.hpp"

extern "C" {
#include "kiss_fft.h"
}

namespace {
const double kPi = 3.14159265358979323846;

// Slaney mel scale, matching librosa.filters.mel(..., htk=False).
const double kFsp = 200.0 / 3.0;
const double kMinLogHz = 1000.0;
const double kMinLogMel = kMinLogHz / kFsp;
const double kLogStep = std::log(6.4) / 27.0;

double hz_to_mel(double hz) {
    if (hz < kMinLogHz) return hz / kFsp;
    return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}

double mel_to_hz(double mel) {
    if (mel < kMinLogMel) return kFsp * mel;
    return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
}

std::vector<float> build_mel_filterbank() {
    const int n_freqs = MelFrontend::kFft / 2 + 1;  // 257
    std::vector<double> mel_hz(MelFrontend::kMels + 2);
    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(MelFrontend::kSampleRate / 2.0);
    for (int i = 0; i < MelFrontend::kMels + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) * i / (MelFrontend::kMels + 1);
        mel_hz[i] = mel_to_hz(mel);
    }

    std::vector<double> fft_freqs(n_freqs);
    for (int k = 0; k < n_freqs; ++k) {
        fft_freqs[k] = (MelFrontend::kSampleRate / 2.0) * k / (n_freqs - 1);
    }

    std::vector<float> weights(static_cast<std::size_t>(MelFrontend::kMels) * n_freqs, 0.0f);
    for (int m = 0; m < MelFrontend::kMels; ++m) {
        const double f_lo = mel_hz[m], f_ctr = mel_hz[m + 1], f_hi = mel_hz[m + 2];
        const double lo_diff = f_ctr - f_lo, hi_diff = f_hi - f_ctr;
        const double enorm = 2.0 / (f_hi - f_lo);  // norm='slaney'
        for (int k = 0; k < n_freqs; ++k) {
            const double lower = (fft_freqs[k] - f_lo) / lo_diff;
            const double upper = (f_hi - fft_freqs[k]) / hi_diff;
            const double value = std::max(0.0, std::min(lower, upper));
            weights[static_cast<std::size_t>(m) * n_freqs + k] = static_cast<float>(value * enorm);
        }
    }
    return weights;
}
}  // namespace

MelFrontend::MelFrontend() : mel_(build_mel_filterbank()) {
    window_.resize(kWin);
    for (int n = 0; n < kWin; ++n) {
        window_[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / (kWin - 1)));
    }
}

bool MelFrontend::load_assets(const std::string& dir) {
    std::vector<float> window;
    std::vector<int> window_shape;
    std::vector<float> filterbank;
    std::vector<int> filter_shape;
    const std::string window_path = dir + "/feature_window.npy";
    const std::string filter_path = dir + "/mel_filterbank.npy";
    const bool window_ok = load_npy_float32(window_path, window, window_shape) && window.size() == kWin;
    const bool filter_ok = load_npy_float32(filter_path, filterbank, filter_shape) &&
                           filterbank.size() == static_cast<std::size_t>(kMels) * (kFft / 2 + 1);
    if (window_ok) window_ = window;
    if (filter_ok) mel_ = filterbank;
    if (!window_ok || !filter_ok) {
        std::fprintf(stderr,
                     "warning: feature assets missing under %s (window=%d filterbank=%d); "
                     "using recomputed values that differ from NeMo inference\n",
                     dir.c_str(), window_ok ? 1 : 0, filter_ok ? 1 : 0);
        return false;
    }
    return true;
}

std::vector<float> MelFrontend::compute(const std::vector<float>& samples, int& num_frames) const {
    const int frames = raw_frames(samples.size());
    num_frames = padded_frames(samples.size());
    std::vector<float> out(static_cast<std::size_t>(num_frames) * kMels, 0.0f);
    if (frames == 0) return out;

    const int center_pad = kFft / 2;             // 256
    const int left_pad = (kFft - kWin) / 2;      // 56
    const std::size_t padded_size = samples.size() + 2 * static_cast<std::size_t>(center_pad);
    std::vector<float> padded(padded_size, 0.0f);

    // Pre-emphasis (0.97) followed by constant center padding.
    padded[center_pad] = samples[0];
    for (std::size_t i = 1; i < samples.size(); ++i) {
        padded[center_pad + i] = samples[i] - 0.97f * samples[i - 1];
    }

    const int threads = std::max(1, std::min(threads_, frames));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));

    const int frames_per_thread = (frames + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const int begin = t * frames_per_thread;
        const int end = std::min(frames, begin + frames_per_thread);
        if (begin >= end) break;
        workers.emplace_back([&, begin, end]() {
            kiss_fft_cfg cfg = kiss_fft_alloc(kFft, 0, nullptr, nullptr);
            std::vector<kiss_fft_cpx> in(kFft), freq(kFft);
            std::vector<float> frame(kFft);
            for (int f = begin; f < end; ++f) {
                const std::size_t start = static_cast<std::size_t>(f) * kHop;
                std::fill(frame.begin(), frame.end(), 0.0f);
                for (int n = 0; n < kWin; ++n) {
                    frame[left_pad + n] = padded[start + left_pad + n] * window_[n];
                }
                for (int n = 0; n < kFft; ++n) {
                    in[n].r = frame[n];
                    in[n].i = 0.0f;
                }
                kiss_fft(cfg, in.data(), freq.data());

                float* row = out.data() + static_cast<std::size_t>(f) * kMels;
                for (int m = 0; m < kMels; ++m) {
                    const float* filter = mel_.data() + static_cast<std::size_t>(m) * (kFft / 2 + 1);
                    float acc = 0.0f;
                    for (int k = 0; k <= kFft / 2; ++k) {
                        const float power = freq[k].r * freq[k].r + freq[k].i * freq[k].i;
                        acc += filter[k] * power;
                    }
                    row[m] = std::log(acc + kLogGuard);
                }
            }
            kiss_fft_free(cfg);
        });
    }
    for (auto& worker : workers) worker.join();
    return out;
}

void MelFrontend::prepare(const std::vector<float>& samples) {
    signal_.resize(samples.size());
    if (samples.empty()) return;
    signal_[0] = samples[0];
    for (std::size_t i = 1; i < samples.size(); ++i) {
        signal_[i] = samples[i] - 0.97f * samples[i - 1];
    }
}

void MelFrontend::compute_range(int from, int to, float* out) const {
    const int total = static_cast<int>(signal_.size());
    compute_range_view(signal_.empty() ? nullptr : signal_.data(), 0, total, from, to, out);
}

void MelFrontend::compute_range_view(const float* signal, int signal_base, int signal_len, int from, int to,
                                     float* out) const {
    if (to <= from) return;
    const int frames = to - from;
    const int threads = std::max(1, std::min(threads_, frames));

    auto compute_block = [&](int begin, int end) {
        kiss_fft_cfg cfg = kiss_fft_alloc(kFft, 0, nullptr, nullptr);
        std::vector<kiss_fft_cpx> in(kFft), freq(kFft);
        std::vector<float> frame(kFft);
        for (int f = begin; f < end; ++f) {
            // Window sample n of frame f maps to signal index f*hop + n - 200:
            // padded p = f*hop + left_pad + n, signal index = p - kFft/2.
            const int base = f * kHop - (kFft / 2 - (kFft - kWin) / 2);
            std::fill(frame.begin(), frame.end(), 0.0f);
            for (int n = 0; n < kWin; ++n) {
                const int index = base + n - signal_base;
                if (index >= 0 && index < signal_len)
                    frame[(kFft - kWin) / 2 + n] = signal[index] * window_[n];
            }
            for (int n = 0; n < kFft; ++n) {
                in[n].r = frame[n];
                in[n].i = 0.0f;
            }
            kiss_fft(cfg, in.data(), freq.data());
            float* row = out + static_cast<std::size_t>(f - from) * kMels;
            for (int m = 0; m < kMels; ++m) {
                const float* filter = mel_.data() + static_cast<std::size_t>(m) * (kFft / 2 + 1);
                float acc = 0.0f;
                for (int k = 0; k <= kFft / 2; ++k) {
                    const float power = freq[k].r * freq[k].r + freq[k].i * freq[k].i;
                    acc += filter[k] * power;
                }
                row[m] = std::log(acc + kLogGuard);
            }
        }
        kiss_fft_free(cfg);
    };

    if (threads == 1) {
        compute_block(from, to);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    const int frames_per_thread = (frames + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const int begin = from + t * frames_per_thread;
        const int end = std::min(to, begin + frames_per_thread);
        if (begin >= end) break;
        workers.emplace_back(compute_block, begin, end);
    }
    for (auto& worker : workers) worker.join();
}

std::vector<float> MelFrontend::debug_filterbank() { return build_mel_filterbank(); }

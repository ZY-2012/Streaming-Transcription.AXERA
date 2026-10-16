#pragma once

#include <cstddef>
#include <string>
#include <vector>

// NeMo-compatible log-mel front-end for Nemotron-3-Diarization streaming (16 kHz).
//
// Mirrors FilterbankFeatures in eval mode: hann(400, periodic=false) window
// centered in a 512-point FFT, hop 160, power spectrum, mel filterbank
// (128 bins, Slaney-normalized), log(x + 2**-24), zero padded to a multiple of 16.
//
// The released .nemo stores a *quantized* window and filterbank; call
// load_assets() to reuse those exact buffers (required to match NeMo inference).
class MelFrontend {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr int kFft = 512;
    static constexpr int kWin = 400;
    static constexpr int kHop = 160;
    static constexpr int kMels = 128;
    static constexpr int kPadTo = 16;
    static constexpr float kLogGuard = 5.960464477539063e-08f;  // 2**-24

    MelFrontend();

    // Load feature_window.npy (400,) and mel_filterbank.npy (128x257) from dir.
    // Returns false and keeps the recomputed values when the assets are missing.
    bool load_assets(const std::string& dir);

    void set_threads(int threads) { threads_ = threads; }

    // Copy + pre-emphasize the waveform once; compute_range then works on frames.
    void prepare(const std::vector<float>& samples);

    // Compute frames [from, to) into out (row-major, (to-from) x 128 float32).
    // Frames are independent, so ranges may be computed incrementally while the
    // NPU runs the previous chunk.
    void compute_range(int from, int to, float* out) const;

    // Streaming variant of compute_range: frames [from, to) are computed
    // against an external pre-emphasized buffer holding absolute signal
    // indices [signal_base, signal_base + signal_len).  Indices outside the
    // buffer are zero-filled, which is exactly how file mode treats the head
    // (negative indices) and the tail beyond end of file, so a rolling buffer
    // that only drops samples no future frame needs is bit-identical to
    // prepare() + compute_range() on the whole waveform.
    void compute_range_view(const float* signal, int signal_base, int signal_len, int from, int to, float* out) const;

    // Mono 16 kHz input -> (num_frames, 128) row-major float32.
    // num_frames = ceil(floor(num_samples / hop) / 16) * 16.
    std::vector<float> compute(const std::vector<float>& samples, int& num_frames) const;

    static int raw_frames(std::size_t num_samples) {
        return static_cast<int>(num_samples / static_cast<std::size_t>(kHop));
    }
    static int padded_frames(std::size_t num_samples) {
        const int frames = raw_frames(num_samples);
        return (frames + kPadTo - 1) / kPadTo * kPadTo;
    }

    // Filterbank accessor for host parity checks (128 x 257, row-major).
    static std::vector<float> debug_filterbank();

private:
    std::vector<float> window_;
    std::vector<float> mel_;
    std::vector<float> signal_;  // pre-emphasized waveform (set by prepare)
    int threads_ = 1;
};

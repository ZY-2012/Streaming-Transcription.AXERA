#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct AudioData {
    int sample_rate = 0;
    std::vector<float> samples;
};

AudioData read_wav_mono(const std::string& path);

// Band-limited resampling (Kaiser-windowed sinc), parameters matched to
// librosa's `kaiser_best` so the C++ and Python frontends agree.
//
// Plain linear interpolation is not usable here: it has no anti-aliasing filter,
// so downsampling 16 kHz -> 8 kHz folds everything above 4 kHz back into the
// band, and its sinc^2 passband droop alone costs -4.2 dB at 3 kHz / -7.4 dB at
// 3.9 kHz. Measured on real 16 kHz speech that cost 0.990 feature cosine.
AudioData resample_audio(const AudioData& input, int target_rate);

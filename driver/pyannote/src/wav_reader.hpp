// Minimal 16-bit PCM WAV reader (mono output, supports stereo downmix).
//
// Vendored from https://github.com/ZY-2012/pyannote-speaker-diarization.axera
// (MIT License, see ../LICENSE).  Local change: malformed files raise an
// exception instead of calling exit(), so the resident diar-worker can report
// an error frame and keep serving later requests.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct WavData {
    int sample_rate = 0;
    std::vector<float> samples;   // mono, [-1, 1]
};

static inline WavData read_wav(const std::string& path) {
    WavData out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    auto rd = [&](void* p, size_t n) {
        if (fread(p, 1, n, f) != n) {
            fclose(f);
            throw std::runtime_error("bad wav " + path);
        }
    };
    char hdr[12];
    rd(hdr, 12);
    if (memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fclose(f);
        throw std::runtime_error("not a RIFF/WAVE file: " + path);
    }
    int channels = 1, bits = 16, rate = 16000;
    std::vector<uint8_t> data;
    while (true) {
        char ch[8];
        if (fread(ch, 1, 8, f) != 8) break;
        uint32_t size = (uint8_t)ch[4] | ((uint8_t)ch[5] << 8) | ((uint8_t)ch[6] << 16) | ((uint8_t)ch[7] << 24);
        if (!memcmp(ch, "fmt ", 4)) {
            std::vector<uint8_t> fmt(size);
            rd(fmt.data(), size);
            channels = fmt[2] | (fmt[3] << 8);
            rate = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
            bits = fmt[14] | (fmt[15] << 8);
        } else if (!memcmp(ch, "data", 4)) {
            data.resize(size);
            rd(data.data(), size);
        } else {
            fseek(f, size, SEEK_CUR);
        }
    }
    fclose(f);
    if (bits != 16)
        throw std::runtime_error("only 16-bit PCM supported (" + path + ")");
    out.sample_rate = rate;
    int n = data.size() / 2;
    out.samples.resize(n / channels);
    for (int i = 0; i < n / channels; ++i) {
        float acc = 0;
        for (int c = 0; c < channels; ++c) {
            int16_t s = data[(i * channels + c) * 2] | (data[(i * channels + c) * 2 + 1] << 8);
            acc += s;
        }
        out.samples[i] = acc / channels / 32768.0f;
    }
    return out;
}

#include "audio_reader.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
uint16_t read_u16(std::ifstream& f) { uint8_t b[2]; f.read(reinterpret_cast<char*>(b),2); return b[0] | (b[1]<<8); }
uint32_t read_u32(std::ifstream& f) { uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4); return b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24); }

const double kPi = 3.14159265358979323846;

double sinc(double x) {
    if (std::fabs(x) < 1e-9) return 1.0;
    return std::sin(kPi * x) / (kPi * x);
}

// Modified Bessel function of the first kind, order 0 (for the Kaiser window).
double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 64; ++k) {
        const double q = x / (2.0 * k);
        term *= q * q;
        sum += term;
        if (term < 1e-14 * sum) break;
    }
    return sum;
}
}


AudioData read_wav_mono(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    char riff[4], wave[4]; f.read(riff,4); (void)read_u32(f); f.read(wave,4);
    if (std::string(riff,4)!="RIFF" || std::string(wave,4)!="WAVE") throw std::runtime_error("not RIFF/WAVE");
    uint16_t format=0, channels=0, bits=0; uint32_t rate=0; std::vector<uint8_t> pcm;
    while (f && !f.eof()) {
        char id[4]; f.read(id,4); if (!f) break; uint32_t n=read_u32(f); std::string tag(id,4);
        if (tag=="fmt ") { format=read_u16(f); channels=read_u16(f); rate=read_u32(f); (void)read_u32(f); (void)read_u16(f); bits=read_u16(f); if(n>16)f.seekg(n-16,std::ios::cur); }
        else if(tag=="data") { pcm.resize(n); f.read(reinterpret_cast<char*>(pcm.data()),n); }
        else f.seekg(n,std::ios::cur);
        if(n&1)f.seekg(1,std::ios::cur);
    }
    if(format!=1 || bits!=16 || channels<1 || rate==0 || pcm.empty()) throw std::runtime_error("only PCM16 WAV supported");
    size_t frames=pcm.size()/(channels*2); AudioData out; out.sample_rate=(int)rate; out.samples.resize(frames);
    for(size_t i=0;i<frames;++i) { int sum=0; for(int c=0;c<channels;++c){size_t p=(i*channels+c)*2; int16_t v=(int16_t)(pcm[p]|(pcm[p+1]<<8)); sum+=v;} out.samples[i]=(float)sum/(channels*32768.0f); }
    return out;
}

AudioData resample_audio(const AudioData& input, int target_rate) {
    if (input.sample_rate == target_rate || input.samples.empty()) return input;

    // Kaiser-windowed sinc with resampy's `kaiser_best` parameters -- the preset
    // librosa.resample uses by default -- so the C++ frontend matches Python.
    const double ratio = static_cast<double>(target_rate) / input.sample_rate;
    const double cutoff = std::min(1.0, ratio) * 0.945;  // rolloff, fraction of input Nyquist
    const double beta = 14.769656459379492;
    const int zeros = 32;                                // sinc zero crossings per side
    const double half_width = zeros / cutoff;            // in input samples
    const int table_res = 1024;                          // table entries per input sample

    // Precompute the right half of the symmetric kernel.
    const int table_len = static_cast<int>(std::ceil(half_width * table_res)) + 2;
    std::vector<float> table(static_cast<size_t>(table_len));
    const double i0_beta = bessel_i0(beta);
    for (int m = 0; m < table_len; ++m) {
        const double u = static_cast<double>(m) / table_res;
        double window = 0.0;
        if (u <= half_width) {
            const double r = u / half_width;
            window = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0_beta;
        }
        table[static_cast<size_t>(m)] = static_cast<float>(cutoff * sinc(cutoff * u) * window);
    }

    const int n_in = static_cast<int>(input.samples.size());
    const int n_out = static_cast<int>(std::llround(n_in * ratio));
    AudioData out;
    out.sample_rate = target_rate;
    out.samples.assign(static_cast<size_t>(n_out), 0.0f);

    const int taps = static_cast<int>(std::ceil(half_width));
    for (int i = 0; i < n_out; ++i) {
        const double t = i / ratio;  // position in input samples
        const int centre = static_cast<int>(std::floor(t));
        const int lo = std::max(0, centre - taps);
        const int hi = std::min(n_in - 1, centre + taps + 1);
        double acc = 0.0;
        for (int k = lo; k <= hi; ++k) {
            const double pos = std::fabs(t - k) * table_res;
            const int m = static_cast<int>(pos);
            if (m + 1 >= table_len) continue;
            const double frac = pos - m;
            const double h = table[static_cast<size_t>(m)] * (1.0 - frac) +
                             table[static_cast<size_t>(m) + 1] * frac;
            acc += h * input.samples[static_cast<size_t>(k)];
        }
        // The kernel carries a `cutoff` gain factor; normalize back to unit passband.
        out.samples[static_cast<size_t>(i)] = static_cast<float>(acc / cutoff);
    }
    return out;
}


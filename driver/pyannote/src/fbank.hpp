// kaldi fbank, aligned with torchaudio.compliance.kaldi.fbank
// (and the Python community1_sdk/fbank.py numpy port).
//
// Settings: 80 mel, frame 25ms/400, shift 10ms/160, hamming, snip_edges,
// dither=0, preemphasis 0.97, remove_dc_offset, use_power, log, + global
// mean-centering over frames (fbank_centering_span=None).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>

#include "kiss_fftr.h"

#include "params.hpp"

static const float EPS = 1.1920929e-07f;   // float32 epsilon

// One 10s window -> (998, 80) fbank, mean-centered over frames.
// `fbank` must be pre-allocated to n_frames * 80 floats.
static inline int kaldi_fbank(const Params& P, const float* wav, int n_samples,
                              float* fbank) {
    const int win = P.win_size, shift = P.win_shift, padded = P.padded;
    const int n_mel = 80;

    // wav is in [-1,1]; torchaudio multiplies by (1<<15)
    std::vector<float> x(n_samples);
    for (int i = 0; i < n_samples; ++i) x[i] = wav[i] * 32768.0f;

    const int m = 1 + (n_samples - win) / shift;   // snip_edges
    const int nfft = padded;

    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, nullptr, nullptr);
    std::vector<kiss_fft_scalar> f_in(padded);
    std::vector<kiss_fft_cpx> f_out(padded / 2 + 1);

    std::vector<float> frame(win);
    std::vector<double> col_sums(80, 0.0);

    #pragma omp parallel
    {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    kiss_fftr_cfg local_cfg = (tid == 0) ? cfg : kiss_fftr_alloc(nfft, 0, nullptr, nullptr);
    std::vector<kiss_fft_scalar> l_in(padded);
    std::vector<kiss_fft_cpx> l_out(padded / 2 + 1);
    std::vector<float> l_frame(win);
    #pragma omp for schedule(static)
    for (int t = 0; t < m; ++t) {
        const float* src = x.data() + t * shift;
        // copy frame + remove DC
        float mean = 0;
        for (int i = 0; i < win; ++i) mean += src[i];
        mean /= win;
        for (int i = 0; i < win; ++i) l_frame[i] = src[i] - mean;
        // preemphasis with replicate pad
        float prev = l_frame[0];
        for (int i = 0; i < win; ++i) {
            float cur = l_frame[i];
            l_frame[i] = cur - 0.97f * prev;
            prev = cur;
        }
        // hamming window + zero pad
        memset(l_in.data(), 0, padded * sizeof(float));
        for (int i = 0; i < win; ++i) l_in[i] = l_frame[i] * P.window[i];
        kiss_fftr(local_cfg, l_in.data(), l_out.data());
        // power spectrum (unnormalized rfft -> |X|^2, clamp eps)
        float pow_spec[513];
        for (int k = 0; k <= padded / 2; ++k) {
            float re = l_out[k].r, im = l_out[k].i;
            pow_spec[k] = std::max(re * re + im * im, EPS);
        }
        // mel filterbank (257 x 80, k-major for rank-1 updates)
        float mel[80] = {0};
        const float* mb = P.mel_banks.data();
        for (int k = 0; k < 257; ++k) {
            float p = pow_spec[k];
            const float* col = mb + (size_t)k * 80;
            for (int b = 0; b < 80; ++b) mel[b] += p * col[b];
        }
        for (int b = 0; b < 80; ++b)
            mel[b] = std::log(std::max(mel[b], EPS));
        float* out = fbank + t * 80;
        for (int b = 0; b < 80; ++b) out[b] = mel[b];
    }
    if (tid != 0) kiss_fftr_free(local_cfg);
    }  // end omp parallel
    kiss_fftr_free(cfg);
    // col sums for centering (serial, cheap)
    for (int t = 0; t < m; ++t)
        for (int b = 0; b < 80; ++b) col_sums[b] += fbank[t * 80 + b];

    // global mean centering over frames (per bin)
    for (int t = 0; t < m; ++t) {
        float* out = fbank + t * 80;
        for (int b = 0; b < 80; ++b) out[b] -= (float)(col_sums[b] / m);
    }
    return m;
}

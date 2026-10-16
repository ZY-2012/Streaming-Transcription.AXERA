// DiarEngine implementation (see diar_engine.hpp for the flow description).
#include "diar_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

namespace c1 {
extern double g_t_unfold, g_t_norm, g_t_ffn, g_t_cells, g_t_npu_other;

static const double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// sub-window cache (see header).  Keys are content hashes of the 10 s of
// samples; identical content in the overlap of consecutive sliding windows
// yields identical masks/embeddings, so serving from cache is bit-identical.
// ---------------------------------------------------------------------------
DiarEngine::CacheKey DiarEngine::HashSubWindow(const float* win) {
    // FNV-1a over the 16-bit linear view of the window, 4 samples (16 bytes)
    // per round to keep hashing cheap (~15 us for 10 s @ 16 kHz).
    std::uint64_t h = 1469598103934665603ULL;
    const int n_short = WINDOW_SAMPLES & ~3;
    const std::int16_t* v = reinterpret_cast<const std::int16_t*>(win);
    for (int i = 0; i < n_short * 2; i += 8) {
        for (int j = 0; j < 8; ++j)
            h = (h ^ (std::uint64_t)(std::uint16_t)v[i + j]) * 1099511628211ULL;
    }
    return h;
}

const DiarEngine::SubWinCache* DiarEngine::CacheGet(CacheKey key) {
    auto it = subwin_cache_.find(key);
    return it == subwin_cache_.end() ? nullptr : &it->second;
}

void DiarEngine::CachePut(CacheKey key, SubWinCache&& entry) {
    auto it = subwin_cache_.find(key);
    if (it != subwin_cache_.end()) {
        it->second = std::move(entry);
        return;
    }
    if (subwin_cache_order_.size() >= kCacheMaxEntries) {
        subwin_cache_.erase(subwin_cache_order_.front());
        subwin_cache_order_.pop_front();
    }
    subwin_cache_.emplace(key, std::move(entry));
    subwin_cache_order_.push_back(key);
}


// ---------------------------------------------------------------------------
// segmentation: one 10s window -> hard 3-speaker masks (589, 3)
// ---------------------------------------------------------------------------
void DiarEngine::SegmentWindow(const float* win, uint8_t* masks) {
    double mean = 0;
    for (int i = 0; i < WINDOW_SAMPLES; ++i) mean += win[i];
    mean /= WINDOW_SAMPLES;
    double var = 0;
    for (int i = 0; i < WINDOW_SAMPLES; ++i) { double d = win[i] - mean; var += d * d; }
    var = var / WINDOW_SAMPLES + 1e-5;
    double sd = std::sqrt(var);
    float nw = P.norm_w, nb = P.norm_b;
    float inv_sd = nw / (float)sd;

    constexpr int N_COLS = 15975, KERNEL = 251;
    if (buf_unfold.empty()) buf_unfold.resize(N_COLS * KERNEL);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N_COLS; ++j) {
        float* row = buf_unfold.data() + j * KERNEL;
        const float* src = win + 10 * j;
        for (int k = 0; k < KERNEL; ++k)
            row[k] = (float)((src[k] - mean) * inv_sd + nb);
    }

    if (buf_a.empty()) buf_a.resize(80 * 5325);
    seg_a.SetInputByName("unfolded", buf_unfold.data());
    seg_a.RunSync();
    seg_a.GetOutputByName("pool0", buf_a.data());
    inst_norm(buf_a.data(), 80, 5325, P.n0w, P.n0b, buf_a.data());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < buf_a.size(); ++i) buf_a[i] = leaky(buf_a[i]);

    if (buf_b1.empty()) buf_b1.resize(60 * 1773);
    seg_b1.SetInputByName("x", buf_a.data());
    seg_b1.RunSync();
    seg_b1.GetOutputByName("pool1", buf_b1.data());
    inst_norm(buf_b1.data(), 60, 1773, P.n1w, P.n1b, buf_b1.data());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < buf_b1.size(); ++i) buf_b1[i] = leaky(buf_b1[i]);

    if (buf_b2.empty()) buf_b2.resize(60 * 589);
    seg_b2.SetInputByName("x", buf_b1.data());
    seg_b2.RunSync();
    seg_b2.GetOutputByName("pool2", buf_b2.data());
    inst_norm(buf_b2.data(), 60, 589, P.n2w, P.n2b, buf_b2.data());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < buf_b2.size(); ++i) buf_b2[i] = leaky(buf_b2[i]);

    // ---- LSTM: 4 layers, 64-step cells, fwd+bwd per call ----
    std::vector<float> x(589 * 60);
    for (int t = 0; t < 589; ++t)
        for (int c = 0; c < 60; ++c) x[t * 60 + c] = buf_b2[c * 589 + t];

    if (buf_cell_x.empty()) buf_cell_x.resize(CELL_CHUNK * 512);
    for (int l = 0; l < 4; ++l) {
        const int in_dim = (l == 0) ? 60 : 256;
        const int T = 589;
        const int n_chunks = (T + CELL_CHUNK - 1) / CELL_CHUNK;
        static std::vector<float> h, c;
        std::vector<float> y((size_t)T * 256);
        std::vector<float> y_tmp(CELL_CHUNK * 256);
        std::vector<float> bwd_seq((size_t)T * in_dim);
        h.assign(256, 0.0f);
        c.assign(256, 0.0f);
        for (int t = 0; t < T; ++t)
            memcpy(bwd_seq.data() + (size_t)(T - 1 - t) * in_dim, x.data() + t * in_dim,
                   in_dim * sizeof(float));
        for (int s = 0; s < n_chunks; ++s) {
            int len = std::min(CELL_CHUNK, T - s * CELL_CHUNK);
            memset(buf_cell_x.data(), 0, (size_t)CELL_CHUNK * 2 * in_dim * sizeof(float));
            for (int i = 0; i < len; ++i) {
                memcpy(buf_cell_x.data() + (size_t)i * 2 * in_dim,
                       x.data() + (size_t)(s * CELL_CHUNK + i) * in_dim, in_dim * sizeof(float));
                memcpy(buf_cell_x.data() + (size_t)i * 2 * in_dim + in_dim,
                       bwd_seq.data() + (size_t)(s * CELL_CHUNK + i) * in_dim,
                       in_dim * sizeof(float));
            }
            cells[l].SetInputByName("x", buf_cell_x.data());
            cells[l].SetInputByName("h0", h.data());
            cells[l].SetInputByName("c0", c.data());
            cells[l].RunSync();
            cells[l].GetOutputByName("y0", y_tmp.data());
            cells[l].GetOutputByName("y1", y_tmp.data() + (CELL_CHUNK / 2) * 256);
            memcpy(y.data() + (size_t)s * CELL_CHUNK * 256, y_tmp.data(),
                   (size_t)len * 256 * sizeof(float));
            cells[l].GetOutputByName("hN", h.data());
            cells[l].GetOutputByName("cN", c.data());
        }
        std::vector<float> next((size_t)T * 256);
        for (int t = 0; t < T; ++t) {
            memcpy(next.data() + (size_t)t * 256, y.data() + (size_t)t * 256,
                   128 * sizeof(float));
            memcpy(next.data() + (size_t)t * 256 + 128,
                   y.data() + (size_t)(T - 1 - t) * 256 + 128, 128 * sizeof(float));
        }
        x.swap(next);
    }

    // ---- FFN + powerset ----
    static std::vector<float> h1, h2, logits;
    h1.resize(589 * 128);
    h2.resize(589 * 128);
    logits.resize(589 * 7);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < 589; ++t) {
        const float* xi = x.data() + t * 256;
        float* ho = h1.data() + t * 128;
        for (int o = 0; o < 128; ++o) ho[o] = P.l0b[o];
        for (int i = 0; i < 256; ++i) {
            float v = xi[i];
            const float* wr = P.l0w.data() + (size_t)i * 128;
            for (int o = 0; o < 128; ++o) ho[o] += v * wr[o];
        }
        for (int o = 0; o < 128; ++o) ho[o] = leaky(ho[o]);
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < 589; ++t) {
        const float* xi = h1.data() + t * 128;
        float* ho = h2.data() + t * 128;
        for (int o = 0; o < 128; ++o) ho[o] = P.l1b[o];
        for (int i = 0; i < 128; ++i) {
            float v = xi[i];
            const float* wr = P.l1w.data() + (size_t)i * 128;
            for (int o = 0; o < 128; ++o) ho[o] += v * wr[o];
        }
        for (int o = 0; o < 128; ++o) ho[o] = leaky(ho[o]);
    }
    for (int t = 0; t < 589; ++t) {
        const float* xi = h2.data() + t * 128;
        float* lo = logits.data() + t * 7;
        for (int o = 0; o < 7; ++o) lo[o] = P.cb[o];
        for (int i = 0; i < 128; ++i) {
            float v = xi[i];
            const float* wr = P.cw.data() + (size_t)i * 7;
            for (int o = 0; o < 7; ++o) lo[o] += v * wr[o];
        }
    }
    for (int t = 0; t < 589; ++t) {
        const float* lo = logits.data() + t * 7;
        int best = 0;
        for (int k = 1; k < 7; ++k) if (lo[k] > lo[best]) best = k;
        for (int s = 0; s < 3; ++s)
            masks[t * 3 + s] = (uint8_t)(P.mapping[best * 3 + s] > 0.5f);
    }
}

// ---------------------------------------------------------------------------
// embeddings
// ---------------------------------------------------------------------------
void DiarEngine::Embeddings(const WavData& wav, const std::vector<uint8_t>& seg_data,
                            int C, std::vector<float>& embs,
                            const std::vector<const SubWinCache*>& cached) {
    embs.assign((size_t)C * 3 * EMB_DIM, 0.0f);
    if (buf_fbank.empty()) buf_fbank.resize(998 * 80);
    static std::vector<float> crop, frames, x, wi, stats, e, cmask, used;
    crop.resize(WINDOW_SAMPLES);
    frames.resize(256 * 10 * 125);
    x.resize(2560 * 125);
    wi.resize(125);
    stats.resize(5120);
    e.resize(EMB_DIM);
    cmask.resize(SEG_FRAMES);
    used.resize(SEG_FRAMES);

    for (int c = 0; c < C; ++c) {
        // Cache hit: all three speaker embeddings are already final.
        if (c < (int)cached.size() && cached[c]) {
            memcpy(embs.data() + (size_t)c * 3 * EMB_DIM,
                   cached[c]->embs.data(), 3 * EMB_DIM * sizeof(float));
            continue;
        }
        int s0 = (int)(c * step_sec_ * SAMPLE_RATE);
        memset(crop.data(), 0, WINDOW_SAMPLES * sizeof(float));
        int avail = std::min(WINDOW_SAMPLES, (int)wav.samples.size() - s0);
        if (avail > 0) memcpy(crop.data(), wav.samples.data() + s0, avail * sizeof(float));
        bool fbank_done = false;
        // the conv stack only depends on the crop: run it at most once per window
        bool conv_done = false;
        for (int spk = 0; spk < 3; ++spk) {
            const uint8_t* base = seg_data.data() + (size_t)c * 589 * 3;
            double msum = 0, csum = 0;
            for (int t = 0; t < 589; ++t) {
                float m = base[t * 3 + spk] ? 1.0f : 0.0f;
                int active = 0;
                for (int s2 = 0; s2 < 3; ++s2) active += base[t * 3 + s2] ? 1 : 0;
                cmask[t] = active < 2 ? m : 0.0f;
                msum += m;
                csum += cmask[t];
            }
            if (csum > MIN_NUM_FRAMES)
                memcpy(used.data(), cmask.data(), 589 * sizeof(float));
            else
                for (int t = 0; t < 589; ++t) used[t] = base[t * 3 + spk] ? 1.0f : 0.0f;
            float usum = 0;
            for (int t = 0; t < 589; ++t) usum += used[t];
            if (usum == 0.0f) {
                memcpy(embs.data() + ((size_t)c * 3 + spk) * EMB_DIM,
                       P.seg1_b.data(), EMB_DIM * sizeof(float));
                continue;
            }
            if (!fbank_done) {
                kaldi_fbank(P, crop.data(), WINDOW_SAMPLES, buf_fbank.data());
                fbank_done = true;
            }
            if (!conv_done) {
                emb.SetInputByName("fbank", buf_fbank.data());
                emb.RunSync();
                emb.GetOutputByName("frames", frames.data());
                for (int f = 0; f < 2560; ++f)
                    memcpy(x.data() + f * 125, frames.data() + f * 125,
                           125 * sizeof(float));
                conv_done = true;
            }
            for (int i = 0; i < 125; ++i) wi[i] = used[(i * 589) / 125];
            double v1 = 1e-8, v2 = 0;
            for (int i = 0; i < 125; ++i) { v1 += wi[i]; v2 += (double)wi[i] * wi[i]; }
            #pragma omp parallel for schedule(static)
            for (int f = 0; f < 2560; ++f) {
                double mean = 0;
                const float* xr = x.data() + f * 125;
                for (int i = 0; i < 125; ++i) mean += (double)xr[i] * wi[i];
                mean /= v1;
                double var = 0;
                for (int i = 0; i < 125; ++i) {
                    double d = xr[i] - mean;
                    var += d * d * wi[i];
                }
                var /= (v1 - v2 / v1 + 1e-8);
                stats[f] = (float)mean;
                stats[2560 + f] = (float)std::sqrt(var);
            }
            for (int o = 0; o < 256; ++o) e[o] = P.seg1_b[o];
            for (int i = 0; i < 5120; ++i) {
                float v = stats[i];
                const float* wr = P.seg1_w.data() + (size_t)i * 256;
                for (int o = 0; o < 256; ++o) e[o] += v * wr[o];
            }
            memcpy(embs.data() + ((size_t)c * 3 + spk) * EMB_DIM, e.data(),
                   EMB_DIM * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// clustering: filter -> AHC -> PLDA -> VBx -> constrained assignment
// ---------------------------------------------------------------------------
void DiarEngine::Cluster(std::vector<float>& embs, const std::vector<uint8_t>& seg_data,
                         int C, std::vector<int>& hard_clusters, int& K,
                         std::vector<float>* centroids_out) {
    const int D = EMB_DIM;
    if (centroids_out) centroids_out->clear();
    std::vector<int> keep_idx;
    for (int c = 0; c < C; ++c) {
        for (int s = 0; s < 3; ++s) {
            int nclean = 0;
            const uint8_t* base = seg_data.data() + (size_t)c * 589 * 3;
            for (int t = 0; t < 589; ++t) {
                int active = 0;
                for (int s2 = 0; s2 < 3; ++s2) active += base[t * 3 + s2] ? 1 : 0;
                if (active == 1 && base[t * 3 + s]) nclean++;
            }
            if (nclean >= MIN_ACTIVE_RATIO * 589) keep_idx.push_back(c * 3 + s);
        }
    }
    const int N = (int)keep_idx.size();
    hard_clusters.assign((size_t)C * 3, 0);

    std::vector<float> train(N * D);
    for (int i = 0; i < N; ++i)
        memcpy(train.data() + i * D, embs.data() + (size_t)keep_idx[i] * D, D * sizeof(float));

    if (N < 2) {
        K = 1;
        // N == 0 happens for short segments whose window padding makes the
        // segmentation produce no clean single-speaker frames; the original
        // RTTM path degenerates to one speaker.  Provide a best-effort
        // centroid from all embeddings so cross-segment identity still works.
        if (centroids_out) {
            if (N == 1) {
                centroids_out->assign(train.begin(), train.end());
            } else if (!embs.empty()) {
                std::vector<double> mean(D, 0.0);
                const int count = static_cast<int>(embs.size() / D);
                for (int i = 0; i < count; ++i)
                    for (int d = 0; d < D; ++d) mean[d] += embs[(size_t)i * D + d];
                std::vector<float> center(D);
                for (int d = 0; d < D; ++d)
                    center[d] = static_cast<float>(mean[d] / std::max(1, count));
                *centroids_out = std::move(center);
            }
        }
        return;
    }

    // ---- AHC centroid linkage (normalized embeddings) + fcluster(0.6) ----
    std::vector<float> train_norm(N * D);
    memcpy(train_norm.data(), train.data(), N * D * sizeof(float));
    l2norm_rows(train_norm, N, D);

    std::vector<double> d2((size_t)N * N, 1e30);
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            double dd = 0;
            const float* a = train_norm.data() + i * D;
            const float* b = train_norm.data() + j * D;
            for (int d = 0; d < D; ++d) { double t = a[d] - b[d]; dd += t * t; }
            d2[i * N + j] = d2[j * N + i] = dd;
        }
    std::vector<int> sz(N, 1), uf(N);
    std::vector<bool> alive(N, true);
    for (int i = 0; i < N; ++i) uf[i] = i;
    auto findf = [&](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    for (int m = 0; m < N - 1; ++m) {
        int bi = -1, bj = -1;
        double best = 1e300;
        for (int i = 0; i < N; ++i) {
            if (!alive[i]) continue;
            for (int j = i + 1; j < N; ++j) {
                if (!alive[j]) continue;
                if (d2[i * N + j] < best) { best = d2[i * N + j]; bi = i; bj = j; }
            }
        }
        if (std::sqrt(best) > 0.6) break;
        uf[findf(bj)] = findf(bi);
        int ni = sz[bi], nj = sz[bj], nk = ni + nj;
        for (int k = 0; k < N; ++k) {
            if (!alive[k] || k == bi || k == bj) continue;
            double dik = d2[bi * N + k], djk = d2[bj * N + k];
            d2[bi * N + k] = d2[k * N + bi] =
                (ni * dik + nj * djk) / nk - (double)(ni * nj) / (nk * nk) * best;
        }
        sz[bi] = nk;
        alive[bj] = false;
    }
    std::vector<int> label_of(N, -1), ahc_labels(N);
    int nl = 0;
    for (int i = 0; i < N; ++i) {
        int r = findf(i);
        if (label_of[r] < 0) label_of[r] = nl++;
        ahc_labels[i] = label_of[r];
    }
    const int S = *std::max_element(ahc_labels.begin(), ahc_labels.end()) + 1;

    // ---- PLDA transform (xvec_tf then plda_tf) ----
    std::vector<float> fea(N * 128);
    for (int i = 0; i < N; ++i) {
        const float* xv = train.data() + i * D;
        std::vector<float> tmp(D), tmp2(128);
        double n = 0;
        for (int d = 0; d < D; ++d) {
            tmp[d] = xv[d] - P.mean1[d];
            n += (double)tmp[d] * tmp[d];
        }
        n = std::sqrt(n) + 1e-30;
        double sq = std::sqrt((double)D);
        for (int d = 0; d < D; ++d) tmp[d] = (float)(tmp[d] / n * sq);
        for (int o = 0; o < 128; ++o) {
            double acc = 0;
            for (int d = 0; d < D; ++d) acc += (double)tmp[d] * P.lda[d * 128 + o];
            tmp2[o] = (float)(acc - P.mean2[o]);
        }
        double n2 = 0;
        for (int o = 0; o < 128; ++o) n2 += (double)tmp2[o] * tmp2[o];
        n2 = std::sqrt(n2) + 1e-30;
        double sq2 = std::sqrt(128.0);
        for (int o = 0; o < 128; ++o)
            tmp2[o] = (float)(tmp2[o] / n2 * sq2 - P.plda_mu[o]);
        for (int o = 0; o < 128; ++o) {
            double acc = 0;
            for (int d = 0; d < 128; ++d)
                acc += (double)tmp2[d] * P.plda_tr[o * 128 + d];
            fea[i * 128 + o] = (float)acc;
        }
    }

    // ---- VBx ----
    std::vector<double> q((size_t)N * S, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int s = 0; s < S; ++s)
            q[i * S + s] = std::exp((s == ahc_labels[i]) ? 7.0 : 0.0);
        double sum = 0;
        for (int s = 0; s < S; ++s) sum += q[i * S + s];
        for (int s = 0; s < S; ++s) q[i * S + s] /= sum;
    }
    std::vector<double> pi(S, 1.0 / S);
    std::vector<double> G(N);
    for (int i = 0; i < N; ++i) {
        double s2 = 0;
        for (int d = 0; d < 128; ++d) s2 += (double)fea[i * 128 + d] * fea[i * 128 + d];
        G[i] = -0.5 * (s2 + 128.0 * std::log(2.0 * PI));
    }
    std::vector<double> Phi(128), V(128), rho(N * 128);
    for (int d = 0; d < 128; ++d) {
        Phi[d] = P.plda_psi[d];
        V[d] = std::sqrt(std::max(Phi[d], 0.0));
    }
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < 128; ++d) rho[i * 128 + d] = fea[i * 128 + d] * V[d];
    const double Fa = 0.07, Fb = 0.8;
    std::vector<double> invL(S * 128), alpha(S * 128);
    std::vector<double> logp(N * S), log_px(N);
    double prev_elbo = -1e300;
    for (int it = 0; it < 20; ++it) {
        std::vector<double> gsum(S, 0.0);
        for (int i = 0; i < N; ++i)
            for (int s = 0; s < S; ++s) gsum[s] += q[i * S + s];
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < 128; ++d)
                invL[s * 128 + d] = 1.0 / (1.0 + Fa / Fb * gsum[s] * Phi[d]);
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < 128; ++d) {
                double acc = 0;
                for (int i = 0; i < N; ++i) acc += q[i * S + s] * rho[i * 128 + d];
                alpha[s * 128 + d] = Fa / Fb * invL[s * 128 + d] * acc;
            }
        std::vector<double> lpi(S);
        for (int s = 0; s < S; ++s) lpi[s] = std::log(pi[s] + 1e-8);
        for (int i = 0; i < N; ++i) {
            double mx = -1e300;
            for (int s = 0; s < S; ++s) {
                double acc = 0, phidot = 0;
                for (int d = 0; d < 128; ++d) {
                    acc += rho[i * 128 + d] * alpha[s * 128 + d];
                    double a2 = alpha[s * 128 + d] * alpha[s * 128 + d];
                    phidot += (invL[s * 128 + d] + a2) * Phi[d];
                }
                logp[i * S + s] = Fa * (acc - 0.5 * phidot) + G[i];
                mx = std::max(mx, logp[i * S + s]);
            }
            double sum = 0;
            for (int s = 0; s < S; ++s)
                sum += std::exp(logp[i * S + s] + lpi[s] - mx);
            log_px[i] = mx + std::log(sum);
            for (int s = 0; s < S; ++s)
                q[i * S + s] = std::exp(logp[i * S + s] + lpi[s] - log_px[i]);
        }
        double psum = 0;
        for (int s = 0; s < S; ++s) {
            pi[s] = 0;
            for (int i = 0; i < N; ++i) pi[s] += q[i * S + s];
            psum += pi[s];
        }
        for (int s = 0; s < S; ++s) pi[s] /= psum;
        double elbo = 0;
        for (int i = 0; i < N; ++i) elbo += log_px[i];
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < 128; ++d) {
                double a2 = alpha[s * 128 + d] * alpha[s * 128 + d];
                elbo += Fb * 0.5 * (std::log(invL[s * 128 + d]) - invL[s * 128 + d] - a2 + 1);
            }
        if (it > 0 && elbo - prev_elbo < 1e-4) break;
        prev_elbo = elbo;
    }
    std::vector<int> kept;
    for (int s = 0; s < S; ++s)
        if (pi[s] > 1e-7) kept.push_back(s);
    K = (int)kept.size();
    if (K == 0) { K = 1; kept.push_back(0); }

    // centroids = W.T @ train / W.sum(0).T
    std::vector<float> centroids((size_t)K * D, 0.0f);
    {
        std::vector<double> wsum(K, 0.0);
        for (int k = 0; k < K; ++k) {
            int s = kept[k];
            for (int i = 0; i < N; ++i) {
                double w = q[i * S + s];
                for (int d = 0; d < D; ++d)
                    centroids[k * D + d] += (float)(w * train[i * D + d]);
                wsum[k] += w;
            }
            for (int d = 0; d < D; ++d)
                centroids[k * D + d] /= (float)(wsum[k] + 1e-30);
        }
    }
    if (centroids_out) *centroids_out = centroids;
    // soft = 2 - cdist(cosine) = 1 + cos
    std::vector<float> soft((size_t)C * 3 * K);
    for (int i = 0; i < C * 3; ++i) {
        const float* e = embs.data() + (size_t)i * D;
        double en = 0;
        for (int d = 0; d < D; ++d) en += (double)e[d] * e[d];
        en = std::sqrt(en) + 1e-30;
        for (int k = 0; k < K; ++k) {
            const float* c = centroids.data() + (size_t)k * D;
            double cn = 0, dot = 0;
            for (int d = 0; d < D; ++d) {
                cn += (double)c[d] * c[d];
                dot += (double)e[d] * c[d];
            }
            cn = std::sqrt(cn) + 1e-30;
            soft[(size_t)i * K + k] = (float)(1.0 + dot / (en * cn));
        }
    }
    float gmin = 1e30;
    for (size_t i = 0; i < soft.size(); ++i) gmin = std::min(gmin, soft[i]);
    const float CNST = gmin - 1.0f;
    for (int c = 0; c < C; ++c) {
        const uint8_t* base = seg_data.data() + (size_t)c * 589 * 3;
        bool any_active = false;
        for (int t = 0; t < 589 && !any_active; ++t)
            for (int s = 0; s < 3; ++s)
                if (base[t * 3 + s]) { any_active = true; break; }
        float cost[3][16];
        for (int s = 0; s < 3; ++s)
            for (int k = 0; k < K && k < 16; ++k)
                cost[s][k] = any_active ? soft[((size_t)c * 3 + s) * K + k] : CNST;
        int best_assign[3] = {0, 0, 0};
        if (K >= 3 && K <= 16) {
            float best_total = -1e30f;
            for (int k0 = 0; k0 < K; ++k0)
                for (int k1 = 0; k1 < K; ++k1) {
                    if (k1 == k0) continue;
                    for (int k2 = 0; k2 < K; ++k2) {
                        if (k2 == k0 || k2 == k1) continue;
                        float tot = cost[0][k0] + cost[1][k1] + cost[2][k2];
                        if (tot > best_total) {
                            best_total = tot;
                            best_assign[0] = k0; best_assign[1] = k1; best_assign[2] = k2;
                        }
                    }
                }
        } else {
            for (int s = 0; s < 3; ++s) {
                int bk = 0;
                for (int k = 1; k < K; ++k)
                    if (cost[s][k] > cost[s][bk]) bk = k;
                best_assign[s] = bk;
            }
        }
        for (int s = 0; s < 3; ++s) hard_clusters[c * 3 + s] = best_assign[s];
    }
}

// ---------------------------------------------------------------------------
// diart-style incremental clustering (config-gated alternative to Cluster)
// ---------------------------------------------------------------------------
void DiarEngine::ClusterIncremental(std::vector<float>& embs,
                                    const std::vector<uint8_t>& seg_data,
                                    int C, std::vector<int>& hard_clusters, int& K,
                                    std::vector<float>* centroids_out,
                                    const IncrementalParams& p) {
    const int D = EMB_DIM;
    const int PD = p.use_plda ? 128 : D;  // matching feature dim (PLDA xvector or raw)
    if (centroids_out) centroids_out->clear();

    // Project every local-speaker embedding once (raw copy or PLDA transform).
    std::vector<double> feats((size_t)C * 3 * PD);
    for (int i = 0; i < C * 3; ++i) {
        const float* xv = embs.data() + (size_t)i * D;
        double* out = feats.data() + (size_t)i * PD;
        if (!p.use_plda) {
            for (int d = 0; d < D; ++d) out[d] = xv[d];
        } else {
            double tmp[256], tmp2[128];
            double n = 0;
            for (int d = 0; d < D; ++d) { tmp[d] = xv[d] - P.mean1[d]; n += tmp[d] * tmp[d]; }
            n = std::sqrt(n) + 1e-30;
            double sq = std::sqrt((double)D);
            for (int d = 0; d < D; ++d) tmp[d] = tmp[d] / n * sq;
            for (int o = 0; o < 128; ++o) {
                double acc = 0;
                for (int d = 0; d < D; ++d) acc += tmp[d] * P.lda[d * 128 + o];
                tmp2[o] = acc - P.mean2[o];
            }
            double n2 = 0;
            for (int o = 0; o < 128; ++o) n2 += tmp2[o] * tmp2[o];
            n2 = std::sqrt(n2) + 1e-30;
            double sq2 = std::sqrt(128.0);
            for (int o = 0; o < 128; ++o) tmp2[o] = tmp2[o] / n2 * sq2 - P.plda_mu[o];
            for (int o = 0; o < 128; ++o) {
                double acc = 0;
                for (int d = 0; d < 128; ++d) acc += tmp2[d] * P.plda_tr[o * 128 + d];
                out[o] = acc;
            }
        }
    }

    // Running global centroids live in the members inc_fc_ / inc_rc_ so a
    // streaming caller can submit sub-windows one at a time and the state
    // persists across Run() calls (see ResetIncremental).

    hard_clusters.assign((size_t)C * 3, -1);

    auto cosd = [](const double* a, const double* b, int dim) {
        double dot = 0, na = 0, nb = 0;
        for (int d = 0; d < dim; ++d) { dot += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; }
        return 1.0 - dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    };

    for (int c = 0; c < C; ++c) {
        const uint8_t* base = seg_data.data() + (size_t)c * 589 * 3;
        int act_frames[3] = {0, 0, 0};
        for (int t = 0; t < 589; ++t)
            for (int s = 0; s < 3; ++s)
                if (base[t * 3 + s]) act_frames[s]++;

        // active local speakers (hard masks: any active frame), long speakers
        // (activation ratio >= rho_update, eligible to create/update centroids).
        std::vector<int> al, ll;
        for (int s = 0; s < 3; ++s) {
            if (act_frames[s] > 0) al.push_back(s);  // tau_active (< 1) <=> any frame
            if (act_frames[s] / 589.0 >= p.rho_update) ll.push_back(s);
        }

        int assign[3] = {-1, -1, -1};
        static const bool kDbgInc = std::getenv("DIAR_DBG") != nullptr;
        char dbgbuf[256];
        if (kDbgInc) {
            double rho[3] = {0, 0, 0};
            for (int s = 0; s < 3; ++s) rho[s] = act_frames[s] / 589.0;
            std::snprintf(dbgbuf, sizeof dbgbuf,
                          "[inc-dbg] win c=%d act=%.2f/%.2f/%.2f", c, rho[0], rho[1], rho[2]);
        } else dbgbuf[0] = 0;
        if (inc_fc_.empty()) {
            for (int s : al) {
                int g = (int)inc_fc_.size();
                const double* f = feats.data() + (size_t)(c * 3 + s) * PD;
                const float* e = embs.data() + (size_t)(c * 3 + s) * D;
                inc_fc_.push_back(std::vector<double>(f, f + PD));
                inc_rc_.push_back(std::vector<double>(e, e + D));
                assign[s] = g;
                if (kDbgInc)
                    std::fprintf(stderr, "%s trk%d FIRST->g%d\n", dbgbuf, s, g);
            }
        } else {
            double near_d[3] = {1e30, 1e30, 1e30};
            int near_g[3] = {-1, -1, -1};
            for (int s : al) {
                const double* f = feats.data() + (size_t)(c * 3 + s) * PD;
                int bg = -1;
                double bd = 1e30;
                for (int g = 0; g < (int)inc_fc_.size(); ++g) {
                    double d = cosd(f, inc_fc_[g].data(), PD);
                    if (d < bd) { bd = d; bg = g; }
                }
                if (bg >= 0 && bd <= p.delta_new) assign[s] = bg;
                near_d[s] = bd; near_g[s] = bg;
                if (kDbgInc)
                    std::fprintf(stderr, "%s trk%d near g%d d=%.3f %s\n", dbgbuf, s, bg, bd,
                                 assign[s] >= 0 ? "MATCH" : "NEW?");
            }
            for (int s : al) {
                if (assign[s] >= 0) continue;
                bool is_long = std::find(ll.begin(), ll.end(), s) != ll.end();
                if ((int)inc_fc_.size() < p.max_speakers && is_long &&
                    near_d[s] > p.delta_min) {
                    int g = (int)inc_fc_.size();
                    const double* f = feats.data() + (size_t)(c * 3 + s) * PD;
                    const float* e = embs.data() + (size_t)(c * 3 + s) * D;
                    inc_fc_.push_back(std::vector<double>(f, f + PD));
                    inc_rc_.push_back(std::vector<double>(e, e + D));
                    assign[s] = g;
                    if (kDbgInc)
                        std::fprintf(stderr, "%s trk%d LONG->new g%d (d=%.3f>%.2f)\n",
                                     dbgbuf, s, g, near_d[s], p.delta_min);
                } else if (!inc_fc_.empty()) {
                    const double* f = feats.data() + (size_t)(c * 3 + s) * PD;
                    int bg = 0;
                    double bd = 1e30;
                    for (int g = 0; g < (int)inc_fc_.size(); ++g) {
                        double d = cosd(f, inc_fc_[g].data(), PD);
                        if (d < bd) { bd = d; bg = g; }
                    }
                    assign[s] = bg;
                    if (kDbgInc)
                        std::fprintf(stderr, "%s trk%d %s g%d d=%.3f\n",
                                     dbgbuf, s,
                                     is_long ? "DRIFT->absorb" : "SHORT->absorb",
                                     bg, bd);
                }
            }
            for (int s : ll) {
                if (assign[s] < 0) continue;
                const double* f = feats.data() + (size_t)(c * 3 + s) * PD;
                const float* e = embs.data() + (size_t)(c * 3 + s) * D;
                for (int d = 0; d < PD; ++d) inc_fc_[assign[s]][d] += f[d];
                for (int d = 0; d < D; ++d) inc_rc_[assign[s]][d] += e[d];
            }
        }

        for (int s = 0; s < 3; ++s) hard_clusters[c * 3 + s] = assign[s];
    }

    K = (int)inc_fc_.size();
    if (centroids_out && K > 0) {
        centroids_out->assign((size_t)K * D, 0.0f);
        for (int g = 0; g < K; ++g) {
            double n = 0;
            for (int d = 0; d < D; ++d) n += inc_rc_[g][d] * inc_rc_[g][d];
            n = std::sqrt(n) + 1e-30;
            for (int d = 0; d < D; ++d)
                (*centroids_out)[(size_t)g * D + d] = (float)(inc_rc_[g][d] / n);
        }
    }
}

void DiarEngine::ResetIncremental() {
    inc_fc_.clear();
    inc_rc_.clear();
}

// ---------------------------------------------------------------------------
// top-level run
// ---------------------------------------------------------------------------
void DiarEngine::Run(const WavData& wav, float step_sec,
                     std::vector<Segment>& rttm,
                     std::vector<Part>* parts,
                     std::vector<float>* centroids,
                     int* num_clusters,
                     int* cache_hits,
                     const IncrementalParams* incremental) {
    step_sec_ = step_sec;

    const int step_samples = (int)(step_sec * SAMPLE_RATE);
    const int n = (int)wav.samples.size();
    int n_full = 0;
    if (n >= WINDOW_SAMPLES) n_full = 1 + (n - WINDOW_SAMPLES) / step_samples;
    int rem = (n >= WINDOW_SAMPLES) ? (n - WINDOW_SAMPLES) % step_samples : n;
    int C = n_full + ((n < WINDOW_SAMPLES || rem > 0) ? 1 : 0);

    std::vector<uint8_t> seg_data((size_t)C * 589 * 3, 0);
    std::vector<float> win(WINDOW_SAMPLES);
    std::vector<const SubWinCache*> cached(C, nullptr);
    std::vector<CacheKey> keys(C, 0);
    int hits = 0;
    for (int c = 0; c < C; ++c) {
        int s0 = c * step_samples;
        memset(win.data(), 0, WINDOW_SAMPLES * sizeof(float));
        int avail = std::min(WINDOW_SAMPLES, n - s0);
        if (avail > 0) memcpy(win.data(), wav.samples.data() + s0, avail * sizeof(float));
        // Only full (unpadded) sub-windows are cacheable: their result is a
        // pure function of the 10 s of samples.
        if (avail == WINDOW_SAMPLES) {
            keys[c] = HashSubWindow(win.data());
            if (const SubWinCache* hit = CacheGet(keys[c])) {
                memcpy(seg_data.data() + (size_t)c * 589 * 3, hit->masks.data(),
                       (size_t)589 * 3);
                cached[c] = hit;
                ++hits;
                continue;
            }
        }
        SegmentWindow(win.data(), seg_data.data() + (size_t)c * 589 * 3);
    }
    if (cache_hits) *cache_hits = hits;

    float t_end = 10.0f + (C - 1) * step_sec + 0.5f * FRAME_DUR;
    int T = closest_frame(t_end) + 1;
    std::vector<float> agg(T, 0.0f), ncont(T, 0.0f);
    std::vector<uint8_t> count(T, 0);
    for (int c = 0; c < C; ++c) {
        int sf = closest_frame(c * step_sec + 0.5f * FRAME_DUR);
        const uint8_t* base = seg_data.data() + (size_t)c * 589 * 3;
        for (int t = 0; t < 589; ++t) {
            int active = 0;
            for (int s = 0; s < 3; ++s) active += base[t * 3 + s] ? 1 : 0;
            agg[sf + t] += active;
            ncont[sf + t] += 1;
        }
    }
    int max_count = 0;
    for (int t = 0; t < T; ++t) {
        int v = (int)std::lrint(agg[t] / std::max(ncont[t], 1e-12f));
        count[t] = (uint8_t)v;
        max_count = std::max(max_count, v);
    }
    if (max_count == 0) {
        rttm.clear();
        if (parts) parts->clear();
        if (centroids) centroids->clear();
        if (num_clusters) *num_clusters = 0;
        return;
    }

    std::vector<float> embs;
    Embeddings(wav, seg_data, C, embs, cached);

    // Store newly computed full sub-windows for the next sliding window.
    // Skipped when the request degenerated before embeddings ran (embs
    // empty); those windows simply recompute next time.
    if (embs.size() == (size_t)C * 3 * EMB_DIM) {
        for (int c = 0; c < C; ++c) {
            if (cached[c] || keys[c] == 0) continue;
            SubWinCache entry;
            entry.masks.assign(seg_data.data() + (size_t)c * 589 * 3,
                               seg_data.data() + (size_t)(c + 1) * 589 * 3);
            entry.embs.assign(embs.data() + (size_t)c * 3 * EMB_DIM,
                              embs.data() + (size_t)(c + 1) * 3 * EMB_DIM);
            CachePut(keys[c], std::move(entry));
        }
    }

    std::vector<int> hard;
    int K = 0;
    if (incremental)
        ClusterIncremental(embs, seg_data, C, hard, K, centroids, *incremental);
    else
        Cluster(embs, seg_data, C, hard, K, centroids);
    if (K <= 0) {
        rttm.clear();
        if (parts) parts->clear();
        if (centroids) centroids->clear();
        if (num_clusters) *num_clusters = 0;
        return;
    }
    if (num_clusters) *num_clusters = K;

    std::vector<float> act((size_t)T * K, 0.0f);
    for (int c = 0; c < C; ++c) {
        int sf = closest_frame(c * step_sec + 0.5f * FRAME_DUR);
        const uint8_t* base = seg_data.data() + (size_t)c * 589 * 3;
        for (int t = 0; t < 589; ++t) {
            for (int k = 0; k < K; ++k) {
                float v = 0;
                for (int s = 0; s < 3; ++s)
                    if (hard[c * 3 + s] == k && base[t * 3 + s]) v = 1.0f;
                act[(size_t)(sf + t) * K + k] += v;
            }
        }
    }
    std::vector<uint8_t> binary((size_t)T * K, 0);
    for (int t = 0; t < T; ++t) {
        std::vector<std::pair<float, int>> order(K);
        for (int k = 0; k < K; ++k) order[k] = {-act[(size_t)t * K + k], k};
        std::stable_sort(order.begin(), order.end());
        int ctop = std::min((int)count[t], K);
        for (int i = 0; i < ctop; ++i) binary[(size_t)t * K + order[i].second] = 1;
    }

    rttm.clear();
    for (int k = 0; k < K; ++k) {
        bool active = false;
        float start = 0;
        for (int t = 0; t < T; ++t) {
            bool on = binary[(size_t)t * K + k] != 0;
            if (on && !active) { start = t * FRAME_STEP + 0.5f * FRAME_DUR; active = true; }
            if (!on && active) {
                float end = t * FRAME_STEP + 0.5f * FRAME_DUR;
                if (end > start) rttm.push_back({start, end - start, k});
                active = false;
            }
        }
        if (active) {
            float end = (T - 1) * FRAME_STEP + 0.5f * FRAME_DUR;
            if (end > start) rttm.push_back({start, end - start, k});
        }
    }
    std::stable_sort(rttm.begin(), rttm.end(),
                     [](const Segment& a, const Segment& b) { return a.start < b.start; });

    if (parts != nullptr) {
        // Reconstruct a non-overlapping speaker sequence from the per-frame
        // activity so ASR can slice the WAV on speaker turns.  A frame that
        // is still covered by the running speaker stays with it; otherwise
        // the longest active run at that frame wins.  Undecided frames
        // inherit the nearest label, so the ranges partition [0, N).
        std::vector<int> label(T, -1);
        for (int t = 0; t < T; ++t) {
            const int prev = t > 0 ? label[t - 1] : -1;
            int chosen = -1;
            if (prev >= 0 && prev < K && binary[(size_t)t * K + prev]) chosen = prev;
            if (chosen < 0) {
                int best_run = -1;
                for (int k = 0; k < K; ++k) {
                    if (!binary[(size_t)t * K + k]) continue;
                    int a = t, b = t;
                    while (a > 0 && binary[(size_t)(a - 1) * K + k]) --a;
                    while (b + 1 < T && binary[(size_t)(b + 1) * K + k]) ++b;
                    const int run = b - a + 1;
                    if (run > best_run) { best_run = run; chosen = k; }
                }
            }
            label[t] = chosen;
        }
        int current = -1;
        for (int t = 0; t < T; ++t) {
            if (label[t] >= 0) current = label[t];
            else if (current >= 0) label[t] = current;
        }
        current = -1;
        for (int t = T - 1; t >= 0; --t) {
            if (label[t] >= 0) current = label[t];
            else if (current >= 0) label[t] = current;
        }

        const std::size_t n_samples = wav.samples.size();
        auto frame_boundary = [T, n_samples](int frame) -> std::size_t {
            if (frame <= 0) return 0;
            if (frame >= T) return n_samples;
            const double seconds = frame * (double)FRAME_STEP + 0.5 * (double)FRAME_DUR;
            long long sample = (long long)std::llround(seconds * SAMPLE_RATE);
            if (sample < 0) sample = 0;
            if ((unsigned long long)sample > n_samples)
                sample = (long long)n_samples;
            return (std::size_t)sample;
        };

        parts->clear();
        int t0 = 0;
        while (t0 < T) {
            int t1 = t0;
            while (t1 < T && label[t1] == label[t0]) ++t1;
            const std::size_t begin = frame_boundary(t0);
            const std::size_t end = frame_boundary(t1);
            if (end > begin && label[t0] >= 0) {
                if ((end - begin) < MIN_PART_SAMPLES && !parts->empty()) {
                    // Absorb short runs (segmentation flicker or overlap
                    // tail) into the preceding speaker part.
                    parts->back().end = end;
                } else {
                    parts->push_back({begin, end, label[t0]});
                }
            }
            t0 = t1;
        }
        // Merge adjacent ranges that share a cluster after absorption.
        std::vector<Part> compact;
        for (const Part& p : *parts) {
            if (p.end <= p.begin) continue;
            if (!compact.empty() && compact.back().cluster == p.cluster &&
                compact.back().end >= p.begin) {
                compact.back().end = std::max(compact.back().end, p.end);
                continue;
            }
            compact.push_back(p);
        }
        parts->swap(compact);
        if (!parts->empty()) {
            parts->front().begin = 0;
            parts->back().end = n_samples;
        }
    }
}

void DiarEngine::DumpFeatures(const WavData& wav, float step_sec,
                              const std::string& path) {
    step_sec_ = step_sec;

    // Same sub-window geometry as Run(): `step_sec` stride, 10 s windows.
    const int step_samples = (int)(step_sec * SAMPLE_RATE);
    const int n = (int)wav.samples.size();
    int n_full = 0;
    if (n >= WINDOW_SAMPLES) n_full = 1 + (n - WINDOW_SAMPLES) / step_samples;
    int rem = (n >= WINDOW_SAMPLES) ? (n - WINDOW_SAMPLES) % step_samples : n;
    int C = n_full + ((n < WINDOW_SAMPLES || rem > 0) ? 1 : 0);

    std::vector<uint8_t> seg_data((size_t)C * 589 * 3, 0);
    std::vector<float> win(WINDOW_SAMPLES);
    std::vector<const SubWinCache*> cached(C, nullptr);
    std::vector<CacheKey> keys(C, 0);
    for (int c = 0; c < C; ++c) {
        int s0 = c * step_samples;
        memset(win.data(), 0, WINDOW_SAMPLES * sizeof(float));
        int avail = std::min(WINDOW_SAMPLES, n - s0);
        if (avail > 0)
            memcpy(win.data(), wav.samples.data() + s0, avail * sizeof(float));
        if (avail == WINDOW_SAMPLES) {
            keys[c] = HashSubWindow(win.data());
            if (const SubWinCache* hit = CacheGet(keys[c])) {
                memcpy(seg_data.data() + (size_t)c * 589 * 3, hit->masks.data(),
                       (size_t)589 * 3);
                cached[c] = hit;
                continue;
            }
        }
        SegmentWindow(win.data(), seg_data.data() + (size_t)c * 589 * 3);
    }

    std::vector<float> embs;
    Embeddings(wav, seg_data, C, embs, cached);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[diar] DumpFeatures: cannot write %s\n",
                     path.c_str());
        return;
    }
    const char magic[8] = {'C', '1', 'D', 'I', 'A', 'R', '1', '\0'};
    std::fwrite(magic, 1, 8, f);
    uint32_t C32 = (uint32_t)C;
    uint32_t step32 = (uint32_t)step_samples;
    std::fwrite(&C32, 4, 1, f);
    std::fwrite(&step32, 4, 1, f);
    for (int c = 0; c < C; ++c) {
        uint32_t s0 = (uint32_t)(c * step_samples);
        std::fwrite(&s0, 4, 1, f);
        std::fwrite(seg_data.data() + (size_t)c * 589 * 3, 1, (size_t)589 * 3, f);
        std::fwrite(embs.data() + (size_t)c * 3 * EMB_DIM, sizeof(float),
                    (size_t)3 * EMB_DIM, f);
    }
    std::fclose(f);
    std::fprintf(stderr, "[diar] DumpFeatures: %s <- C=%d step=%d\n",
                 path.c_str(), C, step_samples);
}

}  // namespace c1

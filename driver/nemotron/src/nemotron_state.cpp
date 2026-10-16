#include "nemotron_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace {
const float kNegInf = -std::numeric_limits<float>::infinity();
const int kMaxIndex = 99999;
const float kLogHalf = std::log(0.5f);

std::vector<int> topk_indices_desc(const std::vector<float>& values, int count, int k) {
    std::vector<int> indices(static_cast<std::size_t>(count));
    std::iota(indices.begin(), indices.end(), 0);
    if (k >= count) return indices;
    // Stable sort: equal scores keep the smaller index (matches the numpy port).
    std::stable_sort(indices.begin(), indices.end(),
                     [&](int a, int b) { return values[a] > values[b]; });
    indices.resize(static_cast<std::size_t>(k));
    return indices;
}

// Returns sorted frame indices selected for the speaker cache, plus the disabled mask.
// ``scores`` is frame-major (t * n_spk + s); selection uses speaker-major flat indices
// (s * n_frames + t) with a stable tie-break to match the numpy implementation.
void get_topk_indices(const std::vector<float>& scores, int n_frames, int n_spk, int spkcache_len,
                      int sil_frames_per_spk, std::vector<int>& topk_sorted, std::vector<char>& is_disabled) {
    const int total = n_frames * n_spk;
    std::vector<int> flat(total);
    for (int s = 0; s < n_spk; ++s) {
        for (int t = 0; t < n_frames; ++t) flat[static_cast<std::size_t>(s) * n_frames + t] = s * n_frames + t;
    }
    const int k = std::min(spkcache_len, total);
    std::stable_sort(flat.begin(), flat.end(), [&](int a, int b) {
        const float sa = scores[static_cast<std::size_t>(a % n_frames) * n_spk + a / n_frames];
        const float sb = scores[static_cast<std::size_t>(b % n_frames) * n_spk + b / n_frames];
        return sa > sb;
    });
    flat.resize(static_cast<std::size_t>(k));

    std::vector<int> indices(flat.size());
    for (std::size_t i = 0; i < flat.size(); ++i) {
        const float value = scores[static_cast<std::size_t>(flat[i] % n_frames) * n_spk + flat[i] / n_frames];
        indices[i] = (value != kNegInf) ? flat[i] : kMaxIndex;
    }
    std::sort(indices.begin(), indices.end());

    const int n_frames_no_sil = n_frames - sil_frames_per_spk;
    topk_sorted.resize(indices.size());
    is_disabled.assign(indices.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const int index = indices[i];
        const bool sentinel = (index == kMaxIndex);
        // Speaker-major flat index (s * n_frames + t): frame is index % n_frames.
        const int frame = sentinel ? 0 : index % n_frames;
        const char disabled = (sentinel || frame >= n_frames_no_sil) ? 1 : 0;
        topk_sorted[i] = disabled ? 0 : frame;
        is_disabled[i] = disabled;
    }
}

void compress_spkcache(const SortformerConfig& config, const std::vector<float>& emb_seq,
                       const std::vector<float>& preds, const std::vector<float>& mean_sil_emb,
                       std::vector<float>& spkcache_out, std::vector<float>& spkcache_preds_out,
                       std::vector<float>* scores_out = nullptr, std::vector<int>* topk_out = nullptr,
                       std::vector<char>* disabled_out = nullptr) {
    const int n_spk = config.num_speakers;
    const int n_frames = static_cast<int>(preds.size()) / n_spk;
    const std::vector<float>& sil_emb =
        (config.use_learnable_sil_emb && !config.learnable_sil_emb.empty()) ? config.learnable_sil_emb : mean_sil_emb;
    const int spkcache_len_per_spk = config.spkcache_len / n_spk - config.spkcache_sil_frames_per_spk;
    const int strong_boost_per_spk = static_cast<int>(std::floor(spkcache_len_per_spk * config.strong_boost_rate));
    const int weak_boost_per_spk = static_cast<int>(std::floor(spkcache_len_per_spk * config.weak_boost_rate));
    const int min_pos_scores_per_spk = static_cast<int>(std::floor(spkcache_len_per_spk * config.min_pos_scores_rate));

    // scores = log(clamp(p,th)) - log(clamp(1-p,th)) + sum_s log(clamp(1-p_s,th)) - log(0.5)
    std::vector<float> scores(preds.size());
    for (int t = 0; t < n_frames; ++t) {
        const float* row = preds.data() + static_cast<std::size_t>(t) * n_spk;
        float log_1_sum = 0.0f;
        for (int s = 0; s < n_spk; ++s) {
            log_1_sum += std::log(std::max(1.0f - row[s], config.pred_score_threshold));
        }
        for (int s = 0; s < n_spk; ++s) {
            const float log_p = std::log(std::max(row[s], config.pred_score_threshold));
            const float log_1 = std::log(std::max(1.0f - row[s], config.pred_score_threshold));
            scores[static_cast<std::size_t>(t) * n_spk + s] = log_p - log_1 + log_1_sum - kLogHalf;
        }
    }

    // Disable non-speech frames and non-positive scores once a speaker has enough positives.
    std::vector<int> pos_counts(n_spk, 0);
    for (int t = 0; t < n_frames; ++t) {
        for (int s = 0; s < n_spk; ++s) {
            const float score = scores[static_cast<std::size_t>(t) * n_spk + s];
            if (preds[static_cast<std::size_t>(t) * n_spk + s] > 0.5f && score > 0.0f) pos_counts[s]++;
        }
    }
    for (int t = 0; t < n_frames; ++t) {
        for (int s = 0; s < n_spk; ++s) {
            const std::size_t index = static_cast<std::size_t>(t) * n_spk + s;
            const bool is_speech = preds[index] > 0.5f;
            if (!is_speech) {
                scores[index] = kNegInf;
                continue;
            }
            const bool is_pos = scores[index] > 0.0f;
            if (!is_pos && pos_counts[s] >= min_pos_scores_per_spk) scores[index] = kNegInf;
        }
    }

    // Boost recently added frames (they live at positions >= spkcache_len).
    if (config.scores_boost_latest > 0.0f) {
        for (int t = config.spkcache_len; t < n_frames; ++t) {
            for (int s = 0; s < n_spk; ++s) scores[static_cast<std::size_t>(t) * n_spk + s] += config.scores_boost_latest;
        }
    }

    // Strong/weak top-k boosting per speaker.
    auto boost = [&](int n_boost, float scale) {
        if (n_boost <= 0) return;
        for (int s = 0; s < n_spk; ++s) {
            std::vector<float> column(static_cast<std::size_t>(n_frames));
            for (int t = 0; t < n_frames; ++t) column[t] = scores[static_cast<std::size_t>(t) * n_spk + s];
            const int k = std::min(n_boost, n_frames);
            const std::vector<int> indices = topk_indices_desc(column, n_frames, k);
            for (int index : indices) scores[static_cast<std::size_t>(index) * n_spk + s] -= scale * kLogHalf;
        }
    };
    boost(strong_boost_per_spk, 2.0f);
    boost(weak_boost_per_spk, 1.0f);

    // Append spkcache_sil_frames_per_spk rows of +inf.
    const int padded_frames = n_frames + config.spkcache_sil_frames_per_spk;
    std::vector<float> padded(static_cast<std::size_t>(padded_frames) * n_spk, 0.0f);
    std::copy(scores.begin(), scores.end(), padded.begin());
    for (int t = n_frames; t < padded_frames; ++t) {
        for (int s = 0; s < n_spk; ++s) padded[static_cast<std::size_t>(t) * n_spk + s] = std::numeric_limits<float>::infinity();
    }

    std::vector<int> topk_sorted;
    std::vector<char> is_disabled;
    // Select from the padded scores (including the +inf silence rows), matching numpy:
    // n_frames here is the padded frame count, so rows >= n_frames - sil_frames are disabled.
    get_topk_indices(padded, padded_frames, n_spk, config.spkcache_len, config.spkcache_sil_frames_per_spk, topk_sorted,
                     is_disabled);
    if (scores_out != nullptr) *scores_out = padded;
    if (topk_out != nullptr) *topk_out = topk_sorted;
    if (disabled_out != nullptr) *disabled_out = is_disabled;

    const int selected = static_cast<int>(topk_sorted.size());
    spkcache_out.assign(static_cast<std::size_t>(selected) * config.fc_d_model, 0.0f);
    spkcache_preds_out.assign(static_cast<std::size_t>(selected) * n_spk, 0.0f);
    for (int j = 0; j < selected; ++j) {
        if (is_disabled[j]) {
            std::copy(sil_emb.begin(), sil_emb.end(),
                      spkcache_out.begin() + static_cast<std::size_t>(j) * config.fc_d_model);
            continue;
        }
        const int frame = topk_sorted[j];
        std::copy(emb_seq.begin() + static_cast<std::size_t>(frame) * config.fc_d_model,
                  emb_seq.begin() + static_cast<std::size_t>(frame + 1) * config.fc_d_model,
                  spkcache_out.begin() + static_cast<std::size_t>(j) * config.fc_d_model);
        std::copy(preds.begin() + static_cast<std::size_t>(frame) * n_spk,
                  preds.begin() + static_cast<std::size_t>(frame + 1) * n_spk,
                  spkcache_preds_out.begin() + static_cast<std::size_t>(j) * n_spk);
    }
}
}  // namespace

void streaming_update(const SortformerConfig& config, StreamingState& state, const std::vector<float>& chunk,
                      const std::vector<float>& preds, int lc, int rc, std::vector<float>& chunk_preds_out) {
    const int n_spk = config.num_speakers;
    const int dim = config.fc_d_model;
    const int chunk_frames = static_cast<int>(chunk.size()) / dim;
    const int spkcache_frames = state.spkcache_frames();
    const int fifo_frames = state.fifo_frames();
    const int chunk_len = chunk_frames - lc - rc;

    // Refresh FIFO predictions from the graph output and append the chunk.
    state.fifo_preds.assign(preds.begin() + static_cast<std::size_t>(spkcache_frames) * n_spk,
                            preds.begin() + static_cast<std::size_t>(spkcache_frames + fifo_frames) * n_spk);

    chunk_preds_out.assign(preds.begin() + static_cast<std::size_t>(spkcache_frames + fifo_frames + lc) * n_spk,
                           preds.begin() + static_cast<std::size_t>(spkcache_frames + fifo_frames + chunk_len + lc) * n_spk);

    state.fifo.insert(state.fifo.end(), chunk.begin() + static_cast<std::size_t>(lc) * dim,
                      chunk.begin() + static_cast<std::size_t>(lc + chunk_len) * dim);
    state.fifo_preds.insert(state.fifo_preds.end(), chunk_preds_out.begin(), chunk_preds_out.end());

    if (fifo_frames + chunk_len > config.fifo_len) {
        int pop_out_len = config.spkcache_update_period;
        pop_out_len = std::max(pop_out_len, chunk_len - config.fifo_len + fifo_frames);
        pop_out_len = std::min(pop_out_len, fifo_frames + chunk_len);

        // Running silence profile from the popped frames (skipped with the
        // learnable silence embedding, matching NeMo).
        if (!config.use_learnable_sil_emb) {
            int sil_frames = 0;
            std::vector<float> sil_sum(dim, 0.0f);
            for (int t = 0; t < pop_out_len; ++t) {
                const float* row = state.fifo_preds.data() + static_cast<std::size_t>(t) * n_spk;
                float sum = 0.0f;
                for (int s = 0; s < n_spk; ++s) sum += row[s];
                if (sum < config.sil_threshold) {
                    sil_frames++;
                    const float* emb = state.fifo.data() + static_cast<std::size_t>(t) * dim;
                    for (int d = 0; d < dim; ++d) sil_sum[d] += emb[d];
                }
            }
            if (sil_frames > 0) {
                const float total_sil = static_cast<float>(state.n_sil_frames + sil_frames);
                for (int d = 0; d < dim; ++d) {
                    state.mean_sil_emb[d] = (state.mean_sil_emb[d] * static_cast<float>(state.n_sil_frames) + sil_sum[d]) /
                                            total_sil;
                }
                state.n_sil_frames += sil_frames;
            }
        }

        // Move popped frames into the speaker cache.
        const std::size_t pop_emb = static_cast<std::size_t>(pop_out_len) * dim;
        const std::size_t pop_preds = static_cast<std::size_t>(pop_out_len) * n_spk;
        state.spkcache.insert(state.spkcache.end(), state.fifo.begin(), state.fifo.begin() + pop_emb);
        if (state.spkcache_compressed) {
            state.spkcache_preds.insert(state.spkcache_preds.end(), state.fifo_preds.begin(),
                                        state.fifo_preds.begin() + pop_preds);
        } else {
            state.spkcache_preds.assign(preds.begin(),
                                        preds.begin() + static_cast<std::size_t>(spkcache_frames) * n_spk);
            state.spkcache_preds.insert(state.spkcache_preds.end(), state.fifo_preds.begin(),
                                        state.fifo_preds.begin() + pop_preds);
        }
        state.fifo.erase(state.fifo.begin(), state.fifo.begin() + pop_emb);
        state.fifo_preds.erase(state.fifo_preds.begin(), state.fifo_preds.begin() + pop_preds);

        if (state.spkcache_frames() > config.spkcache_len) {
            std::vector<float> new_cache, new_preds;
            compress_spkcache(config, state.spkcache, state.spkcache_preds, state.mean_sil_emb, new_cache, new_preds);
            state.spkcache.swap(new_cache);
            state.spkcache_preds.swap(new_preds);
            state.spkcache_compressed = true;
        }
    }
}

void debug_compress(const SortformerConfig& config, const std::vector<float>& emb_seq, const std::vector<float>& preds,
                    const std::vector<float>& mean_sil_emb, std::vector<float>& spkcache_out,
                    std::vector<float>& spkcache_preds_out, std::vector<float>& scores_out,
                    std::vector<int>& topk_indices_out, std::vector<char>& is_disabled_out) {
    compress_spkcache(config, emb_seq, preds, mean_sil_emb, spkcache_out, spkcache_preds_out, &scores_out,
                      &topk_indices_out, &is_disabled_out);
}

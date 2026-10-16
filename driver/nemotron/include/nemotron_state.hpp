#pragma once

#include <cstdint>
#include <vector>

// Nemotron-3-Diarization streaming configuration (offline 30.4 s profile by
// default; the low-latency profile is selected from the command line).
struct SortformerConfig {
    int num_speakers = 8;
    int fc_d_model = 512;
    int subsampling_factor = 8;
    int upsample_factor = 8;       // subpixel Conv1d factor of the high-res head
    int chunk_len = 340;
    int chunk_left_context = 0;
    int chunk_right_context = 40;
    int fifo_len = 40;
    int spkcache_len = 264;
    int spkcache_update_period = 300;
    int spkcache_sil_frames_per_spk = 1;
    float pred_score_threshold = 0.25f;
    float scores_boost_latest = 0.05f;
    float sil_threshold = 0.2f;
    float strong_boost_rate = 0.75f;
    float weak_boost_rate = 1.5f;
    float min_pos_scores_rate = 0.5f;
    // Nemotron uses a learnable silence embedding for disabled cache frames;
    // the running silence profile is not used when this is enabled.
    bool use_learnable_sil_emb = false;
    std::vector<float> learnable_sil_emb;  // (fc_d_model,)
};

struct StreamingState {
    std::vector<float> spkcache;        // (T, 512)
    std::vector<float> spkcache_preds;  // (T, 8)
    bool spkcache_compressed = false;
    std::vector<float> fifo;            // (T, 512)
    std::vector<float> fifo_preds;      // (T, 8)
    std::vector<float> mean_sil_emb;    // (512,)
    std::int64_t n_sil_frames = 0;

    void init(const SortformerConfig& config) { mean_sil_emb.assign(config.fc_d_model, 0.0f); }
    int spkcache_frames() const { return static_cast<int>(spkcache.size()) / 512; }
    int fifo_frames() const { return static_cast<int>(fifo.size()) / 512; }
};

// Port of NeMo SortformerModules.streaming_update (sync path, eval, batch 1).
//   chunk:  (lc + chunk_len + rc, 512)
//   preds:  (spkcache_frames + fifo_frames + chunk_frames, 8)  <- 80 ms grid
//   chunk_preds_out: (chunk_len, 8)
void streaming_update(const SortformerConfig& config, StreamingState& state, const std::vector<float>& chunk,
                      const std::vector<float>& preds, int lc, int rc, std::vector<float>& chunk_preds_out);

// Debug hook for host parity checks: runs _compress_spkcache and exposes the
// padded scores, selected frame indices and disabled mask.
void debug_compress(const SortformerConfig& config, const std::vector<float>& emb_seq, const std::vector<float>& preds,
                    const std::vector<float>& mean_sil_emb, std::vector<float>& spkcache_out,
                    std::vector<float>& spkcache_preds_out, std::vector<float>& scores_out,
                    std::vector<int>& topk_indices_out, std::vector<char>& is_disabled_out);

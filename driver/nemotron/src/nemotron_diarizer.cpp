#include "nemotron_diarizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "mel_frontend.hpp"

#include "chunk_layout.hpp"

namespace {
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// FeatureStacking pre-encode: one output frame per 8 input mel frames (ceil).
int pre_encode_length(int mel_frames) { return (mel_frames + 7) / 8; }

struct PendingChunk {
    std::vector<float> embeddings;  // valid pre-encoded frames
    int valid_frames = 0;
};
}  // namespace

SortformerDiarizer::SortformerDiarizer(const std::string& preencode_model, const std::string& encoder_model,
                                       const SortformerConfig& config, int threads, const std::string& assets_dir)
    : preencode_(preencode_model),
      encoder_(encoder_model),
      config_(config),
      threads_(std::max(1, threads)),
      assets_dir_(assets_dir) {
    if (!preencode_.has_input("chunk")) throw std::runtime_error("preencode model has no 'chunk' input");
    if (!encoder_.has_input("seq") || !encoder_.has_input("total_lengths")) {
        throw std::runtime_error("encoder model must take 'seq' and 'total_lengths'");
    }
}

std::vector<float> SortformerDiarizer::run(const std::vector<float>& samples, DiarizerReport& report) {
    const int subsampling = config_.subsampling_factor;
    const int upsample = config_.upsample_factor;
    const int mel_dim = MelFrontend::kMels;
    const int chunk_mel = (config_.chunk_left_context + config_.chunk_len + config_.chunk_right_context) * subsampling;
    const int state_len = config_.spkcache_len + config_.fifo_len;
    const int dim = config_.fc_d_model;
    const int n_spk = config_.num_speakers;

    const auto total_start = Clock::now();

    // ---- Front-end: pre-emphasize once, compute mel ranges incrementally in
    // the producer thread so the FFT work overlaps the NPU encoder. ----
    MelFrontend frontend;
    frontend.set_threads(threads_);
    if (!assets_dir_.empty()) frontend.load_assets(assets_dir_);
    frontend.prepare(samples);
    const int raw_frames = MelFrontend::raw_frames(samples.size());
    const int padded_frames = MelFrontend::padded_frames(samples.size());
    std::vector<float> features(static_cast<std::size_t>(padded_frames) * mel_dim, 0.0f);
    int mel_computed = 0;
    report.mel_seconds = 0.0;

    // ---- Chunk layout (matches NeMo streaming_feat_loader; shared with the
    // streaming worker through chunk_layout.hpp so both paths stay in lock
    // step) ----
    const std::vector<DiarChunk> chunks = BuildChunkLayout(raw_frames, config_);

    const std::size_t pre_out_elems = preencode_.output_elems("chunk_pre_encode_embs");
    const int chunk_embs_cap = static_cast<int>(pre_out_elems) / dim;
    const std::size_t seq_elems = encoder_.input_elems("seq");
    const std::size_t preds_elems = encoder_.output_elems("high_res_preds");
    if (static_cast<int>(seq_elems) != (state_len + chunk_embs_cap) * dim) {
        throw std::runtime_error("encoder 'seq' input shape mismatch");
    }
    if (preds_elems != seq_elems / dim * upsample * n_spk) {
        throw std::runtime_error("encoder 'high_res_preds' shape mismatch");
    }

    StreamingState state;
    state.init(config_);
    std::vector<float> all_preds;
    all_preds.reserve(static_cast<std::size_t>(chunks.size()) * config_.chunk_len * upsample * n_spk);

    std::vector<float> chunk_input(static_cast<std::size_t>(chunk_mel) * mel_dim, 0.0f);
    std::vector<float> chunk_embs(static_cast<std::size_t>(chunk_embs_cap) * dim, 0.0f);
    std::vector<float> preds80(static_cast<std::size_t>(state_len + chunk_embs_cap) * n_spk, 0.0f);
    std::vector<float> chunk_preds;

    auto ensure_features = [&](const DiarChunk& chunk) {
        const int need = chunk.end + chunk.right;
        if (need <= mel_computed) return;
        const auto mel_start = Clock::now();
        frontend.compute_range(mel_computed, need, features.data() + static_cast<std::size_t>(mel_computed) * mel_dim);
        report.mel_seconds += seconds_since(mel_start);
        mel_computed = need;
    };

    auto build_chunk_input = [&](const DiarChunk& chunk) {
        std::fill(chunk_input.begin(), chunk_input.end(), 0.0f);
        const int rows = chunk.end + chunk.right - (chunk.start - chunk.left);
        std::memcpy(chunk_input.data(), features.data() + static_cast<std::size_t>(chunk.start - chunk.left) * mel_dim,
                    static_cast<std::size_t>(rows) * mel_dim * sizeof(float));
    };

    auto preencode_chunk = [&](const DiarChunk& chunk, PendingChunk& pending) {
        ensure_features(chunk);
        build_chunk_input(chunk);
        std::memcpy(preencode_.input("chunk"), chunk_input.data(), chunk_input.size() * sizeof(float));
        preencode_.run();
        const int mel_rows = chunk.end + chunk.right - (chunk.start - chunk.left);
        pending.valid_frames = pre_encode_length(mel_rows);
        const float* output = preencode_.output("chunk_pre_encode_embs");
        pending.embeddings.assign(output, output + static_cast<std::size_t>(pending.valid_frames) * dim);
    };

    auto encode_chunk = [&](const DiarChunk& chunk, const PendingChunk& pending) {
        const int spkcache_frames = state.spkcache_frames();
        const int fifo_frames = state.fifo_frames();
        const int total_lengths = spkcache_frames + fifo_frames + pending.valid_frames;

        // Write the packed sequence straight into the NPU input buffer (no staging copy).
        float* seq_input = encoder_.input("seq");
        std::fill(seq_input, seq_input + seq_elems, 0.0f);
        std::memcpy(seq_input, state.spkcache.data(), state.spkcache.size() * sizeof(float));
        std::memcpy(seq_input + static_cast<std::size_t>(spkcache_frames) * dim, state.fifo.data(),
                    state.fifo.size() * sizeof(float));
        std::memcpy(seq_input + static_cast<std::size_t>(spkcache_frames + fifo_frames) * dim,
                    pending.embeddings.data(), pending.embeddings.size() * sizeof(float));
        *encoder_.input("total_lengths") = static_cast<float>(total_lengths);
        encoder_.run();

        const float* high_res = encoder_.output("high_res_preds");
        const int lc_enc = static_cast<int>(std::lround(static_cast<double>(chunk.left) / subsampling));
        const int rc_enc = static_cast<int>(std::ceil(static_cast<double>(chunk.right) / subsampling));
        const int chunk_len_enc = pending.valid_frames - lc_enc - rc_enc;

        // 10 ms chunk output for the transcript, sliced from the packed
        // high-resolution predictions (pre-update state lengths).
        const std::size_t start = static_cast<std::size_t>(spkcache_frames + fifo_frames + lc_enc) * upsample;
        const std::size_t count = static_cast<std::size_t>(chunk_len_enc) * upsample;
        all_preds.insert(all_preds.end(), high_res + start * n_spk, high_res + (start + count) * n_spk);

        // 80 ms predictions over the packed sequence feed the AOSC state update:
        // mean-pool each valid window of `upsample` high-resolution frames; the
        // padded tail beyond total_lengths stays zero (apply_mask_to_preds).
        std::fill(preds80.begin(), preds80.end(), 0.0f);
        for (int t = 0; t < total_lengths; ++t) {
            for (int s = 0; s < n_spk; ++s) {
                float acc = 0.0f;
                const float* base = high_res + (static_cast<std::size_t>(t) * upsample) * n_spk + s;
                for (int f = 0; f < upsample; ++f) acc += base[static_cast<std::size_t>(f) * n_spk];
                preds80[static_cast<std::size_t>(t) * n_spk + s] = acc / static_cast<float>(upsample);
            }
        }
        streaming_update(config_, state, pending.embeddings, preds80, lc_enc, rc_enc, chunk_preds);
    };

    if (threads_ <= 1 || chunks.empty()) {
        PendingChunk pending;
        for (const DiarChunk& chunk : chunks) {
            const auto pre_start = Clock::now();
            preencode_chunk(chunk, pending);
            report.pre_seconds += seconds_since(pre_start);
            const auto enc_start = Clock::now();
            encode_chunk(chunk, pending);
            report.enc_seconds += seconds_since(enc_start);
        }
    } else {
        // Producer/consumer pipeline: worker thread runs the (small) pre-encode
        // graph ahead while the main thread runs the encoder + state update.
        std::deque<PendingChunk> queue;
        std::mutex mutex;
        std::condition_variable not_empty, not_full;
        bool done = false;
        std::exception_ptr producer_error;
        const std::size_t max_queue = 4;

        std::thread producer([&]() {
            try {
                for (const DiarChunk& chunk : chunks) {
                    PendingChunk pending;
                    const auto pre_start = Clock::now();
                    preencode_chunk(chunk, pending);
                    const double pre_elapsed = seconds_since(pre_start);
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        while (queue.size() >= max_queue && !done) not_full.wait(lock);
                        if (done) return;
                        report.pre_seconds += pre_elapsed;
                        queue.push_back(std::move(pending));
                    }
                    not_empty.notify_one();
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                producer_error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                done = true;
            }
            not_empty.notify_all();
        });

        for (std::size_t i = 0; i < chunks.size(); ++i) {
            PendingChunk pending;
            {
                std::unique_lock<std::mutex> lock(mutex);
                not_empty.wait(lock, [&]() { return !queue.empty() || done || producer_error; });
                if (producer_error) {
                    done = true;
                    not_full.notify_all();
                    producer.join();
                    std::rethrow_exception(producer_error);
                }
                if (queue.empty()) continue;  // producer finished or failed
                pending = std::move(queue.front());
                queue.pop_front();
            }
            not_full.notify_one();
            const auto enc_start = Clock::now();
            encode_chunk(chunks[i], pending);
            report.enc_seconds += seconds_since(enc_start);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            done = true;
        }
        not_full.notify_all();
        producer.join();
    }

    // NeMo truncates the accumulated predictions to the preprocessor frame count.
    const std::size_t valid = static_cast<std::size_t>(raw_frames) * n_spk;
    if (all_preds.size() > valid) all_preds.resize(valid);

    report.steps = static_cast<int>(chunks.size());
    report.frames = raw_frames;
    report.audio_seconds = static_cast<double>(samples.size()) / MelFrontend::kSampleRate;
    report.total_seconds = seconds_since(total_start);
    report.threads = threads_;
    return all_preds;
}

// Streaming speaker-diarization worker for the AX650N true-streaming
// transcription pipeline (Nemotron-3 end-to-end streaming Sortformer).
//
// The file-mode binary (nemotron_ax650) consumes a whole WAV; this worker
// keeps the encoder state alive across an arbitrarily long PCM stream and
// emits the committed 10 ms speaker probabilities as soon as each chunk
// stabilizes.  Chunk boundaries, left/right context, the packing of
// spkcache+fifo+chunk into "seq" and the AOSC state machine are shared with
// the file-mode path through chunk_layout.hpp + nemotron_state.cpp, so for
// the same audio the emitted predictions are identical to the file-mode
// ones (verify on board with --dump-preds + cmp).
//
// Latency (low-latency preset): a chunk commits 0.48 s of new audio after it
// has seen 0.48 s + 0.32 s of look-ahead, so labels for instant t are
// available around t + 0.8 s + one encoder step.
//
// Resident protocol (streaming mode, no --wav):
//   stdin   : uint32 payload_bytes, then payload (16 kHz mono S16LE PCM).
//             payload_bytes == 0 or stdin EOF  => flush the tail and finish.
//   response: frame = type byte + uint32 length + payload (default fd 3)
//             'R' "ready"                 once, after the models are loaded
//             'S' uint32 start_frame, uint32 n_frames,
//                 then n_frames * num_speakers float32 probabilities,
//                 row-major, absolute 10 ms frame index from stream start
//                 (frame i covers [i*10ms, (i+1)*10ms))
//             'D' stats JSON text         once, after the flush
//             'E' error text              fatal, worker exits afterwards
//
// One-shot debug mode (no NPU lock ownership games, useful for board A/B):
//   stream-diar-worker --preencode ... --encoder ... --wav in.wav \
//       [--rttm out.rttm] [--dump-preds out.bin] [--realtime] [--lowlat]
//   feeds the file through the exact streaming path; --rttm/--dump-preds must
//   match nemotron_ax650's outputs.  --realtime paces the feed to wall clock
//   so the 'S' arrival times show the true end-to-end label latency.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "audio_reader.hpp"
#include "ax_engine_runner.hpp"
#include "chunk_layout.hpp"
#include "mel_frontend.hpp"
#include "nemotron_postprocess.hpp"
#include "nemotron_state.hpp"
#include "npy_reader.hpp"
#include "streaming_frontend.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// FeatureStacking pre-encode: one output frame per 8 input mel frames (ceil);
// mirrors the helper in nemotron_diarizer.cpp.
int pre_encode_length(int mel_frames) { return (mel_frames + 7) / 8; }

struct Options {
    std::string preencode_path, encoder_path;
    std::string sil_emb_path, assets_dir;
    std::string wav_path, rttm_path, preds_path;
    bool realtime = false;
    int threads = 8;
    int response_fd = 3;  // -1 disables the response channel (wav debug mode)
    bool response_fd_explicit = false;
    SortformerConfig config;
};

void usage() {
    std::fprintf(stderr,
                 "usage: stream-diar-worker --preencode <preencode.axmodel> --encoder <encoder.axmodel>\n"
                 "                        [--sil-emb <learnable_sil_emb.npy>] [--assets <dir>]\n"
                 "                        [--preset lowlat|offline] [--threads N] [--response-fd N]\n"
                 "  streaming mode reads framed 16 kHz mono S16LE PCM on stdin\n"
                 "  debug mode: --wav <in.wav> [--rttm <out.rttm>] [--dump-preds <out.bin>] [--realtime]\n"
                 "  preset overrides: --chunk-len N --chunk-left-context N --chunk-right-context N\n"
                 "                    --fifo-len N --spkcache-len N --spkcache-update-period N\n");
}

std::string arg_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

void parse_options(int argc, char** argv, Options* opts) {
    SortformerConfig& config = opts->config;
    std::string preset = "lowlat";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--preencode") opts->preencode_path = arg_value(i, argc, argv);
        else if (arg == "--encoder") opts->encoder_path = arg_value(i, argc, argv);
        else if (arg == "--sil-emb") opts->sil_emb_path = arg_value(i, argc, argv);
        else if (arg == "--assets") opts->assets_dir = arg_value(i, argc, argv);
        else if (arg == "--wav") opts->wav_path = arg_value(i, argc, argv);
        else if (arg == "--rttm") opts->rttm_path = arg_value(i, argc, argv);
        else if (arg == "--dump-preds") opts->preds_path = arg_value(i, argc, argv);
        else if (arg == "--realtime") opts->realtime = true;
        else if (arg == "--threads") opts->threads = std::atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "--response-fd") {
            opts->response_fd = std::atoi(arg_value(i, argc, argv).c_str());
            opts->response_fd_explicit = true;
        }
        else if (arg == "--preset") preset = arg_value(i, argc, argv);
        else if (arg == "--chunk-len") config.chunk_len = std::atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "--chunk-left-context") config.chunk_left_context = std::atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "--chunk-right-context") config.chunk_right_context = std::atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "--fifo-len") config.fifo_len = std::atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "--spkcache-len") config.spkcache_len = std::atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "--spkcache-update-period")
            config.spkcache_update_period = std::atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "-h" || arg == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (preset == "lowlat") {
        config.chunk_len = 6; config.chunk_left_context = 0; config.chunk_right_context = 4;
        config.fifo_len = 264; config.spkcache_len = 264; config.spkcache_update_period = 222;
    } else if (preset == "offline") {
        config.chunk_len = 340; config.chunk_left_context = 0; config.chunk_right_context = 40;
        config.fifo_len = 40; config.spkcache_len = 264; config.spkcache_update_period = 300;
    } else {
        throw std::runtime_error("unknown preset: " + preset);
    }
    if (opts->preencode_path.empty() || opts->encoder_path.empty()) {
        usage();
        throw std::runtime_error("--preencode and --encoder are required");
    }
    // Streaming mode answers on fd 3 by default; the one-shot wav mode disables
    // the response channel unless the caller asked for one explicitly.
    if (opts->wav_path.empty()) {
        if (!opts->response_fd_explicit && opts->response_fd < 0) opts->response_fd = 3;
    } else {
        if (!opts->response_fd_explicit) opts->response_fd = -1;
    }
}

bool read_all(int fd, void* data, std::size_t size) {
    auto* out = static_cast<unsigned char*>(data);
    while (size > 0) {
        const ssize_t n = read(fd, out, size);
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        out += n;
        size -= static_cast<std::size_t>(n);
    }
    return true;
}

bool write_all(int fd, const void* data, std::size_t size) {
    const auto* in = static_cast<const unsigned char*>(data);
    while (size > 0) {
        const ssize_t n = write(fd, in, size);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        in += n;
        size -= static_cast<std::size_t>(n);
    }
    return true;
}

bool write_frame(int fd, char type, const void* payload, std::size_t len) {
    const uint32_t size = static_cast<uint32_t>(len);
    return write_all(fd, &type, sizeof(type)) && write_all(fd, &size, sizeof(size)) &&
           write_all(fd, payload, len);
}

// ---------------------------------------------------------------------------
// Streaming diarizer core.
// ---------------------------------------------------------------------------
class StreamDiarizer {
public:
    using EmitFn = std::function<void(int start_frame, const float* rows, int n_rows)>;

    StreamDiarizer(const Options& opts)
        : opts_(opts), config_(opts.config), mel_frontend_(), mel_(mel_frontend_),
          preencode_(opts.preencode_path), encoder_(opts.encoder_path) {
        if (!mel_frontend_.load_assets(opts.assets_dir) && !opts.assets_dir.empty())
            std::fprintf(stderr, "warning: continuing without checkpoint assets\n");
        mel_frontend_.set_threads(std::max(1, std::min(opts.threads, 8)));
        if (!preencode_.has_input("chunk"))
            throw std::runtime_error("preencode model has no 'chunk' input");
        if (!encoder_.has_input("seq") || !encoder_.has_input("total_lengths"))
            throw std::runtime_error("encoder model must take 'seq' and 'total_lengths'");

        const int dim = config_.fc_d_model;
        const int n_spk = config_.num_speakers;
        const int state_len = config_.spkcache_len + config_.fifo_len;
        chunk_mel_ = (config_.chunk_left_context + config_.chunk_len + config_.chunk_right_context) *
                     config_.subsampling_factor;
        const std::size_t pre_out_elems = preencode_.output_elems("chunk_pre_encode_embs");
        chunk_embs_cap_ = static_cast<int>(pre_out_elems) / dim;
        seq_elems_ = encoder_.input_elems("seq");
        const std::size_t preds_elems = encoder_.output_elems("high_res_preds");
        if (seq_elems_ != static_cast<std::size_t>(state_len + chunk_embs_cap_) * dim)
            throw std::runtime_error("encoder 'seq' input shape mismatch");
        if (preds_elems != seq_elems_ / dim * config_.upsample_factor * n_spk)
            throw std::runtime_error("encoder 'high_res_preds' shape mismatch");

        state_.init(config_);
        chunk_input_.assign(static_cast<std::size_t>(chunk_mel_) * MelFrontend::kMels, 0.0f);
        preds80_.assign(static_cast<std::size_t>(state_len + chunk_embs_cap_) * n_spk, 0.0f);
    }

    void append(const float* samples, std::size_t n) { mel_.append(samples, n); }

    // Compute everything computable and run as many chunks as possible.  With
    // flush=true the remaining tail frames (zero-filled like a file ending
    // here) are computed and split exactly like file mode.
    void process_available(bool flush, const EmitFn& emit) {
        const auto mel_start = Clock::now();
        mel_.advance(flush ? mel_.total_frames() : mel_.computable_frames(), flush);
        mel_seconds_ += seconds_since(mel_start);

        for (;;) {
            DiarChunk chunk;
            if (!NextChunk(cursor_, mel_.ready(), config_, flush, &chunk)) break;
            process_chunk(chunk, emit);
            const int left_needed = config_.chunk_left_context * config_.subsampling_factor;
            mel_.trim(std::max(0, cursor_ - left_needed));
        }
    }

    // Stats in the same JSON shape as nemotron_ax650 for easy comparison.
    std::string stats_json(double total_seconds) const {
        char buffer[512];
        const double audio_seconds = static_cast<double>(mel_.total_samples()) / MelFrontend::kSampleRate;
        const int frames = emitted_frames_;
        const double host_s = std::max(0.0, total_seconds - mel_seconds_ - pre_seconds_ - enc_seconds_);
        std::snprintf(buffer, sizeof(buffer),
                      "{\"steps\":%d,\"frames\":%d,\"audio_s\":%.3f,\"mel_s\":%.3f,\"pre_s\":%.3f,"
                      "\"enc_s\":%.3f,\"host_s\":%.3f,\"total_s\":%.3f,\"rtf\":%.5f,\"ms_per_step\":%.3f,"
                      "\"threads\":%d}",
                      steps_, frames, audio_seconds, mel_seconds_, pre_seconds_, enc_seconds_, host_s,
                      total_seconds, audio_seconds > 0 ? total_seconds / audio_seconds : 0.0,
                      steps_ > 0 ? 1000.0 * total_seconds / steps_ : 0.0, opts_.threads);
        return buffer;
    }

    int emitted_frames() const { return emitted_frames_; }
    std::int64_t total_samples() const { return mel_.total_samples(); }

private:
    void process_chunk(const DiarChunk& chunk, const EmitFn& emit) {
        const int subsampling = config_.subsampling_factor;
        const int upsample = config_.upsample_factor;
        const int dim = config_.fc_d_model;
        const int n_spk = config_.num_speakers;
        const int mel_dim = MelFrontend::kMels;

        // --- pre-encode (FeatureStacking) over [start-left, end+right) ---
        const int from = chunk.start - chunk.left;
        const int rows = chunk.end + chunk.right - from;
        std::fill(chunk_input_.begin(), chunk_input_.end(), 0.0f);
        for (int f = 0; f < rows; ++f) {
            const float* src = mel_.row(from + f);
            if (src == nullptr) throw std::runtime_error("mel frame trimmed before use");
            std::memcpy(chunk_input_.data() + static_cast<std::size_t>(f) * mel_dim, src,
                        static_cast<std::size_t>(mel_dim) * sizeof(float));
        }
        const auto pre_start = Clock::now();
        std::memcpy(preencode_.input("chunk"), chunk_input_.data(), chunk_input_.size() * sizeof(float));
        preencode_.run();
        const int valid_frames = pre_encode_length(rows);
        const float* emb_out = preencode_.output("chunk_pre_encode_embs");
        pending_.embeddings.assign(emb_out, emb_out + static_cast<std::size_t>(valid_frames) * dim);
        pending_.valid_frames = valid_frames;
        pre_seconds_ += seconds_since(pre_start);

        // --- encode: pack spkcache + fifo + chunk into "seq" ---
        const int spkcache_frames = state_.spkcache_frames();
        const int fifo_frames = state_.fifo_frames();
        const int total_lengths = spkcache_frames + fifo_frames + valid_frames;

        float* seq_input = encoder_.input("seq");
        std::fill(seq_input, seq_input + seq_elems_, 0.0f);
        std::memcpy(seq_input, state_.spkcache.data(), state_.spkcache.size() * sizeof(float));
        std::memcpy(seq_input + static_cast<std::size_t>(spkcache_frames) * dim, state_.fifo.data(),
                    state_.fifo.size() * sizeof(float));
        std::memcpy(seq_input + static_cast<std::size_t>(spkcache_frames + fifo_frames) * dim,
                    pending_.embeddings.data(), pending_.embeddings.size() * sizeof(float));
        *encoder_.input("total_lengths") = static_cast<float>(total_lengths);

        const auto enc_start = Clock::now();
        encoder_.run();
        enc_seconds_ += seconds_since(enc_start);

        const float* high_res = encoder_.output("high_res_preds");
        const int lc_enc = static_cast<int>(std::lround(static_cast<double>(chunk.left) / subsampling));
        const int rc_enc = static_cast<int>(std::ceil(static_cast<double>(chunk.right) / subsampling));
        const int chunk_len_enc = valid_frames - lc_enc - rc_enc;

        // --- committed 10 ms rows for this chunk's body [start, end) ---
        const std::size_t start = static_cast<std::size_t>(spkcache_frames + fifo_frames + lc_enc) * upsample;
        const std::size_t count = static_cast<std::size_t>(chunk_len_enc) * upsample;
        const int raw_limit = flush_limit(chunk.start + static_cast<int>(count));
        const int emit_count = raw_limit - chunk.start;
        if (emit_count > 0)
            emit(chunk.start, high_res + start * n_spk, emit_count);

        // --- 80 ms predictions over the packed sequence feed the AOSC update ---
        std::fill(preds80_.begin(), preds80_.end(), 0.0f);
        for (int t = 0; t < total_lengths; ++t) {
            for (int s = 0; s < n_spk; ++s) {
                float acc = 0.0f;
                const float* base = high_res + (static_cast<std::size_t>(t) * upsample) * n_spk + s;
                for (int f = 0; f < upsample; ++f) acc += base[static_cast<std::size_t>(f) * n_spk];
                preds80_[static_cast<std::size_t>(t) * n_spk + s] = acc / static_cast<float>(upsample);
            }
        }
        streaming_update(config_, state_, pending_.embeddings, preds80_, lc_enc, rc_enc, chunk_preds_);

        cursor_ = chunk.end;
        ++steps_;
        emitted_frames_ += emit_count;
    }

    // File-mode semantics: predictions only exist up to total_frames()
    // (num_samples / hop); the last partial chunk can produce more rows than
    // that and file mode truncates them, so clamp the same way here.
    int flush_limit(int at_most) const {
        const int raw = mel_.total_frames();
        return std::min(at_most, raw);
    }

    const Options& opts_;
    SortformerConfig config_;
    MelFrontend mel_frontend_;
    StreamingMel mel_;
    AxEngineRunner preencode_;
    AxEngineRunner encoder_;

    StreamingState state_;
    struct Pending {
        std::vector<float> embeddings;
        int valid_frames = 0;
    } pending_;

    int cursor_ = 0;  // next chunk start (absolute 10 ms frame)
    int chunk_mel_ = 0;
    int chunk_embs_cap_ = 0;
    std::size_t seq_elems_ = 0;
    std::vector<float> chunk_input_;
    std::vector<float> preds80_;
    std::vector<float> chunk_preds_;

    int steps_ = 0;
    int emitted_frames_ = 0;
    double mel_seconds_ = 0.0, pre_seconds_ = 0.0, enc_seconds_ = 0.0;
};

// ---------------------------------------------------------------------------
// Feeding modes.
// ---------------------------------------------------------------------------
int run_wav_mode(const Options& opts) {
    AudioData audio = read_wav_mono(opts.wav_path);
    if (audio.sample_rate != MelFrontend::kSampleRate)
        audio = resample_audio(audio, MelFrontend::kSampleRate);

    StreamDiarizer diarizer(opts);
    std::vector<float> all_preds;
    all_preds.reserve(static_cast<std::size_t>(MelFrontend::raw_frames(audio.samples.size())) *
                      opts.config.num_speakers);

    const auto sink = [&](int start_frame, const float* rows, int n_rows) {
        const std::size_t expect = static_cast<std::size_t>(start_frame) * opts.config.num_speakers;
        if (expect != all_preds.size())
            throw std::runtime_error("prediction stream gap at frame " + std::to_string(start_frame));
        all_preds.insert(all_preds.end(), rows, rows + static_cast<std::size_t>(n_rows) * opts.config.num_speakers);
    };

    const auto total_start = Clock::now();
    const std::size_t block = 1600;  // 100 ms
    std::size_t fed = 0;
    while (fed < audio.samples.size()) {
        if (opts.realtime) {
            const double target = static_cast<double>(fed) / MelFrontend::kSampleRate;
            std::this_thread::sleep_until(total_start + std::chrono::duration<double>(target));
        }
        const std::size_t n = std::min(block, audio.samples.size() - fed);
        diarizer.append(audio.samples.data() + fed, n);
        fed += n;
        diarizer.process_available(false, sink);
    }
    diarizer.process_available(true, sink);
    const double total_seconds = seconds_since(total_start);

    if (!opts.rttm_path.empty() || !opts.preds_path.empty()) {
        const int frames = static_cast<int>(all_preds.size()) / opts.config.num_speakers;
        if (!opts.rttm_path.empty()) {
            const std::string uri = opts.wav_path.substr(opts.wav_path.find_last_of('/') + 1);
            const std::string stem = uri.substr(0, uri.find_last_of('.'));
            std::vector<std::string> lines;
            PostProcessingParams params;
            preds_to_rttm_lines(all_preds, frames, opts.config.num_speakers, stem, params, lines);
            std::ofstream out(opts.rttm_path.c_str());
            if (!out) throw std::runtime_error("cannot write " + opts.rttm_path);
            for (const std::string& line : lines) out << line << "\n";
        }
        if (!opts.preds_path.empty()) {
            std::ofstream out(opts.preds_path.c_str(), std::ios::binary);
            if (!out) throw std::runtime_error("cannot write " + opts.preds_path);
            out.write(reinterpret_cast<const char*>(all_preds.data()),
                      static_cast<std::streamsize>(all_preds.size() * sizeof(float)));
        }
    }

    std::printf("%s\n", diarizer.stats_json(total_seconds).c_str());
    return 0;
}

int run_stream_mode(const Options& opts) {
    StreamDiarizer diarizer(opts);
    const int response_fd = opts.response_fd;

    if (!write_frame(response_fd, 'R', "ready", 5)) {
        std::fprintf(stderr, "cannot write startup frame to fd %d\n", response_fd);
        return 1;
    }

    std::vector<float> samples;
    std::vector<unsigned char> pcm;
    const auto sink = [&](int start_frame, const float* rows, int n_rows) {
        const int n_spk = opts.config.num_speakers;
        const std::size_t payload_bytes = 2 * sizeof(uint32_t) +
                                          static_cast<std::size_t>(n_rows) * n_spk * sizeof(float);
        std::vector<unsigned char> payload(payload_bytes);
        uint32_t start = static_cast<uint32_t>(start_frame);
        uint32_t count = static_cast<uint32_t>(n_rows);
        std::memcpy(payload.data(), &start, sizeof(start));
        std::memcpy(payload.data() + sizeof(start), &count, sizeof(count));
        std::memcpy(payload.data() + 2 * sizeof(uint32_t), rows,
                    static_cast<std::size_t>(n_rows) * n_spk * sizeof(float));
        if (!write_frame(response_fd, 'S', payload.data(), payload.size()))
            throw std::runtime_error("cannot write 'S' frame to fd " + std::to_string(response_fd));
    };

    const auto total_start = Clock::now();
    bool finished = false;
    uint32_t length = 0;
    while (!finished && read_all(STDIN_FILENO, &length, sizeof(length))) {
        if (length == 0) break;  // explicit flush request
        pcm.resize(length);
        if (!read_all(STDIN_FILENO, pcm.data(), pcm.size()))
            throw std::runtime_error("truncated PCM frame on stdin");
        if (length % sizeof(int16_t) != 0)
            throw std::runtime_error("PCM frame size not a multiple of 2 bytes");
        samples.resize(length / sizeof(int16_t));
        for (std::size_t i = 0; i < samples.size(); ++i) {
            int16_t value;
            std::memcpy(&value, pcm.data() + i * sizeof(int16_t), sizeof(value));
            samples[i] = static_cast<float>(value) / 32768.0f;
        }
        diarizer.append(samples.data(), samples.size());
        diarizer.process_available(false, sink);
    }
    diarizer.process_available(true, sink);

    const std::string stats = diarizer.stats_json(seconds_since(total_start));
    write_frame(response_fd, 'D', stats.data(), stats.size());
    std::fprintf(stderr, "[stream-diar] %s\n", stats.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    try {
        parse_options(argc, argv, &opts);
        if (!opts.sil_emb_path.empty()) {
            std::vector<float> sil_emb;
            std::vector<int> shape;
            if (!load_npy_float32(opts.sil_emb_path, sil_emb, shape) ||
                sil_emb.size() != static_cast<std::size_t>(opts.config.fc_d_model)) {
                throw std::runtime_error("cannot load learnable silence embedding: " + opts.sil_emb_path);
            }
            opts.config.learnable_sil_emb = sil_emb;
            opts.config.use_learnable_sil_emb = true;
        } else {
            std::fprintf(stderr, "warning: --sil-emb not given; using the running silence profile\n");
        }
        return opts.wav_path.empty() ? run_stream_mode(opts) : run_wav_mode(opts);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        if (opts.response_fd >= 0) {
            const std::string message = error.what();
            write_frame(opts.response_fd, 'E', message.data(), message.size());
        }
        return 1;
    }
}

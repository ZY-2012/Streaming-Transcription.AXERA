// stream-asr-worker: resident streaming SenseVoice worker for the
// Streaming-Transcription pipeline (protocol: docs/stream-asr-worker-protocol.md).
//
//   stdin    : uint32 payload_bytes + 16 kHz mono S16LE PCM; zero length = flush
//   response : 'R' "ready"; 'T' flags(4B) + coverage_samples(4B) + UTF-8 text;
//              'E' error.  flags bit0 = partial (the session text so far),
//              bit1 = final (after a flush; the worker re-arms immediately).
//
// The dedicated streaming model is stateless: one run sees a [1,26,560]
// LFR-feature window (26 x 60 ms) and emits CTC logits for its 26 frames
// (output positions 4..29 of [1,30,25055]; positions 0..3 are the language
// query slots, dropped like the offline postprocess).  Streaming therefore
// runs the last-26-frame window every `kStep` new frames and keeps a
// per-frame latest-token map; a frame's token commits (appends to the
// session text, CTC-deduped across the commit boundary) once it has
// `kCommitContext` frames of right-context, and the flush zero-pads enough
// silence to give every remaining frame full 26-frame context before the
// final commit.  Partial text = committed text + the provisional tail from
// the newest window, so captions track the fed audio at window cadence.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "ax_runner.hpp"
#include "frontend.hpp"

namespace {

constexpr int kVocabSize = 25055;
constexpr int kVocabBlank = 0;
constexpr int kVocabSpecial = 2;  // 0 <unk>, 1 <s>, 2 </s>: never emit
constexpr int kQuery = StreamFrontend::kQueryNum;      // 4
constexpr int kWindow = StreamFrontend::kWindowFrames;  // 26
constexpr int kStep = 6;            // window cadence: every 6 new LFR frames (360 ms)
constexpr int kCommitContext = 16;  // right-context before a token freezes
constexpr int kEarlyRun = 8;        // partial windows from 8 LFR frames (0.5 s):
                                    // anchors must cover the epoch opening or
                                    // a turn boundary there maps to zero chars

bool ReadAllStdin(void* data, std::size_t n) {
    auto* p = static_cast<unsigned char*>(data);
    while (n > 0) {
        const ssize_t r = read(STDIN_FILENO, p, n);
        if (r == 0) return false;
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

bool WriteAll(int fd, const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    while (n > 0) {
        const ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w;
        n -= static_cast<std::size_t>(w);
    }
    return true;
}

bool WriteFrame(int fd, char type, const void* payload, std::size_t len) {
    const uint32_t size = static_cast<uint32_t>(len);
    return WriteAll(fd, &type, sizeof(type)) && WriteAll(fd, &size, sizeof(size)) &&
           WriteAll(fd, payload, len);
}

std::string ArgValue(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

std::vector<std::string> LoadTokens(const std::string& path) {
    std::vector<std::string> tokens;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open tokens: " + path);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        tokens.push_back(line);
    }
    return tokens;
}

// The streaming session: front-end state + the per-frame token map + the
// commit machinery.  A flush finalizes it; Reset() re-arms for the next
// session (fresh fbank).
struct Session {
    StreamFrontend* frontend = nullptr;
    const std::vector<std::string>* tokens = nullptr;

    std::vector<int> frame_token;      // authoritative decode per LFR frame:
                                       // taken while the frame sat at the
                                       // window CENTER (>=10 frames of context
                                       // on both sides); -1 = not yet
    std::vector<int> latest_token;     // freshest decode from ANY position
                                       // (caption tail only - may be noisier)
    std::string committed;             // committed session text
    int committed_frames = 0;          // frames whose tokens are final
    int last_committed_token = -1;     // CTC dedup state across the commit boundary
    std::int64_t fed_samples = 0;      // PCM fed into this session
    std::int64_t run_coverage = 0;     // samples fed at the last window run
    int next_run = kEarlyRun;         // LFR total at which the next run fires
    int total_lfr = 0;                // real LFR frames (frame_token is padded)

    void Reset() {
        frame_token.clear();
        latest_token.clear();
        total_lfr = 0;
        committed.clear();
        committed_frames = 0;
        last_committed_token = -1;
        fed_samples = 0;
        run_coverage = 0;
        next_run = kWindow;
        frontend->Reset();
    }
};

}  // namespace

namespace {

// Minimal 16 kHz mono S16 WAV reader for the one-shot debug mode.
std::vector<float> ReadWav16k(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    char header[44];
    if (std::fread(header, 1, sizeof(header), f) != sizeof(header) ||
        std::memcmp(header, "RIFF", 4) != 0 || std::memcmp(header + 8, "WAVE", 4) != 0)
        throw std::runtime_error("not a plain 44-byte wav header: " + path);
    std::vector<float> pcm;
    int16_t buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 2, 4096, f)) > 0)
        for (std::size_t i = 0; i < n; ++i) pcm.push_back(static_cast<float>(buf[i]));
    std::fclose(f);
    return pcm;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = ".", language = "zh", wav_path;
    int response_fd = 3;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model-dir") model_dir = ArgValue(i, argc, argv);
            else if (arg == "--language") language = ArgValue(i, argc, argv);
            else if (arg == "--response-fd") response_fd = std::atoi(ArgValue(i, argc, argv).c_str());
            else if (arg == "--wav") wav_path = ArgValue(i, argc, argv);
            else if (arg == "--threads") (void)ArgValue(i, argc, argv);
            else throw std::runtime_error("unknown argument: " + arg);
        }
        const std::string model_path = model_dir + "/sensevoice/streaming_sensevoice.axmodel";
        const std::string tokens_path = model_dir + "/sensevoice/tokens.txt";

        StreamFrontend frontend;
        AxRunner runner(model_path);
        const std::vector<std::string> tokens = LoadTokens(tokens_path);
        if (tokens.size() < kVocabSize) throw std::runtime_error("tokens.txt too small");
        const std::map<std::string, int> lid{{"auto", 0}, {"zh", 3}, {"en", 4},
                                              {"yue", 7}, {"ja", 11}, {"ko", 12}};
        const auto lang_it = lid.find(language);
        const int lang_token = lang_it != lid.end() ? lang_it->second : 0;

        const bool debug = !wav_path.empty();
        auto emit = [&](char type, const void* payload, std::size_t len) {
            if (debug) {
                if (type == 'T') {
                    const uint8_t* p = static_cast<const uint8_t*>(payload);
                    const uint32_t flags = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
                    const uint32_t cov = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
                    const char* text = reinterpret_cast<const char*>(p + 8);
                    std::printf("[%6.2fs]%s %.*s\n", cov / 16000.0f,
                                (flags & 2u) ? " FINAL" : "", static_cast<int>(len - 8), text);
                }
                return;
            }
            WriteFrame(response_fd, type, payload, len);
        };
        if (!debug && !WriteFrame(response_fd, 'R', "ready", 5)) return 1;

        Session session;
        session.frontend = &frontend;
        session.tokens = &tokens;

        float* input_feats = runner.Input("speech");
        int32_t* input_mask = runner.InputInt("mask");
        int32_t* input_lang = runner.InputInt("language");

        // Run the window ending at `lfr_end` (the last kWindow frames) and
        // refresh the per-frame token map for the frames it covers.
        auto run_window = [&](int lfr_end, int total_mel) {
            const int count = std::min(kWindow, lfr_end);
            if (count <= 0) return;
            const int begin = lfr_end - count;
            frontend.PackWindow(begin, count, total_mel, input_feats);
            for (int i = 0; i < kQuery + kWindow; ++i)
                input_mask[i] = i < kQuery + count ? 1 : 0;
            *input_lang = lang_token;
            runner.Run();
            const int out_len = static_cast<int>(*reinterpret_cast<const int32_t*>(
                runner.Output("encoder_out_lens")));
            const float* logits = runner.Output("ctc_logits");
            const int valid = std::max(0, std::min(out_len - kQuery, count));
            // Authoritative strip: with the full window the frame at
            // position p has p frames of left and (26-1-p) of right context;
            // the center positions (10..15) carry >=10 on both sides, which
            // is where the decode is reliable.  Partial windows (< 26) have
            // no center, so every valid position is provisional material.
            int auth_begin = 0, auth_end = valid;
            if (count >= kWindow && begin > 0) {
                auth_begin = std::max(0, kWindow - 16);
                auth_end = std::max(auth_begin, std::min(valid, kWindow - 10));
            }
            for (int j = 0; j < valid; ++j) {
                const float* row = logits + static_cast<std::size_t>(kQuery + j) * kVocabSize;
                int best = 0;
                float best_v = row[0];
                for (int v = 1; v < kVocabSize; ++v)
                    if (row[v] > best_v) { best_v = row[v]; best = v; }
                const int frame = begin + j;
                if (frame >= session.total_lfr) continue;  // padded silence rows
                session.latest_token[frame] = best;
                if (j >= auth_begin && j < auth_end) session.frame_token[frame] = best;
            }
            session.run_coverage = session.fed_samples;
        };

        // Emit a partial: committed text + the provisional tail from the
        // freshest decodes (CTC dedup continues across the boundary).
        auto emit_partial = [&]() {
            std::string partial = session.committed;
            int prev = session.last_committed_token;
            for (int f = session.committed_frames; f < session.total_lfr; ++f) {
                const int token = session.latest_token[f];  // freshest, not
                if (token <= kVocabSpecial || token == prev) continue;  // the strip
                prev = token;
                partial.append(tokens[token]);
            }
            uint8_t head[8];
            uint32_t flags = 1;
            uint32_t coverage = static_cast<uint32_t>(session.run_coverage);
            std::memcpy(head, &flags, 4);
            std::memcpy(head + 4, &coverage, 4);
            std::string payload;
            payload.resize(8 + partial.size());
            std::memcpy(&payload[0], head, 8);
            std::memcpy(&payload[8], partial.data(), partial.size());
            emit('T', payload.data(), payload.size());
        };

        // Flush: zero-pad silence so every real frame gets a full-window
        // decode, commit everything, emit the final, re-arm the session.
        auto flush = [&]() {
            const int real_lfr = session.total_lfr;
            const int target = real_lfr + kWindow - 1;  // full right-context for the last real frame
            const std::vector<float> silence(960, 0.0f);  // 60 ms per LFR frame + tail
            for (int pad = real_lfr; pad < target; ++pad)
                frontend.AcceptWaveform(silence.data(), silence.size());
            frontend.InputFinished();
            const int mel = frontend.NumMelFrames();
            const int lfr = frontend.NumLfrFrames(mel);
            if (static_cast<int>(session.frame_token.size()) < lfr) {
                session.frame_token.resize(static_cast<std::size_t>(lfr), -1);
                session.latest_token.resize(session.frame_token.size(), -1);
            }
            // Frames the streaming runs already gave a center strip end at
            // about (next_run - kStep - 11); sweep only from there so every
            // remaining real frame lands in one strip (each step tiles the
            // next 6 frames; the padded tail provides their right context).
            const int sweep_from = std::max(kWindow, real_lfr - 6);
            for (int end = sweep_from; end <= lfr; end += kStep) run_window(end, mel);
            run_window(lfr, mel);
            // Rebuild the WHOLE session text from the latest per-frame decodes:
            // every frame's map entry holds its most recent window decode (~25
            // frames of right-context - the last window that still covered it),
            // while the incremental commits froze tokens at 16-context for the
            // sake of caption latency; those earlier picks lose punctuation and
            // function words.  The final text is the authoritative rebuild.
            session.committed.clear();
            int prev = -1;
            for (int frame = 0; frame < real_lfr; ++frame) {
                const int token = session.frame_token[frame];
                if (token <= kVocabSpecial || token == prev) continue;
                prev = token;
                session.committed.append(tokens[token]);
            }
            session.committed_frames = real_lfr;
            session.last_committed_token = prev;
            uint8_t head[8];
            uint32_t flags = 2;
            uint32_t coverage = static_cast<uint32_t>(session.fed_samples);
            std::memcpy(head, &flags, 4);
            std::memcpy(head + 4, &coverage, 4);
            std::string payload;
            payload.resize(8 + session.committed.size());
            std::memcpy(&payload[0], head, 8);
            std::memcpy(&payload[8], session.committed.data(), session.committed.size());
            emit('T', payload.data(), payload.size());
            session.Reset();
        };

        if (debug) {
            const std::vector<float> audio = ReadWav16k(wav_path);
            const std::size_t step = 1600;  // 100 ms chunks
            for (std::size_t off = 0; off < audio.size(); off += step) {
                const std::size_t n = std::min(step, audio.size() - off);
                frontend.AcceptWaveform(audio.data() + off, n);
                session.fed_samples += n;
                const int mel = frontend.NumMelFrames();
                const int total_lfr = frontend.NumLfrFrames(mel);
                if (total_lfr > session.total_lfr) {
                    session.total_lfr = total_lfr;
                    if (static_cast<int>(session.frame_token.size()) < total_lfr) {
                        session.frame_token.resize(static_cast<std::size_t>(total_lfr) + 64, -1);
                        session.latest_token.resize(session.frame_token.size(), -1);
                    }
                }
                if (total_lfr >= kEarlyRun && total_lfr >= session.next_run) {
                    run_window(total_lfr, mel);
                    session.next_run = total_lfr + kStep;
                    const int can_commit = total_lfr - kCommitContext;
                    while (session.committed_frames < can_commit &&
                           session.committed_frames < total_lfr) {
                        const int frame = session.committed_frames++;
                        const int token = session.frame_token[frame];
                        if (token <= kVocabSpecial || token == session.last_committed_token) continue;
                        session.last_committed_token = token;
                        session.committed.append(tokens[token]);
                    }
                    emit_partial();
                }
            }
            flush();
            return 0;
        }
        std::vector<float> pcm;
        std::vector<unsigned char> buf;
        uint32_t length = 0;
        while (ReadAllStdin(&length, sizeof(length))) {
            if (length == 0) { flush(); continue; }
            buf.resize(length);
            if (!ReadAllStdin(buf.data(), buf.size())) throw std::runtime_error("truncated PCM frame");
            pcm.resize(length / 2);
            for (std::size_t i = 0; i < pcm.size(); ++i) {
                int16_t v;
                std::memcpy(&v, buf.data() + 2 * i, sizeof(v));
                pcm[i] = static_cast<float>(v);
            }
            frontend.AcceptWaveform(pcm.data(), pcm.size());
            session.fed_samples += pcm.size();
            const int mel = frontend.NumMelFrames();
            const int total_lfr = frontend.NumLfrFrames(mel);
            if (total_lfr > session.total_lfr) {
                session.total_lfr = total_lfr;
                if (static_cast<int>(session.frame_token.size()) < total_lfr) {
                    session.frame_token.resize(static_cast<std::size_t>(total_lfr) + 64, -1);
                    session.latest_token.resize(session.frame_token.size(), -1);
                }
            }
            if (total_lfr >= kEarlyRun && total_lfr >= session.next_run) {
                run_window(total_lfr, mel);
                session.next_run = total_lfr + kStep;
                // commit frames with enough right-context, then emit
                const int can_commit = total_lfr - kCommitContext;
                while (session.committed_frames < can_commit &&
                       session.committed_frames < total_lfr) {
                    const int frame = session.committed_frames++;
                    const int token = session.frame_token[frame];
                    if (token <= kVocabSpecial || token == session.last_committed_token) continue;
                    session.last_committed_token = token;
                    session.committed.append(tokens[token]);
                }
                emit_partial();
            }
        }
        flush();
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        std::fprintf(stderr, "[asr-worker] %s\n", msg.c_str());
        WriteFrame(response_fd, 'E', msg.data(), msg.size());
        return 1;
    }
    return 0;
}

// Resident speaker-diarization worker for the AX650 audio chain.
//
// Protocol (identical framing to qwen3-worker):
//   startup        : one response frame 'R' "ready" on --response-fd
//   request        : uint32 length + UTF-8 WAV path on stdin
//   success        : response frame 'R' with text payload
//                    "<count>\n<begin> <end> <speaker-id>\n..."
//   failure        : response frame 'E' with an error message
//
// The worker owns the persistent speaker table and serializes nothing itself:
// the audio pipeline holds the NPU2 Cell lock while a request is in flight.
//
// One-shot debug mode (no pipeline):
//   diar-worker --model DIR --wav in.wav [--out out.rttm]
// prints "begin end speaker" lines to stdout and optionally writes RTTM.
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "ax_sys_api.h"

#include "diar_engine.hpp"
#include "speaker_table.hpp"
#include "wav_reader.hpp"

namespace {

struct Options {
    std::string model_dir;
    std::string table_path = "/tmp/ax-audio-sdk/speakers.json";
    float step_sec = 2.0f;
    float speaker_match_threshold = 0.55f;
    // "greedy": legacy per-cluster threshold matching (default, unchanged).
    // "online": diart-style joint Hungarian assignment with speech-gated
    // speaker creation and centroid updates (realtime online mode).
    std::string assign_mode = "greedy";
    float online_min_new_sec = 2.0f;
    float online_min_update_sec = 1.0f;
    int online_max_speakers = 20;
    int omp_threads = 4;
    int response_fd = 3;
    // Evaluation helper: also dump the engine's native (overlap-capable) RTTM
    // with persistent speaker ids for every processed WAV.
    std::string rttm_dir;
    // One-shot debug mode.
    bool oneshot = false;
    std::string wav_path;
    std::string out_rttm;
    // Offline-prototyping helper: dump per-sub-window masks + embeddings for
    // --wav to this binary path (see DiarEngine::DumpFeatures) instead of
    // running clustering.  Requires --wav; does not touch any label timeline.
    std::string dump_features;
    // diart-style incremental clustering inside DiarEngine::Run (config-gated
    // alternative to the batch AHC+PLDA+VBx).  Emits the first label from the
    // first sub-window instead of waiting for the full window buffer.
    bool incremental = false;
    bool incremental_plda = false;
    float incremental_delta_new = 0.5f;
    float incremental_delta_min = 0.75f;
    float incremental_rho_update = 0.4f;
};

bool ReadAll(int fd, void* data, std::size_t size) {
    auto* output = static_cast<unsigned char*>(data);
    while (size > 0) {
        const ssize_t count = read(fd, output, size);
        if (count == 0) return false;
        if (count < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        output += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

bool WriteAll(int fd, const void* data, std::size_t size) {
    const auto* input = static_cast<const unsigned char*>(data);
    while (size > 0) {
        const ssize_t count = write(fd, input, size);
        if (count < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        input += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

bool WriteFrame(int fd, char type, const std::string& payload) {
    const uint32_t size = static_cast<uint32_t>(payload.size());
    return WriteAll(fd, &type, sizeof(type)) &&
           WriteAll(fd, &size, sizeof(size)) &&
           WriteAll(fd, payload.data(), payload.size());
}

const char* RequireValue(int argc, char** argv, int* index) {
    if (*index + 1 >= argc)
        throw std::invalid_argument(std::string("missing value for ") + argv[*index]);
    return argv[++*index];
}

int ParseInt(const char* value, const char* name) {
    std::size_t consumed = 0;
    const std::string text(value);
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size())
        throw std::invalid_argument(std::string("invalid integer for ") + name);
    return parsed;
}

float ParseFloat(const char* value, const char* name) {
    std::size_t consumed = 0;
    const std::string text(value);
    const float parsed = std::stof(text, &consumed);
    if (consumed != text.size())
        throw std::invalid_argument(std::string("invalid number for ") + name);
    return parsed;
}

Options ParseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--model") {
            options.model_dir = RequireValue(argc, argv, &i);
        } else if (arg == "--table") {
            options.table_path = RequireValue(argc, argv, &i);
        } else if (arg == "--step") {
            options.step_sec = ParseFloat(RequireValue(argc, argv, &i), "--step");
        } else if (arg == "--threshold") {
            options.speaker_match_threshold =
                ParseFloat(RequireValue(argc, argv, &i), "--threshold");
        } else if (arg == "--assign") {
            options.assign_mode = RequireValue(argc, argv, &i);
            if (options.assign_mode != "greedy" && options.assign_mode != "online")
                throw std::invalid_argument("--assign must be greedy or online");
        } else if (arg == "--min-new-sec") {
            options.online_min_new_sec =
                ParseFloat(RequireValue(argc, argv, &i), "--min-new-sec");
        } else if (arg == "--update-min-sec") {
            options.online_min_update_sec =
                ParseFloat(RequireValue(argc, argv, &i), "--update-min-sec");
        } else if (arg == "--max-speakers") {
            options.online_max_speakers =
                ParseInt(RequireValue(argc, argv, &i), "--max-speakers");
        } else if (arg == "--threads") {
            options.omp_threads = ParseInt(RequireValue(argc, argv, &i), "--threads");
        } else if (arg == "--response-fd") {
            options.response_fd = ParseInt(RequireValue(argc, argv, &i), "--response-fd");
        } else if (arg == "--rttm-dir") {
            options.rttm_dir = RequireValue(argc, argv, &i);
        } else if (arg == "--wav") {
            options.oneshot = true;
            options.wav_path = RequireValue(argc, argv, &i);
        } else if (arg == "--out") {
            options.out_rttm = RequireValue(argc, argv, &i);
        } else if (arg == "--dump-features") {
            options.dump_features = RequireValue(argc, argv, &i);
        } else if (arg == "--incremental") {
            options.incremental = true;
        } else if (arg == "--incremental-plda") {
            options.incremental_plda = true;
        } else if (arg == "--delta-new") {
            options.incremental_delta_new =
                ParseFloat(RequireValue(argc, argv, &i), "--delta-new");
        } else if (arg == "--delta-min") {
            options.incremental_delta_min =
                ParseFloat(RequireValue(argc, argv, &i), "--delta-min");
        } else if (arg == "--rho-update") {
            options.incremental_rho_update =
                ParseFloat(RequireValue(argc, argv, &i), "--rho-update");
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.model_dir.empty())
        throw std::invalid_argument("--model is required");
    if (options.omp_threads < 1) options.omp_threads = 1;
    if (options.step_sec <= 0.0f) throw std::invalid_argument("--step must be positive");
    return options;
}

struct DiarResult {
    std::vector<c1::Part> parts;
    std::vector<c1::Segment> rttm;
    std::vector<std::string> speakers;   // one id per local cluster
    int clusters = 0;
};

DiarResult RunRequest(c1::DiarEngine* engine, diar::SpeakerTable* table,
                      const Options& options, const std::string& wav_path) {
    DiarResult result;
    const auto begin = std::chrono::steady_clock::now();
    WavData wav = read_wav(wav_path);
    std::vector<float> centroids;
    std::vector<c1::Segment> rttm;
    std::vector<c1::Part> parts;
    int clusters = 0;
    int cache_hits = -1;
    c1::IncrementalParams inc;
    const c1::IncrementalParams* inc_ptr = nullptr;
    if (options.incremental) {
        inc.use_plda = options.incremental_plda;
        inc.delta_new = options.incremental_delta_new;
        inc.delta_min = options.incremental_delta_min;
        inc.rho_update = options.incremental_rho_update;
        inc_ptr = &inc;
    }
    engine->Run(wav, options.step_sec, rttm, &parts, &centroids, &clusters,
                &cache_hits, inc_ptr);

    std::vector<long long> weights(clusters, 0);
    for (const c1::Part& part : parts)
        if (part.cluster >= 0 && part.cluster < clusters)
            weights[part.cluster] += static_cast<long long>(part.end - part.begin);
    const long long now = static_cast<long long>(std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now()));
    if (options.assign_mode == "online") {
        diar::SpeakerTable::OnlineParams params;
        params.threshold = options.speaker_match_threshold;
        params.min_new_sec = options.online_min_new_sec;
        params.min_update_sec = options.online_min_update_sec;
        params.max_speakers = options.online_max_speakers;
        // Per-segment raw-window fallback (WAVs the pipeline names "*.raw.wav")
        // has no birth rights: a fallback segment blends multiple voices into
        // one centroid, so it must never create a phantom speaker nor drift an
        // existing one.  Identify-only against the nearest table entry.
        const bool identify_only =
            wav_path.size() >= 8 &&
            wav_path.compare(wav_path.size() - 8, 8, ".raw.wav") == 0;
        if (identify_only) {
            params.max_speakers = 0;
            params.min_update_sec = 1e9f;
        }
        result.speakers = table->AssignOnline(centroids, c1::EMB_DIM, weights,
                                              params, now);
    } else {
        result.speakers = table->Assign(centroids, c1::EMB_DIM, weights,
                                        options.speaker_match_threshold, now);
    }
    if (!parts.empty()) table->Save(options.table_path);

    result.parts = std::move(parts);
    result.rttm = std::move(rttm);
    result.clusters = clusters;

    if (!options.rttm_dir.empty()) {
        std::string uri = wav_path.substr(wav_path.find_last_of('/') + 1);
        const std::size_t dot = uri.find_last_of('.');
        if (dot != std::string::npos) uri = uri.substr(0, dot);
        std::ofstream out(options.rttm_dir + "/" + uri + ".rttm");
        if (out) {
            for (const c1::Segment& segment : result.rttm) {
                const std::string id =
                    (segment.spk >= 0 &&
                     segment.spk < static_cast<int>(result.speakers.size()))
                        ? result.speakers[segment.spk] : std::string("speaker-0");
                out << "SPEAKER " << uri << " 1 " << std::fixed
                    << std::setprecision(3) << segment.start << " " << segment.dur
                    << " <NA> <NA> " << id << " <NA> <NA>\n";
            }
        }
    }

    const double seconds = wav.samples.size() / (double)c1::SAMPLE_RATE;
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    std::fprintf(stderr,
                 "[diar] wav=%s audio=%.2fs clusters=%d centroids=%zu parts=%zu took=%.0fms rtf=%.4f cache=%d\n",
                 wav_path.c_str(), seconds, clusters,
                 centroids.size() / c1::EMB_DIM, result.parts.size(),
                 elapsed * 1000.0, seconds > 0 ? elapsed / seconds : 0.0,
                 cache_hits);
    return result;
}

std::string PartsPayload(const DiarResult& result) {
    // Identity lookup can fail when the engine has no centroid at all; treat
    // the whole response as "no speaker parts" so the pipeline falls back to
    // whole-segment ASR instead of receiving an unparsable empty id.
    for (const c1::Part& part : result.parts) {
        if (part.cluster < 0 ||
            part.cluster >= static_cast<int>(result.speakers.size()) ||
            result.speakers[part.cluster].empty())
            return "0\n";
    }
    std::ostringstream out;
    out << result.parts.size() << "\n";
    for (const c1::Part& part : result.parts) {
        out << part.begin << " " << part.end << " "
            << result.speakers[part.cluster] << "\n";
    }
    return out.str();
}

void WriteRttm(const std::string& path, const std::string& uri,
               const std::vector<c1::Segment>& rttm) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    for (const auto& s : rttm)
        std::fprintf(f, "SPEAKER %s 1 %7.3f %7.3f <NA> <NA> SPEAKER_%02d <NA> <NA>\n",
                     uri.c_str(), s.start, s.dur, s.spk);
    std::fclose(f);
}

int RunOneshot(c1::DiarEngine* engine, diar::SpeakerTable* table,
               const Options& options) {
    const DiarResult result = RunRequest(engine, table, options, options.wav_path);
    std::fputs(PartsPayload(result).c_str(), stdout);
    if (!options.out_rttm.empty()) {
        std::string uri = options.wav_path.substr(options.wav_path.find_last_of('/') + 1);
        const std::size_t dot = uri.find_last_of('.');
        if (dot != std::string::npos) uri = uri.substr(0, dot);
        WriteRttm(options.out_rttm, uri, result.rttm);
    }
    return 0;
}

int RunDumpFeatures(c1::DiarEngine* engine, const Options& options) {
    WavData wav = read_wav(options.wav_path);
    engine->DumpFeatures(wav, options.step_sec, options.dump_features);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    int response_fd = 3;
    try {
        Options options = ParseArgs(argc, argv);
        response_fd = options.response_fd;
        omp_set_num_threads(options.omp_threads);

        if (AX_SYS_Init() != 0) throw std::runtime_error("AX_SYS_Init failed");
        c1::DiarEngine engine;
        if (!engine.Init(options.model_dir))
            throw std::runtime_error("diar engine init failed: " + options.model_dir);
        diar::SpeakerTable table;
        table.Load(options.table_path);
        std::fprintf(stderr,
                     "[diar] model=%s table=%s step=%.2f threshold=%.2f threads=%d "
                     "speakers=%zu\n",
                     options.model_dir.c_str(), options.table_path.c_str(),
                     options.step_sec, options.speaker_match_threshold,
                     options.omp_threads, table.size());
        if (options.assign_mode == "online") {
            std::fprintf(stderr,
                         "[diar] online assign: hungarian min_new=%.1fs "
                         "min_update=%.1fs max_speakers=%d\n",
                         options.online_min_new_sec,
                         options.online_min_update_sec,
                         options.online_max_speakers);
        }
        if (options.incremental) {
            std::fprintf(stderr,
                         "[diar] incremental clustering: %s delta_new=%.2f delta_min=%.2f "
                         "rho_update=%.2f\n",
                         options.incremental_plda ? "plda" : "raw",
                         options.incremental_delta_new,
                         options.incremental_delta_min,
                         options.incremental_rho_update);
        }

        if (!options.dump_features.empty())
            return RunDumpFeatures(&engine, options);
        if (options.oneshot) {
            // One-shot analyses a whole session in one Run() call; start from
            // an empty incremental speaker set.
            engine.ResetIncremental();
            return RunOneshot(&engine, &table, options);
        }
        // Streaming loop: sub-window-at-a-time requests keep the incremental
        // clustering state across calls (see --incremental).  A session reset
        // is signalled by an empty request (size 0 below).
        if (options.incremental) engine.ResetIncremental();
        if (!WriteFrame(response_fd, 'R', "ready")) return 1;

        uint32_t request_size = 0;
        while (ReadAll(STDIN_FILENO, &request_size, sizeof(request_size))) {
            if (request_size == 0) {
                // Empty request = session boundary: reset the incremental
                // clustering state and the persistent speaker table.
                if (options.incremental) engine.ResetIncremental();
                table.Clear();
                if (!WriteFrame(response_fd, 'R', "reset")) break;
                continue;
            }
            if (request_size > 64U * 1024U) {
                if (!WriteFrame(response_fd, 'E', "invalid request size")) break;
                continue;
            }
            std::string wav_path(request_size, '\0');
            if (!ReadAll(STDIN_FILENO, wav_path.data(), wav_path.size())) break;
            try {
                const DiarResult result = RunRequest(&engine, &table, options, wav_path);
                if (!WriteFrame(response_fd, 'R', PartsPayload(result))) break;
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[diar] request failed: %s\n", error.what());
                if (!WriteFrame(response_fd, 'E', error.what())) break;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        WriteFrame(response_fd, 'E', error.what());
        std::fprintf(stderr, "[diar] worker error: %s\n", error.what());
        return 1;
    }
}

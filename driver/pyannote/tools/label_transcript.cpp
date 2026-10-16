// label-transcript: turn an audio file into a speaker-labelled transcript
//
//   Speaker_0: [0.000 63.810] 文本...
//   Speaker_1: [63.810 70.471] 文本...
//
// Pipeline:
//   1. fork build/diar-worker (one-shot) -> global speaker timeline
//      (stdout: partition parts; --out: native RTTM with overlap)
//   2. intersect the partition with the RTTM activity, merge/split into turns
//   3. AX_ASR_RunFile per turn chunk -> text
//   4. print "Speaker_K: [start end] text" and optionally write RTTM
//
// The AX Engine interposer below mirrors framework/audio_pipeline.cpp: the
// prebuilt SenseVoice library initializes AX Engine with VNPU disabled, but
// this tool (like the chain) runs in BIG_LITTLE mode.
#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "ax_asr_api.h"

// ---------------------------------------------------------------------------
// AX Engine interposer (same policy as framework/audio_pipeline.cpp)
// ---------------------------------------------------------------------------
void* ResolveAxEngineSymbol(const char* name) {
  static void* engine = dlopen("libax_engine.so", RTLD_NOW | RTLD_LOCAL);
  return engine != nullptr ? dlsym(engine, name) : nullptr;
}

extern "C" int32_t AX_ENGINE_Init(void* attributes) {
  using InitFn = int32_t (*)(void*);
  static auto real_init =
      reinterpret_cast<InitFn>(ResolveAxEngineSymbol("AX_ENGINE_Init"));
  if (real_init == nullptr) return -1;
  constexpr int32_t kBigLittleMode = 2;
  if (attributes == nullptr) return real_init(attributes);
  *static_cast<int32_t*>(attributes) = kBigLittleMode;
  const int32_t status = real_init(attributes);
  if (status == 0)
    std::cerr << "[npu] AX Engine mode=BIG_LITTLE (2+1)\n";
  else
    std::cerr << "error: AX Engine BIG_LITTLE init failed: 0x" << std::hex
              << static_cast<uint32_t>(status) << std::dec << "\n";
  return status;
}

extern "C" int32_t AX_ENGINE_CreateHandle(void** handle, const void* model,
                                          uint32_t size) {
  using CreateFn = int32_t (*)(void**, const void*, uint32_t);
  using SetAffinityFn = int32_t (*)(void*, uint32_t);
  using GetAffinityFn = int32_t (*)(void*, uint32_t*);
  using GetModelTypeFn = int32_t (*)(void*, int32_t*);
  using DestroyFn = int32_t (*)(void*);
  static auto real_create =
      reinterpret_cast<CreateFn>(ResolveAxEngineSymbol("AX_ENGINE_CreateHandle"));
  static auto set_affinity =
      reinterpret_cast<SetAffinityFn>(ResolveAxEngineSymbol("AX_ENGINE_SetAffinity"));
  static auto get_affinity =
      reinterpret_cast<GetAffinityFn>(ResolveAxEngineSymbol("AX_ENGINE_GetAffinity"));
  static auto get_model_type =
      reinterpret_cast<GetModelTypeFn>(ResolveAxEngineSymbol("AX_ENGINE_GetHandleModelType"));
  static auto destroy =
      reinterpret_cast<DestroyFn>(ResolveAxEngineSymbol("AX_ENGINE_DestroyHandle"));
  if (real_create == nullptr || set_affinity == nullptr ||
      get_affinity == nullptr || destroy == nullptr)
    return -1;
  const int32_t status = real_create(handle, model, size);
  if (status != 0) return status;
  int32_t model_type = -1;
  const int32_t model_type_status =
      get_model_type ? get_model_type(*handle, &model_type) : -1;
  const uint32_t affinity = (model_type == 1) ? 0x1U : 0x2U;
  const int32_t affinity_status = set_affinity(*handle, affinity);
  uint32_t actual = 0;
  const int32_t query_status = get_affinity(*handle, &actual);
  if (affinity_status != 0 || query_status != 0 || actual != affinity) {
    std::cerr << "error: cannot bind model to affinity=0x" << std::hex
              << affinity << std::dec << "\n";
    destroy(*handle);
    *handle = nullptr;
    return affinity_status != 0 ? affinity_status : -1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
namespace {

constexpr int kRate = 16000;

struct Interval {
  std::size_t begin = 0;    // 16k samples
  std::size_t end = 0;
  std::string speaker;
};

const char* RequireValue(int argc, char** argv, int* index) {
  if (*index + 1 >= argc)
    throw std::invalid_argument(std::string("missing value for ") + argv[*index]);
  return argv[++*index];
}

std::vector<int16_t> ReadWav16(const std::string& path, int* rate) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  char hdr[12];
  in.read(hdr, 12);
  if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0)
    throw std::runtime_error("not RIFF/WAVE: " + path);
  int channels = 1, bits = 16;
  std::vector<int16_t> interleaved;
  auto le32 = [](const char* p) {
    return static_cast<int32_t>(static_cast<uint8_t>(p[0]) |
                                (static_cast<uint8_t>(p[1]) << 8) |
                                (static_cast<uint8_t>(p[2]) << 16) |
                                (static_cast<uint8_t>(p[3]) << 24));
  };
  while (true) {
    char ch[8];
    if (!in.read(ch, 8)) break;
    const int32_t size = le32(ch + 4);
    if (std::memcmp(ch, "fmt ", 4) == 0) {
      std::vector<char> fmt(size);
      in.read(fmt.data(), size);
      channels = static_cast<uint8_t>(fmt[2]) | (static_cast<uint8_t>(fmt[3]) << 8);
      *rate = le32(fmt.data() + 4);
      bits = static_cast<uint8_t>(fmt[14]) | (static_cast<uint8_t>(fmt[15]) << 8);
    } else if (std::memcmp(ch, "data", 4) == 0) {
      interleaved.resize(size / 2);
      in.read(reinterpret_cast<char*>(interleaved.data()), size);
    } else {
      in.seekg(size, std::ios::cur);
    }
  }
  if (bits != 16) throw std::runtime_error("only 16-bit PCM supported");
  if (channels <= 1) return interleaved;
  std::vector<int16_t> mono(interleaved.size() / channels);
  for (std::size_t i = 0; i < mono.size(); ++i) {
    int acc = 0;
    for (int c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
    mono[i] = static_cast<int16_t>(acc / channels);
  }
  return mono;
}

bool WriteWav(const std::string& path, const int16_t* samples, std::size_t count) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  const uint32_t bytes = static_cast<uint32_t>(count * sizeof(int16_t));
  const uint32_t riff = 36 + bytes;
  out.write("RIFF", 4);
  out.write(reinterpret_cast<const char*>(&riff), 4);
  out.write("WAVEfmt ", 8);
  const uint32_t fmt_size = 16;
  const uint16_t format = 1, channels = 1, bits = 16, block = 2;
  const uint32_t rate = kRate, byte_rate = kRate * 2;
  out.write(reinterpret_cast<const char*>(&fmt_size), 4);
  out.write(reinterpret_cast<const char*>(&format), 2);
  out.write(reinterpret_cast<const char*>(&channels), 2);
  out.write(reinterpret_cast<const char*>(&rate), 4);
  out.write(reinterpret_cast<const char*>(&byte_rate), 4);
  out.write(reinterpret_cast<const char*>(&block), 2);
  out.write(reinterpret_cast<const char*>(&bits), 2);
  out.write("data", 4);
  out.write(reinterpret_cast<const char*>(&bytes), 4);
  out.write(reinterpret_cast<const char*>(samples), bytes);
  return static_cast<bool>(out);
}

// Runs the resident diar-worker in one-shot mode.  Returns the partition parts
// (stdout) and writes the native RTTM to rttm_path.
bool RunDiarWorker(const std::string& worker, const std::string& model_dir,
                   const std::string& wav, float threshold, int threads,
                   const std::string& rttm_path,
                   std::vector<Interval>* parts,
                   std::vector<Interval>* activity) {
  int out_pipe[2] = {-1, -1};
  if (pipe(out_pipe) != 0) return false;
  const pid_t pid = fork();
  if (pid < 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    return false;
  }
  if (pid == 0) {
    dup2(out_pipe[1], STDOUT_FILENO);
    close(out_pipe[0]);
    close(out_pipe[1]);
    std::string thr = std::to_string(threshold);
    std::string t = std::to_string(threads);
    execl(worker.c_str(), worker.c_str(), "--model", model_dir.c_str(),
          "--wav", wav.c_str(), "--out", rttm_path.c_str(),
          "--threshold", thr.c_str(), "--threads", t.c_str(),
          static_cast<char*>(nullptr));
    _exit(127);
  }
  close(out_pipe[1]);
  std::string payload;
  char buffer[4096];
  ssize_t n = 0;
  while ((n = read(out_pipe[0], buffer, sizeof(buffer))) > 0)
    payload.append(buffer, static_cast<std::size_t>(n));
  close(out_pipe[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "[label] diar worker failed (status=" << status << ")\n";
    return false;
  }
  std::istringstream in(payload);
  std::size_t count = 0;
  if (!(in >> count)) {
    std::cerr << "[label] no parts from diar worker\n";
    return false;
  }
  for (std::size_t i = 0; i < count; ++i) {
    std::size_t begin = 0, end = 0;
    std::string speaker;
    if (!(in >> begin >> end >> speaker)) break;
    if (end > begin) parts->push_back({begin, end, speaker});
  }
  std::ifstream rttm(rttm_path);
  std::string line;
  while (std::getline(rttm, line)) {
    std::istringstream ls(line);
    std::string token, uri, type, label, nav;
    double start = 0, dur = 0;
    if (!(ls >> token >> uri >> type >> start >> dur >> nav >> nav >> label))
      continue;
    const std::size_t begin = static_cast<std::size_t>(start * kRate);
    const std::size_t end = static_cast<std::size_t>((start + dur) * kRate);
    if (end > begin) activity->push_back({begin, end, label});
  }
  return true;
}

std::string Transcribe(AX_ASR_HANDLE handle, const std::string& language,
                       const int16_t* samples, std::size_t count,
                       const std::string& tmp_wav,
                       int (*run_file)(AX_ASR_HANDLE, const char*,
                                       const char*, char**)) {
  if (!WriteWav(tmp_wav, samples, count)) return {};
  char* result = nullptr;
  const int status = run_file(handle, tmp_wav.c_str(), language.c_str(),
                              &result);
  if (status != 0 || result == nullptr) {
    if (result != nullptr) std::free(result);
    return {};
  }
  std::string text(result);
  std::free(result);
  return text;
}

std::string Trim(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

void MergeTurns(std::vector<Interval>* turns, std::size_t max_gap,
                std::size_t min_len) {
  std::vector<Interval> merged;
  for (auto& t : *turns) {
    if (t.end <= t.begin) continue;
    if (!merged.empty() && merged.back().speaker == t.speaker &&
        t.begin <= merged.back().end + max_gap) {
      // Adjacent or overlapping pieces of the same speaker (the native RTTM
      // emits overlapping activity intervals) collapse into one turn.
      merged.back().end = std::max(merged.back().end, t.end);
      continue;
    }
    merged.push_back(t);
  }
  std::vector<Interval> kept;
  for (std::size_t i = 0; i < merged.size(); ++i) {
    if (merged[i].end - merged[i].begin >= min_len ||
        merged.size() == 1) {
      kept.push_back(merged[i]);
    } else if (!kept.empty()) {
      // Absorb a too-short turn into the previous one.
      kept.back().end = merged[i].end;
    } else if (i + 1 < merged.size()) {
      merged[i + 1].begin = merged[i].begin;
    }
  }
  *turns = std::move(kept);
}

// Silence-aware split point: center of the quietest 200 ms window in [lo, hi]
// that is actually below the silence threshold (word-internal stop closures
// are near-zero-energy too; a 200 ms window averages them with voiced
// neighbours).  Returns `hi` (hard cut) when no real pause is found.
std::size_t FindQuietPoint(const std::vector<int16_t>& pcm, std::size_t lo,
                           std::size_t hi) {
  constexpr std::size_t kWin = 3200;   // 200 ms @ 16 kHz
  constexpr std::size_t kStep = 320;   // 20 ms
  constexpr double kSilenceRms = 200.0;
  if (hi <= lo + kWin || pcm.size() <= kWin) return hi;
  const std::size_t last = std::min(hi, pcm.size() - kWin);
  std::size_t best = hi;
  double best_rms = kSilenceRms;
  for (std::size_t c = lo; c <= last; c += kStep) {
    double acc = 0.0;
    for (std::size_t k = c; k < c + kWin; ++k) {
      const double s = static_cast<double>(pcm[k]);
      acc += s * s;
    }
    const double rms = std::sqrt(acc / kWin);
    if (rms < best_rms) {
      best_rms = rms;
      best = c + kWin / 2;
    }
  }
  return best;
}

// Merge a freshly transcribed chunk into the accumulated turn text, dropping
// the overlap repeated at the seam (each chunk re-feeds its left context).
std::string MergeChunkText(const std::string& acc, const std::string& piece,
                           const std::string& language) {
  if (acc.empty()) return piece;
  if (piece.empty()) return acc;
  if (language == "en") {
    std::vector<std::string> a, b;
    std::istringstream sa(acc), sb(piece);
    std::string w;
    while (sa >> w) a.push_back(w);
    while (sb >> w) b.push_back(w);
    const std::size_t maxl = std::min<std::size_t>(6, std::min(a.size(), b.size()));
    for (std::size_t l = maxl; l >= 1; --l) {
      bool same = true;
      for (std::size_t i = 0; i < l; ++i) {
        if (a[a.size() - l + i] != b[i]) {
          same = false;
          break;
        }
      }
      if (same) {
        std::string merged = acc;
        for (std::size_t i = l; i < b.size(); ++i) merged += " " + b[i];
        return merged;
      }
    }
    return acc + " " + piece;
  }
  const std::size_t maxl =
      std::min<std::size_t>(8, std::min(acc.size(), piece.size()));
  for (std::size_t l = maxl; l >= 1; --l) {
    if (acc.compare(acc.size() - l, l, piece, 0, l) == 0)
      return acc + piece.substr(l);
  }
  return acc + piece;
}

}  // namespace

int main(int argc, char** argv) {
  std::string wav, model_dir, worker = "build/diar-worker";
  std::string asr_lib = "driver/asr/sensevoice/lib/libax_asr_api.so";
  std::string asr_model = "models-for-asr-chain/asr/sensevoice";
  std::string asr_share_dir;
  std::string language = "zh";
  std::string out_path;
  std::string rttm_out;
  float threshold = 0.55f;
  int threads = 4;
  double max_chunk_sec = 30.0;
  double merge_gap_sec = 0.5;
  double min_turn_sec = 0.3;
  double pad_sec = 0.3;
  double chunk_search_sec = 0.0;
  bool inner_pad = false;
  bool dedupe = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--wav") wav = RequireValue(argc, argv, &i);
      else if (arg == "--model") model_dir = RequireValue(argc, argv, &i);
      else if (arg == "--worker") worker = RequireValue(argc, argv, &i);
      else if (arg == "--asr-lib") asr_lib = RequireValue(argc, argv, &i);
      else if (arg == "--asr-model") asr_model = RequireValue(argc, argv, &i);
      else if (arg == "--asr-share-dir") asr_share_dir = RequireValue(argc, argv, &i);
      else if (arg == "--language") language = RequireValue(argc, argv, &i);
      else if (arg == "--out") out_path = RequireValue(argc, argv, &i);
      else if (arg == "--rttm") rttm_out = RequireValue(argc, argv, &i);
      else if (arg == "--threshold") threshold = std::stof(RequireValue(argc, argv, &i));
      else if (arg == "--threads") threads = std::stoi(RequireValue(argc, argv, &i));
      else if (arg == "--max-chunk") max_chunk_sec = std::stod(RequireValue(argc, argv, &i));
      else if (arg == "--merge-gap") merge_gap_sec = std::stod(RequireValue(argc, argv, &i));
      else if (arg == "--min-turn") min_turn_sec = std::stod(RequireValue(argc, argv, &i));
      else if (arg == "--pad") pad_sec = std::stod(RequireValue(argc, argv, &i));
      else if (arg == "--chunk-search") chunk_search_sec = std::stod(RequireValue(argc, argv, &i));
      else if (arg == "--inner-pad") inner_pad = std::stoi(RequireValue(argc, argv, &i)) != 0;
      else if (arg == "--dedupe") dedupe = std::stoi(RequireValue(argc, argv, &i)) != 0;
      else {
        std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
        return 2;
      }
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "argument error: %s\n", error.what());
    return 2;
  }
  if (wav.empty() || out_path.empty()) {
    std::fprintf(stderr,
                 "usage: label-transcript --wav in.wav --out out.txt "
                 "[--model DIR] [--worker build/diar-worker] "
                 "[--asr-lib LIB] [--asr-model DIR] [--language zh] "
                 "[--rttm out.rttm] [--max-chunk 30] [--merge-gap 0.5] "
                 "[--pad 0.3] [--chunk-search 0] [--inner-pad 0] [--dedupe 0]\n");
    return 2;
  }
  if (!asr_share_dir.empty()) setenv("AX_SENSEVOICE_SHARE_DIR",
                                     asr_share_dir.c_str(), 1);
  if (model_dir.empty()) model_dir = "models-for-asr-chain/pyannote";

  try {
    // Initialize ASR first so AX Engine is in BIG_LITTLE before the diar
    // worker (which then auto-falls back to the same mode).
    void* lib = dlopen(asr_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
      std::fprintf(stderr, "cannot load ASR library %s: %s\n", asr_lib.c_str(),
                   dlerror());
      return 1;
    }
    auto asr_init = reinterpret_cast<AX_ASR_HANDLE (*)(AX_ASR_TYPE_E, const char*)>(
        dlsym(lib, "AX_ASR_Init"));
    auto asr_uninit = reinterpret_cast<void (*)(AX_ASR_HANDLE)>(
        dlsym(lib, "AX_ASR_Uninit"));
    auto asr_run_file = reinterpret_cast<int (*)(AX_ASR_HANDLE, const char*,
                                                 const char*, char**)>(
        dlsym(lib, "AX_ASR_RunFile"));
    if (asr_init == nullptr || asr_uninit == nullptr || asr_run_file == nullptr) {
      std::fprintf(stderr, "ASR library has no AX_ASR_Init/Uninit/RunFile\n");
      return 1;
    }
    AX_ASR_HANDLE asr = asr_init(AX_SENSEVOICE, asr_model.c_str());
    if (asr == nullptr) {
      std::fprintf(stderr, "AX_ASR_Init failed: %s\n", asr_model.c_str());
      return 1;
    }

    int rate = kRate;
    const std::vector<int16_t> pcm = ReadWav16(wav, &rate);
    if (rate != kRate)
      std::fprintf(stderr, "[label] warning: input is %d Hz (model expects 16k)\n",
                   rate);

    const std::string rttm_path = out_path + ".raw.rttm";
    std::vector<Interval> parts, activity;
    if (!RunDiarWorker(worker, model_dir, wav, threshold, threads, rttm_path,
                       &parts, &activity)) {
      asr_uninit(asr);
      return 1;
    }
    if (parts.empty() || activity.empty()) {
      std::fprintf(stderr, "[label] diarization returned no speech\n");
      asr_uninit(asr);
      return 1;
    }

    // Intersect the partition with the RTTM activity: every speech frame gets
    // the partition's speaker.
    std::vector<Interval> speech;
    for (const auto& part : parts) {
      for (const auto& act : activity) {
        const std::size_t begin = std::max(part.begin, act.begin);
        const std::size_t end = std::min(part.end, act.end);
        if (end > begin) speech.push_back({begin, end, part.speaker});
      }
    }
    std::stable_sort(speech.begin(), speech.end(),
                     [](const Interval& a, const Interval& b) {
                       return a.begin < b.begin;
                     });
    MergeTurns(&speech, static_cast<std::size_t>(merge_gap_sec * kRate),
               static_cast<std::size_t>(min_turn_sec * kRate));

    // Speaker id mapping by first appearance (Speaker_0, Speaker_1, ...).
    std::vector<std::string> order;
    for (const auto& turn : speech)
      if (std::find(order.begin(), order.end(), turn.speaker) == order.end())
        order.push_back(turn.speaker);

    std::ofstream out(out_path);
    std::ofstream rttm(rttm_out.empty() ? "/dev/null" : rttm_out);
    const std::size_t max_chunk = static_cast<std::size_t>(max_chunk_sec * kRate);
    const std::size_t pad = static_cast<std::size_t>(pad_sec * kRate);
    const std::size_t search = static_cast<std::size_t>(chunk_search_sec * kRate);
    std::size_t turn_index = 0;
    for (std::size_t ti = 0; ti < speech.size(); ++ti) {
      const auto& turn = speech[ti];
      const auto it = std::find(order.begin(), order.end(), turn.speaker);
      const std::size_t speaker_index =
          static_cast<std::size_t>(std::distance(order.begin(), it));
      std::string text;
      std::size_t begin = turn.begin;
      while (begin < turn.end) {
        const std::size_t limit = std::min(begin + max_chunk, turn.end);
        std::size_t end = limit;
        // Long turn: cut at the quietest point within the last
        // `--chunk-search` seconds before the size limit instead of an
        // arbitrary hard cut (keeps chunk seams inside pauses).
        if (limit < turn.end && search > 0 && limit > begin + search)
          end = FindQuietPoint(pcm, limit - search, limit);
        // --pad: left context for every chunk and trailing context at the
        // turn end.  Plain clamp to the audio range only: empirically the
        // "never cross the neighbour speaker" clamp loses recall (R8003
        // equal 79.3% -> 77.3%) because the 0.3 s of neighbour context is
        // more often a word onset than cross-talk.
        std::size_t pbegin = begin;
        if (pad > 0 && (begin == turn.begin || inner_pad)) {
          pbegin = begin > pad ? begin - pad : 0;
        }
        std::size_t pend = end;
        if (end == turn.end && pad > 0)
          pend = std::min(pcm.size(), end + pad);
        if (pend <= pbegin) {
          begin = end;
          continue;
        }
        const std::string piece =
            Transcribe(asr, language, pcm.data() + pbegin, pend - pbegin,
                       out_path + ".chunk.wav", asr_run_file);
        if (!piece.empty()) {
          if (dedupe) {
            text = MergeChunkText(text, piece, language);
          } else {
            if (!text.empty() && language == "en") text += ' ';
            text += piece;
          }
        }
        begin = end;
      }
      text = Trim(text);
      std::ostringstream stamp;
      stamp.setf(std::ios::fixed);
      stamp.precision(3);
      stamp << "Speaker_" << speaker_index << ": ["
            << turn.begin / static_cast<double>(kRate) << " "
            << turn.end / static_cast<double>(kRate) << "] " << text << "\n";
      if (!text.empty()) {
        out << stamp.str();
        if (rttm.is_open())
          rttm << "SPEAKER " << wav << " 1 "
               << turn.begin / static_cast<double>(kRate) << " "
               << (turn.end - turn.begin) / static_cast<double>(kRate)
               << " <NA> <NA> Speaker_" << speaker_index << " <NA> <NA>\n";
        ++turn_index;
      }
    }
    std::remove((out_path + ".chunk.wav").c_str());
    std::remove(rttm_path.c_str());
    std::cerr << "[label] turns=" << turn_index << " speakers=" << order.size()
              << " -> " << out_path << "\n";
    asr_uninit(asr);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "label-transcript: %s\n", error.what());
    return 1;
  }
}

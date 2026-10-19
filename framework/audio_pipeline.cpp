#include "configui/apm-config.h"
#include "text_agreement.hpp"
#include "fastenhance_c_api.h"
#include "ten_vad.h"
#include "ax_asr_api.h"
#include <future>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <vector>


namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using WallClock = std::chrono::system_clock;

// FastEnhance and the prebuilt TEN-VAD library both initialize AX Engine in
// this process. TEN-VAD was built with VIRTUAL_NPU_DISABLE hard-coded, so
// interpose its AX Engine calls and enforce one process-wide BIG_LITTLE
// policy. On AX650, affinity 0x1 is the BIG two-core partition required by
// NPU2 SenseVoice models, while affinity 0x2 is the LITTLE single-core
// partition used by the NPU1 FastEnhance and TEN-VAD models.

void* ResolveAxEngineSymbol(const char* name) {
  static void* engine = dlopen("libax_engine.so", RTLD_NOW | RTLD_LOCAL);
  return engine != nullptr ? dlsym(engine, name) : nullptr;
}

// Active device mode picked by the Init interposer: 0 = DISABLE (unified),
// 2 = BIG_LITTLE.  The CreateHandle interposer consults it to decide whether
// the partition-affinity bind is a hard requirement.
int32_t g_ax_engine_mode = -1;

extern "C" int32_t AX_ENGINE_Init(void* attributes) {
  using InitFn = int32_t (*)(void*);
  static auto real_init = reinterpret_cast<InitFn>(
      ResolveAxEngineSymbol("AX_ENGINE_Init"));
  if (real_init == nullptr) return -1;
  if (attributes == nullptr) return real_init(attributes);
  constexpr int32_t kDisableMode = 0;
  constexpr int32_t kBigLittleMode = 2;
  // Prefer VNPU DISABLE: every chain model (NPU1/NPU2 compiled) loads there,
  // and the co-resident tools (the streaming-ASR asr_server, the nemotron
  // diarization worker) initialize DISABLE themselves, so one shared mode
  // keeps all processes out of the device-global mode conflict.  Fall back
  // to BIG_LITTLE when another process already holds DISABLE.
  *static_cast<int32_t*>(attributes) = kDisableMode;
  int32_t status = real_init(attributes);
  if (status == 0) {
    g_ax_engine_mode = kDisableMode;
    std::cerr << "[npu] AX Engine mode=DISABLE (unified NPU)\n";
    return 0;
  }
  std::cerr << "[npu] VNPU DISABLE init failed: 0x" << std::hex
            << static_cast<uint32_t>(status) << std::dec
            << "; retrying BIG_LITTLE\n";
  *static_cast<int32_t*>(attributes) = kBigLittleMode;
  status = real_init(attributes);
  if (status == 0) {
    g_ax_engine_mode = kBigLittleMode;
    std::cerr << "[npu] AX Engine mode=BIG_LITTLE (2+1)\n";
    return 0;
  }
  std::cerr << "error: AX Engine init failed with both VNPU DISABLE and "
               "BIG_LITTLE: 0x" << std::hex << static_cast<uint32_t>(status)
            << std::dec
            << " (reboot may be required after using another VNPU mode)\n";
  return status;
}

extern "C" int32_t AX_ENGINE_CreateHandle(void** handle, const void* model,
                                            uint32_t size) {
  using CreateFn = int32_t (*)(void**, const void*, uint32_t);
  using SetAffinityFn = int32_t (*)(void*, uint32_t);
  using GetAffinityFn = int32_t (*)(void*, uint32_t*);
  using GetModelTypeFn = int32_t (*)(void*, int32_t*);
  using DestroyFn = int32_t (*)(void*);
  static auto real_create = reinterpret_cast<CreateFn>(
      ResolveAxEngineSymbol("AX_ENGINE_CreateHandle"));
  static auto set_affinity = reinterpret_cast<SetAffinityFn>(
      ResolveAxEngineSymbol("AX_ENGINE_SetAffinity"));
  static auto get_affinity = reinterpret_cast<GetAffinityFn>(
      ResolveAxEngineSymbol("AX_ENGINE_GetAffinity"));
  static auto get_model_type = reinterpret_cast<GetModelTypeFn>(
      ResolveAxEngineSymbol("AX_ENGINE_GetHandleModelType"));
  static auto destroy = reinterpret_cast<DestroyFn>(
      ResolveAxEngineSymbol("AX_ENGINE_DestroyHandle"));
  if (real_create == nullptr || set_affinity == nullptr ||
      get_affinity == nullptr || destroy == nullptr)
    return -1;
  const int32_t status = real_create(handle, model, size);
  if (status != 0) return status;

  int32_t model_type = -1;
  const int32_t model_type_status =
      get_model_type ? get_model_type(*handle, &model_type) : -1;
  const uint32_t kRealtimeVnpu = (model_type == 1) ? 0x1U : 0x2U;
  if (g_ax_engine_mode == 0) {
    // Unified NPU (DISABLE): the partition affinity is at best a scheduling
    // hint, so bind best-effort and never fail on it.
    set_affinity(*handle, kRealtimeVnpu);
    uint32_t actual = 0;
    get_affinity(*handle, &actual);
    std::cerr << "[npu] model loaded (DISABLE mode, affinity hint 0x"
              << std::hex << kRealtimeVnpu << " actual 0x" << actual
              << std::dec;
    if (model_type_status == 0) std::cerr << " model_type=" << model_type;
    std::cerr << ")\n";
    return 0;
  }
  const int32_t affinity_status = set_affinity(*handle, kRealtimeVnpu);
  uint32_t actual = 0;
  const int32_t query_status = get_affinity(*handle, &actual);
  if (affinity_status != 0 || query_status != 0 ||
      actual != kRealtimeVnpu) {
    std::cerr << "error: cannot bind realtime model to affinity=0x"
              << std::hex << kRealtimeVnpu << ", set_status=0x"
              << affinity_status << ", query_status=0x" << query_status
              << ", actual=0x" << actual << std::dec;
    if (model_type_status == 0) std::cerr << ", model_type=" << model_type;
    std::cerr << "\n";
    destroy(*handle);
    *handle = nullptr;
    return affinity_status != 0 ? affinity_status : -1;
  }
  std::cerr << "[npu] realtime affinity=0x" << std::hex << kRealtimeVnpu
            << std::dec;
  std::cerr << (kRealtimeVnpu == 0x1 ? " (NPU2/BIG, two cores)" : " (NPU1/LITTLE, one core)");
  if (model_type_status == 0) std::cerr << " model_type=" << model_type;
  std::cerr << "\n";
  return 0;
}

namespace {

constexpr int kRateIn = 48000;
constexpr int kRateOut = 16000;
constexpr size_t kEnhanceHop = 512;

// diar-worker assignment args (defined after the worker classes; the resident
// worker's StartProcess needs it before its definition).
void AddAssignArgs(std::vector<std::string>* args,
                   const apm_example::AppConfig::Diarization& config);

std::string ShellQuote(const std::string& s);
std::string TimeStamp(WallClock::time_point t);

std::string ResolveLibrary(const std::string& path) {
  if (path.empty() || path.find('/') != std::string::npos) return path;
  return "./" + path;
}

std::string AbsolutePath(const std::string& path) {
  if (path.empty()) return path;
  std::error_code ec;
  const fs::path absolute = fs::absolute(fs::path(path), ec);
  return ec ? path : absolute.lexically_normal().string();
}

std::string FindExecutable(const std::string& name) {
  if (name.empty()) return {};
  if (name.find('/') != std::string::npos) {
    return access(name.c_str(), X_OK) == 0 ? name : std::string{};
  }
  const char* path_env = std::getenv("PATH");
  if (!path_env) return {};
  std::stringstream paths(path_env);
  std::string directory;
  while (std::getline(paths, directory, ':')) {
    if (directory.empty()) directory = ".";
    const fs::path candidate = fs::path(directory) / name;
    if (access(candidate.c_str(), X_OK) == 0) return candidate.string();
  }
  return {};
}

bool FfmpegHasMp3Lame(const std::string& ffmpeg) {
  const std::string command = ShellQuote(ffmpeg) +
      " -hide_banner -encoders 2>/dev/null | grep -q libmp3lame";
  return std::system(command.c_str()) == 0;
}

std::atomic<bool> g_stop{false};

void HandleSignal(int) { g_stop.store(true); }

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity = 0) : capacity_(capacity) {}
  bool Push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&] { return closed_ || capacity_ == 0 || q_.size() < capacity_; });
    if (closed_) return false;
    q_.push_back(std::move(value));
    not_empty_.notify_one();
    return true;
  }
  bool TryPush(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || (capacity_ != 0 && q_.size() >= capacity_)) return false;
    q_.push_back(std::move(value));
    not_empty_.notify_one();
    return true;
  }
  bool TryPopValue(T* value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (q_.empty()) return false;
    *value = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }
  bool Pop(T* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return closed_ || !q_.empty(); });
    if (q_.empty()) return false;
    *value = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }
  bool TryDropOldest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (q_.empty()) return false;
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }
  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }
  size_t Size() const { std::lock_guard<std::mutex> lock(mutex_); return q_.size(); }
 private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_, not_full_;
  std::deque<T> q_;
  bool closed_ = false;
};

// One lease covers a whole ASR Cell, including PCM delivery.
// ResultWorker admits Cells in FIFO order; flock extends exclusion to other
// pipeline processes. Audio capture, NPU1 and playback remain independent.
class Npu2CellArbiter {
 public:
  class Lease {
   public:
    explicit Lease(Npu2CellArbiter* owner) : owner_(owner), lock_(owner->mutex_) {
      while (flock(owner_->lock_fd_, LOCK_EX) != 0) {
        if (errno != EINTR) throw std::runtime_error("cannot acquire NPU2 Cell lock");
      }
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { flock(owner_->lock_fd_, LOCK_UN); }
   private:
    Npu2CellArbiter* owner_;
    std::unique_lock<std::mutex> lock_;
  };
  explicit Npu2CellArbiter(const std::string& lock_file) {
    std::error_code ec;
    const fs::path parent = fs::path(lock_file).parent_path();
    if (!parent.empty()) fs::create_directories(parent, ec);
    lock_fd_ = open(lock_file.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (lock_fd_ < 0) throw std::runtime_error("cannot open NPU2 lock: " + lock_file);
  }
  ~Npu2CellArbiter() { if (lock_fd_ >= 0) close(lock_fd_); }
  Lease Acquire() { return Lease(this); }
 private:
  int lock_fd_ = -1;
  std::mutex mutex_;
};

class NpuArbiter {
 public:
  void FastLock() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++fast_waiters_;
    cv_.wait(lock, [&] { return !active_; });
    --fast_waiters_;
    active_ = true;
  }
  void VadLock() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !active_ && fast_waiters_ == 0; });
    active_ = true;
  }
  void Unlock() { std::lock_guard<std::mutex> lock(mutex_); active_ = false; cv_.notify_all(); }
 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool active_ = false;
  int fast_waiters_ = 0;
};

class Timing {
 public:
  explicit Timing(std::string name) : name_(std::move(name)), start_(Clock::now()) {}
  void Add(uint64_t ns) { std::lock_guard<std::mutex> lock(mutex_); ++calls_; total_ns_ += ns; max_ns_ = std::max(max_ns_, ns); last_ns_ = ns; }
  void Report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const double total = static_cast<double>(total_ns_) / 1e6;
    const double elapsed = std::max(1.0, static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count()) / 1e6);
    std::cerr << "[npu] " << name_ << " calls=" << calls_
              << " last=" << last_ns_ / 1e6 << "ms avg=" << (calls_ ? total / calls_ : 0)
              << "ms max=" << max_ns_ / 1e6 << "ms busy=" << total << "ms util=" << total * 100 / elapsed << "%\n";
  }
 private:
  std::string name_;
  mutable std::mutex mutex_;
  Clock::time_point start_;
  uint64_t calls_ = 0, total_ns_ = 0, max_ns_ = 0, last_ns_ = 0;
};

class FastEnhance {
 public:
  FastEnhance(const apm_example::AppConfig& config, NpuArbiter* arbiter)
      : enabled(config.fastenhance.enabled), arbiter_(arbiter), timing_("fastenhance") {
    if (!enabled) return;
    const std::string library = ResolveLibrary(config.fastenhance.library);
    library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error("cannot load FastEnhance: " + std::string(dlerror()));
    create_ = reinterpret_cast<Create>(dlsym(library_, "fastenhance_create"));
    create_error_ = reinterpret_cast<CreateError>(
        dlsym(library_, "fastenhance_create_last_error"));
    process_ = reinterpret_cast<ProcessFn>(dlsym(library_, "fastenhance_process"));
    destroy_ = reinterpret_cast<Destroy>(dlsym(library_, "fastenhance_destroy"));
    if (!create_ || !process_ || !destroy_) throw std::runtime_error("invalid FastEnhance ABI");
    handle_ = create_(config.fastenhance.model_path.c_str(), config.fastenhance.npu_core);
    if (!handle_) {
      const char* detail = create_error_ ? create_error_() : nullptr;
      throw std::runtime_error(std::string("fastenhance_create failed") +
                               (detail && *detail ? ": " + std::string(detail)
                                                  : std::string{}));
    }
  }
  ~FastEnhance() { Close(); }
  std::array<int16_t, kEnhanceHop> Process(const std::array<int16_t, kEnhanceHop>& in) {
    std::array<int16_t, kEnhanceHop> out = in;
    if (!enabled) return out;
    std::array<float, kEnhanceHop> source{}, target{};
    for (size_t i = 0; i < kEnhanceHop; ++i) source[i] = in[i] / 32768.0f;
    const auto begin = Clock::now();
    arbiter_->FastLock();
    const int status = process_(handle_, source.data(), target.data());
    arbiter_->Unlock();
    timing_.Add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
    if (status != 0) throw std::runtime_error("fastenhance_process failed: " + std::to_string(status));
    for (size_t i = 0; i < kEnhanceHop; ++i) {
      const float x = std::isfinite(target[i]) ? target[i] : 0.0f;
      out[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, std::round(x * 32768.0f))));
    }
    return out;
  }
  void Close() { if (handle_ && destroy_) destroy_(handle_); handle_ = nullptr; if (library_) dlclose(library_); library_ = nullptr; }
  void Report() const { timing_.Report(); }
 private:
  using Create = void* (*)(const char*, int);
  using CreateError = const char* (*)();
  using ProcessFn = int (*)(void*, const float*, float*);
  using Destroy = void (*)(void*);
  bool enabled;
  NpuArbiter* arbiter_;
  Timing timing_;
  void* library_ = nullptr;
  void* handle_ = nullptr;
  Create create_ = nullptr;
  CreateError create_error_ = nullptr;
  ProcessFn process_ = nullptr;
  Destroy destroy_ = nullptr;
};

class TenVad {
 public:
  TenVad(const apm_example::AppConfig& config, NpuArbiter* arbiter)
      : enabled_(config.vad.enabled), hop_(std::max(1, static_cast<int>(std::lround(kRateOut * config.vad.hop_ms / 1000.0)))), threshold_(config.vad.threshold), arbiter_(arbiter), timing_("ten-vad") {
    if (!enabled_) return;
    const std::string library = ResolveLibrary(config.vad.library);
    library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error("cannot load TEN-VAD: " + std::string(dlerror()));
    create_ = reinterpret_cast<Create>(dlsym(library_, "ten_vad_create"));
    process_ = reinterpret_cast<ProcessFn>(dlsym(library_, "ten_vad_process"));
    destroy_ = reinterpret_cast<Destroy>(dlsym(library_, "ten_vad_destroy"));
    if (!create_ || !process_ || !destroy_) throw std::runtime_error("invalid TEN-VAD ABI");
    const fs::path old = fs::current_path();
    fs::current_path(fs::path(config.vad.model_path).parent_path());
    if (create_(&handle_, hop_, threshold_) != 0) { fs::current_path(old); throw std::runtime_error("ten_vad_create failed"); }
    fs::current_path(old);
  }
  ~TenVad() { Close(); }
  bool Process(const std::vector<int16_t>& samples) {
    if (!enabled_) return true;
    float probability = 0.0f; int flag = 0;
    const auto begin = Clock::now();
    arbiter_->VadLock();
    const int status = process_(handle_, samples.data(), samples.size(), &probability, &flag);
    arbiter_->Unlock();
    timing_.Add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
    if (status != 0) throw std::runtime_error("ten_vad_process failed: " + std::to_string(status));
    return flag != 0;
  }
  int Hop() const { return hop_; }
  void Close() { if (handle_ && destroy_) destroy_(&handle_); handle_ = nullptr; if (library_) dlclose(library_); library_ = nullptr; }
  void Report() const { timing_.Report(); }
 private:
  using Create = int (*)(ten_vad_handle_t*, size_t, float);
  using ProcessFn = int (*)(ten_vad_handle_t, const int16_t*, size_t, float*, int*);
  using Destroy = int (*)(ten_vad_handle_t*);
  bool enabled_;
  int hop_;
  float threshold_;
  NpuArbiter* arbiter_;
  Timing timing_;
  void* library_ = nullptr;
  ten_vad_handle_t handle_ = nullptr;
  Create create_ = nullptr;
  ProcessFn process_ = nullptr;
  Destroy destroy_ = nullptr;
};

class Decimator {
 public:
  Decimator() {
    constexpr int taps = 63;
    coeff_.resize(taps);
    const double fc = 7200.0 / kRateIn, mid = (taps - 1) / 2.0;
    for (int i = 0; i < taps; ++i) {
      const double x = i - mid;
      constexpr double pi = 3.14159265358979323846;
      const double sinc = x == 0 ? 2 * fc : std::sin(2 * pi * fc * x) / (pi * x);
      const double w = 0.42 - 0.5 * std::cos(2 * pi * i / (taps - 1)) + 0.08 * std::cos(4 * pi * i / (taps - 1));
      coeff_[i] = sinc * w;
    }
    double gain = 0; for (double x : coeff_) gain += x; for (double& x : coeff_) x /= gain;
    history_.assign(taps - 1, 0.0f);
  }
  std::vector<int16_t> Process(const std::vector<int16_t>& input) {
    std::vector<float> values = history_; for (int16_t x : input) values.push_back(x);
    std::vector<int16_t> out;
    const size_t limit = values.size() - history_.size();
    for (size_t i = phase_; i < limit; i += 3) {
      double sum = 0; for (size_t j = 0; j < coeff_.size(); ++j) sum += values[i + j] * coeff_[j];
      out.push_back(static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, std::round(sum)))));
    }
    history_.assign(values.end() - history_.size(), values.end());
    const size_t consumed = phase_ < limit ? phase_ + ((limit - phase_ + 2) / 3) * 3 : phase_;
    phase_ = consumed >= limit ? consumed - limit : 0;
    return out;
  }
 private:
  std::vector<double> coeff_;
  std::vector<float> history_;
  size_t phase_ = 0;
};

// JSON-lines event stream for the live demo viewer (tools/live-demo).  Every
// line carries `t` = seconds since the first input sample reached the
// pipeline, so a viewer can compare emit time against audio time.
class EventLog {
 public:
  static EventLog& Get() { static EventLog log; return log; }
  void Open(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (path.empty()) return;
    out_.open(path, std::ios::trunc);
  }
  void MarkStart() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) return;
    started_ = true;
    t0_ = std::chrono::steady_clock::now();
    if (!out_.is_open()) return;
    const double epoch = std::chrono::duration<double>(
        WallClock::now().time_since_epoch()).count();
    out_ << "{\"ev\":\"start\",\"t\":0,\"epoch\":" << std::fixed
         << std::setprecision(3) << epoch << "}\n" << std::flush;
  }
  // `body` is the JSON object content without braces and without "t".
  void Write(const std::string& ev, const std::string& body) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!out_.is_open()) return;
    const double t = started_ ? std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_).count() : 0.0;
    out_ << "{\"ev\":\"" << ev << "\",\"t\":" << std::fixed
         << std::setprecision(3) << t << "," << body << "}\n" << std::flush;
  }
  static std::string Quote(const std::string& s) {
    std::string o = "\"";
    for (unsigned char ch : s) {
      if (ch == '"') o += "\\\"";
      else if (ch == '\\') o += "\\\\";
      else if (ch == '\n') o += "\\n";
      else if (ch < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", ch); o += b; }
      else o += static_cast<char>(ch);
    }
    return o + "\"";
  }
  static std::string Sec(std::size_t samples, int rate) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.3f", samples / static_cast<double>(rate));
    return b;
  }
 private:
  std::mutex mutex_;
  std::ofstream out_;
  bool started_ = false;
  std::chrono::steady_clock::time_point t0_;
};

struct Segment {
  struct Part { std::size_t begin = 0, end = 0; std::string speaker_id; };
  size_t start = 0, end = 0;
  std::vector<int16_t> samples;
  std::vector<std::size_t> join_points;
  // (concat_offset, source_offset) pairs: start of every voiced run.  Used by
  // the offline evaluation harness to map segment coordinates back to source
  // time; harmless in production and written as <wav>.runs.txt.
  std::vector<std::pair<std::size_t, std::size_t>> runs;
  // Raw (silence-preserving) audio from the first voiced sample to the segment
  // end, plus its source offset.  Diarization runs on this window so the model
  // keeps turn-taking and overlap cues; ASR still uses the voiced-concatenated
  // `samples` above.
  std::vector<int16_t> raw_samples;
  std::size_t raw_start = 0;
  std::vector<Part> parts;
  // Streaming first-word fast path: a short "preview" slice emitted early by
  // the accumulator so the ASR can produce opening text before the speaker
  // label is available.  Previews skip diarization entirely.
  bool is_preview = false;
  WallClock::time_point wall_start, wall_end;
};

// Speaker-timeline view for text tracks: the streaming ASR track asks the
// diarization source (NemotronStreamDiarizer) which speaker is active where,
// and waits for the timeline frontier to pass an epoch's voiced audio before
// attributing its final line(s).  Same shape as the ResultWorker timeline
// paths so both consumers stay interchangeable.
struct SpeakerLookup {
  virtual ~SpeakerLookup() = default;
  // Absolute speaker parts (16 kHz sample coordinates).
  virtual std::vector<Segment::Part> Timeline() const = 0;
  // Samples with final labels (timeline frontier).
  virtual std::size_t Committed() const = 0;
  // Wait until the frontier covers `end` or the source finishes; true when
  // covered.
  virtual bool WaitCovered(std::size_t end, std::chrono::milliseconds timeout) = 0;
  // True once no further timeline updates will arrive.
  virtual bool Finished() const = 0;
};

// P2c ASR slice pad: a slice boundary that lands inside a word onset leaves
// the ASR with the tail of the word only (see the comment at the definition
// site below for the measured impact).  Timestamps keep the unpadded bounds;
// only the ASR input grows, so labeling and DER scoring are unaffected.
constexpr std::size_t kAsrPadSamples = kRateOut / 5;  // 0.2 s

// Forward declaration: the streaming preview slices map their concatenated
// offsets back to source time stamps through the run map (defined below).
std::size_t ConcatToSource(
    const std::vector<std::pair<std::size_t, std::size_t>>& runs,
    std::size_t concat, bool is_end);

class Accumulator {
 public:
  Accumulator(const apm_example::AppConfig& c,
              std::function<void(Segment)> submit)
      : min_burst_(c.vad.min_speech_ms * kRateOut / 1000),
        min_segment_(std::max(1, static_cast<int>(std::lround(c.vad.min_segment_seconds * kRateOut)))),
        max_(std::max(1, static_cast<int>(std::lround(c.vad.max_segment_seconds * kRateOut)))),
        trailing_(std::max(1, static_cast<int>(std::lround(c.vad.trailing_silence_seconds * kRateOut)))),
        preview_(c.vad.preview_seconds > 0.0f
                     ? std::max(1, static_cast<int>(std::lround(c.vad.preview_seconds * kRateOut)))
                     : 0),
        pause_split_(c.vad.pause_split_seconds > 0.0f
                         ? std::max(1, static_cast<int>(std::lround(c.vad.pause_split_seconds * kRateOut)))
                         : 0),
        // Word-safe pause length: real inter-word gaps measured on the 2P
        // clip are >= ~190 ms while VAD flicker inside words stays < 100 ms,
        // so 200 ms separates the two; slice cut points land on it.
        pause_len_(std::max(1, kRateOut / 5)),
        submit_(std::move(submit)),
        start_wall_(WallClock::now()) {
    std::cerr << "[vad] accumulator min=" << min_segment_ / static_cast<double>(kRateOut)
              << "s max=" << max_ / static_cast<double>(kRateOut)
              << "s trailing=" << trailing_ / static_cast<double>(kRateOut)
              << "s pause_split=" << (pause_split_ > 0
                  ? std::to_string(pause_split_ / static_cast<double>(kRateOut)) + "s"
                  : std::string("off")) << "\n";
    // Opening slice: prefer a word-safe pause-cut once half the interval
    // has voiced audio (measured pauses cluster at 1.2-2.2 s), falling back
    // to a blind cut after the first full `preview_` voiced samples (fast
    // first words; the segment starts on a pause so its begin is clean).
    // Later slices prefer the word-safe pause-cut in the unvoiced branch
    // and only blind-cut as a fallback after twice the interval.
    if (preview_ > 0) preview_next_ = static_cast<std::size_t>(preview_);
  }
  void Push(const std::vector<int16_t>& hop, bool voiced) {
    if (voiced) {
      if (!in_voice_) {
        burst_start_ = cursor_;
        if (burst_.empty()) {
          // First voiced run of a new segment: start recording the raw window
          // (with silences) for diarization.
          raw_.clear();
          raw_start_ = cursor_;
          raw_active_ = true;
        }
        if (raw_active_) raw_.insert(raw_.end(), hop.begin(), hop.end());
        if (!burst_.empty()) {
          // Speech resumed after a pause shorter than the trailing-silence
          // threshold.  Record the removed-silence boundary for the .joins
          // sidecar; diarization no longer depends on it.
          const std::size_t point = buffer_.size() + burst_.size();
          runs_.push_back({point, cursor_});
          if (point > 0 &&
              (join_points_.empty() || join_points_.back() != point))
            join_points_.push_back(point);
        } else {
          runs_.push_back({0, cursor_});
        }
      } else {
        if (raw_active_) raw_.insert(raw_.end(), hop.begin(), hop.end());
      }
      burst_.insert(burst_.end(), hop.begin(), hop.end());
      last_voice_ = cursor_ + hop.size();
      have_voice_ = true;
      in_voice_ = true;
      const size_t total = buffer_.size() + burst_.size();
      // Streaming fast path (blind-cut fallback): without a pause, a slice
      // only cuts after twice `preview_` NEW voiced samples past the last
      // emitted preview (the opening slice uses 1x for fast first words).
      // Normally the word-safe pause-cut in the unvoiced branch fires first;
      // this path only keeps text flowing through fully fluent stretches.
      // The first slice of a segment starts at its beginning (not at the
      // last preview end) so openings are not lost.
      if (preview_ > 0 && total >= preview_next_ &&
          total < static_cast<size_t>(max_)) {
        EmitPreview();
      }
      if (total >= static_cast<size_t>(max_)) {
        if (buffer_.empty()) start_ = burst_start_;
        buffer_.insert(buffer_.end(), burst_.begin(), burst_.end());
        burst_.clear();
        Emit();
      }
    } else {
      if (raw_active_) raw_.insert(raw_.end(), hop.begin(), hop.end());
      in_voice_ = false;
      const size_t total = buffer_.size() + burst_.size();
      const size_t since_voice = cursor_ + hop.size() - last_voice_;
      // Do not start the silence timer until the accumulated speech has
      // reached the configured minimum.  Once it has, a trailing-silence hop
      // closes and submits the utterance immediately.
      const bool close_end = !burst_.empty() &&
          total >= static_cast<size_t>(min_segment_) &&
          since_voice >= static_cast<size_t>(trailing_);
      // Long-monologue split: once the segment is past pause_split_, a short
      // natural pause (>= 0.2 s) is enough to close it, instead of waiting
      // for the full trailing silence or the hard max-length cut (which can
      // land mid-word and garble the ASR at both sides of the cut).
      const bool close_split = !burst_.empty() && pause_split_ > 0 &&
          total >= static_cast<size_t>(pause_split_) &&
          since_voice >= static_cast<size_t>(pause_len_);
      if (close_end || close_split) {
        // Word-safe tail slice: the same pause that closes the segment also
        // closes the pending preview slice, so its text lands ~0.2 s after
        // the pause instead of waiting for the speaker label (the final
        // supersedes it 0.4-1.4 s later).  Without this, pause_split would
        // starve the chained previews: any pause late enough to satisfy the
        // slice interval already closes the segment.  The opening slice only
        // needs half the interval: measured pauses cluster at 1.2-2.2 s of
        // voiced audio, so this cuts there (word-safe and usually EARLIER
        // than the old fixed 2 s blind cut) instead of mid-word.
        if (preview_ > 0 && burst_.size() >= static_cast<size_t>(min_burst_) &&
            total - preview_done_ >=
                static_cast<size_t>(preview_done_ == 0 ? preview_ / 2
                                                       : preview_)) {
          EmitPreview();
        }
        if (burst_.size() >= static_cast<size_t>(min_burst_)) {
          if (buffer_.empty()) start_ = burst_start_;
          buffer_.insert(buffer_.end(), burst_.begin(), burst_.end());
        } else {
          std::cerr << "[vad] discarded short burst=" << burst_.size() / static_cast<double>(kRateOut) << "s\n";
          runs_.clear();
          raw_.clear();
          raw_active_ = false;
          preview_done_ = 0; preview_next_ = static_cast<std::size_t>(preview_);
        }
        burst_.clear();
        // A trailing-silence (or pause-split) boundary completes the current
        // utterance.  The previous implementation only emitted here after
        // max_ samples had accumulated, so normal utterances shorter than
        // max_segment_seconds stayed queued indefinitely.
        if (!buffer_.empty()) Emit();
      } else if (preview_ > 0 && !burst_.empty() &&
                 total - preview_done_ >=
                     static_cast<size_t>(preview_done_ == 0 ? preview_ / 2
                                                            : preview_) &&
                 since_voice >= static_cast<size_t>(pause_len_)) {
        // Word-safe fast-text cut: the slice interval (half of it for the
        // opening slice - pauses cluster just before/after the 2 s mark) has
        // elapsed and the talker paused (>= 0.2 s, the word-safe threshold).
        // Cut the preview here instead of blind-cutting mid-word: cut points
        // land on pauses, so both slice sides keep whole words for the ASR.
        EmitPreview();
      }
    }
    cursor_ += hop.size();
  }
  void Flush() { if (burst_.size() >= static_cast<size_t>(min_burst_)) { if (buffer_.empty()) start_ = burst_start_; buffer_.insert(buffer_.end(), burst_.begin(), burst_.end()); } else { runs_.clear(); raw_.clear(); raw_active_ = false; preview_done_ = 0; preview_next_ = static_cast<std::size_t>(preview_); } if (!buffer_.empty()) Emit(); burst_.clear(); }
 private:
  // Text-only fast path: emit the next slice of the accumulated voiced audio
  // so the ASR can produce words before the speaker label is known.  Slices
  // chain through the segment: each one fires either on a >= 0.2 s pause
  // (word-safe cut point, preferred) or - if the talker never pauses - as a
  // blind cut in the voiced branch after twice the interval.
  void EmitPreview() {
    const std::size_t total = buffer_.size() + burst_.size();
    const std::size_t begin = preview_done_;
    const std::size_t end = total;
    preview_done_ = end;
    // Word-safe chaining: after any emitted slice the next one only
    // blind-cuts mid-speech after twice the interval - normally the
    // pause-cut in the unvoiced branch fires first.  (The ctor/emit reset
    // keeps 1x for the opening slice: fast first words, and the segment
    // itself starts on a pause so the slice begin is clean.)
    preview_next_ = end + 2 * static_cast<std::size_t>(preview_);
    // Chained slices start mid-speech, so pad the ASR input backwards the
    // same 0.2 s the per-speaker parts use; timestamps stay unpadded.
    const std::size_t sample_begin =
        begin > kAsrPadSamples ? begin - kAsrPadSamples : 0;
    Segment p;
    p.is_preview = true;
    p.start = ConcatToSource(runs_, begin, false);
    p.end = ConcatToSource(runs_, end, true);
    p.samples.reserve(end - sample_begin);
    if (sample_begin < buffer_.size())
      p.samples.assign(buffer_.begin() + sample_begin,
                       buffer_.begin() + std::min(end, buffer_.size()));
    if (end > buffer_.size()) {
      const std::size_t from =
          sample_begin > buffer_.size() ? sample_begin - buffer_.size() : 0;
      p.samples.insert(p.samples.end(), burst_.begin() + from,
                       burst_.begin() + (end - buffer_.size()));
    }
    p.wall_start = start_wall_ + std::chrono::milliseconds(
        static_cast<int64_t>(p.start) * 1000 / kRateOut);
    p.wall_end = start_wall_ + std::chrono::milliseconds(
        static_cast<int64_t>(p.end) * 1000 / kRateOut);
    std::cerr << "[vad] preview submitted duration="
              << p.samples.size() / static_cast<double>(kRateOut) << "s\n";
    submit_(std::move(p));
  }
  void Emit() {
    if (buffer_.empty()) return;
    Segment s;
    s.start = start_;
    s.end = have_voice_ ? last_voice_ : cursor_;
    s.samples = std::move(buffer_);
    s.runs = std::move(runs_);
    runs_.clear();
    s.raw_samples = std::move(raw_);
    s.raw_start = raw_start_;
    raw_.clear();
    raw_active_ = false;
    for (std::size_t point : join_points_)
      if (point > 0 && point < s.samples.size()) s.join_points.push_back(point);
    s.wall_start = start_wall_ + std::chrono::milliseconds(static_cast<int64_t>(s.start) * 1000 / kRateOut);
    s.wall_end = start_wall_ + std::chrono::milliseconds(static_cast<int64_t>(s.end) * 1000 / kRateOut);
    // Speaker parts are produced by the resident diar-worker in the result
    // worker, after the segment WAV is written and off the capture path.
    std::cerr << "[vad] segment submitted duration="
              << s.samples.size() / static_cast<double>(kRateOut) << "s\n";
    submit_(std::move(s));
    buffer_.clear();
    join_points_.clear();
    // A max-length split also ends the current burst logically.  This makes
    // the next voiced hop establish a fresh start offset and end timestamp.
    in_voice_ = false;
    have_voice_ = false;
    preview_done_ = 0; preview_next_ = static_cast<std::size_t>(preview_);
  }
  int min_burst_, min_segment_, max_, trailing_, preview_, pause_split_, pause_len_; std::function<void(Segment)> submit_; WallClock::time_point start_wall_; std::vector<int16_t> buffer_, burst_; std::vector<std::size_t> join_points_; std::vector<std::pair<std::size_t, std::size_t>> runs_; std::vector<int16_t> raw_; std::size_t raw_start_ = 0; bool raw_active_ = false; size_t start_ = 0, burst_start_ = 0, cursor_ = 0, last_voice_ = 0; bool in_voice_ = false, have_voice_ = false; std::size_t preview_done_ = 0, preview_next_ = 0;
};

bool WriteWav(const fs::path& path, const std::vector<int16_t>& samples, int rate) {
  std::ofstream out(path, std::ios::binary); if (!out) return false;
  const uint32_t bytes = samples.size() * sizeof(int16_t), riff = 36 + bytes;
  out.write("RIFF", 4); out.write(reinterpret_cast<const char*>(&riff), 4); out.write("WAVEfmt ", 8);
  const uint32_t fmt_size = 16; const uint16_t format = 1, channels = 1; const uint32_t byte_rate = rate * 2; const uint16_t block = 2, bits = 16;
  out.write(reinterpret_cast<const char*>(&fmt_size),4); out.write(reinterpret_cast<const char*>(&format),2); out.write(reinterpret_cast<const char*>(&channels),2); out.write(reinterpret_cast<const char*>(&rate),4); out.write(reinterpret_cast<const char*>(&byte_rate),4); out.write(reinterpret_cast<const char*>(&block),2); out.write(reinterpret_cast<const char*>(&bits),2); out.write("data",4); out.write(reinterpret_cast<const char*>(&bytes),4); out.write(reinterpret_cast<const char*>(samples.data()), bytes); return static_cast<bool>(out);
}

bool WriteJoinPoints(const fs::path& wav,
                     const std::vector<std::size_t>& join_points, int rate) {
  const fs::path path = wav.string() + ".joins.txt";
  std::ofstream out(path);
  if (!out) return false;
  out << "# sample_index milliseconds\n";
  for (std::size_t point : join_points)
    out << point << ' ' << point * 1000 / static_cast<std::size_t>(rate) << '\n';
  return static_cast<bool>(out);
}

bool WriteRuns(const fs::path& wav,
               const std::vector<std::pair<std::size_t, std::size_t>>& runs) {
  const fs::path path = wav.string() + ".runs.txt";
  std::ofstream out(path);
  if (!out) return false;
  out << "# concat_offset source_offset\n";
  for (const auto& run : runs) out << run.first << ' ' << run.second << '\n';
  return static_cast<bool>(out);
}

bool WriteParts(const fs::path& wav, const std::vector<Segment::Part>& parts) {
  const fs::path path = wav.string() + ".parts.txt";
  std::ofstream out(path);
  if (!out) return false;
  out << parts.size() << "\n";
  for (const auto& part : parts)
    out << part.begin << ' ' << part.end << ' ' << part.speaker_id << '\n';
  return static_cast<bool>(out);
}

// Project a speaker timeline onto the VAD-voiced concatenated timeline used by
// ASR.  `timeline` entries are relative to `timeline_offset` (0 for session
// diarization in file coordinates, the window start for per-segment raw
// windows).  `runs` maps concatenated offsets to absolute pipeline-stream
// offsets.  On overlap the longest timeline range wins, then unassigned
// samples inherit the nearest label so the resulting ranges partition
// [0, trimmed_samples).
//
// P2 part cleanup (docs/diarization-progress.md): sliding-window commits cut
// a continuous speech region at every core boundary and speaker changes cut
// the stream into many small parts, each of which becomes one ASR request
// (hence junk "。-only" or empty transcript lines).  (A) adjacent parts with
// the same speaker are re-joined — the samples and their speaker are
// unchanged, only the arbitrary split disappears; (B) parts shorter than
// kMinPartSamples are absorbed into the longer neighbour — speaker
// attribution moves by <0.3 s at a time.  Measured on the 2P run:
// 418 -> 236 parts, v8 agreement 91.6% -> 91.5%; T3 conf is unchanged up to
// 0.3 s and degrades beyond it (threshold sweep in the eval assets).
constexpr std::size_t kMinPartSamples = kRateOut * 3 / 10;  // 0.3 s
// A closed speaker part is released once the timeline frontier passes its end
// by this margin.  Provisional (open-run) parts always end at the frontier,
// so they can never satisfy it and are only released after closing; for
// closed parts a later same-speaker merge only extends the end, and the
// released prefix is clamped by `done`, so a small margin is safe.
constexpr std::size_t kPartReleaseMargin = kRateOut / 10;  // 0.1 s

// Concatenated (voiced-only) offset -> absolute source offset via `runs`.
// `is_end` maps an exclusive end so it never jumps into the next run.

std::size_t ConcatToSource(
    const std::vector<std::pair<std::size_t, std::size_t>>& runs,
    std::size_t concat, bool is_end) {
  if (runs.empty()) return concat;
  const std::size_t probe = (is_end && concat > 0) ? concat - 1 : concat;
  std::size_t r = 0;
  while (r + 1 < runs.size() && runs[r + 1].first <= probe) ++r;
  return runs[r].second + (probe - runs[r].first) + ((is_end && concat > 0) ? 1 : 0);
}

// Largest concatenated prefix whose source samples are already covered by the
// time line frontier (source samples with final speaker labels).
std::size_t ConcatCoveredBySource(
    const std::vector<std::pair<std::size_t, std::size_t>>& runs,
    std::size_t frontier, std::size_t trimmed_samples) {
  std::size_t cut = 0;
  for (std::size_t r = 0; r < runs.size(); ++r) {
    const std::size_t concat = runs[r].first;
    const std::size_t source = runs[r].second;
    if (source >= frontier) break;
    const std::size_t concat_end =
        (r + 1 < runs.size()) ? runs[r + 1].first : trimmed_samples;
    cut = concat + std::min(concat_end - concat, frontier - source);
  }
  return std::min(cut, trimmed_samples);
}

std::vector<Segment::Part> MapTimelineToTrimmed(
    const std::vector<Segment::Part>& timeline,
    std::size_t timeline_offset,
    const std::vector<std::pair<std::size_t, std::size_t>>& runs,
    std::size_t trimmed_samples) {
  std::vector<Segment::Part> parts;
  if (timeline.empty() || runs.empty() || trimmed_samples == 0) return parts;
  std::vector<int> label(trimmed_samples, -1);
  std::vector<std::size_t> order(timeline.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return (timeline[a].end - timeline[a].begin) >
           (timeline[b].end - timeline[b].begin);
  });
  for (std::size_t idx : order) {
    const auto& entry = timeline[idx];
    if (entry.end <= entry.begin) continue;
    const std::size_t part_begin = entry.begin + timeline_offset;
    const std::size_t part_end = entry.end + timeline_offset;
    for (std::size_t r = 0; r < runs.size(); ++r) {
      const std::size_t run_concat = runs[r].first;
      const std::size_t run_source = runs[r].second;
      const std::size_t run_concat_end =
          (r + 1 < runs.size()) ? runs[r + 1].first : trimmed_samples;
      if (run_concat_end <= run_concat) continue;
      const std::size_t run_source_end = run_source + (run_concat_end - run_concat);
      const std::size_t begin = std::max(part_begin, run_source);
      const std::size_t end = std::min(part_end, run_source_end);
      if (end <= begin) continue;
      const std::size_t concat_begin = run_concat + (begin - run_source);
      const std::size_t concat_end = run_concat + (end - run_source);
      for (std::size_t s = concat_begin; s < concat_end && s < trimmed_samples; ++s)
        if (label[s] < 0) label[s] = static_cast<int>(idx);
    }
  }
  int current = -1;
  for (std::size_t i = 0; i < trimmed_samples; ++i) {
    if (label[i] >= 0) current = label[i];
    else if (current >= 0) label[i] = current;
  }
  current = -1;
  for (std::size_t i = trimmed_samples; i-- > 0;) {
    if (label[i] >= 0) current = label[i];
    else if (current >= 0) label[i] = current;
  }
  std::size_t i = 0;
  while (i < trimmed_samples) {
    const int id = label[i];
    std::size_t j = i;
    while (j < trimmed_samples && label[j] == id) ++j;
    if (id >= 0 && j > i) {
      const std::string& speaker =
          timeline[static_cast<std::size_t>(id)].speaker_id;
      if (!speaker.empty()) {
        // (A) re-join regions split at a commit boundary: adjacent timeline
        // entries with the same speaker are one utterance.
        if (!parts.empty() && parts.back().end == i &&
            parts.back().speaker_id == speaker) {
          parts.back().end = j;
        } else {
          Segment::Part part;
          part.begin = i;
          part.end = j;
          part.speaker_id = speaker;
          parts.push_back(std::move(part));
        }
      }
    }
    i = j;
  }
  // (B) absorb sub-0.5 s fragments into the longer neighbour; such requests
  // only produce junk ASR lines and their speaker attribution is unreliable.
  while (parts.size() > 1) {
    std::size_t victim = parts.size();
    for (std::size_t p = 0; p < parts.size(); ++p) {
      if (parts[p].end - parts[p].begin < kMinPartSamples) {
        victim = p;
        break;
      }
    }
    if (victim == parts.size()) break;
    bool absorb_left = victim > 0;
    if (victim > 0 && victim + 1 < parts.size()) {
      const std::size_t prev_span = parts[victim - 1].end - parts[victim - 1].begin;
      const std::size_t next_span = parts[victim + 1].end - parts[victim + 1].begin;
      absorb_left = prev_span >= next_span;
    }
    if (absorb_left) {
      parts[victim - 1].end = parts[victim].end;
    } else {
      parts[victim + 1].begin = parts[victim].begin;
    }
    parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(victim));
  }
  return parts;
}

std::string ShellQuote(const std::string& s) { std::string out = "'"; for (char c : s) { if (c == '\'') out += "'\\''"; else out += c; } return out + "'"; }

std::string TimeStamp(WallClock::time_point t) { const auto tt = WallClock::to_time_t(t); std::tm tm{}; localtime_r(&tt, &tm); const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()) % 1000; std::ostringstream out; out << std::put_time(&tm, "%Y%m%d-%H-%M-%S-") << std::setfill('0') << std::setw(3) << ms.count(); return out.str(); }

std::string SpeakerTimeStamp(WallClock::time_point t) {
  const auto tt = WallClock::to_time_t(t); std::tm tm{}; localtime_r(&tt, &tm);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()) % 1000;
  std::ostringstream out; out << std::put_time(&tm, "%y%m%d_%H-%M-%S-")
      << std::setfill('0') << std::setw(3) << ms.count(); return out.str();
}

// Streaming transcript sidecar: "transcript.txt" -> "transcript_streaming.txt".
std::string DeriveStreamingTextFile(const std::string& text_file) {
  const fs::path p(text_file);
  return (p.parent_path() / (p.stem().string() + "_streaming" + p.extension().string())).string();
}

class SenseVoiceRunner {
 public:
  explicit SenseVoiceRunner(const apm_example::AppConfig& c)
      : enabled_(c.asr.enabled),
        c_library_(AbsolutePath(c.asr.library)),
        model_(AbsolutePath(c.asr.model_path)),
        language_(c.asr.language) {}
  ~SenseVoiceRunner() { Close(); }
  bool Start() {
    if (!enabled_) return true;
    library_ = dlopen(c_library_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) { std::cerr << "[asr] load SenseVoice library failed: " << dlerror() << "\n"; return false; }
    init_ = reinterpret_cast<Init>(dlsym(library_, "AX_ASR_Init"));
    uninit_ = reinterpret_cast<Uninit>(dlsym(library_, "AX_ASR_Uninit"));
    run_ = reinterpret_cast<Run>(dlsym(library_, "AX_ASR_RunFile"));
    if (!init_ || !uninit_ || !run_) return false;
    handle_ = init_(AX_SENSEVOICE, model_.c_str());
    if (!handle_) { std::cerr << "[asr] AX_ASR_Init(SenseVoice) failed\n"; return false; }
    run_pcm_ = reinterpret_cast<RunPcm>(dlsym(library_, "AX_ASR_RunPCM"));
    return true;
  }
  std::string Transcribe(const fs::path& wav) {
    if (!enabled_) return "";
    if (!handle_) return "[ASR_ERROR] sensevoice unavailable";
    char* result = nullptr;
    int status = -1;
    try {
      status = run_(handle_, wav.c_str(), language_.c_str(), &result);
    } catch (const std::exception& error) {
      return "[ASR_ERROR] " + std::string(error.what());
    }
    if (status != 0 || !result) return "[ASR_ERROR] sensevoice status=" + std::to_string(status);
    std::string text(result); std::free(result); return text;
  }
  // In-memory transcription of 16k mono PCM16 (streaming-slicer tick path;
  // measured on-board: ~21 ms per second of audio, 3 s window -> ~150 ms).
  std::string TranscribePcm(const std::vector<int16_t>& pcm) {
    if (!enabled_) return "";
    if (!handle_ || !run_pcm_) return "[ASR_ERROR] sensevoice unavailable";
    static std::vector<float> buf;
    buf.resize(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) buf[i] = pcm[i] / 32768.0f;
    char* result = nullptr;
    int status = -1;
    try {
      status = run_pcm_(handle_, buf.data(), static_cast<int>(buf.size()),
                         kRateOut, language_.c_str(), &result);
    } catch (const std::exception& error) {
      return "[ASR_ERROR] " + std::string(error.what());
    }
    if (status != 0 || !result) return "[ASR_ERROR] sensevoice status=" + std::to_string(status);
    std::string text(result); std::free(result); return text;
  }
  void Close() {
    if (handle_ && uninit_) uninit_(handle_);
    if (library_) dlclose(library_);
    handle_ = nullptr;
    library_ = nullptr;
  }
 private:
  static bool WriteAll(int fd, const void* data, size_t n) { const char* p = static_cast<const char*>(data); while (n) { const ssize_t w = write(fd, p, n); if (w <= 0) return false; p += w; n -= w; } return true; }
  bool enabled_;
  using Init = AX_ASR_HANDLE (*)(AX_ASR_TYPE_E, const char*);
  using Uninit = void (*)(AX_ASR_HANDLE);
  using Run = int (*)(AX_ASR_HANDLE, const char*, const char*, char**);
  using RunPcm = int (*)(AX_ASR_HANDLE, float*, int, int, const char*, char**);
  std::string c_library_;
  void* library_ = nullptr; AX_ASR_HANDLE handle_ = nullptr; Init init_ = nullptr; Uninit uninit_ = nullptr; Run run_ = nullptr; RunPcm run_pcm_ = nullptr;
  std::string model_, language_;
  pid_t pid_ = -1;
  int in_fd_ = -1, out_fd_ = -1;
};

// Fixed-cadence streaming text (streaming-slicer refactor).
//
// Feeds every enhanced 16k hop (silence included, straight from the VAD hop
// stream) and produces text on a fixed beat instead of the VAD-segment beat:
//
//   epoch  = speech run delimited by silences >= stream_pause_seconds (word
//            safe cut points; the epoch window [epoch_start, now) only ever
//            grows, so the left edge never eats a word mid-epoch)
//   tick   = every stream_tick_seconds of audio inside an epoch the whole
//            epoch window is re-transcribed in-memory (RunPCM, ~21 ms/s) and
//            the text is merged with the previous hypothesis
//   merge  = LocalAgreement-2: the longest common prefix of two consecutive
//            hypotheses becomes confirmed text (append-only; the remainder is
//            shown as a provisional tail until the next tick)
//   final  = when the closing pause is confirmed, the epoch hypothesis is
//            committed in full (it is the best-quality view of the epoch)
//
// Replaces the old preview path's pause-gated output cadence: fluent
// monologues now produce text every tick instead of stalling 4-5 s until the
// speaker happens to pause.  Events: "stext" (tick; commit/tail delta) and
// "sfinal" (epoch close; full authoritative-for-epoch text).
class StreamSlicer {
 public:
  StreamSlicer(const apm_example::AppConfig& c, SenseVoiceRunner* asr,
               Npu2CellArbiter* arbiter)
      : enabled_(c.vad.stream_tick_seconds > 0.0f &&
                 c.asr.mode != "streaming"),
        asr_(asr), arbiter_(arbiter),
        streaming_text_file_(DeriveStreamingTextFile(c.output.text_file)),
        tick_(std::max<size_t>(1, static_cast<size_t>(
            c.vad.stream_tick_seconds * kRateOut))),
        pause_close_(std::max<size_t>(1, static_cast<size_t>(
            c.vad.stream_pause_seconds * kRateOut))),
        epoch_cap_(std::max<size_t>(1, static_cast<size_t>(
            c.vad.stream_epoch_max_seconds * kRateOut))) {}
  ~StreamSlicer() { Close(false); }

  void Start() {
    if (!enabled_ || thread_.joinable()) return;
    thread_ = std::thread([this] { Loop(); });
  }

  // Producer side (VAD loop thread).  Hops are queued; the queue never blocks
  // the VAD pipeline (drops oldest, mirroring the vad_jobs pattern).
  void Push(const std::vector<int16_t>& hop, bool voiced) {
    if (!enabled_ || !thread_.joinable()) return;
    Item item;
    item.samples = hop;
    item.voiced = voiced;
    if (!queue_.TryPush(std::move(item))) {
      queue_.TryDropOldest();
      queue_.TryPush(std::move(item));
    }
  }

  // End of stream: drain, finalize the open epoch, stop the thread.
  void FlushAndClose() {
    if (!enabled_) return;
    Item flush;
    flush.flush = true;
    if (thread_.joinable()) {
      queue_.Push(std::move(flush));
      thread_.join();
    } else {
      queue_.Close();
    }
  }
  void Close(bool) { FlushAndClose(); }

 private:
  struct Item {
    std::vector<int16_t> samples;
    bool voiced = false;
    bool flush = false;
  };

  // Punctuation-loose LA-2 helpers live in text_agreement.hpp (shared with
  // the streaming ASR track).

  void Loop() {
    Item item;
    while (queue_.Pop(&item)) {
      if (item.flush) break;
      abs_ += item.samples.size();
      Feed(item.samples, item.voiced);
    }
    if (item.flush) {
      // Drain anything still queued, then finalize the open epoch.
      while (queue_.TryPopValue(&item)) {
        if (item.flush) continue;
        abs_ += item.samples.size();
        Feed(item.samples, item.voiced);
      }
      FinalizeEpoch(true);
      queue_.Close();
    }
  }

  void Feed(const std::vector<int16_t>& hop, bool voiced) {
    if (!in_epoch_) {
      if (!voiced) return;
      OpenEpoch();
    }
    epoch_.insert(epoch_.end(), hop.begin(), hop.end());
    if (voiced) {
      voiced_ += hop.size();
      silence_ = 0;
      voiced_end_ = abs_;
    } else {
      silence_ += hop.size();
      if (silence_ >= pause_close_) { FinalizeEpoch(false); return; }
    }
    if (epoch_.size() >= epoch_cap_) { FinalizeEpoch(false); return; }
    if (voiced_ >= min_voiced_ && abs_ >= next_tick_) TickEpoch();
  }

  void OpenEpoch() {
    in_epoch_ = true;
    epoch_.clear();
    voiced_ = 0;
    silence_ = 0;
    voiced_end_ = abs_;
    confirmed_.clear();
    tail_.clear();
    last_hyp_.clear();
    emitted_text_.clear();
    emitted_tail_.clear();
    next_tick_ = abs_ + tick_;
    epoch_start_ = abs_;
  }

  // Window handed to the ASR: epoch audio up to the last voiced sample plus a
  // 0.2 s pad (leading/trailing silence beyond that only costs NPU time).
  std::vector<int16_t> AsrWindow() const {
    const size_t pad = kRateOut / 5;
    const size_t voiced_len = voiced_end_ - epoch_start_;
    const size_t len = std::min(epoch_.size(), voiced_len + pad);
    return std::vector<int16_t>(epoch_.begin(), epoch_.begin() + len);
  }

  void TickEpoch() {
    next_tick_ = abs_ + tick_;
    if (voiced_ < min_voiced_) return;
    const std::vector<int16_t> window = AsrWindow();
    last_window_len_ = window.size();
    std::string hyp;
    {
      auto cell = arbiter_->Acquire();
      hyp = asr_->TranscribePcm(window);
    }
    if (hyp.empty() || hyp.rfind("[ASR_ERROR]", 0) == 0) return;
    // LocalAgreement-2 (punctuation-loose): confirm the content both this and
    // the previous hypothesis agree on; the newer hypothesis's punctuation
    // wins for display.  Length-preference revision: when consecutive
    // hypotheses agree on a LONGER prefix that differs from the committed one
    // (single-char jitter, e.g. the unstable speaker name), the longer
    // agreement replaces it - the page renders whole-line so a revision is
    // cheap, and the epoch close rewrites the line anyway.
    std::string add;
    if (!last_hyp_.empty()) {
      const size_t p = textutil::CommonPrefixLoose(last_hyp_, hyp);
      if (p > 0) {
        const std::string nc = hyp.substr(0, p);
        const std::string ncs = textutil::StripPunct(nc);
        const std::string ocs = textutil::StripPunct(confirmed_);
        if (ncs.size() >= ocs.size() && nc != confirmed_) {
          add = ncs.compare(0, ocs.size(), ocs) == 0 && ncs.size() > ocs.size()
                    ? ncs.substr(ocs.size())
                    : "";
          confirmed_ = nc;
        }
      }
    }
    last_hyp_ = hyp;
    // Locate the confirmed content inside this hypothesis (punctuation may
    // have moved) and show the rest as the provisional tail.
    tail_ = hyp.substr(textutil::ContentMatchOffset(hyp, textutil::StripPunct(confirmed_)));
    const bool changed = confirmed_ != emitted_text_ || tail_ != emitted_tail_;
    if (!changed) return;
    emitted_text_ = confirmed_;
    emitted_tail_ = tail_;
    const std::string b = EventLog::Sec(epoch_start_, kRateOut);
    const std::string e = EventLog::Sec(abs_, kRateOut);
    if (!streaming_text_file_.empty()) {
      std::ofstream stxt(streaming_text_file_, std::ios::app);
      stxt << "tick@[" << b << "," << e << "] " << confirmed_ << " | " << tail_
           << "\n";
    }
    EventLog::Get().Write("stext",
        "\"b\":" + b + ",\"e\":" + e +
        ",\"add\":" + EventLog::Quote(add) +
        ",\"text\":" + EventLog::Quote(confirmed_) +
        ",\"tail\":" + EventLog::Quote(tail_));
  }

  // Epoch closes on a confirmed pause (word-safe), on the hard epoch cap, or
  // at end of stream.  The final full-epoch hypothesis is committed in full;
  // if it disagrees with the already-committed prefix the sfinal text simply
  // replaces the epoch line on the page (append-only within ticks, but the
  // epoch close is the authoritative-for-epoch view).
  void FinalizeEpoch(bool end_of_stream) {
    if (!in_epoch_) return;
    in_epoch_ = false;
    const bool voiced_enough = voiced_ >= min_voiced_ * 3 / 5;
    if (voiced_enough) {
      std::string text;
      if (NeedFinalAsr()) {
        const std::vector<int16_t> window = AsrWindow();
        auto cell = arbiter_->Acquire();
        text = asr_->TranscribePcm(window);
      } else {
        text = last_hyp_;  // nothing new since the last tick's hypothesis
      }
      if (text.empty() || text.rfind("[ASR_ERROR]", 0) == 0) text = last_hyp_;
      if (!text.empty()) {
        confirmed_ = text;
        const std::string b = EventLog::Sec(epoch_start_, kRateOut);
        const std::string e = EventLog::Sec(
            end_of_stream ? abs_ : voiced_end_, kRateOut);
        if (!streaming_text_file_.empty()) {
          std::ofstream stxt(streaming_text_file_, std::ios::app);
          stxt << "final@[" << b << "," << e << "] " << text << "\n";
        }
        EventLog::Get().Write("sfinal",
            "\"b\":" + b + ",\"e\":" + e +
            ",\"text\":" + EventLog::Quote(text));
      }
    }
    epoch_.clear();
    confirmed_.clear();
    tail_.clear();
    last_hyp_.clear();
  }

  // Skip the final ASR pass when the last tick's window already covered all
  // voiced audio (pause close right after a tick): its hypothesis is the
  // final text.  Compare against the last window length in samples.
  bool NeedFinalAsr() const {
    if (last_hyp_.empty()) return true;
    const size_t voiced_len = voiced_end_ - epoch_start_;
    return voiced_len > last_window_len_;
  }

  bool enabled_ = false;
  SenseVoiceRunner* asr_ = nullptr;
  Npu2CellArbiter* arbiter_ = nullptr;
  std::string streaming_text_file_;
  const size_t tick_, pause_close_, epoch_cap_;
  const size_t min_voiced_ = kRateOut / 2;   // 0.5 s of voiced audio
  BlockingQueue<Item> queue_{256};  // ~4 s of hops: ASR/diar NPU waits must
                                    // not drop epoch audio (garbled text)
  std::thread thread_;

  // Slicer-thread state (fed through queue_).
  size_t abs_ = 0;          // absolute 16k sample count fed so far
  bool in_epoch_ = false;
  std::vector<int16_t> epoch_;  // samples from epoch_start_ (includes pauses)
  size_t epoch_start_ = 0, voiced_ = 0, silence_ = 0, voiced_end_ = 0;
  size_t next_tick_ = 0;
  size_t last_window_len_ = 0;
  std::string confirmed_, tail_, last_hyp_;
  std::string emitted_text_, emitted_tail_;
};

// ---------------------------------------------------------------------------
// Streaming ASR (true-streaming text track).
//
// Replaces the "cut segments then transcribe" text path with a continuous
// session over the streaming SenseVoice model (26-frame rolling window,
// NPU2).  Audio is fed hop by hop exactly like the diarization stream; the
// session returns incremental hypotheses which are stabilized with the same
// punctuation-loose LocalAgreement-2 merge the offline slicer uses, so the
// user-visible text only ever grows.  The VAD no longer decides where ASR
// cuts happen - epochs close on >= stream_pause_seconds silences (word-safe
// phrase boundaries) and the diarization timeline attributes speakers, so
// recognition never sees an arbitrary mid-word window again.
//
// asr.mode: streaming selects this track; the per-segment offline ASR path
// in ResultWorker is skipped (it still archives WAV/MP3).
// ---------------------------------------------------------------------------

// One streaming recognition session.  A session covers one epoch (phrase);
// Finish() flushes the tail and returns the final text for it.
class StreamingAsrSession {
 public:
  virtual ~StreamingAsrSession() = default;
  // Feed epoch audio (16 kHz mono S16LE); called hop by hop.
  virtual void Feed(const std::vector<int16_t>& pcm) = 0;
  // Current hypothesis for everything fed so far (may run inference).
  virtual std::string RequestPartial() = 0;
  // Flush the tail: returns the epoch's final text and resets the session.
  virtual std::string Finish() = 0;
  virtual void Close() = 0;
};

namespace {

// Minimal JSON string-field extractor: finds "key":"value" and decodes the
// standard escapes (the asr_server response is exactly {"text":"..."}).
bool ParseJsonStringField(const std::string& json, const char* key, std::string* out) {
  const std::string needle = std::string("\"") + key + "\"";
  std::size_t p = json.find(needle);
  if (p == std::string::npos) return false;
  p = json.find(':', p + needle.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) ++p;
  if (p >= json.size() || json[p] != '"') return false;
  ++p;
  out->clear();
  while (p < json.size() && json[p] != '"') {
    if (json[p] == '\\' && p + 1 < json.size()) {
      ++p;
      const char e = json[p];
      switch (e) {
        case 'n': out->push_back('\n'); break;
        case 't': out->push_back('\t'); break;
        case 'r': out->push_back('\r'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'u': {
          if (p + 4 >= json.size()) return false;
          unsigned cp = 0;
          for (int i = 0; i < 4; ++i) {
            ++p;
            const char h = json[p];
            cp = cp * 16 + (h <= '9' ? static_cast<unsigned>(h - '0')
                                     : static_cast<unsigned>((h | 32) - 'a' + 10));
          }
          if (cp < 0x80) {
            out->push_back(static_cast<char>(cp));
          } else if (cp < 0x800) {
            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          } else {
            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: out->push_back(e); break;
      }
      ++p;
    } else {
      out->push_back(json[p]);
      ++p;
    }
  }
  return p < json.size();
}

std::string ServerPortOf(const std::string& server) {
  const std::size_t p = server.rfind(':');
  if (p == std::string::npos) return "8080";
  std::string port = server.substr(p + 1);
  const std::size_t slash = port.find('/');
  if (slash != std::string::npos) port = port.substr(0, slash);
  return port.empty() ? std::string("8080") : port;
}

}  // namespace

// backend = "http": the bundled OpenAI-compatible asr_server binary.  Each
// partial request POSTs the growing epoch window; the server feeds it through
// the streaming model (rolling-window CTC, no mid-word cuts).  The pipeline
// spawns the server on demand and reuses an already-running one.
class HttpAsrSession : public StreamingAsrSession {
 public:
  HttpAsrSession(const apm_example::AppConfig::Asr::Streaming& cfg,
                 const std::string& model_dir, const std::string& language,
                 const std::string& work_dir)
      : cfg_(cfg), model_dir_(model_dir), language_(language), work_dir_(work_dir) {
    server_ = cfg_.server;
    fs::create_directories(work_dir_, ec_);
  }

  bool Start() {
    if (ServerAlive()) {
      std::cerr << "[asr-stream] using external asr_server at " << server_ << "\n";
      return true;
    }
    const std::string exe = FindExecutable(cfg_.runner);
    if (exe.empty()) {
      std::cerr << "[asr-stream] asr_server is not executable: " << cfg_.runner << "\n";
      return false;
    }
    const std::string port = ServerPortOf(server_);
    const std::string log = (fs::path(work_dir_) / "asr-server.log").string();
    pid_ = fork();
    if (pid_ < 0) return false;
    if (pid_ == 0) {
      const int null_fd = open("/dev/null", O_RDONLY);
      const int log_fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (null_fd >= 0) dup2(null_fd, STDIN_FILENO);
      if (log_fd >= 0) { dup2(log_fd, STDOUT_FILENO); dup2(log_fd, STDERR_FILENO); }
      execl(exe.c_str(), exe.c_str(), "-p", port.c_str(), "-m", model_dir_.c_str(),
            "--payload_limit_mb", "200", static_cast<char*>(nullptr));
      _exit(127);
    }
    for (int i = 0; i < 60; ++i) {
      if (ServerAlive()) {
        std::cerr << "[asr-stream] asr_server spawned on port " << port
                  << " (model " << model_dir_ << ")\n";
        return true;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cerr << "[asr-stream] asr_server did not come up on port " << port << "\n";
    return false;
  }

  void Feed(const std::vector<int16_t>& pcm) override {
    buffer_.insert(buffer_.end(), pcm.begin(), pcm.end());
  }

  std::string RequestPartial() override { return Post(buffer_); }

  std::string Finish() override {
    const std::string text = Post(buffer_);
    buffer_.clear();
    return text;
  }

  void Close() override {
    buffer_.clear();
    if (pid_ > 0) {
      kill(pid_, SIGTERM);
      int status = 0;
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
      pid_ = -1;
    }
  }

 private:
  bool ServerAlive() {
    const std::string cmd = "curl -sf --max-time 3 " +
        ShellQuote(server_ + "/v1/models") + " > /dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
  }

  std::string Post(const std::vector<int16_t>& pcm) {
    if (pcm.empty()) return "";
    const fs::path wav = fs::path(work_dir_) / "asr-req.wav";
    if (!WriteWav(wav, pcm, kRateOut)) {
      std::cerr << "[asr-stream] request WAV failed\n";
      return "";
    }
    const fs::path resp = fs::path(work_dir_) / "asr-req.json";
    std::string cmd = "curl -sf --max-time 60 -X POST";
    cmd += " -F " + ShellQuote("file=@" + wav.string());
    cmd += " -F " + ShellQuote("model=" + cfg_.model_name);
    cmd += " -F " + ShellQuote("language=" + language_);
    cmd += " " + ShellQuote(server_ + "/v1/audio/transcriptions");
    cmd += " -o " + ShellQuote(resp.string());
    cmd += " > /dev/null 2>&1";
    const int status = std::system(cmd.c_str());
    if (status != 0) {
      std::cerr << "[asr-stream] transcriptions POST failed status=" << status << "\n";
      return "";
    }
    std::ifstream in(resp.c_str(), std::ios::binary);
    const std::string json((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    std::string text;
    if (!ParseJsonStringField(json, "text", &text)) {
      std::cerr << "[asr-stream] response parse failed: " << json.substr(0, 120) << "\n";
      return "";
    }
    return text;
  }

  apm_example::AppConfig::Asr::Streaming cfg_;
  std::string model_dir_, language_, work_dir_, server_;
  std::error_code ec_;
  std::vector<int16_t> buffer_;
  pid_t pid_ = -1;
};

// backend = "worker": the resident stream-asr-worker.  Framed PCM on stdin,
// incremental text events on fd 3 (protocol: docs/stream-asr-worker-protocol.md).
class StdioWorkerSession : public StreamingAsrSession {
 public:
  StdioWorkerSession(const apm_example::AppConfig::Asr::Streaming& cfg,
                     const std::string& model_dir, const std::string& language)
      : cfg_(cfg), model_dir_(model_dir), language_(language),
        push_(std::max<size_t>(1, static_cast<size_t>(0.1f * kRateOut))) {}

  bool Start() {
    const std::string exe = FindExecutable(cfg_.worker);
    if (exe.empty()) {
      std::cerr << "[asr-stream] worker is not executable: " << cfg_.worker << "\n";
      return false;
    }
    int in_pipe[2] = {-1, -1}, resp_pipe[2] = {-1, -1};
    if (pipe2(in_pipe, O_CLOEXEC) != 0 || pipe2(resp_pipe, O_CLOEXEC) != 0) return false;
    pid_ = fork();
    if (pid_ < 0) return false;
    if (pid_ == 0) {
      dup2(in_pipe[0], STDIN_FILENO);
      dup2(resp_pipe[1], 3);
      for (int fd : in_pipe) if (fd != STDIN_FILENO && fd != 3) close(fd);
      for (int fd : resp_pipe) if (fd != STDIN_FILENO && fd != 3) close(fd);
      execl(exe.c_str(), exe.c_str(), "--model-dir", model_dir_.c_str(),
            "--language", language_.c_str(), "--response-fd", "3",
            static_cast<char*>(nullptr));
      _exit(127);
    }
    close(in_pipe[0]);
    close(resp_pipe[1]);
    in_fd_ = in_pipe[1];
    resp_fd_ = resp_pipe[0];
    char type = 0;
    std::string payload;
    if (!ReadFrame(&type, &payload) || type != 'R') {
      std::cerr << "[asr-stream] worker startup failed: "
                << (payload.empty() ? "worker exited" : payload) << "\n";
      return false;
    }
    reader_ = std::thread([this] { ReaderLoop(); });
    return true;
  }

  void Feed(const std::vector<int16_t>& pcm) override {
    pending_.insert(pending_.end(), pcm.begin(), pcm.end());
    if (pending_.size() >= push_) FlushPcm();
  }

  std::string RequestPartial() override {
    std::lock_guard<std::mutex> lock(mutex_);
    return partial_;
  }

  std::string Finish() override {
    FlushPcm();
    const uint32_t zero = 0;
    if (!WriteAll(in_fd_, &zero, sizeof(zero))) return PartialLocked();
    std::unique_lock<std::mutex> lock(mutex_);
    if (!done_.wait_for(lock, std::chrono::seconds(15), [&] { return final_ready_ || dead_; }))
      std::cerr << "[asr-stream] worker finish timeout\n";
    const std::string text = final_ready_ ? final_ : partial_;
    final_.clear();
    final_ready_ = false;
    partial_.clear();
    return text;
  }

  void Close() override {
    if (in_fd_ >= 0) close(in_fd_);
    in_fd_ = -1;
    if (reader_.joinable()) reader_.join();
    if (resp_fd_ >= 0) close(resp_fd_);
    resp_fd_ = -1;
    if (pid_ > 0) {
      int status = 0;
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
      pid_ = -1;
    }
  }

 private:
  static bool WriteAll(int fd, const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    while (n > 0) {
      const ssize_t w = write(fd, p, n);
      if (w <= 0) { if (w < 0 && errno == EINTR) continue; return false; }
      p += w; n -= static_cast<std::size_t>(w);
    }
    return true;
  }

  static bool ReadAll(int fd, void* data, std::size_t n) {
    auto* p = static_cast<unsigned char*>(data);
    while (n > 0) {
      const ssize_t r = read(fd, p, n);
      if (r <= 0) { if (r < 0 && errno == EINTR) continue; return false; }
      p += r; n -= static_cast<std::size_t>(r);
    }
    return true;
  }

  bool ReadFrame(char* type, std::string* payload) {
    uint32_t size = 0;
    if (!ReadAll(resp_fd_, type, sizeof(*type)) ||
        !ReadAll(resp_fd_, &size, sizeof(size)) || size > 16U * 1024U * 1024U)
      return false;
    payload->assign(size, '\0');
    return ReadAll(resp_fd_, &(*payload)[0], payload->size());
  }

  void FlushPcm() {
    if (pending_.empty()) return;
    const uint32_t bytes = static_cast<uint32_t>(pending_.size() * sizeof(int16_t));
    if (!WriteAll(in_fd_, &bytes, sizeof(bytes)) ||
        !WriteAll(in_fd_, pending_.data(), bytes)) {
      std::cerr << "[asr-stream] worker PCM pipe write failed\n";
      std::lock_guard<std::mutex> lock(mutex_);
      dead_ = true;
      done_.notify_all();
    }
    pending_.clear();
  }

  std::string PartialLocked() {
    std::lock_guard<std::mutex> lock(mutex_);
    return partial_;
  }

  void ReaderLoop() {
    char type = 0;
    std::string payload;
    while (ReadFrame(&type, &payload)) {
      if (type == 'T' && payload.size() >= sizeof(uint32_t)) {
        uint32_t flags = 0;
        std::memcpy(&flags, payload.data(), sizeof(flags));
        const std::string text = payload.substr(sizeof(flags));
        std::lock_guard<std::mutex> lock(mutex_);
        if (flags & 2u) { final_ = text; final_ready_ = true; }
        else partial_ = text;
        done_.notify_all();
      } else if (type == 'E') {
        std::cerr << "[asr-stream] worker error: " << payload << "\n";
        std::lock_guard<std::mutex> lock(mutex_);
        dead_ = true;
        done_.notify_all();
        return;
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    dead_ = true;
    done_.notify_all();
  }

  apm_example::AppConfig::Asr::Streaming cfg_;
  std::string model_dir_, language_;
  const std::size_t push_;
  int in_fd_ = -1, resp_fd_ = -1;
  pid_t pid_ = -1;
  std::thread reader_;
  std::vector<int16_t> pending_;
  mutable std::mutex mutex_;
  std::condition_variable done_;
  std::string partial_, final_;
  bool final_ready_ = false, dead_ = false;
};

// Speaker helpers over an absolute parts timeline.

// Speaker with the most coverage of [b, e); empty when nothing covers it.
std::string DominantSpeaker(const std::vector<Segment::Part>& parts,
                            std::size_t b, std::size_t e) {
  if (parts.empty() || e <= b) return "";
  std::vector<std::pair<std::string, std::size_t>> cover;
  for (const auto& p : parts) {
    const std::size_t cb = std::max(p.begin, b), ce = std::min(p.end, e);
    if (ce <= cb || p.speaker_id.empty()) continue;
    bool found = false;
    for (auto& c : cover) {
      if (c.first == p.speaker_id) { c.second += ce - cb; found = true; break; }
    }
    if (!found) cover.emplace_back(p.speaker_id, ce - cb);
  }
  if (cover.empty()) return "";
  std::stable_sort(cover.begin(), cover.end(),
                   [](const auto& a, const auto& b2) { return a.second > b2.second; });
  return cover.front().first;
}

// Non-overlapping speaker segmentation of [b, e): longest part wins on
// overlap, uncovered gaps become empty-speaker pieces so turn changes stay
// visible.  Pieces are sorted by time and adjacent same-speaker pieces are
// merged.
std::vector<Segment::Part> ResolveSpeakerSegments(
    const std::vector<Segment::Part>& parts, std::size_t b, std::size_t e) {
  std::vector<Segment::Part> clipped;
  for (const auto& p : parts) {
    const std::size_t cb = std::max(p.begin, b), ce = std::min(p.end, e);
    if (ce > cb) clipped.push_back({cb, ce, p.speaker_id});
  }
  std::stable_sort(clipped.begin(), clipped.end(),
                   [](const Segment::Part& a, const Segment::Part& c) {
                     return (a.end - a.begin) > (c.end - c.begin);
                   });
  std::vector<Segment::Part> taken;
  for (const auto& p : clipped) {
    bool free_range = true;
    for (const auto& t : taken) {
      if (p.begin < t.end && t.begin < p.end) { free_range = false; break; }
    }
    if (free_range) taken.push_back(p);
  }
  std::stable_sort(taken.begin(), taken.end(),
                   [](const Segment::Part& a, const Segment::Part& c) { return a.begin < c.begin; });
  std::vector<Segment::Part> out;
  std::size_t cursor = b;
  for (const auto& t : taken) {
    if (t.begin > cursor) out.push_back({cursor, t.begin, ""});
    out.push_back(t);
    cursor = std::max(cursor, t.end);
  }
  if (cursor < e) out.push_back({cursor, e, ""});
  for (std::size_t i = out.size() - 1; i > 0; --i) {
    if (out[i].speaker_id == out[i - 1].speaker_id) {
      out[i - 1].end = out[i].end;
      out.erase(out.begin() + static_cast<std::ptrdiff_t>(i));
    }
  }
  return out;
}

class StreamingAsrTrack {
 public:
  StreamingAsrTrack(const apm_example::AppConfig& c, SenseVoiceRunner* asr,
                    Npu2CellArbiter* arbiter)
      : refine_(c.asr.refine == "epoch"),
        split_turns_(c.asr.streaming.split_turns),
        poll_(std::max<size_t>(1, static_cast<size_t>(c.asr.streaming.poll_seconds * kRateOut))),
        pause_close_(std::max<size_t>(1, static_cast<size_t>(
            c.asr.streaming.epoch_pause_seconds * kRateOut))),
        epoch_cap_(std::max<size_t>(1, static_cast<size_t>(c.asr.streaming.epoch_cap_seconds * kRateOut))),
        label_wait_ms_(static_cast<int>(c.asr.streaming.label_wait_seconds * 1000.0f)),
        cfg_(c.asr), asr_(asr), arbiter_(arbiter),
        text_file_(c.output.text_file),
        streaming_text_file_(DeriveStreamingTextFile(c.output.text_file)),
        work_dir_(c.output.temp_dir) {
    fs::create_directories(work_dir_);
    if (!text_file_.empty())
      fs::create_directories(fs::path(text_file_).parent_path());
  }
  ~StreamingAsrTrack() { FlushAndClose(); }

  // Set after ResultWorker::StartDiarization created the nemotron worker.
  void SetSpeakerSource(SpeakerLookup* speakers) { speakers_ = speakers; }

  void Start() {
    if (thread_.joinable()) return;
    thread_ = std::thread([this] { Loop(); });
  }

  // Producer side (VAD loop thread); never blocks it.
  void Push(const std::vector<int16_t>& hop, bool voiced) {
    if (!thread_.joinable()) return;
    Item item;
    item.samples = hop;
    item.voiced = voiced;
    if (!queue_.TryPush(std::move(item))) {
      queue_.TryDropOldest();
      queue_.TryPush(std::move(item));
    }
  }

  void FlushAndClose() {
    if (!thread_.joinable()) return;
    Item flush;
    flush.flush = true;
    queue_.Push(std::move(flush));
    thread_.join();
  }
  void Close(bool) { FlushAndClose(); }

 private:
  struct Item {
    std::vector<int16_t> samples;
    bool voiced = false;
    bool flush = false;
  };

  void Loop() {
    if (!MakeSession()) {
      queue_.Close();
      return;
    }
    Item item;
    while (queue_.Pop(&item)) {
      if (item.flush) break;
      TrackWall();
      abs_ += item.samples.size();
      Feed(item);
    }
    if (item.flush) {
      while (queue_.TryPopValue(&item)) {
        if (item.flush) continue;
        TrackWall();
        abs_ += item.samples.size();
        Feed(item);
      }
      FinalizeEpoch(true);
      queue_.Close();
    }
    if (session_) session_->Close();
  }

  bool MakeSession() {
    const std::string model_dir = cfg_.streaming.model_dir.empty()
                                     ? cfg_.model_path
                                     : cfg_.streaming.model_dir;
    const std::string language = cfg_.streaming.language.empty()
                                     ? cfg_.language
                                     : cfg_.streaming.language;
    if (cfg_.streaming.backend == "worker") {
      auto session = std::make_unique<StdioWorkerSession>(cfg_.streaming, model_dir, language);
      if (!session->Start()) return false;
      session_ = std::move(session);
      return true;
    }
    if (cfg_.streaming.backend == "http") {
      auto session = std::make_unique<HttpAsrSession>(cfg_.streaming, model_dir, language, work_dir_);
      if (!session->Start()) return false;
      session_ = std::move(session);
      return true;
    }
    std::cerr << "[asr-stream] backend '" << cfg_.streaming.backend
              << "' is not wired yet (use http or worker)\n";
    return false;
  }

  void TrackWall() {
    if (have_wall_) return;
    base_wall_ = WallClock::now();
    base_abs_ = abs_;
    have_wall_ = true;
  }

  WallClock::time_point WallAt(std::size_t pos) const {
    return base_wall_ + std::chrono::milliseconds(
        static_cast<long long>((pos - base_abs_) * 1000 / kRateOut));
  }

  void Feed(const Item& item) {
    if (!in_epoch_) {
      if (!item.voiced) return;
      OpenEpoch();
    }
    epoch_.insert(epoch_.end(), item.samples.begin(), item.samples.end());
    if (session_) session_->Feed(item.samples);
    if (item.voiced) {
      voiced_ += item.samples.size();
      silence_ = 0;
      voiced_end_ = abs_;
    } else {
      silence_ += item.samples.size();
      if (silence_ >= pause_close_) { FinalizeEpoch(false); return; }
    }
    if (epoch_.size() >= epoch_cap_) { FinalizeEpoch(false); return; }
    if (voiced_ >= min_voiced_ && abs_ >= next_poll_) PollPartial();
  }

  void OpenEpoch() {
    in_epoch_ = true;
    epoch_.clear();
    voiced_ = 0;
    silence_ = 0;
    voiced_end_ = abs_;
    confirmed_.clear();
    tail_.clear();
    last_hyp_.clear();
    emitted_text_.clear();
    emitted_tail_.clear();
    anchors_.clear();
    next_poll_ = abs_ + poll_;
    epoch_start_ = abs_;
  }

  void PollPartial() {
    next_poll_ = abs_ + poll_;
    if (voiced_ < min_voiced_) return;
    if (!session_) return;
    const std::string hyp = session_->RequestPartial();
    MergeHypothesis(hyp);
  }

  // Punctuation-loose LocalAgreement-2 (shared helpers, same policy as the
  // offline slicer); also maintains the char<->time anchors.
  void MergeHypothesis(const std::string& hyp) {
    if (hyp.empty() || hyp.rfind("[ASR_ERROR]", 0) == 0) return;
    std::string add;
    if (!last_hyp_.empty()) {
      const std::size_t p = textutil::CommonPrefixLoose(last_hyp_, hyp);
      if (p > 0) {
        const std::string nc = hyp.substr(0, p);
        const std::string ncs = textutil::StripPunct(nc);
        const std::string ocs = textutil::StripPunct(confirmed_);
        if (ncs.size() >= ocs.size() && nc != confirmed_) {
          add = ncs.compare(0, ocs.size(), ocs) == 0 && ncs.size() > ocs.size()
                    ? ncs.substr(ocs.size())
                    : "";
          confirmed_ = nc;
        }
      }
    }
    last_hyp_ = hyp;
    tail_ = hyp.substr(textutil::ContentMatchOffset(hyp, textutil::StripPunct(confirmed_)));
    const std::size_t chars = textutil::StripPunct(confirmed_).size();
    if (chars > 0) {
      if (anchors_.empty()) {
        anchors_.push_back({chars, abs_});
      } else if (chars > anchors_.back().chars) {
        if (anchors_.back().pos >= abs_) anchors_.back().chars = chars;
        else anchors_.push_back({chars, abs_});
      }
    }
    const bool changed = confirmed_ != emitted_text_ || tail_ != emitted_tail_;
    if (!changed) return;
    emitted_text_ = confirmed_;
    emitted_tail_ = tail_;
    const std::string b = EventLog::Sec(epoch_start_, kRateOut);
    const std::string e = EventLog::Sec(abs_, kRateOut);
    const std::string spk = ProvisionalSpeaker();
    if (!streaming_text_file_.empty()) {
      std::ofstream stxt(streaming_text_file_, std::ios::app);
      stxt << "tick@[" << b << "," << e << "] " << (spk.empty() ? std::string("-") : spk)
           << " " << confirmed_ << " | " << tail_ << "\n";
    }
    EventLog::Get().Write("stext",
        "\"b\":" + b + ",\"e\":" + e +
        ",\"add\":" + EventLog::Quote(add) +
        ",\"text\":" + EventLog::Quote(confirmed_) +
        ",\"tail\":" + EventLog::Quote(tail_) +
        (spk.empty() ? "" : ",\"spk\":" + EventLog::Quote(spk)));
  }

  // Best-effort label of the epoch so far: the dominant speaker of the
  // already-covered prefix of the diarization timeline.
  std::string ProvisionalSpeaker() const {
    if (!speakers_) return "";
    const std::size_t committed = speakers_->Committed();
    if (committed <= epoch_start_) return "";
    return DominantSpeaker(speakers_->Timeline(), epoch_start_,
                           std::min(voiced_end_, committed));
  }

  // Window handed to the offline refine pass (if enabled).
  std::vector<int16_t> EpochWindow() const {
    const std::size_t pad = kRateOut / 5;
    const std::size_t voiced_len = voiced_end_ - epoch_start_;
    const std::size_t len = std::min(epoch_.size(), voiced_len + pad);
    return std::vector<int16_t>(epoch_.begin(), epoch_.begin() + len);
  }

  void FinalizeEpoch(bool end_of_stream) {
    if (!in_epoch_) return;
    in_epoch_ = false;
    const bool voiced_enough = voiced_ >= min_voiced_ * 3 / 5;
    std::string text;
    if (session_) text = session_->Finish();
    if (!voiced_enough) {
      ResetEpochState();
      return;
    }
    if (refine_ && asr_ && !epoch_.empty()) {
      auto lease = arbiter_->Acquire();
      const std::string r = asr_->TranscribePcm(EpochWindow());
      if (!r.empty() && r.rfind("[ASR_ERROR]", 0) != 0) text = r;
    }
    if (text.empty() || text.rfind("[ASR_ERROR]", 0) == 0) text = last_hyp_;
    if (!text.empty()) EmitFinal(text);
    ResetEpochState();
  }

  void ResetEpochState() {
    epoch_.clear();
    confirmed_.clear();
    tail_.clear();
    last_hyp_.clear();
    anchors_.clear();
  }

  struct Piece {
    std::size_t begin = 0, end = 0;
    std::string speaker, text;
  };

  // Split one epoch's text at diarization turn boundaries via the coarse
  // char<->time map; falls back to an empty vector (single dominant piece)
  // whenever the split does not line up piece-for-piece.
  std::vector<Piece> SplitByTurns(const std::string& text,
                                  const std::vector<Segment::Part>& parts,
                                  std::size_t b, std::size_t e) const {
    if (!speakers_) return {};
    const auto resolved = ResolveSpeakerSegments(parts, b, e);
    if (resolved.size() < 2) return {};
    std::vector<std::size_t> cuts;
    for (std::size_t i = 1; i < resolved.size(); ++i)
      cuts.push_back(textutil::CharsAt(anchors_, resolved[i].begin));
    const auto pieces = textutil::SplitAt(text, cuts);
    if (pieces.size() != resolved.size()) return {};
    std::vector<Piece> out;
    for (std::size_t i = 0; i < pieces.size(); ++i) {
      if (pieces[i].empty()) continue;
      out.push_back({resolved[i].begin, resolved[i].end, resolved[i].speaker_id, pieces[i]});
    }
    return out;
  }

  void EmitFinal(const std::string& text) {
    const std::size_t b = epoch_start_, e = voiced_end_;
    std::vector<Segment::Part> parts;
    if (speakers_) {
      if (!speakers_->Finished())
        speakers_->WaitCovered(e, std::chrono::milliseconds(label_wait_ms_));
      parts = speakers_->Timeline();
    }
    std::vector<Piece> pieces;
    if (split_turns_) pieces = SplitByTurns(text, parts, b, e);
    if (pieces.empty())
      pieces.push_back({b, e, DominantSpeaker(parts, b, e), text});

    const std::string bs = EventLog::Sec(b, kRateOut);
    const std::string es = EventLog::Sec(e, kRateOut);
    const std::string dominant = DominantSpeaker(parts, b, e);
    if (!streaming_text_file_.empty()) {
      std::ofstream stxt(streaming_text_file_, std::ios::app);
      stxt << "final@[" << bs << "," << es << "] "
           << (dominant.empty() ? std::string("-") : dominant) << " " << text << "\n";
    }
    EventLog::Get().Write("sfinal",
        "\"b\":" + bs + ",\"e\":" + es +
        ",\"text\":" + EventLog::Quote(text) +
        (dominant.empty() ? "" : ",\"spk\":" + EventLog::Quote(dominant)));

    std::ofstream txt(text_file_, std::ios::app);
    for (const Piece& piece : pieces) {
      if (piece.text.empty()) continue;
      const auto wall = WallAt(piece.begin);
      if (piece.speaker.empty()) {
        txt << "[" << TimeStamp(wall) << " --> "
            << TimeStamp(WallAt(piece.end)) << "] " << piece.text << "\n";
      } else {
        txt << piece.speaker << "@" << SpeakerTimeStamp(wall) << " " << piece.text << "\n";
      }
      txt.flush();
      EventLog::Get().Write("final",
          "\"b\":" + EventLog::Sec(piece.begin, kRateOut) +
          ",\"e\":" + EventLog::Sec(piece.end, kRateOut) +
          ",\"spk\":" + EventLog::Quote(piece.speaker) +
          ",\"text\":" + EventLog::Quote(piece.text));
    }
  }

  const bool refine_;
  const bool split_turns_;
  const std::size_t poll_, pause_close_, epoch_cap_;
  const int label_wait_ms_;
  const std::size_t min_voiced_ = kRateOut / 2;   // 0.5 s of voiced audio
  apm_example::AppConfig::Asr cfg_;
  SenseVoiceRunner* asr_ = nullptr;
  Npu2CellArbiter* arbiter_ = nullptr;
  SpeakerLookup* speakers_ = nullptr;
  std::unique_ptr<StreamingAsrSession> session_;
  const std::string text_file_, streaming_text_file_, work_dir_;
  BlockingQueue<Item> queue_{512};  // ~8 s of hops: session POSTs and diar
                                    // waits must not drop epoch audio
  std::thread thread_;

  // Track-thread state (fed through queue_).
  std::size_t abs_ = 0;          // absolute 16k sample count fed so far
  bool have_wall_ = false;
  WallClock::time_point base_wall_;
  std::size_t base_abs_ = 0;
  bool in_epoch_ = false;
  std::vector<int16_t> epoch_;  // samples from epoch_start_ (includes pauses)
  std::size_t epoch_start_ = 0, voiced_ = 0, silence_ = 0, voiced_end_ = 0;
  std::size_t next_poll_ = 0;
  std::string confirmed_, tail_, last_hyp_;
  std::string emitted_text_, emitted_tail_;
  std::vector<textutil::CharTimeAnchor> anchors_;
};


// Resident pyannote diarization worker client.  The worker keeps the eight
// axmodels loaded and owns the persistent speaker table; one request maps a
// segment WAV to non-overlapping speaker ranges.  A request holds the NPU2
// Cell lock so diarization never overlaps SenseVoice inference.
class DiarWorker {
 public:
  DiarWorker(const apm_example::AppConfig::Diarization& config,
             Npu2CellArbiter* arbiter)
      : config_(config), arbiter_(arbiter) {}
  ~DiarWorker() { Close(); }

  // Starts the worker process.  Must be called after the pipeline has
  // initialized AX Engine in BIG_LITTLE mode: the community-1 models require
  // a VNPU-disabled init, and initializing that before BIG_LITTLE makes the
  // pipeline's own AX_ENGINE_Init fail on this board.
  bool Start() {
    if (!config_.enabled || active_) return active_;
    if (!StartProcess()) {
      std::cerr << "[diar] worker startup failed\n";
      return false;
    }
    active_ = true;
    return true;
  }

  bool active() const { return active_; }

  // Returns sample ranges labelled with persistent speaker ids.  An empty
  // result means diarization was unavailable for this segment; the caller
  // falls back to whole-segment ASR.
  std::vector<Segment::Part> Label(const std::string& wav_path,
                                   std::size_t sample_count) {
    if (!active_) return {};
    std::lock_guard<std::mutex> guard(mutex_);
    try {
      auto lease = arbiter_->Acquire();
      const uint32_t size = static_cast<uint32_t>(wav_path.size());
      if (!WriteAllFd(input_fd_, &size, sizeof(size)) ||
          !WriteAllFd(input_fd_, wav_path.data(), wav_path.size()))
        throw std::runtime_error("worker input pipe closed");
      char type = 0;
      std::string payload;
      if (!ReadFrame(response_fd_, &type, &payload))
        throw std::runtime_error("worker response pipe closed");
      if (type != 'R')
        throw std::runtime_error(payload.empty() ? "worker error" : payload);
      return ParseParts(payload, sample_count);
    } catch (const std::exception& error) {
      std::cerr << "[diar] " << error.what() << "\n";
      return {};
    }
  }

  void Close() {
    if (closed_) return;
    closed_ = true;
    if (input_fd_ >= 0) close(input_fd_);
    if (response_fd_ >= 0) close(response_fd_);
    input_fd_ = response_fd_ = -1;
    if (pid_ > 0) {
      int status = 0;
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
      pid_ = -1;
    }
    active_ = false;
  }

 private:
  static bool WriteAllFd(int fd, const void* data, std::size_t size) {
    const auto* input = static_cast<const unsigned char*>(data);
    while (size > 0) {
      const ssize_t count = write(fd, input, size);
      if (count <= 0) {
        if (count < 0 && errno == EINTR) continue;
        return false;
      }
      input += count;
      size -= static_cast<std::size_t>(count);
    }
    return true;
  }

  static bool ReadAllFd(int fd, void* data, std::size_t size) {
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

  static bool ReadFrame(int fd, char* type, std::string* payload) {
    uint32_t size = 0;
    if (!ReadAllFd(fd, type, sizeof(*type)) ||
        !ReadAllFd(fd, &size, sizeof(size)) || size > 64U * 1024U * 1024U)
      return false;
    payload->assign(size, '\0');
    return ReadAllFd(fd, payload->data(), payload->size());
  }

  static std::vector<Segment::Part> ParseParts(const std::string& payload,
                                               std::size_t sample_count) {
    std::vector<Segment::Part> parts;
    std::istringstream in(payload);
    std::size_t count = 0;
    if (!(in >> count)) throw std::runtime_error("invalid diarization response");
    parts.reserve(std::min<std::size_t>(count, 256));
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t begin = 0, end = 0;
      std::string speaker;
      if (!(in >> begin >> end >> speaker)) {
        std::string detail = payload.substr(0, 160);
        for (char& c : detail) if (c == '\n') c = '|';
        throw std::runtime_error("truncated diarization response count=" +
                                 std::to_string(count) + " index=" +
                                 std::to_string(i) + " payload='" + detail + "'");
      }
      if (end <= begin) continue;
      if (begin > sample_count) begin = sample_count;
      if (end > sample_count) end = sample_count;
      if (end <= begin) continue;
      Segment::Part part;
      part.begin = begin;
      part.end = end;
      part.speaker_id = std::move(speaker);
      parts.push_back(std::move(part));
    }
    return parts;
  }

  static void AddArg(std::vector<std::string>* args,
                     const std::string& name, const std::string& value) {
    args->push_back(name);
    args->push_back(value);
  }

  bool StartProcess() {
    const std::string executable = FindExecutable(config_.runner);
    if (executable.empty()) {
      std::cerr << "[diar] runner is not executable: " << config_.runner << "\n";
      return false;
    }
    int input_pipe[2] = {-1, -1};
    int response_pipe[2] = {-1, -1};
    if (pipe2(input_pipe, O_CLOEXEC) != 0 ||
        pipe2(response_pipe, O_CLOEXEC) != 0) {
      std::cerr << "[diar] cannot create worker pipes\n";
      for (int fd : input_pipe) if (fd >= 0) close(fd);
      for (int fd : response_pipe) if (fd >= 0) close(fd);
      return false;
    }

    std::vector<std::string> args{executable};
    AddArg(&args, "--model", AbsolutePath(config_.model_path));
    AddArg(&args, "--table", config_.speaker_table_path);
    AddArg(&args, "--step", std::to_string(config_.step_sec));
    AddArg(&args, "--threshold", std::to_string(config_.speaker_match_threshold));
    AddArg(&args, "--threads", std::to_string(config_.omp_threads));
    AddArg(&args, "--response-fd", "3");
    AddAssignArgs(&args, config_);

    pid_ = fork();
    if (pid_ < 0) {
      std::cerr << "[diar] fork failed\n";
      for (int fd : input_pipe) close(fd);
      for (int fd : response_pipe) close(fd);
      return false;
    }
    if (pid_ == 0) {
      dup2(input_pipe[0], STDIN_FILENO);
      dup2(response_pipe[1], 3);
      for (int fd : input_pipe)
        if (fd != STDIN_FILENO && fd != 3) close(fd);
      for (int fd : response_pipe)
        if (fd != STDIN_FILENO && fd != 3) close(fd);
      std::vector<char*> argv;
      argv.reserve(args.size() + 1);
      for (std::string& arg : args) argv.push_back(arg.data());
      argv.push_back(nullptr);
      execv(executable.c_str(), argv.data());
      const std::string message = "exec failed: " + std::string(std::strerror(errno));
      const char type = 'E';
      const uint32_t size = static_cast<uint32_t>(message.size());
      WriteAllFd(3, &type, sizeof(type));
      WriteAllFd(3, &size, sizeof(size));
      WriteAllFd(3, message.data(), message.size());
      _exit(127);
    }
    close(input_pipe[0]);
    close(response_pipe[1]);
    input_fd_ = input_pipe[1];
    response_fd_ = response_pipe[0];

    char type = 0;
    std::string message;
    if (!ReadFrame(response_fd_, &type, &message) || type != 'R') {
      std::cerr << "[diar] startup failed: "
                << (message.empty() ? "worker exited" : message) << "\n";
      close(input_fd_);
      close(response_fd_);
      input_fd_ = response_fd_ = -1;
      int status = 0;
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
      pid_ = -1;
      return false;
    }
    std::cerr << "[diar] pyannote worker loaded; speaker table="
              << config_.speaker_table_path << "\n";
    return true;
  }

  apm_example::AppConfig::Diarization config_;
  Npu2CellArbiter* arbiter_;
  int input_fd_ = -1;
  int response_fd_ = -1;
  pid_t pid_ = -1;
  bool active_ = false;
  bool closed_ = false;
  std::mutex mutex_;
};

// Extra diar-worker args selecting the assignment strategy.  The "online"
// mode uses diart-style joint Hungarian assignment with speech-gated speaker
// creation; every other mode keeps the legacy greedy matcher untouched.
void AddAssignArgs(std::vector<std::string>* args,
                   const apm_example::AppConfig::Diarization& config) {
  if (config.mode == "online") {
    args->push_back("--assign");
    args->push_back("online");
    args->push_back("--min-new-sec");
    args->push_back(std::to_string(config.online_min_new_speech_sec));
    args->push_back("--update-min-sec");
    args->push_back(std::to_string(config.online_update_min_sec));
    args->push_back("--max-speakers");
    args->push_back(std::to_string(config.online_max_speakers));
  }
  if (config.incremental) {
    // diart-style incremental clustering (engine-side), replacing the batch
    // AHC+PLDA+VBx path.  Independent of the speaker-table assignment mode.
    args->push_back("--incremental");
    if (config.incremental_plda) args->push_back("--incremental-plda");
    args->push_back("--delta-new");
    args->push_back(std::to_string(config.incremental_delta_new));
    args->push_back("--delta-min");
    args->push_back(std::to_string(config.incremental_delta_min));
    args->push_back("--rho-update");
    args->push_back(std::to_string(config.incremental_rho_update));
  }
}

// Reads the sample rate from a RIFF/WAVE file; returns 0 when the file is not
// a readable WAV.  Session diarization needs the source sample rate to map the
// worker's sample coordinates onto the pipeline's 16 kHz stream.
int ReadWavSampleRate(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return 0;
  char header[12];
  if (!in.read(header, sizeof(header)) ||
      std::memcmp(header, "RIFF", 4) != 0 ||
      std::memcmp(header + 8, "WAVE", 4) != 0)
    return 0;
  while (in) {
    char chunk[8];
    if (!in.read(chunk, sizeof(chunk))) break;
    const uint32_t size =
        static_cast<uint8_t>(chunk[4]) |
        (static_cast<uint8_t>(chunk[5]) << 8) |
        (static_cast<uint8_t>(chunk[6]) << 16) |
        (static_cast<uint8_t>(chunk[7]) << 24);
    if (std::memcmp(chunk, "fmt ", 4) == 0) {
      std::vector<char> fmt(size);
      if (!in.read(fmt.data(), static_cast<std::streamsize>(fmt.size())) ||
          fmt.size() < 8)
        return 0;
      return static_cast<int>(
          static_cast<uint8_t>(fmt[4]) |
          (static_cast<uint8_t>(fmt[5]) << 8) |
          (static_cast<uint8_t>(fmt[6]) << 16) |
          (static_cast<uint8_t>(fmt[7]) << 24));
    }
    in.seekg(size, std::ios::cur);
  }
  return 0;
}

// Runs the diar-worker binary once over a whole source file (one-shot mode)
// and returns its non-overlapping speaker ranges scaled to 16 kHz pipeline
// coordinates.  Used by diarization.mode=session for file inputs.
std::vector<Segment::Part> RunSessionDiarization(
    const apm_example::AppConfig::Diarization& config,
    const std::string& wav_path, int source_rate) {
  std::vector<Segment::Part> parts;
  const std::string executable = FindExecutable(config.runner);
  if (executable.empty()) {
    std::cerr << "[diar] session runner is not executable: " << config.runner
              << "\n";
    return parts;
  }
  int out_pipe[2] = {-1, -1};
  if (pipe2(out_pipe, O_CLOEXEC) != 0) {
    std::cerr << "[diar] session pipe failed\n";
    return parts;
  }
  std::vector<std::string> args{executable};
  args.push_back("--model");
  args.push_back(AbsolutePath(config.model_path));
  args.push_back("--table");
  args.push_back(config.speaker_table_path);
  args.push_back("--step");
  args.push_back(std::to_string(config.step_sec));
  args.push_back("--threshold");
  args.push_back(std::to_string(config.speaker_match_threshold));
  args.push_back("--threads");
  args.push_back(std::to_string(config.omp_threads));
  args.push_back("--wav");
  args.push_back(AbsolutePath(wav_path));
  AddAssignArgs(&args, config);
  const pid_t pid = fork();
  if (pid < 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    std::cerr << "[diar] session fork failed\n";
    return parts;
  }
  if (pid == 0) {
    dup2(out_pipe[1], STDOUT_FILENO);
    close(out_pipe[0]);
    close(out_pipe[1]);
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (std::string& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    execv(executable.c_str(), argv.data());
    _exit(127);
  }
  close(out_pipe[1]);
  std::string payload;
  char buffer[4096];
  ssize_t count = 0;
  while ((count = read(out_pipe[0], buffer, sizeof(buffer))) > 0)
    payload.append(buffer, static_cast<std::size_t>(count));
  close(out_pipe[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "[diar] session diarization failed (status=" << status << ")\n";
    return {};
  }
  std::istringstream in(payload);
  std::size_t size = 0;
  if (!(in >> size)) {
    std::cerr << "[diar] session diarization returned no parsable parts\n";
    return {};
  }
  const double scale =
      source_rate > 0 ? static_cast<double>(kRateOut) / source_rate : 1.0;
  for (std::size_t i = 0; i < size; ++i) {
    std::size_t begin = 0, end = 0;
    std::string speaker;
    if (!(in >> begin >> end >> speaker)) break;
    if (end <= begin) continue;
    Segment::Part part;
    part.begin = static_cast<std::size_t>(std::llround(begin * scale));
    part.end = static_cast<std::size_t>(std::llround(end * scale));
    part.speaker_id = std::move(speaker);
    parts.push_back(std::move(part));
  }
  std::cerr << "[diar] session timeline parts=" << parts.size()
            << " source_rate=" << source_rate << "\n";
  return parts;
}

// Streaming session diarization for realtime inputs: buffers raw 16 kHz audio
// (silences preserved), sends fixed chunks to the resident diar-worker, and
// appends the chunk's speaker parts (matched to the persistent table) to an
// absolute timeline.  ResultWorker maps each VAD segment onto that timeline.
class SessionDiarizer {
 public:
  // `chunk_sec` is the clustering window (analysis context); `step_sec` is the
  // commit step (0 = legacy: one commit per full chunk).  When 0 < step <
  // chunk, the worker clusters a full `chunk_sec` window ending at every step
  // boundary but only the newest step-sized core region is committed to the
  // timeline (diart-style decoupling: label latency ~ step + inference while
  // the clustering context stays a full chunk long).  The very first window
  // commits in full so speaker discovery bootstraps with the whole context.
  // `first_window_sec` (0 = off, must be < chunk_sec) shortens that very first
  // commit only: with it set, the opening labels land after ~first_window +
  // inference instead of a whole chunk, and every later commit still clusters
  // a full chunk of context.
  SessionDiarizer(double chunk_sec, double step_sec, double first_window_sec)
      : window_samples_(static_cast<std::size_t>(
            std::max(1.0, chunk_sec) * kRateOut)),
        step_samples_(static_cast<std::size_t>(
            std::max(0.0, step_sec) * kRateOut)),
        first_window_samples_(static_cast<std::size_t>(
            std::max(0.0, first_window_sec) * kRateOut)) {
    if (step_samples_ == 0 || step_samples_ >= window_samples_)
      step_samples_ = window_samples_;  // legacy: commit per full chunk
    if (first_window_samples_ == 0 || first_window_samples_ >= window_samples_)
      first_window_samples_ = window_samples_;
  }
  ~SessionDiarizer() { Stop(); }

  void Start(DiarWorker* worker, const std::string& work_dir) {
    worker_ = worker;
    work_dir_ = work_dir;
    std::error_code ec;
    fs::create_directories(work_dir_, ec);
    thread_ = std::thread([this] { Loop(); });
  }

  // Called from the capture thread for every 16 kHz hop (voiced or silent).
  void Append(const std::vector<int16_t>& hop) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    if (!stream_started_ && !hop.empty()) {
      stream_started_ = true;
      stream_t0_ = std::chrono::steady_clock::now();
    }
    buffer_.insert(buffer_.end(), hop.begin(), hop.end());
    ready_.notify_all();
  }

  bool Covers(std::size_t end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return committed_ >= end;
  }

  // Timeline frontier: every source sample below this has a final label.
  std::size_t Committed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return committed_;
  }

  bool Finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_;
  }

  // Waits until the timeline covers `end`; false on timeout or once the
  // final chunk has committed without covering it (Finish).  Input closure
  // alone (Stop) does NOT end the wait: the final chunk's timeline is still
  // being produced by ProcessRemaining, and waking waiters early would send
  // them to the per-segment fallback with phantom-speaker side effects.
  bool WaitCovered(std::size_t end, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool notified = ready_.wait_for(lock, timeout,
        [&] { return finished_ || committed_ >= end; });
    return notified && committed_ >= end;
  }

  // Final verdict: no further timeline commits will arrive.  Unblocks the
  // remaining waiters so they can fall back per-segment.
  void Finish() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_ = true;
    }
    ready_.notify_all();
  }

  std::vector<Segment::Part> Timeline() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timeline_;
  }

  // Processes the final partial core region (pipeline shutdown); call after
  // Stop().  Clusters the last full window of retained context.
  void ProcessRemaining() {
    std::vector<int16_t> tail;
    std::size_t base = 0, frontier = 0, committed = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      base = base_;
      frontier = base_ + buffer_.size();
      committed = committed_;
      tail = std::move(buffer_);
      buffer_.clear();
      base_ = frontier;
    }
    if (!worker_ || committed >= frontier) return;
    const std::size_t window_begin =
        std::max(base, frontier > window_samples_ ? frontier - window_samples_
                                                  : static_cast<std::size_t>(0));
    if (window_begin >= frontier) return;
    tail.erase(tail.begin(),
               tail.begin() + static_cast<std::ptrdiff_t>(window_begin - base));
    if (tail.size() < static_cast<std::size_t>(kRateOut)) return;
    RunWindow(window_begin, committed, frontier, tail);
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    ready_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

 private:
  // Absolute source position the frontier must reach before the next commit.
  std::size_t NeedNoLock() const {
    // Cold start: a full window unless the ramp-up first window is enabled.
    if (committed_ == 0) return first_window_samples_;
    return committed_ + step_samples_;
  }

  // Earliest source position still needed: the next window ends at
  // committed_ + step and spans window_samples_.
  std::size_t KeepFromNoLock() const {
    if (step_samples_ >= window_samples_) return committed_;
    if (committed_ + step_samples_ > window_samples_)
      return committed_ + step_samples_ - window_samples_;
    return 0;
  }

  double WallSecNoLock() const {
    if (!stream_started_) return 0.0;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         stream_t0_).count();
  }

  void Loop() {
    while (true) {
      std::vector<int16_t> window;
      std::size_t window_begin = 0, core_begin = 0, core_end = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] {
          return closed_ ||
                 base_ + buffer_.size() >= NeedNoLock();
        });
        if (closed_) return;
        core_begin = committed_;
        core_end = NeedNoLock();
        window_begin = (core_end > window_samples_)
                           ? core_end - window_samples_
                           : static_cast<std::size_t>(0);
        window.assign(
            buffer_.begin() + static_cast<std::ptrdiff_t>(window_begin - base_),
            buffer_.begin() + static_cast<std::ptrdiff_t>(core_end - base_));
      }
      RunWindow(window_begin, core_begin, core_end, window);
    }
  }

  // Clusters `window` (absolute [window_begin, core_end)) and commits only
  // the parts intersecting the core region [core_begin, core_end).
  void RunWindow(std::size_t window_begin, std::size_t core_begin,
                 std::size_t core_end, const std::vector<int16_t>& window) {
    if (!worker_) return;
    const fs::path wav = fs::path(work_dir_) /
        ("session-" + std::to_string(window_begin) + ".wav");
    std::vector<Segment::Part> parts;
    if (WriteWav(wav, window, kRateOut)) {
      parts = worker_->Label(wav.string(), window.size());
      std::error_code ec;
      fs::remove(wav, ec);
    } else {
      std::cerr << "[diar] session chunk WAV failed: " << wav << "\n";
    }
    std::vector<Segment::Part> commit;
    commit.reserve(parts.size());
    for (const auto& part : parts) {
      const std::size_t begin = window_begin + part.begin;
      const std::size_t end = window_begin + part.end;
      Segment::Part entry;
      entry.begin = std::max(begin, core_begin);
      entry.end = std::min(end, core_end);
      entry.speaker_id = part.speaker_id;
      if (entry.end > entry.begin) commit.push_back(std::move(entry));
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& entry : commit) timeline_.push_back(entry);
      std::stable_sort(timeline_.begin(), timeline_.end(),
                       [](const Segment::Part& a, const Segment::Part& b) {
                         return a.begin < b.begin;
                       });
      committed_ = std::max(committed_, core_end);
      const std::size_t keep_from = KeepFromNoLock();
      if (base_ < keep_from) {
        const std::size_t erase_n = std::min(buffer_.size(), keep_from - base_);
        buffer_.erase(buffer_.begin(),
                      buffer_.begin() + static_cast<std::ptrdiff_t>(erase_n));
        base_ += erase_n;
      }
      const double wall = WallSecNoLock();
      std::cerr << "[diar] session window start=" << window_begin / kRateOut
                << "s len=" << window.size() / kRateOut << "s core=["
                << core_begin / kRateOut << "s," << core_end / kRateOut
                << "s) parts=" << commit.size()
                << " committed=" << committed_ / kRateOut
                << "s wall=+" << wall << "s\n";
    }
    {
      std::string body = "\"core\":[" + EventLog::Sec(core_begin, kRateOut) +
                         "," + EventLog::Sec(core_end, kRateOut) + "],\"parts\":[";
      for (std::size_t i = 0; i < commit.size(); ++i) {
        if (i) body += ",";
        body += "[" + EventLog::Sec(commit[i].begin, kRateOut) + "," +
                EventLog::Sec(commit[i].end, kRateOut) + "," +
                EventLog::Quote(commit[i].speaker_id) + "]";
      }
      EventLog::Get().Write("diar", body + "]");
    }
    ready_.notify_all();
  }

  std::size_t window_samples_;          // clustering context (analysis window)
  std::size_t step_samples_;            // commit step (== window: legacy)
  std::size_t first_window_samples_;    // ramp-up first commit (== window: off)
  DiarWorker* worker_ = nullptr;
  std::string work_dir_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::vector<int16_t> buffer_;          // starts at absolute `base_`
  std::size_t base_ = 0;                 // absolute position of buffer_[0]
  std::size_t committed_ = 0;            // samples with final labels
  std::vector<Segment::Part> timeline_;  // absolute speaker parts
  bool closed_ = false;
  bool finished_ = false;               // final verdict; no commits follow
  bool stream_started_ = false;
  std::chrono::steady_clock::time_point stream_t0_;
  std::thread thread_;
};

// True-streaming diarization source: the resident Nemotron worker
// (end-to-end streaming Sortformer, driver/nemotron).  Interface-compatible
// with SessionDiarizer so the ResultWorker timeline paths stay shared:
// Append() feeds 16 kHz PCM, the reader thread folds the incremental 10 ms
// speaker-activity frames into the global timeline, and Committed()/
// WaitCovered()/Timeline() expose the same sample-indexed speaker parts.
// Label latency is ~0.9 s (low-latency preset: 0.48 s chunk + 0.32 s
// look-ahead + one encoder step) instead of a full clustering window.
class NemotronStreamDiarizer : public SpeakerLookup {
 public:
  explicit NemotronStreamDiarizer(
      const apm_example::AppConfig::Diarization& config)
      : config_(config),
        onset_(config.nemotron.onset),
        offset_(config.nemotron.offset) {}
  ~NemotronStreamDiarizer() { Stop(); }

  // Fork the worker; throws when it cannot start (missing binary/models,
  // NPU mode conflict, ...).  Called after the pipeline's own AX Engine init
  // like DiarWorker::Start().
  void Start() {
    if (active_) return;
    if (!StartProcess())
      throw std::runtime_error(
          "Nemotron streaming diarization worker is unavailable");
    active_ = true;
    writer_ = std::thread([this] { WriterLoop(); });
    reader_ = std::thread([this] { ReaderLoop(); });
    std::cerr << "[diar] nemotron streaming worker loaded; preset="
              << config_.nemotron.preset << " variant="
              << config_.nemotron.variant << " model_dir="
              << config_.nemotron.model_dir << "\n";
  }

  bool active() const { return active_; }

  // Capture thread: every 16 kHz hop, voiced or silent.
  void Append(const std::vector<int16_t>& hop) {
    if (!active_ || hop.empty()) return;
    std::lock_guard<std::mutex> lock(write_mutex_);
    pcm_buffer_.insert(pcm_buffer_.end(), hop.begin(), hop.end());
    write_ready_.notify_one();
  }

  std::size_t Committed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return committed_;
  }

  bool Finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_;
  }

  // Same semantics as SessionDiarizer::WaitCovered.
  bool WaitCovered(std::size_t end, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool notified = ready_.wait_for(
        lock, timeout, [&] { return finished_ || committed_ >= end; });
    return notified && committed_ >= end;
  }

  // No further timeline updates will arrive.  The reader thread already
  // sets finished_ itself when the worker sends its 'D' stats frame (or
  // exits); Finish only re-notifies, kept for interface parity.
  void Finish() { ready_.notify_all(); }

  std::vector<Segment::Part> Timeline() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timeline_;
  }

  // Input closure: stop feeding PCM so the worker flushes its tail.  The
  // reader keeps draining the response pipe until the worker exits and only
  // then sets finished_, so pending WaitCovered callers still receive the
  // final chunk's labels.
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(write_mutex_);
      closed_ = true;
      write_ready_.notify_all();
    }
    if (writer_.joinable()) writer_.join();
    if (reader_.joinable()) reader_.join();
    CloseProcess();
    active_ = false;
  }

 private:
  static constexpr int kSpeakers = 8;       // Nemotron slot count
  static constexpr std::size_t kFrameSamples = kRateOut / 100;  // 10 ms hop
  static constexpr std::size_t kNoFrame =
      std::numeric_limits<std::size_t>::max();
  // Same-speaker runs separated by less silence than this merge into one
  // timeline part (the frame hysteresis alone leaves brief threshold dips
  // as separate turns, which only fragments ASR requests).
  static constexpr std::size_t kMergeGapFrames = 50;  // 0.5 s

  static bool WriteAllFd(int fd, const void* data, std::size_t size) {
    const auto* input = static_cast<const unsigned char*>(data);
    while (size > 0) {
      const ssize_t count = write(fd, input, size);
      if (count <= 0) {
        if (count < 0 && errno == EINTR) continue;
        return false;
      }
      input += count;
      size -= static_cast<std::size_t>(count);
    }
    return true;
  }

  static bool ReadAllFd(int fd, void* data, std::size_t size) {
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

  static bool ReadFrame(int fd, char* type, std::string* payload) {
    uint32_t size = 0;
    if (!ReadAllFd(fd, type, sizeof(*type)) ||
        !ReadAllFd(fd, &size, sizeof(size)) || size > 64U * 1024U * 1024U)
      return false;
    payload->assign(size, '\0');
    return ReadAllFd(fd, payload->data(), payload->size());
  }

  bool StartProcess() {
    const auto& nemo = config_.nemotron;
    const std::string executable = FindExecutable(nemo.runner);
    if (executable.empty()) {
      std::cerr << "[diar] nemotron runner is not executable: "
                << nemo.runner << "\n";
      return false;
    }
    const std::string dir = AbsolutePath(nemo.model_dir);
    // lowlat/offline selects the streaming profile; variant selects the
    // Pulsar2 compile target (npu2 = BIG_LITTLE alongside the pipeline,
    // npu3 = the original VNPU-DISABLE-only models for standalone runs).
    const std::string suffix = nemo.preset == "lowlat" ? "_lowlat" : "";
    const std::string variant = nemo.variant == "npu3" ? "" : "_npu2";
    const std::string preencode = dir + "/preencode" + suffix + variant + ".axmodel";
    const std::string encoder = dir + "/encoder" + suffix + variant + ".axmodel";
    const std::string sil_emb = dir + "/learnable_sil_emb.npy";
    for (const std::string& path : {preencode, encoder, sil_emb}) {
      if (!fs::exists(path)) {
        std::cerr << "[diar] nemotron model file missing: " << path << "\n";
        return false;
      }
    }
    std::vector<std::string> args{executable,
      "--preencode", preencode, "--encoder", encoder,
      "--sil-emb", sil_emb, "--assets", dir,
      "--preset", nemo.preset,
      "--threads", std::to_string(nemo.threads),
      "--response-fd", "3"};

    int input_pipe[2] = {-1, -1};
    int response_pipe[2] = {-1, -1};
    if (pipe2(input_pipe, O_CLOEXEC) != 0 ||
        pipe2(response_pipe, O_CLOEXEC) != 0) {
      std::cerr << "[diar] cannot create nemotron worker pipes\n";
      for (int fd : input_pipe) if (fd >= 0) close(fd);
      for (int fd : response_pipe) if (fd >= 0) close(fd);
      return false;
    }

    pid_ = fork();
    if (pid_ < 0) {
      std::cerr << "[diar] fork failed\n";
      for (int fd : input_pipe) close(fd);
      for (int fd : response_pipe) close(fd);
      return false;
    }
    if (pid_ == 0) {
      dup2(input_pipe[0], STDIN_FILENO);
      dup2(response_pipe[1], 3);
      for (int fd : input_pipe)
        if (fd != STDIN_FILENO && fd != 3) close(fd);
      for (int fd : response_pipe)
        if (fd != STDIN_FILENO && fd != 3) close(fd);
      std::vector<char*> argv;
      argv.reserve(args.size() + 1);
      for (std::string& arg : args) argv.push_back(arg.data());
      argv.push_back(nullptr);
      execv(executable.c_str(), argv.data());
      const std::string message =
          "exec failed: " + std::string(std::strerror(errno));
      const char type = 'E';
      const uint32_t size = static_cast<uint32_t>(message.size());
      WriteAllFd(3, &type, sizeof(type));
      WriteAllFd(3, &size, sizeof(size));
      WriteAllFd(3, message.data(), message.size());
      _exit(127);
    }
    close(input_pipe[0]);
    close(response_pipe[1]);
    input_fd_ = input_pipe[1];
    response_fd_ = response_pipe[0];

    char type = 0;
    std::string message;
    if (!ReadFrame(response_fd_, &type, &message) || type != 'R') {
      std::cerr << "[diar] nemotron worker startup failed: "
                << (message.empty() ? "worker exited" : message) << "\n";
      close(input_fd_);
      close(response_fd_);
      input_fd_ = response_fd_ = -1;
      int status = 0;
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
      pid_ = -1;
      return false;
    }
    return true;
  }

  void CloseProcess() {
    if (input_fd_ >= 0) close(input_fd_);
    if (response_fd_ >= 0) close(response_fd_);
    input_fd_ = response_fd_ = -1;
    if (pid_ > 0) {
      int status = 0;
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
      pid_ = -1;
    }
  }

  // Feeds buffered PCM to the worker as framed messages; on closure sends
  // the zero-length flush frame and closes the pipe (the worker then emits
  // its tail predictions and a final 'D' stats frame).
  void WriterLoop() {
    std::vector<int16_t> pending;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(write_mutex_);
        write_ready_.wait(
            lock, [&] { return closed_ || !pcm_buffer_.empty(); });
        pending.swap(pcm_buffer_);
        if (pending.empty() && closed_) break;
      }
      const uint32_t bytes =
          static_cast<uint32_t>(pending.size() * sizeof(int16_t));
      if (!WriteAllFd(input_fd_, &bytes, sizeof(bytes)) ||
          !WriteAllFd(input_fd_, pending.data(), bytes)) {
        // Worker died; the reader observes EOF and finishes the timeline.
        std::cerr << "[diar] nemotron worker pipe write failed\n";
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        ready_.notify_all();
        return;
      }
      pending.clear();
    }
    const uint32_t zero = 0;
    if (!WriteAllFd(input_fd_, &zero, sizeof(zero)))
      std::cerr << "[diar] nemotron flush frame write failed\n";
    close(input_fd_);
    input_fd_ = -1;
  }

  // Consumes incremental speaker-activity frames until the worker finishes.
  void ReaderLoop() {
    char type = 0;
    std::string payload;
    while (ReadFrame(response_fd_, &type, &payload)) {
      if (type == 'S') {
        if (payload.size() < 2 * sizeof(uint32_t)) break;
        uint32_t start_frame = 0, n_frames = 0;
        std::memcpy(&start_frame, payload.data(), sizeof(start_frame));
        std::memcpy(&n_frames, payload.data() + sizeof(start_frame),
                    sizeof(n_frames));
        const std::size_t expected = 2 * sizeof(uint32_t) +
            static_cast<std::size_t>(n_frames) * kSpeakers * sizeof(float);
        if (payload.size() != expected) break;  // protocol corruption
        ConsumeRows(start_frame, n_frames,
                    reinterpret_cast<const float*>(payload.data() +
                                                   2 * sizeof(uint32_t)));
      } else if (type == 'D') {
        std::cerr << "[diar] nemotron worker " << payload << "\n";
        break;
      } else if (type == 'E') {
        std::cerr << "[diar] nemotron worker error: " << payload << "\n";
        break;
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_ = true;
      CloseOpenPartsNoLock(frames_received_);
    }
    ready_.notify_all();
  }

  // Folds one in-order activity block into the timeline.
  void ConsumeRows(uint32_t start_frame, uint32_t n_frames, const float* rows) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t i = 0; i < n_frames; ++i) {
      const std::size_t frame = start_frame + i;
      if (frame != frames_received_) {
        // The protocol guarantees contiguous in-order frames; a gap means
        // the stream is corrupt, so freeze the timeline at the frontier.
        std::cerr << "[diar] nemotron frame gap: expected "
                  << frames_received_ << " got " << frame << "\n";
        CloseOpenPartsNoLock(frames_received_);
        finished_ = true;
        ready_.notify_all();
        return;
      }
      const float* probs = rows + static_cast<std::size_t>(i) * kSpeakers;
      for (int s = 0; s < kSpeakers; ++s) {
        SpeakerRun& run = runs_[static_cast<std::size_t>(s)];
        const float p = probs[s];
        // Hysteresis identical to the NeMo post-processing default: above
        // `onset` forces the speaker on, below `offset` forces it off, in
        // between keeps the previous state.
        if (p > onset_ || p < offset_) run.on = p > onset_;
        if (run.on) {
          if (run.start == kNoFrame) {
            run.start = frame;
            open_part_[static_cast<std::size_t>(s)] = PushPartNoLock(s, frame);
          }
          // An ongoing run is exposed as a provisional part that grows with
          // the frontier.  Without this the timeline's leading edge has no
          // entry yet, and MapTimelineToTrimmed would inherit the label of
          // the *previous* speaker for the newest audio (mislabeling the
          // opening of a new speaker's turn, or dropping it entirely when
          // the timeline was empty there).
          timeline_[open_part_[static_cast<std::size_t>(s)]].end =
              (frame + 1) * kFrameSamples;
        } else if (run.start != kNoFrame) {
          run.start = kNoFrame;
          open_part_[static_cast<std::size_t>(s)] = kNoFrame;
        }
      }
      frames_received_ = frame + 1;
    }
    committed_ = frames_received_ * kFrameSamples;
    // Live-demo event: same shape as the pyannote session "diar" events
    // (core = [prev, committed] seconds, parts = newly closed speaker runs)
    // so tools/live-demo renders the streaming timeline unchanged.  Parts
    // still being extended by an open run are held back until they close.
    {
      std::string body = "\"core\":[" + EventLog::Sec(event_prev_committed_, kRateOut) +
                         "," + EventLog::Sec(committed_, kRateOut) + "],\"parts\":[";
      bool first = true;
      while (event_parts_sent_ < timeline_.size() &&
             !IsOpenPartNoLock(event_parts_sent_)) {
        if (!first) body += ",";
        first = false;
        const Segment::Part& part = timeline_[event_parts_sent_];
        body += "[" + EventLog::Sec(part.begin, kRateOut) + "," +
                EventLog::Sec(part.end, kRateOut) + "," +
                EventLog::Quote(part.speaker_id) + "]";
        ++event_parts_sent_;
      }
      event_prev_committed_ = committed_;
      EventLog::Get().Write("diar", body + "]");
    }
    ready_.notify_all();
  }

  void CloseOpenPartsNoLock(std::size_t /*end_frame*/) {
    // Open runs are already exposed in the timeline with their final end
    // (extended per frame); only the run bookkeeping needs resetting.
    for (int s = 0; s < kSpeakers; ++s) {
      SpeakerRun& run = runs_[static_cast<std::size_t>(s)];
      run.start = kNoFrame;
      run.on = false;
      open_part_[static_cast<std::size_t>(s)] = kNoFrame;
    }
  }

  bool IsOpenPartNoLock(std::size_t index) const {
    for (std::size_t s = 0; s < kSpeakers; ++s)
      if (open_part_[s] == index) return true;
    return false;
  }

  // Opens a provisional part for a run starting at begin_frame, merging with
  // the speaker's previous part when the silence gap is short; returns the
  // timeline index.  The caller extends .end while the run stays on.
  std::size_t PushPartNoLock(int speaker, std::size_t begin_frame) {
    const std::size_t begin = begin_frame * kFrameSamples;
    const std::size_t spk = static_cast<std::size_t>(speaker);
    const std::size_t last = last_part_[spk];
    // Merge with the speaker's previous run when the silence gap between the
    // two is short; per-speaker last-part index keeps this O(1) and timeline_
    // never erases, so indices stay valid.
    if (last != kNoFrame && last < timeline_.size() &&
        timeline_[last].end + kMergeGapFrames * kFrameSamples >= begin) {
      timeline_[last].end = std::max(timeline_[last].end, begin);
      return last;
    }
    Segment::Part part;
    part.begin = begin;
    part.end = begin;
    part.speaker_id = "speaker_" + std::to_string(speaker);
    last_part_[spk] = timeline_.size();
    timeline_.push_back(std::move(part));
    return timeline_.size() - 1;
  }

  struct SpeakerRun {
    bool on = false;
    std::size_t start = kNoFrame;
  };

  apm_example::AppConfig::Diarization config_;
  float onset_, offset_;
  int input_fd_ = -1;
  int response_fd_ = -1;
  pid_t pid_ = -1;
  bool active_ = false;
  bool closed_ = false;
  bool finished_ = false;

  mutable std::mutex mutex_;                 // timeline state
  std::condition_variable ready_;
  std::vector<Segment::Part> timeline_;      // absolute 16 kHz sample parts
  std::size_t committed_ = 0;                // samples with final labels
  std::size_t frames_received_ = 0;          // absolute 10 ms frame frontier
  std::size_t event_parts_sent_ = 0;          // live-demo event bookkeeping
  std::size_t event_prev_committed_ = 0;
  std::vector<SpeakerRun> runs_{static_cast<std::size_t>(kSpeakers)};
  std::vector<std::size_t> last_part_{static_cast<std::size_t>(kSpeakers),
                                      kNoFrame};
  std::vector<std::size_t> open_part_{static_cast<std::size_t>(kSpeakers),
                                      kNoFrame};

  std::mutex write_mutex_;                   // PCM feeding state
  std::condition_variable write_ready_;
  std::vector<int16_t> pcm_buffer_;
  std::thread writer_;
  std::thread reader_;
};

class ResultWorker {
 public:
  ResultWorker(const apm_example::AppConfig& c,
               Npu2CellArbiter* npu2_arbiter)
      : c_(c), npu2_arbiter_(npu2_arbiter),
        text_owned_by_track_(c.asr.mode == "streaming"),
        asr_(c),
        diarrer_(c.diarization, npu2_arbiter),
        jobs_(0) {  // unbounded; segments arrive at VAD pace
    fs::create_directories(c_.output.temp_dir);
    fs::create_directories(c_.output.mp3_dir);
    streaming_text_file_ = DeriveStreamingTextFile(c_.output.text_file);
    if (!c_.output.text_file.empty())
      EventLog::Get().Open(
          (fs::path(c_.output.text_file).parent_path() / "events.jsonl").string());
    const std::string configured = FindExecutable(c_.output.mp3_encoder);
    if (!configured.empty() && fs::path(configured).filename() == "lame") {
      encoder_ = configured;
      encoder_kind_ = EncoderKind::kLame;
    } else {
      const std::string lame = FindExecutable("lame");
      if (!lame.empty()) {
        encoder_ = lame;
        encoder_kind_ = EncoderKind::kLame;
      } else if (!configured.empty() && FfmpegHasMp3Lame(configured)) {
        encoder_ = configured;
        encoder_kind_ = EncoderKind::kFfmpeg;
      } else {
        const std::string ffmpeg = FindExecutable("ffmpeg");
        if (!ffmpeg.empty() && FfmpegHasMp3Lame(ffmpeg)) {
          encoder_ = ffmpeg;
          encoder_kind_ = EncoderKind::kFfmpeg;
        }
      }
    }
    if (encoder_.empty()) {
      std::cerr << "[mp3] no usable encoder found; WAV files will be retained\n";
    } else {
      std::cerr << "[mp3] encoder=" << encoder_ << "\n";
    }
    if (!asr_.Start()) {
      std::cerr << "[asr] failed to start SenseVoice; transcription will be skipped\n";
    }
    thread_ = std::thread([this] { Loop(); });
  }
  ~ResultWorker() { Close(); }
  void Submit(Segment s) { jobs_.Push(std::move(s)); }
  // Streaming slicer shares the resident SenseVoice handle (its thread runs
  // ASR between this worker's jobs; both acquire the NPU2 arbiter).
  SenseVoiceRunner& Asr() { return asr_; }
  // Speaker timeline for the streaming ASR track (null when the diarization
  // engine does not expose one).
  SpeakerLookup* speaker_source() { return nemotron_ ? nemotron_.get() : nullptr; }
  // Called once the pipeline's own AX Engine (BIG_LITTLE) is initialized.
  void StartDiarization() {
    if (!c_.diarization.enabled) return;
    if (c_.diarization.engine == "nemotron") {
      // True streaming: the end-to-end Sortformer worker owns the whole
      // speaker timeline; pyannote paths stay untouched below.
      nemotron_ = std::make_unique<NemotronStreamDiarizer>(c_.diarization);
      nemotron_->Start();
      return;
    }
    bool streaming_session = false;
    if (c_.diarization.mode == "session" || c_.diarization.mode == "online") {
      const bool file_input = c_.runtime.input == "file";
      if (file_input && c_.runtime.file_input_mode != "realtime") {
        // Batch file input: diarize the whole source once and reuse the
        // timeline for every segment.
        const int rate = ReadWavSampleRate(c_.runtime.input_wav);
        if (rate <= 0) {
          std::cerr << "[diar] cannot read input WAV rate; "
                       "using per-segment mode\n";
        } else {
          std::cerr << "[diar] session diarization on " << c_.runtime.input_wav
                    << " (rate=" << rate << ")\n";
          session_parts_ =
              RunSessionDiarization(c_.diarization, c_.runtime.input_wav, rate);
          session_ready_ = !session_parts_.empty();
        }
      } else {
        streaming_session = true;
      }
    }
    if (!diarrer_.Start())
      throw std::runtime_error(
          "Speaker diarization is enabled but its worker is unavailable");
    if (streaming_session) {
      session_ = std::make_unique<SessionDiarizer>(
          c_.diarization.session_chunk_sec, c_.diarization.session_step_sec,
          c_.diarization.session_first_window_sec);
      session_->Start(&diarrer_, c_.output.temp_dir);
      std::cerr << "[diar] streaming session diarization mode="
                << c_.diarization.mode << " chunk="
                << c_.diarization.session_chunk_sec << "s step="
                << (c_.diarization.session_step_sec > 0.0f
                        ? c_.diarization.session_step_sec
                        : c_.diarization.session_chunk_sec)
                << "s first_win="
                << (c_.diarization.session_first_window_sec > 0.0f
                        ? c_.diarization.session_first_window_sec
                        : c_.diarization.session_chunk_sec)
                << "s max_wait=" << c_.diarization.session_max_wait_sec
                << "s";
      if (c_.diarization.incremental)
        std::cerr << " incremental(delta_new="
                  << c_.diarization.incremental_delta_new
                  << ",delta_min=" << c_.diarization.incremental_delta_min
                  << ",rho_update=" << c_.diarization.incremental_rho_update
                  << (c_.diarization.incremental_plda ? ",plda" : ",raw")
                  << ")";
      std::cerr << "\n";
    }
  }

  // Capture thread: feed every 16 kHz hop (voiced or silent) to the streaming
  // session diarizer when it is active.
  void AppendSessionSamples(const std::vector<int16_t>& hop) {
    if (nemotron_) nemotron_->Append(hop);
    else if (session_) session_->Append(hop);
  }
  void Close(bool abort = false) {
    if (closed_) return;
    closed_ = true;
    if (session_) {
      if (!abort) {
        session_->Stop();
        session_->ProcessRemaining();
      }
      // Final verdict after the last commit: unblocks any remaining timeline
      // waiters (they then fall back per-segment).  Kept alive until the
      // result thread joins because that thread owns pending waits.
      session_->Finish();
    }
    if (nemotron_) {
      if (!abort) {
        // Flush the worker's tail (zero-length frame) and drain its final
        // labels; the reader sets finished_ when the worker's 'D' arrives.
        nemotron_->Stop();
      }
      nemotron_->Finish();
    }
    jobs_.Close();
    if (thread_.joinable()) thread_.join();
    EventLog::Get().Write("end", "\"abort\":" + std::string(abort ? "true" : "false"));
    session_.reset();
    nemotron_.reset();
    diarrer_.Close();
    asr_.Close();
  }
 private:
  void Loop() {
    Segment s;
    while (jobs_.Pop(&s)) {
      // Streaming fast path: text-only preview slice, transcribed immediately
      // so the opening words land at ~preview_seconds.  No speaker label (that
      // still arrives with the normal segment ~10 s later); the preview is
      // written to the streaming transcript and skips diarization + MP3.
      if (s.is_preview) {
        if (text_owned_by_track_) continue;  // streaming track owns fast text
        const std::string stamp =
            TimeStamp(s.wall_start) + "_" + TimeStamp(s.wall_end);
        const fs::path preview_wav =
            fs::path(c_.output.temp_dir) / (stamp + ".preview.wav");
        if (WriteWav(preview_wav, s.samples, kRateOut)) {
          auto cell = npu2_arbiter_->Acquire();
          const std::string text = asr_.Transcribe(preview_wav);
          if (!streaming_text_file_.empty()) {
            std::ofstream stxt(streaming_text_file_, std::ios::app);
            stxt << "pending@[" << EventLog::Sec(s.start, kRateOut) << ","
                 << EventLog::Sec(s.end, kRateOut) << "] " << text << "\n";
          }
          std::cerr << "[asr] preview text: " << text << "\n";
          EventLog::Get().Write("preview",
              "\"b\":" + EventLog::Sec(s.start, kRateOut) +
              ",\"e\":" + EventLog::Sec(s.end, kRateOut) +
              ",\"text\":" + EventLog::Quote(text));
          RemoveWav(preview_wav);
        }
        continue;
      }
      const std::string stamp = TimeStamp(s.wall_start) + "_" + TimeStamp(s.wall_end);
      const fs::path wav = fs::path(c_.output.temp_dir) / (stamp + ".wav");
      if (!WriteWav(wav, s.samples, kRateOut)) {
        std::cerr << "[save] WAV failed: " << wav << "\n";
        RemoveWav(wav);
        continue;
      }
      std::cerr << "[save] WAV saved: " << wav << " (" << s.samples.size() / static_cast<double>(kRateOut) << "s)\n";
      if (!WriteJoinPoints(wav, s.join_points, kRateOut))
        std::cerr << "[save] join-point file failed: " << wav << ".joins.txt\n";
      if (!WriteRuns(wav, s.runs))
        std::cerr << "[save] run-map file failed: " << wav << ".runs.txt\n";
      if (!text_owned_by_track_) {
      fs::create_directories(fs::path(c_.output.text_file).parent_path());
      std::ofstream txt(c_.output.text_file, std::ios::app);
      bool streamed = false;
      if (diarrer_.active() || session_ready_ || session_ || nemotron_) {
        bool labeled = false;
        // Plan B: release every speaker part to ASR the moment the timeline
        // covers it instead of waiting for the whole segment (see
        // StreamSessionParts).
        if ((session_ || nemotron_) && !s.runs.empty())
          streamed = StreamSessionParts(s, stamp, wav, txt);
        labeled = streamed;
        // Whole-file session pre-run (batch file inputs).
        if (!labeled && session_ready_ && !s.runs.empty()) {
          s.parts = MapTimelineToTrimmed(session_parts_, 0, s.runs,
                                         s.samples.size());
          labeled = !s.parts.empty();
        }
        // Per-segment raw window fallback (silences preserved).
        if (!labeled && !s.raw_samples.empty()) {
          const fs::path raw_wav = wav.string() + ".raw.wav";
          if (WriteWav(raw_wav, s.raw_samples, kRateOut)) {
            const auto raw_parts =
                diarrer_.Label(raw_wav.string(), s.raw_samples.size());
            std::error_code ec;
            fs::remove(raw_wav, ec);
            s.parts = MapTimelineToTrimmed(raw_parts, s.raw_start, s.runs,
                                           s.samples.size());
            labeled = !s.parts.empty();
          } else {
            std::cerr << "[diar] raw window WAV failed: " << raw_wav << "\n";
          }
        }
        if (!labeled && diarrer_.active())
          s.parts = diarrer_.Label(wav.string(), s.samples.size());
        if (!s.parts.empty()) WriteParts(wav, s.parts);
        if (s.parts.empty())
          std::cerr << "[diar] no speaker parts for " << wav
                    << "; whole-segment ASR fallback\n";
      }
      if (s.parts.empty()) {
        auto cell = npu2_arbiter_->Acquire();
        const std::string text = asr_.Transcribe(wav);
        txt << "[" << TimeStamp(s.wall_start) << " --> " << TimeStamp(s.wall_end) << "] " << text << "\n";
        EventLog::Get().Write("final",
            "\"b\":" + EventLog::Sec(s.start, kRateOut) +
            ",\"e\":" + EventLog::Sec(s.end, kRateOut) +
            ",\"spk\":\"\",\"text\":" + EventLog::Quote(text));
      } else if (!streamed) {
        // Speaker parts are ASR-only slices.  The original VAD WAV below is
        // still the sole input to MP3 encoding, so speaker labeling cannot
        // change the established audio archival behavior.  Each slice reaches
        // back by kAsrPadSamples so a boundary that fell inside a word onset
        // does not hand the ASR a headless word; see kAsrPadSamples.  The reach
        // deliberately overlaps the previous part's tail — the clipped onset
        // samples live there — and 0.2 s is short enough not to pull the
        // previous speaker's words into the transcript.
        for (std::size_t i = 0; i < s.parts.size(); ++i)
          TranscribePart(s, stamp, i, s.parts[i].begin, s.parts[i].end,
                         s.parts[i].speaker_id, txt);
      }
      if (streamed)
        std::cerr << "[save] transcript saved (streamed, " << s.parts.size()
                  << " parts)\n";
      }  // !text_owned_by_track_ (streaming ASR track owns the transcript)
      if (encoder_.empty()) {
        std::cerr << "[save] transcript saved; MP3 skipped because no encoder is available\n";
        RemoveWav(wav);
        continue;
      }
      const fs::path mp3 = fs::path(c_.output.mp3_dir) / (stamp + ".mp3");
      std::string cmd;
      if (encoder_kind_ == EncoderKind::kLame) {
        cmd = ShellQuote(encoder_) + " --silent " + ShellQuote(wav.string()) + " " + ShellQuote(mp3.string());
      } else {
        cmd = ShellQuote(encoder_) + " -y -loglevel error -i " + ShellQuote(wav.string()) + " -codec:a libmp3lame " + ShellQuote(mp3.string());
      }
      const int status = std::system(cmd.c_str());
      if (status != 0) {
        std::cerr << "[mp3] encoder failed status=" << status << ", WAV retained: " << wav << "\n";
        std::error_code ec;
        fs::remove(mp3, ec);
      } else if (fs::is_regular_file(mp3)) {
        std::cerr << "[save] MP3 saved: " << mp3 << "\n";
        RemoveWav(wav);
      } else {
        std::cerr << "[mp3] encoder returned success but file is missing: " << mp3 << "\n";
      }
    }
  }
  // ASR on one speaker part [begin, end) of the concatenated segment.  The
  // slice reaches back by kAsrPadSamples (see there); `index` only names the
  // temporary WAV.
  void TranscribePart(const Segment& s, const std::string& stamp,
                      std::size_t index, std::size_t part_begin,
                      std::size_t part_end, const std::string& speaker,
                      std::ofstream& txt) {
    const std::size_t begin = std::min(part_begin, s.samples.size());
    const std::size_t end = std::min(std::max(part_end, begin), s.samples.size());
    if (end <= begin) return;
    const std::size_t slice_begin =
        begin > kAsrPadSamples ? begin - kAsrPadSamples : 0;
    std::vector<int16_t> pcm(s.samples.begin() + slice_begin, s.samples.begin() + end);
    const auto part_start = s.wall_start + std::chrono::milliseconds(static_cast<int64_t>(begin) * 1000 / kRateOut);
    const fs::path part_wav = fs::path(c_.output.temp_dir) / (stamp + "." + std::to_string(index) + ".wav");
    if (!WriteWav(part_wav, pcm, kRateOut)) return;
    std::string text;
    {
      auto cell = npu2_arbiter_->Acquire();
      text = asr_.Transcribe(part_wav);
    }
    txt << speaker << "@" << SpeakerTimeStamp(part_start) << " " << text << "\n";
    txt.flush();
    EventLog::Get().Write("final",
        "\"b\":" + EventLog::Sec(ConcatToSource(s.runs, begin, false), kRateOut) +
        ",\"e\":" + EventLog::Sec(ConcatToSource(s.runs, end, true), kRateOut) +
        ",\"spk\":" + EventLog::Quote(speaker) +
        ",\"text\":" + EventLog::Quote(text));
    RemoveWav(part_wav);
  }

  // Plan B (docs/realtime-pipeline-timing.md §6): release each speaker part to
  // ASR as soon as the session timeline covers it instead of waiting for the
  // whole segment.  A part is released once it is closed, i.e. it ends at
  // least kMinPartSamples before the covered prefix so later commits cannot
  // re-merge or absorb it; the rest is released once the segment is fully
  // covered, using the same mapping as before.  Returns false when nothing
  // was released (caller falls back to the per-segment paths).
  bool StreamSessionParts(Segment& s, const std::string& stamp,
                          const fs::path& wav, std::ofstream& txt) {
    const std::size_t n = s.samples.size();
    const std::size_t segment_end =
        s.runs.back().second + (n - s.runs.back().first);
    const auto deadline = Clock::now() + std::chrono::milliseconds(
        static_cast<std::int64_t>(c_.diarization.session_max_wait_sec * 1000.0f));
    std::vector<Segment::Part> released;
    std::size_t done = 0;  // concat offset already transcribed
    auto release = [&](std::size_t b, std::size_t e, const std::string& spk) {
      b = std::max(b, done);
      if (e <= b) return;
      TranscribePart(s, stamp, released.size(), b, e, spk, txt);
      released.push_back({b, e, spk});
      done = e;
    };
    while (true) {
      const std::size_t frontier = StreamCommitted();
      const auto parts = MapTimelineToTrimmed(StreamTimeline(), 0, s.runs, n);
      if (frontier >= segment_end) {
        for (const auto& p : parts) release(p.begin, p.end, p.speaker_id);
        break;
      }
      const std::size_t cut = ConcatCoveredBySource(s.runs, frontier, n);
      for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        if (parts[i].end + kPartReleaseMargin > cut) break;
        release(parts[i].begin, parts[i].end, parts[i].speaker_id);
      }
      const auto now = Clock::now();
      if (StreamFinished() || now >= deadline ||
          c_.diarization.session_max_wait_sec <= 0.0f) {
        std::cerr << "[diar] session timeline wait timed out for " << wav
                  << "; released " << released.size() << " parts early\n";
        // Partial coverage: finish with the best mapping available.  Even
        // with nothing released incrementally the timeline's parts are the
        // authoritative labels; falling back to whole-segment diarization
        // here would discard correct per-part speakers (e.g. a new speaker
        // whose parts merged into a single unreleased part).  The caller's
        // per-segment fallback still applies when no part maps at all.
        for (const auto& p : parts) release(p.begin, p.end, p.speaker_id);
        break;
      }
      StreamWaitCovered(frontier + 1,
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
    }
    s.parts = std::move(released);
    return !s.parts.empty();
  }

  // Streaming timeline source dispatch: the pyannote session diarizer and
  // the nemotron worker expose the same interface, so the release logic
  // above is shared verbatim.
  std::size_t StreamCommitted() const {
    return nemotron_ ? nemotron_->Committed() : session_->Committed();
  }
  std::vector<Segment::Part> StreamTimeline() const {
    return nemotron_ ? nemotron_->Timeline() : session_->Timeline();
  }
  bool StreamFinished() const {
    return nemotron_ ? nemotron_->Finished() : session_->Finished();
  }
  bool StreamWaitCovered(std::size_t end, std::chrono::milliseconds timeout) {
    return nemotron_ ? nemotron_->WaitCovered(end, timeout)
                     : session_->WaitCovered(end, timeout);
  }

  static void RemoveWav(const fs::path& wav) {
    std::error_code ec;
    if (fs::remove(wav, ec)) {
      std::cerr << "[save] temporary WAV removed: " << wav << "\n";
    } else if (ec) {
      std::cerr << "[save] failed to remove temporary WAV " << wav << ": "
                << ec.message() << "\n";
    }
  }
  enum class EncoderKind { kNone, kLame, kFfmpeg };
  apm_example::AppConfig c_;
  Npu2CellArbiter* npu2_arbiter_;
  SenseVoiceRunner asr_;
  DiarWorker diarrer_;
  BlockingQueue<Segment> jobs_;
  std::thread thread_;
  std::vector<Segment::Part> session_parts_;
  std::unique_ptr<SessionDiarizer> session_;
  std::unique_ptr<NemotronStreamDiarizer> nemotron_;
  bool session_ready_ = false;
  const bool text_owned_by_track_ = false;
  std::string streaming_text_file_;
  std::string encoder_;
  EncoderKind encoder_kind_ = EncoderKind::kNone;
  bool closed_ = false;
};

struct Child { pid_t pid = -1; int in = -1, out = -1; };

bool ReadAll(int fd, void* data, size_t n) { char* p = static_cast<char*>(data); while (n && !g_stop.load()) { const ssize_t r = read(fd, p, n); if (r <= 0) return false; p += r; n -= r; } return n == 0; }
bool WriteAll(int fd, const void* data, size_t n) { const char* p = static_cast<const char*>(data); while (n && !g_stop.load()) { const ssize_t w = write(fd, p, n); if (w <= 0) return false; p += w; n -= w; } return n == 0; }

Child StartApm(const apm_example::AppConfig& c, const std::string& config_path) {
  int in[2], out[2]; if (pipe(in) || pipe(out)) throw std::runtime_error("pipe failed"); Child child; child.pid = fork(); if (child.pid == 0) { dup2(in[0], STDIN_FILENO); dup2(out[1], STDOUT_FILENO); close(in[1]); close(out[0]); if (c.fastenhance.monitor_enabled) execl(c.runtime.apm_binary.c_str(), c.runtime.apm_binary.c_str(), "--stdout", "--monitor-stdin", config_path.c_str(), static_cast<char*>(nullptr)); else execl(c.runtime.apm_binary.c_str(), c.runtime.apm_binary.c_str(), "--stdout", config_path.c_str(), static_cast<char*>(nullptr)); _exit(127); } close(in[0]); close(out[1]); child.in = in[1]; child.out = out[0]; if (child.pid < 0) throw std::runtime_error("fork failed"); return child;
}

int StopChild(Child* child) { int status = 0; if (child->in >= 0) { close(child->in); child->in = -1; } if (child->out >= 0) { close(child->out); child->out = -1; } if (child->pid > 0) { kill(child->pid, SIGTERM); if (waitpid(child->pid, &status, 0) < 0) status = -1; child->pid = -1; } return status; }

class Pipeline {
 public:
  Pipeline(apm_example::AppConfig c, std::string config_path) : config_(std::move(c)), config_path_(std::move(config_path)), arbiter_(), npu2_arbiter_(config_.runtime.npu_lock_file), results_(config_, &npu2_arbiter_), slicer_(config_, &results_.Asr(), &npu2_arbiter_), enhancer_(config_, &arbiter_), vad_(config_, &arbiter_), accumulator_(config_, [this](Segment s) { results_.Submit(std::move(s)); }), vad_jobs_(std::max<size_t>(4, config_.vad.input_buffer_ms / std::max(1, config_.vad.hop_ms))), monitor_queue_(monitor_queue_capacity(config_)) { results_.StartDiarization();
    if (config_.asr.mode == "streaming") {
      strack_ = std::make_unique<StreamingAsrTrack>(config_, &results_.Asr(), &npu2_arbiter_);
      strack_->SetSpeakerSource(results_.speaker_source());
      strack_->Start();
      std::cerr << "[asr-stream] streaming ASR track started (backend="
                << config_.asr.streaming.backend << ")\n";
    } else {
      slicer_.Start();
    }
    vad_thread_ = std::thread([this] { VadLoop(); }); }
  void Run() { child_ = StartApm(config_, config_path_); std::thread monitor; if (config_.fastenhance.monitor_enabled) monitor = std::thread([this] { MonitorLoop(); }); std::vector<int16_t> raw(kRateIn / 100); while (!g_stop.load() && ReadAll(child_.out, raw.data(), raw.size() * sizeof(int16_t))) Process(raw); monitor_queue_.Close(); const int child_status = StopChild(&child_); if (monitor.joinable()) monitor.join(); const bool interrupted = g_stop.load(); Close(interrupted); if (!interrupted && (child_status == -1 || !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)) throw std::runtime_error("APM input process exited with an error"); }
  void Close(bool abort = false) { if (closed_) return; closed_ = true; if (!abort) { if (!enhance_remainder_.empty()) { const size_t n = enhance_remainder_.size(); enhance_remainder_.resize(kEnhanceHop, 0); std::vector<int16_t> tail = enhance_remainder_; enhance_remainder_.clear(); Process(tail); (void)n; } if (!vad_remainder_.empty()) { vad_remainder_.resize(vad_.Hop(), 0); if (BatchInput()) vad_jobs_.Push(vad_remainder_); else vad_jobs_.TryPush(vad_remainder_); vad_remainder_.clear(); } } vad_jobs_.Close(); if (vad_thread_.joinable()) vad_thread_.join(); if (!abort) accumulator_.Flush(); if (strack_) strack_->FlushAndClose(); else slicer_.FlushAndClose(); results_.Close(abort); enhancer_.Report(); vad_.Report(); }
 private:
  void Process(const std::vector<int16_t>& input) { EventLog::Get().MarkStart(); { auto diar_down = diar_decimator_.Process(input); diar_remainder_.insert(diar_remainder_.end(), diar_down.begin(), diar_down.end()); while (diar_remainder_.size() >= static_cast<size_t>(vad_.Hop())) { std::vector<int16_t> d(diar_remainder_.begin(), diar_remainder_.begin() + vad_.Hop()); diar_remainder_.erase(diar_remainder_.begin(), diar_remainder_.begin() + vad_.Hop()); results_.AppendSessionSamples(d); } } enhance_remainder_.insert(enhance_remainder_.end(), input.begin(), input.end()); while (enhance_remainder_.size() >= kEnhanceHop) { std::array<int16_t, kEnhanceHop> hop{}; std::copy_n(enhance_remainder_.begin(), kEnhanceHop, hop.begin()); enhance_remainder_.erase(enhance_remainder_.begin(), enhance_remainder_.begin() + kEnhanceHop); auto output = enhancer_.Process(hop); std::vector<int16_t> rendered(output.begin(), output.end()); if (!monitor_queue_.TryPush(rendered)) { monitor_queue_.TryDropOldest(); monitor_queue_.TryPush(std::move(rendered)); } std::vector<int16_t> enhanced(output.begin(), output.end()); auto down = decimator_.Process(enhanced); vad_remainder_.insert(vad_remainder_.end(), down.begin(), down.end()); while (vad_remainder_.size() >= static_cast<size_t>(vad_.Hop())) { std::vector<int16_t> v(vad_remainder_.begin(), vad_remainder_.begin() + vad_.Hop()); vad_remainder_.erase(vad_remainder_.begin(), vad_remainder_.begin() + vad_.Hop()); if (BatchInput()) vad_jobs_.Push(std::move(v)); else if (!vad_jobs_.TryPush(v)) { vad_jobs_.TryDropOldest(); vad_jobs_.TryPush(std::move(v)); } } } }
  void VadLoop() { std::vector<int16_t> hop; while (vad_jobs_.Pop(&hop)) { bool voiced = true; try { voiced = vad_.Process(hop); } catch (const std::exception& e) { std::cerr << "[ten-vad] " << e.what() << "\n"; } accumulator_.Push(hop, voiced); if (strack_) strack_->Push(hop, voiced); else slicer_.Push(hop, voiced); } }
  void MonitorLoop() { std::vector<int16_t> hop; while (!g_stop.load() && monitor_queue_.Pop(&hop)) { if (!WriteAll(child_.in, hop.data(), hop.size() * sizeof(int16_t))) break; } }
  bool BatchInput() const { return config_.runtime.input == "file" && config_.runtime.file_input_mode == "batch"; }
  static size_t monitor_queue_capacity(const apm_example::AppConfig& c) { return std::max<size_t>(2, static_cast<size_t>(c.buffers.application_buffer_ms) * kRateIn / 1000 / kEnhanceHop); }
  apm_example::AppConfig config_; std::string config_path_; NpuArbiter arbiter_; Npu2CellArbiter npu2_arbiter_; ResultWorker results_; StreamSlicer slicer_; std::unique_ptr<StreamingAsrTrack> strack_; FastEnhance enhancer_; TenVad vad_; Decimator decimator_; Decimator diar_decimator_; Accumulator accumulator_; BlockingQueue<std::vector<int16_t>> vad_jobs_; BlockingQueue<std::vector<int16_t>> monitor_queue_; std::thread vad_thread_; std::vector<int16_t> enhance_remainder_, vad_remainder_, diar_remainder_; Child child_; bool closed_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  std::string config_path = "config/sdk-config.yaml";
  for (int i = 1; i < argc; ++i) { std::string arg = argv[i]; if (arg == "--config" && i + 1 < argc) config_path = argv[++i]; else { std::cerr << "Usage: " << argv[0] << " [--config PATH]\n"; return 2; } }
  apm_example::AppConfig config; std::string error; if (!apm_example::LoadAppConfig(config_path, &config, &error)) { std::cerr << "Configuration error: " << error << "\n"; return 1; }
  if (config.runtime.input == "file") { std::error_code ec; if (!fs::is_regular_file(config.runtime.input_wav, ec)) { std::cerr << "Configuration error: input WAV does not exist or is not a regular file: " << config.runtime.input_wav << "\n"; return 1; } }
  std::signal(SIGINT, HandleSignal); std::signal(SIGTERM, HandleSignal);
  std::signal(SIGPIPE, SIG_IGN);
  try { Pipeline pipeline(std::move(config), config_path); pipeline.Run(); } catch (const std::exception& e) { std::cerr << "audio-pipeline: " << e.what() << "\n"; return 1; }
  return 0;
}

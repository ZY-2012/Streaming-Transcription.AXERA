#include "apm-config.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace apm_example {
namespace {

std::string Trim(const std::string& value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](char c) {
    return std::isspace(static_cast<unsigned char>(c));
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](char c) {
                      return std::isspace(static_cast<unsigned char>(c));
                    }).base();
  return first < last ? std::string(first, last) : std::string();
}

std::string StripComment(const std::string& line) {
  bool single_quote = false;
  bool double_quote = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '\'' && !double_quote) {
      single_quote = !single_quote;
    } else if (line[i] == '"' && !single_quote &&
               (i == 0 || line[i - 1] != '\\')) {
      double_quote = !double_quote;
    } else if (line[i] == '#' && !single_quote && !double_quote) {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string Unquote(std::string value) {
  value = Trim(value);
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    const std::string quoted = value.substr(1, value.size() - 2);
    value.clear();
    for (std::size_t i = 0; i < quoted.size(); ++i) {
      if (quoted[i] != '\\' || i + 1 >= quoted.size()) {
        value.push_back(quoted[i]);
        continue;
      }
      const char escaped = quoted[++i];
      if (escaped == 'n') value.push_back('\n');
      else if (escaped == 'r') value.push_back('\r');
      else if (escaped == 't') value.push_back('\t');
      else if (escaped == '"') value.push_back('"');
      else if (escaped == '\\') value.push_back('\\');
      else { value.push_back('\\'); value.push_back(escaped); }
    }
  } else if (value.size() >= 2 && value.front() == '\'' &&
             value.back() == '\'') {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

bool ParseYaml(const std::string& path,
               std::map<std::string, std::string>* values,
               std::string* error) {
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open config file: " + path;
    return false;
  }

  struct Parent {
    int indent;
    std::string key;
  };
  std::vector<Parent> parents;
  std::string raw;
  int line_number = 0;
  while (std::getline(input, raw)) {
    ++line_number;
    const std::string line = StripComment(raw);
    if (Trim(line).empty()) {
      continue;
    }
    const size_t first = line.find_first_not_of(' ');
    if (first == std::string::npos || line.find('\t', 0) != std::string::npos) {
      *error = "tabs are not allowed in YAML indentation at line " +
               std::to_string(line_number);
      return false;
    }
    const int indent = static_cast<int>(first);
    const std::string content = line.substr(first);
    const size_t colon = content.find(':');
    if (colon == std::string::npos) {
      *error = "expected key: value at line " + std::to_string(line_number);
      return false;
    }
    const std::string key = Trim(content.substr(0, colon));
    const std::string value = Trim(content.substr(colon + 1));
    if (key.empty()) {
      *error = "empty YAML key at line " + std::to_string(line_number);
      return false;
    }
    while (!parents.empty() && indent <= parents.back().indent) {
      parents.pop_back();
    }
    std::string full_key;
    for (const auto& parent : parents) {
      if (!full_key.empty()) full_key += '.';
      full_key += parent.key;
    }
    if (!full_key.empty()) full_key += '.';
    full_key += key;
    if (value.empty()) {
      parents.push_back({indent, key});
    } else {
      if (!values->emplace(full_key, Unquote(value)).second) {
        *error = "duplicate YAML key: " + full_key;
        return false;
      }
    }
  }
  return true;
}

bool GetString(const std::map<std::string, std::string>& values,
               const std::string& key,
               std::string* output,
               std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  if (it->second.empty()) {
    *error = key + " must not be empty";
    return false;
  }
  *output = it->second;
  return true;
}

bool GetStringAllowEmpty(const std::map<std::string, std::string>& values,
                         const std::string& key,
                         std::string* output,
                         std::string* /*error*/) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  *output = it->second;
  return true;
}

bool GetBool(const std::map<std::string, std::string>& values,
             const std::string& key,
             bool* output,
             std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  std::string value = it->second;
  std::transform(value.begin(), value.end(), value.begin(), [](char c) {
    return std::tolower(static_cast<unsigned char>(c));
  });
  if (value == "true") {
    *output = true;
  } else if (value == "false") {
    *output = false;
  } else {
    *error = key + " must be true or false";
    return false;
  }
  return true;
}

template <typename T>
bool GetNumber(const std::map<std::string, std::string>& values,
               const std::string& key,
               T* output,
               std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  char* end = nullptr;
  errno = 0;
  const double number = std::strtod(it->second.c_str(), &end);
  if (errno != 0 || end == it->second.c_str() || *end != '\0') {
    *error = key + " must be numeric";
    return false;
  }
  *output = static_cast<T>(number);
  return true;
}

bool GetNumber(const std::map<std::string, std::string>& values,
               const std::string& key,
               int* output,
               std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  char* end = nullptr;
  errno = 0;
  const double number = std::strtod(it->second.c_str(), &end);
  if (errno != 0 || end == it->second.c_str() || *end != '\0' ||
      !std::isfinite(number) || std::floor(number) != number ||
      number < static_cast<double>(std::numeric_limits<int>::min()) ||
      number > static_cast<double>(std::numeric_limits<int>::max())) {
    *error = key + " must be an integer-valued number";
    return false;
  }
  *output = static_cast<int>(number);
  return true;
}

bool IsNativeRate(int rate) {
  return rate == 8000 || rate == 16000 || rate == 32000 || rate == 48000;
}

}  // namespace

bool ValidateApmConfig(const AppConfig& c, std::string* error);

bool ValidateAppConfig(const AppConfig& c, std::string* error) {
  if (!ValidateApmConfig(c, error)) return false;
  if (c.fastenhance.npu_core != 0 || c.vad.npu_core != 0 ||
      c.asr.npu_core != 1) {
    *error = "FastEnhance/TEN-VAD must use NPU1; SenseVoice must use the NPU2 slot";
    return false;
  }
  if (c.vad.hop_ms <= 0 || c.vad.hop_ms > 100 || c.vad.threshold < 0.0f ||
      c.vad.threshold > 1.0f || c.vad.min_speech_ms <= 0 ||
      c.vad.min_segment_seconds <= 0 || c.vad.max_segment_seconds <= 0 ||
      c.vad.trailing_silence_seconds <= 0 ||
      c.vad.input_buffer_ms <= 0) {
    *error = "VAD hop/threshold/segment values are invalid";
    return false;
  }
  if (c.vad.max_segment_seconds <=
      c.vad.min_segment_seconds + c.vad.trailing_silence_seconds) {
    *error = "VAD max accumulation must be greater than min accumulation plus trailing silence";
    return false;
  }
  if (c.vad.preview_seconds < 0.0f ||
      (c.vad.preview_seconds > 0.0f &&
       c.vad.preview_seconds >= c.vad.max_segment_seconds)) {
    *error = "VAD preview_seconds must be 0 (off) or less than max_segment_seconds";
    return false;
  }
  if (c.vad.pause_split_seconds < 0.0f ||
      (c.vad.pause_split_seconds > 0.0f &&
       (c.vad.pause_split_seconds < c.vad.min_segment_seconds ||
        c.vad.pause_split_seconds >= c.vad.max_segment_seconds))) {
    *error = "VAD pause_split_seconds must be 0 (off) or in "
             "[min_segment_seconds, max_segment_seconds)";
    return false;
  }
  if (c.vad.stream_tick_seconds < 0.0f ||
      c.vad.stream_pause_seconds <= 0.0f ||
      c.vad.stream_epoch_max_seconds <= c.vad.stream_pause_seconds) {
    *error = "VAD stream_tick_seconds must be >= 0, stream_pause_seconds > 0 "
             "and stream_epoch_max_seconds > stream_pause_seconds";
    return false;
  }
  if (c.diarization.speaker_match_threshold < -1.0f ||
      c.diarization.speaker_match_threshold > 1.0f ||
      c.diarization.step_sec <= 0.0f || c.diarization.step_sec > 10.0f ||
      c.diarization.omp_threads < 1 || c.diarization.omp_threads > 64 ||
      (c.diarization.mode != "segment" && c.diarization.mode != "session" &&
       c.diarization.mode != "online") ||
      c.diarization.session_chunk_sec < 10.0f ||
      c.diarization.session_chunk_sec > 600.0f ||
      c.diarization.session_step_sec < 0.0f ||
      c.diarization.session_step_sec > c.diarization.session_chunk_sec ||
      c.diarization.session_first_window_sec < 0.0f ||
      c.diarization.session_first_window_sec >= c.diarization.session_chunk_sec ||
      c.diarization.session_max_wait_sec < 0.0f ||
      c.diarization.session_max_wait_sec > 900.0f ||
      c.diarization.online_min_new_speech_sec < 0.5f ||
      c.diarization.online_min_new_speech_sec > 30.0f ||
      c.diarization.online_update_min_sec < 0.0f ||
      c.diarization.online_update_min_sec > 30.0f ||
      c.diarization.online_max_speakers < 2 ||
      c.diarization.online_max_speakers > 40 ||
      c.diarization.incremental_delta_new <= 0.0f ||
      c.diarization.incremental_delta_new > 2.0f ||
      c.diarization.incremental_delta_min < 0.0f ||
      c.diarization.incremental_delta_min > 2.0f ||
      c.diarization.incremental_rho_update <= 0.0f ||
      c.diarization.incremental_rho_update >= 1.0f ||
      c.diarization.runner.empty() || c.diarization.model_path.empty() ||
      c.diarization.speaker_table_path.empty()) {
    *error = "Diarization worker/model values are invalid";
    return false;
  }
  if (c.diarization.engine != "pyannote" && c.diarization.engine != "nemotron") {
    *error = "diarization.engine must be pyannote or nemotron";
    return false;
  }
  if (c.diarization.engine == "nemotron") {
    if ((c.diarization.nemotron.preset != "lowlat" &&
         c.diarization.nemotron.preset != "offline") ||
        (c.diarization.nemotron.variant != "npu2" &&
         c.diarization.nemotron.variant != "npu3") ||
        c.diarization.nemotron.threads < 1 ||
        c.diarization.nemotron.threads > 64 ||
        c.diarization.nemotron.onset <= 0.0f ||
        c.diarization.nemotron.onset > 1.0f ||
        c.diarization.nemotron.offset <= 0.0f ||
        c.diarization.nemotron.offset > c.diarization.nemotron.onset ||
        c.diarization.nemotron.runner.empty() ||
        c.diarization.nemotron.model_dir.empty()) {
      *error = "diarization.nemotron values are invalid "
               "(preset lowlat|offline, threads 1-64, 0 < offset <= onset <= 1)";
      return false;
    }
  }
  if (c.output.max_mp3_storage_mb < 0 || c.asr.worker_threads <= 0) {
    *error = "output storage and ASR worker values are invalid";
    return false;
  }
  if ((c.asr.mode != "offline" && c.asr.mode != "streaming") ||
      (c.asr.refine != "off" && c.asr.refine != "epoch")) {
    *error = "asr.mode must be offline or streaming; asr.refine must be off or epoch";
    return false;
  }
  if (c.asr.mode == "streaming") {
    if ((c.asr.streaming.backend != "http" &&
         c.asr.streaming.backend != "worker" &&
         c.asr.streaming.backend != "lib") ||
        (c.asr.streaming.confirm != "immediate" &&
         c.asr.streaming.confirm != "agreement") ||
        c.asr.streaming.poll_seconds < 0.05f ||
        c.asr.streaming.poll_seconds > 5.0f ||
        c.asr.streaming.epoch_cap_seconds < 2.0f ||
        c.asr.streaming.epoch_cap_seconds > 60.0f ||
        c.asr.streaming.epoch_pause_seconds < 0.1f ||
        c.asr.streaming.epoch_pause_seconds > 2.0f ||
        c.asr.streaming.label_wait_seconds < 0.0f ||
        c.asr.streaming.label_wait_seconds > 30.0f ||
        c.asr.streaming.runner.empty() ||
        c.asr.streaming.worker.empty() ||
        c.asr.streaming.server.empty() ||
        c.asr.streaming.model_name.empty()) {
      *error = "asr.streaming values are invalid";
      return false;
    }
  }
  if (c.runtime.npu_lock_file.empty()) {
    *error = "runtime.npu_lock_file must not be empty";
    return false;
  }
  return true;
}

bool ValidateApmConfig(const AppConfig& c, std::string* error) {
  if (!error) return false;
  if (c.runtime.input != "alsa" && c.runtime.input != "file") {
    *error = "runtime.input must be alsa or file";
    return false;
  }
  if (c.runtime.file_input_mode != "realtime" &&
      c.runtime.file_input_mode != "batch") {
    *error = "runtime.file_input_mode must be realtime or batch";
    return false;
  }
  if (c.runtime.input == "file" && c.runtime.input_wav.empty()) {
    *error = "runtime.input_wav must not be empty for file input";
    return false;
  }
  if (!IsNativeRate(c.audio.playback_sample_rate_hz) ||
      !IsNativeRate(c.audio.capture_sample_rate_hz)) {
    *error = "S16 APM sample rates must be one of 8000, 16000, 32000, 48000";
    return false;
  }
  if (c.audio.playback_sample_rate_hz != c.audio.capture_sample_rate_hz) {
    *error = "S16 APM requires playback and capture sample rates to match";
    return false;
  }
  if (c.audio.playback_channels < 1 || c.audio.playback_channels > 8 ||
      c.audio.capture_channels < 1 || c.audio.capture_channels > 8) {
    *error = "playback/capture channels must be in [1, 8]";
    return false;
  }
  if (c.buffers.period_ms <= 0 ||
      c.buffers.alsa_buffer_ms < c.buffers.period_ms ||
      c.buffers.application_buffer_ms <= 0) {
    *error = "buffer values are invalid";
    return false;
  }
  if (c.apm.delay.base_ms < 0 || c.apm.delay.base_ms > 500) {
    *error = "apm.delay.base_ms must be in [0, 500]";
    return false;
  }
  if (c.apm.aec.mode != "aec3" && c.apm.aec.mode != "mobile") {
    *error = "apm.aec.mode must be aec3 or mobile";
    return false;
  }
  const std::set<std::string> ns_levels = {"low", "moderate", "high", "very_high"};
  if (!ns_levels.count(c.apm.ns.level)) {
    *error = "apm.ns.level must be low, moderate, high, or very_high";
    return false;
  }
  const std::set<std::string> agc1_modes = {
      "adaptive_analog", "adaptive_digital", "fixed_digital"};
  if (!agc1_modes.count(c.apm.agc1.mode)) {
    *error = "apm.agc1.mode is invalid";
    return false;
  }
  if (c.apm.agc1.target_level_dbfs < 0 ||
      c.apm.agc1.target_level_dbfs > 31 ||
      c.apm.agc1.compression_gain_db < 0 ||
      c.apm.agc1.compression_gain_db > 90) {
    *error = "AGC1 target must be [0,31] and compression gain [0,90]";
    return false;
  }
  if (c.apm.agc2.fixed_gain_db < 0 || c.apm.agc2.fixed_gain_db >= 50 ||
      c.apm.agc2.adaptive_headroom_db < 0 ||
      c.apm.agc2.adaptive_max_gain_db <= 0 ||
      c.apm.agc2.adaptive_initial_gain_db < 0 ||
      c.apm.agc2.adaptive_max_gain_change_db_per_second <= 0 ||
      c.apm.agc2.adaptive_max_output_noise_level_dbfs > 0) {
    *error = "AGC2 values are outside their supported ranges";
    return false;
  }
  if (!std::isfinite(c.apm.gain_adjustment.pre_gain_factor) ||
      !std::isfinite(c.apm.gain_adjustment.post_gain_factor) ||
      c.apm.gain_adjustment.pre_gain_factor <= 0 ||
      c.apm.gain_adjustment.post_gain_factor <= 0 ||
      c.apm.gain_adjustment.pre_gain_factor > 1000 ||
      c.apm.gain_adjustment.post_gain_factor > 1000) {
    *error = "capture pre/post gain factors must be finite and in (0, 1000]";
    return false;
  }
  return true;
}

bool LoadAppConfig(const std::string& path,
                   AppConfig* c,
                   std::string* error) {
  if (!c || !error) return false;
  std::map<std::string, std::string> v;
  if (!ParseYaml(path, &v, error)) return false;

  const std::set<std::string> known = {
      "audio.playback.device", "audio.playback.sample_rate_hz",
      "audio.playback.channels", "audio.capture.device",
      "audio.capture.sample_rate_hz", "audio.capture.channels",
      "buffers.period_ms", "buffers.alsa_buffer_ms",
      "buffers.application_buffer_ms", "apm.aec.enabled", "apm.aec.mode",
      "apm.aec.multi_channel_render", "apm.aec.multi_channel_capture",
      "apm.delay.base_ms", "apm.ns.enabled", "apm.ns.level",
      "apm.agc1.enabled", "apm.agc1.mode", "apm.agc1.target_level_dbfs",
      "apm.agc1.compression_gain_db", "apm.agc1.limiter",
      "apm.agc2.enabled", "apm.agc2.fixed_gain_db",
      "apm.agc2.adaptive_digital", "apm.agc2.adaptive_headroom_db",
      "apm.agc2.adaptive_max_gain_db", "apm.agc2.adaptive_initial_gain_db",
      "apm.agc2.adaptive_max_gain_change_db_per_second",
      "apm.agc2.adaptive_max_output_noise_level_dbfs",
      "apm.gain_adjustment.enabled",
      "apm.gain_adjustment.pre_gain_factor",
      "apm.gain_adjustment.post_gain_factor",
      "apm.high_pass_filter_enabled",
      "fastenhance.enabled", "fastenhance.model_path",
      "fastenhance.library", "fastenhance.npu_core",
      "fastenhance.monitor_enabled",
      "vad.enabled", "vad.model_path", "vad.library", "vad.npu_core",
      "vad.input_buffer_ms", "vad.hop_ms", "vad.threshold", "vad.min_speech_ms",
      "vad.min_segment_seconds", "vad.max_segment_seconds",
      "vad.trailing_silence_seconds", "vad.preview_seconds",
      "vad.pause_split_seconds", "vad.stream_tick_seconds",
      "vad.stream_pause_seconds", "vad.stream_epoch_max_seconds",
      "diarization.enabled", "diarization.logging_enabled",
      "diarization.mode", "diarization.engine",
      "diarization.runner", "diarization.model_path",
      "diarization.nemotron.runner", "diarization.nemotron.model_dir",
      "diarization.nemotron.preset", "diarization.nemotron.variant",
      "diarization.nemotron.threads",
      "diarization.nemotron.onset", "diarization.nemotron.offset",
      "diarization.speaker_table_path", "diarization.speaker_match_threshold",
      "diarization.step_sec", "diarization.omp_threads",
      "diarization.session_chunk_sec", "diarization.session_max_wait_sec",
      "diarization.session_step_sec",
      "diarization.session_first_window_sec",
      "diarization.online_min_new_speech_sec",
      "diarization.online_update_min_sec",
      "diarization.online_max_speakers",
      "diarization.incremental", "diarization.incremental_plda",
      "diarization.incremental_delta_new", "diarization.incremental_delta_min",
      "diarization.incremental_rho_update",
      "asr.enabled", "asr.model_path", "asr.model_type", "asr.language",
      "asr.library", "asr.npu_core", "asr.worker_threads",
      "asr.mode", "asr.refine",
      "asr.streaming.backend", "asr.streaming.runner",
      "asr.streaming.worker", "asr.streaming.model_dir",
      "asr.streaming.server", "asr.streaming.model_name",
      "asr.streaming.language", "asr.streaming.poll_seconds", "asr.streaming.confirm",
      "asr.streaming.epoch_cap_seconds", "asr.streaming.epoch_pause_seconds",
      "asr.streaming.label_wait_seconds",
      "asr.streaming.split_turns",
      "output.temp_dir", "output.text_file", "output.mp3_dir",
      "output.max_mp3_storage_mb", "output.mp3_encoder",
      "runtime.input", "runtime.input_wav", "runtime.file_input_mode",
      "runtime.apm_binary", "runtime.npu_lock_file",
      "runtime.log_level", "runtime.playback_enabled"};
  for (const auto& item : v) {
    if (!known.count(item.first)) {
      *error = "unknown config key: " + item.first;
      return false;
    }
  }

#define GET_STRING(key, field) if (!GetString(v, key, &field, error)) return false
#define GET_OPTIONAL_STRING(key, field) \
  if (!GetStringAllowEmpty(v, key, &field, error)) return false
#define GET_BOOL(key, field) if (!GetBool(v, key, &field, error)) return false
#define GET_NUM(key, field) if (!GetNumber(v, key, &field, error)) return false
  GET_STRING("audio.playback.device", c->audio.playback_device);
  GET_NUM("audio.playback.sample_rate_hz", c->audio.playback_sample_rate_hz);
  GET_NUM("audio.playback.channels", c->audio.playback_channels);
  GET_STRING("audio.capture.device", c->audio.capture_device);
  GET_NUM("audio.capture.sample_rate_hz", c->audio.capture_sample_rate_hz);
  GET_NUM("audio.capture.channels", c->audio.capture_channels);
  GET_NUM("buffers.period_ms", c->buffers.period_ms);
  GET_NUM("buffers.alsa_buffer_ms", c->buffers.alsa_buffer_ms);
  GET_NUM("buffers.application_buffer_ms", c->buffers.application_buffer_ms);
  GET_BOOL("apm.aec.enabled", c->apm.aec.enabled);
  GET_STRING("apm.aec.mode", c->apm.aec.mode);
  GET_BOOL("apm.aec.multi_channel_render", c->apm.aec.multi_channel_render);
  GET_BOOL("apm.aec.multi_channel_capture", c->apm.aec.multi_channel_capture);
  GET_NUM("apm.delay.base_ms", c->apm.delay.base_ms);
  GET_BOOL("apm.ns.enabled", c->apm.ns.enabled);
  GET_STRING("apm.ns.level", c->apm.ns.level);
  GET_BOOL("apm.agc1.enabled", c->apm.agc1.enabled);
  GET_STRING("apm.agc1.mode", c->apm.agc1.mode);
  GET_NUM("apm.agc1.target_level_dbfs", c->apm.agc1.target_level_dbfs);
  GET_NUM("apm.agc1.compression_gain_db", c->apm.agc1.compression_gain_db);
  GET_BOOL("apm.agc1.limiter", c->apm.agc1.limiter);
  GET_BOOL("apm.agc2.enabled", c->apm.agc2.enabled);
  GET_NUM("apm.agc2.fixed_gain_db", c->apm.agc2.fixed_gain_db);
  GET_BOOL("apm.agc2.adaptive_digital", c->apm.agc2.adaptive_digital);
  GET_NUM("apm.agc2.adaptive_headroom_db", c->apm.agc2.adaptive_headroom_db);
  GET_NUM("apm.agc2.adaptive_max_gain_db", c->apm.agc2.adaptive_max_gain_db);
  GET_NUM("apm.agc2.adaptive_initial_gain_db",
          c->apm.agc2.adaptive_initial_gain_db);
  GET_NUM("apm.agc2.adaptive_max_gain_change_db_per_second",
          c->apm.agc2.adaptive_max_gain_change_db_per_second);
  GET_NUM("apm.agc2.adaptive_max_output_noise_level_dbfs",
          c->apm.agc2.adaptive_max_output_noise_level_dbfs);
  GET_BOOL("apm.gain_adjustment.enabled",
           c->apm.gain_adjustment.enabled);
  GET_NUM("apm.gain_adjustment.pre_gain_factor",
          c->apm.gain_adjustment.pre_gain_factor);
  GET_NUM("apm.gain_adjustment.post_gain_factor",
          c->apm.gain_adjustment.post_gain_factor);
  GET_BOOL("apm.high_pass_filter_enabled",
           c->apm.high_pass_filter_enabled);
  GET_BOOL("fastenhance.enabled", c->fastenhance.enabled);
  GET_STRING("fastenhance.model_path", c->fastenhance.model_path);
  GET_OPTIONAL_STRING("fastenhance.library", c->fastenhance.library);
  GET_NUM("fastenhance.npu_core", c->fastenhance.npu_core);
  GET_BOOL("fastenhance.monitor_enabled", c->fastenhance.monitor_enabled);
  GET_BOOL("vad.enabled", c->vad.enabled);
  GET_STRING("vad.model_path", c->vad.model_path);
  GET_OPTIONAL_STRING("vad.library", c->vad.library);
  GET_NUM("vad.npu_core", c->vad.npu_core);
  GET_NUM("vad.input_buffer_ms", c->vad.input_buffer_ms);
  GET_NUM("vad.hop_ms", c->vad.hop_ms);
  GET_NUM("vad.threshold", c->vad.threshold);
  GET_NUM("vad.min_speech_ms", c->vad.min_speech_ms);
  GET_NUM("vad.min_segment_seconds", c->vad.min_segment_seconds);
  GET_NUM("vad.max_segment_seconds", c->vad.max_segment_seconds);
  GET_NUM("vad.trailing_silence_seconds", c->vad.trailing_silence_seconds);
  GET_NUM("vad.preview_seconds", c->vad.preview_seconds);
  GET_NUM("vad.pause_split_seconds", c->vad.pause_split_seconds);
  GET_NUM("vad.stream_tick_seconds", c->vad.stream_tick_seconds);
  GET_NUM("vad.stream_pause_seconds", c->vad.stream_pause_seconds);
  GET_NUM("vad.stream_epoch_max_seconds", c->vad.stream_epoch_max_seconds);
  GET_BOOL("diarization.enabled", c->diarization.enabled);
  GET_BOOL("diarization.logging_enabled", c->diarization.logging_enabled);
  GET_STRING("diarization.mode", c->diarization.mode);
  GET_STRING("diarization.engine", c->diarization.engine);
  GET_STRING("diarization.nemotron.runner", c->diarization.nemotron.runner);
  GET_STRING("diarization.nemotron.model_dir", c->diarization.nemotron.model_dir);
  GET_STRING("diarization.nemotron.preset", c->diarization.nemotron.preset);
  GET_STRING("diarization.nemotron.variant", c->diarization.nemotron.variant);
  GET_NUM("diarization.nemotron.threads", c->diarization.nemotron.threads);
  GET_NUM("diarization.nemotron.onset", c->diarization.nemotron.onset);
  GET_NUM("diarization.nemotron.offset", c->diarization.nemotron.offset);
  GET_STRING("diarization.runner", c->diarization.runner);
  GET_STRING("diarization.model_path", c->diarization.model_path);
  GET_STRING("diarization.speaker_table_path", c->diarization.speaker_table_path);
  GET_NUM("diarization.speaker_match_threshold", c->diarization.speaker_match_threshold);
  GET_NUM("diarization.step_sec", c->diarization.step_sec);
  GET_NUM("diarization.omp_threads", c->diarization.omp_threads);
  GET_NUM("diarization.session_chunk_sec", c->diarization.session_chunk_sec);
  GET_NUM("diarization.session_step_sec", c->diarization.session_step_sec);
  GET_NUM("diarization.session_first_window_sec",
          c->diarization.session_first_window_sec);
  GET_NUM("diarization.session_max_wait_sec", c->diarization.session_max_wait_sec);
  GET_NUM("diarization.online_min_new_speech_sec", c->diarization.online_min_new_speech_sec);
  GET_NUM("diarization.online_update_min_sec", c->diarization.online_update_min_sec);
  GET_NUM("diarization.online_max_speakers", c->diarization.online_max_speakers);
  GET_BOOL("diarization.incremental", c->diarization.incremental);
  GET_BOOL("diarization.incremental_plda", c->diarization.incremental_plda);
  GET_NUM("diarization.incremental_delta_new", c->diarization.incremental_delta_new);
  GET_NUM("diarization.incremental_delta_min", c->diarization.incremental_delta_min);
  GET_NUM("diarization.incremental_rho_update", c->diarization.incremental_rho_update);
  GET_BOOL("asr.enabled", c->asr.enabled);
  GET_STRING("asr.mode", c->asr.mode);
  GET_STRING("asr.refine", c->asr.refine);
  GET_STRING("asr.streaming.backend", c->asr.streaming.backend);
  GET_STRING("asr.streaming.runner", c->asr.streaming.runner);
  GET_STRING("asr.streaming.worker", c->asr.streaming.worker);
  GET_STRING("asr.streaming.model_dir", c->asr.streaming.model_dir);
  GET_STRING("asr.streaming.server", c->asr.streaming.server);
  GET_STRING("asr.streaming.model_name", c->asr.streaming.model_name);
  GET_STRING("asr.streaming.language", c->asr.streaming.language);
  GET_NUM("asr.streaming.poll_seconds", c->asr.streaming.poll_seconds);
  GET_STRING("asr.streaming.confirm", c->asr.streaming.confirm);
  GET_NUM("asr.streaming.epoch_cap_seconds", c->asr.streaming.epoch_cap_seconds);
  GET_NUM("asr.streaming.epoch_pause_seconds", c->asr.streaming.epoch_pause_seconds);
  GET_NUM("asr.streaming.label_wait_seconds", c->asr.streaming.label_wait_seconds);
  GET_BOOL("asr.streaming.split_turns", c->asr.streaming.split_turns);
  GET_STRING("asr.model_path", c->asr.model_path);
  GET_STRING("asr.model_type", c->asr.model_type);
  GET_STRING("asr.language", c->asr.language);
  GET_OPTIONAL_STRING("asr.library", c->asr.library);
  GET_NUM("asr.npu_core", c->asr.npu_core);
  GET_NUM("asr.worker_threads", c->asr.worker_threads);
  GET_STRING("runtime.npu_lock_file", c->runtime.npu_lock_file);
  GET_STRING("output.temp_dir", c->output.temp_dir);
  GET_STRING("output.text_file", c->output.text_file);
  GET_STRING("output.mp3_dir", c->output.mp3_dir);
  GET_NUM("output.max_mp3_storage_mb", c->output.max_mp3_storage_mb);
  GET_STRING("output.mp3_encoder", c->output.mp3_encoder);
  GET_STRING("runtime.input", c->runtime.input);
  GET_OPTIONAL_STRING("runtime.input_wav", c->runtime.input_wav);
  GET_STRING("runtime.file_input_mode", c->runtime.file_input_mode);
  GET_STRING("runtime.apm_binary", c->runtime.apm_binary);
  GET_STRING("runtime.log_level", c->runtime.log_level);
  GET_BOOL("runtime.playback_enabled", c->runtime.playback_enabled);
#undef GET_STRING
#undef GET_OPTIONAL_STRING
#undef GET_BOOL
#undef GET_NUM
  return ValidateAppConfig(*c, error);
}

bool LoadApmConfig(const std::string& path,
                   AppConfig* c,
                   std::string* error) {
  if (!c || !error) return false;
  std::map<std::string, std::string> v;
  if (!ParseYaml(path, &v, error)) return false;

  const std::set<std::string> owned = {
      "audio.playback.device", "audio.playback.sample_rate_hz",
      "audio.playback.channels", "audio.capture.device",
      "audio.capture.sample_rate_hz", "audio.capture.channels",
      "buffers.period_ms", "buffers.alsa_buffer_ms",
      "buffers.application_buffer_ms", "apm.aec.enabled", "apm.aec.mode",
      "apm.aec.multi_channel_render", "apm.aec.multi_channel_capture",
      "apm.delay.base_ms", "apm.ns.enabled", "apm.ns.level",
      "apm.agc1.enabled", "apm.agc1.mode", "apm.agc1.target_level_dbfs",
      "apm.agc1.compression_gain_db", "apm.agc1.limiter",
      "apm.agc2.enabled", "apm.agc2.fixed_gain_db",
      "apm.agc2.adaptive_digital", "apm.agc2.adaptive_headroom_db",
      "apm.agc2.adaptive_max_gain_db", "apm.agc2.adaptive_initial_gain_db",
      "apm.agc2.adaptive_max_gain_change_db_per_second",
      "apm.agc2.adaptive_max_output_noise_level_dbfs",
      "apm.gain_adjustment.enabled",
      "apm.gain_adjustment.pre_gain_factor",
      "apm.gain_adjustment.post_gain_factor",
      "apm.high_pass_filter_enabled", "runtime.input", "runtime.input_wav",
      "runtime.file_input_mode", "runtime.playback_enabled"};
  for (const auto& item : v) {
    const bool apm_owned = item.first.rfind("audio.", 0) == 0 ||
                           item.first.rfind("buffers.", 0) == 0 ||
                           item.first.rfind("apm.", 0) == 0;
    if (apm_owned && !owned.count(item.first)) {
      *error = "unknown APM config key: " + item.first;
      return false;
    }
  }

#define APM_GET_STRING(key, field) if (!GetString(v, key, &field, error)) return false
#define APM_GET_OPTIONAL_STRING(key, field) \
  if (!GetStringAllowEmpty(v, key, &field, error)) return false
#define APM_GET_BOOL(key, field) if (!GetBool(v, key, &field, error)) return false
#define APM_GET_NUM(key, field) if (!GetNumber(v, key, &field, error)) return false
  APM_GET_STRING("audio.playback.device", c->audio.playback_device);
  APM_GET_NUM("audio.playback.sample_rate_hz", c->audio.playback_sample_rate_hz);
  APM_GET_NUM("audio.playback.channels", c->audio.playback_channels);
  APM_GET_STRING("audio.capture.device", c->audio.capture_device);
  APM_GET_NUM("audio.capture.sample_rate_hz", c->audio.capture_sample_rate_hz);
  APM_GET_NUM("audio.capture.channels", c->audio.capture_channels);
  APM_GET_NUM("buffers.period_ms", c->buffers.period_ms);
  APM_GET_NUM("buffers.alsa_buffer_ms", c->buffers.alsa_buffer_ms);
  APM_GET_NUM("buffers.application_buffer_ms", c->buffers.application_buffer_ms);
  APM_GET_BOOL("apm.aec.enabled", c->apm.aec.enabled);
  APM_GET_STRING("apm.aec.mode", c->apm.aec.mode);
  APM_GET_BOOL("apm.aec.multi_channel_render", c->apm.aec.multi_channel_render);
  APM_GET_BOOL("apm.aec.multi_channel_capture", c->apm.aec.multi_channel_capture);
  APM_GET_NUM("apm.delay.base_ms", c->apm.delay.base_ms);
  APM_GET_BOOL("apm.ns.enabled", c->apm.ns.enabled);
  APM_GET_STRING("apm.ns.level", c->apm.ns.level);
  APM_GET_BOOL("apm.agc1.enabled", c->apm.agc1.enabled);
  APM_GET_STRING("apm.agc1.mode", c->apm.agc1.mode);
  APM_GET_NUM("apm.agc1.target_level_dbfs", c->apm.agc1.target_level_dbfs);
  APM_GET_NUM("apm.agc1.compression_gain_db", c->apm.agc1.compression_gain_db);
  APM_GET_BOOL("apm.agc1.limiter", c->apm.agc1.limiter);
  APM_GET_BOOL("apm.agc2.enabled", c->apm.agc2.enabled);
  APM_GET_NUM("apm.agc2.fixed_gain_db", c->apm.agc2.fixed_gain_db);
  APM_GET_BOOL("apm.agc2.adaptive_digital", c->apm.agc2.adaptive_digital);
  APM_GET_NUM("apm.agc2.adaptive_headroom_db", c->apm.agc2.adaptive_headroom_db);
  APM_GET_NUM("apm.agc2.adaptive_max_gain_db", c->apm.agc2.adaptive_max_gain_db);
  APM_GET_NUM("apm.agc2.adaptive_initial_gain_db", c->apm.agc2.adaptive_initial_gain_db);
  APM_GET_NUM("apm.agc2.adaptive_max_gain_change_db_per_second", c->apm.agc2.adaptive_max_gain_change_db_per_second);
  APM_GET_NUM("apm.agc2.adaptive_max_output_noise_level_dbfs", c->apm.agc2.adaptive_max_output_noise_level_dbfs);
  APM_GET_BOOL("apm.gain_adjustment.enabled", c->apm.gain_adjustment.enabled);
  APM_GET_NUM("apm.gain_adjustment.pre_gain_factor", c->apm.gain_adjustment.pre_gain_factor);
  APM_GET_NUM("apm.gain_adjustment.post_gain_factor", c->apm.gain_adjustment.post_gain_factor);
  APM_GET_BOOL("apm.high_pass_filter_enabled", c->apm.high_pass_filter_enabled);
  APM_GET_STRING("runtime.input", c->runtime.input);
  APM_GET_OPTIONAL_STRING("runtime.input_wav", c->runtime.input_wav);
  APM_GET_STRING("runtime.file_input_mode", c->runtime.file_input_mode);
  APM_GET_BOOL("runtime.playback_enabled", c->runtime.playback_enabled);
#undef APM_GET_STRING
#undef APM_GET_OPTIONAL_STRING
#undef APM_GET_BOOL
#undef APM_GET_NUM
  return ValidateApmConfig(*c, error);
}

}  // namespace apm_example

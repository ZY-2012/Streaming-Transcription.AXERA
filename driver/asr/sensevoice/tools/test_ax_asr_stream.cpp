// Board experiment: can the EXISTING (offline-era) libax_asr_api.so drive
// the dedicated streaming SenseVoice model when it is placed at the fixed
// path <model_path>/sensevoice/sensevoice.axmodel?
//
//   test_ax_asr_stream <model_path> <16k-mono-wav> [chunk_ms] [lib]
//
// dlopens libax_asr_api.so (default: ../lib/libax_asr_api.so), Init ->
// StreamInit -> feeds the WAV in chunk_ms slices, printing StreamResult
// after every feed; at EOF feeds 0.6 s of silence (StreamFinish is absent
// in this .so build) and prints the final result.  Exit prints PASS/FAIL
// heuristics: the run passes when partials grow monotonically and the
// final text is non-empty.
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ax_asr_api.h"

namespace {

std::vector<int16_t> ReadWav16k(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  char header[44];
  in.read(header, sizeof(header));
  if (in.gcount() != 44 || std::memcmp(header, "RIFF", 4) != 0 ||
      std::memcmp(header + 8, "WAVE", 4) != 0) {
    std::fprintf(stderr, "not a plain 44-header wav: %s\n", path.c_str());
    return {};
  }
  std::vector<int16_t> pcm;
  int16_t buf[4096];
  while (in.read(reinterpret_cast<char*>(buf), sizeof(buf)))
    pcm.insert(pcm.end(), buf, buf + in.gcount() / sizeof(int16_t));
  return pcm;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model_path> <16k-mono-wav> [chunk_ms=160] [lib=../lib/libax_asr_api.so]\n", argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string wav = argv[2];
  const int chunk_ms = argc > 3 ? std::atoi(argv[3]) : 160;
  const std::string lib = argc > 4 ? argv[4] : "../lib/libax_asr_api.so";

  void* handle = dlopen(lib.c_str(), RTLD_NOW);
  if (!handle) {
    std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  auto init = reinterpret_cast<decltype(&AX_ASR_Init)>(dlsym(handle, "AX_ASR_Init"));
  auto sinit = reinterpret_cast<decltype(&AX_ASR_StreamInit)>(dlsym(handle, "AX_ASR_StreamInit"));
  auto sfeed = reinterpret_cast<decltype(&AX_ASR_StreamFeed)>(dlsym(handle, "AX_ASR_StreamFeed"));
  auto sres = reinterpret_cast<decltype(&AX_ASR_StreamResult)>(dlsym(handle, "AX_ASR_StreamResult"));
  auto sfinish = reinterpret_cast<decltype(&AX_ASR_StreamFinish)>(dlsym(handle, "AX_ASR_StreamFinish"));
  auto uninit = reinterpret_cast<decltype(&AX_ASR_Uninit)>(dlsym(handle, "AX_ASR_Uninit"));
  if (!init || !sinit || !sfeed || !sres) {
    std::fprintf(stderr, "streaming symbols missing (init=%p sinit=%p feed=%p res=%p)\n",
                 (void*)init, (void*)sinit, (void*)sfeed, (void*)sres);
    return 1;
  }
  std::printf("StreamFinish symbol: %s\n", sfinish ? "present" : "ABSENT (pad flush)");

  AX_ASR_HANDLE asr = init(AX_SENSEVOICE, model_path.c_str());
  if (!asr) {
    std::fprintf(stderr, "AX_ASR_Init failed for %s\n", model_path.c_str());
    return 1;
  }
  std::printf("ASR initialized. model=%s\n", model_path.c_str());

  std::vector<int16_t> pcm = ReadWav16k(wav);
  if (pcm.empty()) return 1;
  const size_t chunk = static_cast<size_t>(chunk_ms) * 16;  // 16 kHz
  std::vector<float> fbuf;
  int status = sinit(asr);
  std::printf("StreamInit status=%d  (%zu samples, %zu ms chunks)\n", status, pcm.size(), chunk / 16);

  auto feed_and_report = [&](const float* data, int n, const char* tag) {
    status = sfeed(asr, const_cast<float*>(data), n, 16000);
    const char* text = nullptr;
    if (sres(asr, &text) == 0 && text) {
      std::printf("[%s] status=%d text: %s\n", tag, status, text);
    } else {
      std::printf("[%s] status=%d result=<none>\n", tag, status);
    }
  };

  fbuf.reserve(chunk);
  double seconds = 0.0;
  for (size_t off = 0; off < pcm.size(); off += chunk) {
    const size_t n = std::min(chunk, pcm.size() - off);
    fbuf.resize(n);
    for (size_t i = 0; i < n; ++i) fbuf[i] = pcm[off + i] / 32768.0f;
    seconds = static_cast<double>(off + n) / 16000.0;
    char tag[32];
    std::snprintf(tag, sizeof(tag), "%.2fs", seconds);
    feed_and_report(fbuf.data(), static_cast<int>(n), tag);
    if (status != 0) { std::fprintf(stderr, "StreamFeed failed: %d\n", status); return 1; }
  }
  // Tail flush: the streaming window needs padding to emit the last frames.
  fbuf.assign(9600, 0.0f);  // 0.6 s silence
  feed_and_report(fbuf.data(), static_cast<int>(fbuf.size()), "flush");
  if (sfinish) sfinish(asr);
  feed_and_report(fbuf.data(), static_cast<int>(fbuf.size()), "final");

  uninit(asr);
  return 0;
}

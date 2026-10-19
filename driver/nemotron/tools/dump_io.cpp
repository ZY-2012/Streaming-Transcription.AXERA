// Dump an axmodel's input/output tensor contract (names + shapes + types)
// by loading it through AX Engine on the board.  Used to recover the
// streaming SenseVoice graph interface (inputs_info is not stored as a
// plain string in the axmodel header).
//
//   dump_io <model.axmodel>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "ax_engine_api.h"
#include "ax_sys_api.h"

static bool WriteAll(int fd, const void* data, std::size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  while (n) {
    const ssize_t w = write(fd, p, n);
    if (w <= 0) return false;
    p += w; n -= static_cast<std::size_t>(w);
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <model.axmodel>\n", argv[0]); return 2; }
  std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  std::vector<char> model(static_cast<std::size_t>(in.tellg()));
  in.seekg(0); in.read(model.data(), static_cast<std::streamsize>(model.size()));
  std::printf("model %s: %zu bytes\n", argv[1], model.size());

  AX_S32 ret = AX_SYS_Init();
  if (ret != 0) { std::fprintf(stderr, "AX_SYS_Init: 0x%x\n", ret); return 1; }
  AX_ENGINE_NPU_ATTR_T attr{};
  attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
  ret = AX_ENGINE_Init(&attr);
  if (ret != 0) {
    std::memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;
    ret = AX_ENGINE_Init(&attr);
    if (ret != 0) { std::fprintf(stderr, "AX_ENGINE_Init: 0x%x\n", ret); return 1; }
  }
  AX_ENGINE_HANDLE handle = nullptr;
  ret = AX_ENGINE_CreateHandle(&handle, model.data(), static_cast<AX_U32>(model.size()));
  if (ret != 0) { std::fprintf(stderr, "CreateHandle: 0x%x\n", ret); return 1; }
  ret = AX_ENGINE_CreateContext(handle);
  if (ret != 0) { std::fprintf(stderr, "CreateContext: 0x%x\n", ret); return 1; }
  AX_ENGINE_IO_INFO_T* info = nullptr;
  ret = AX_ENGINE_GetIOInfo(handle, &info);
  if (ret != 0) { std::fprintf(stderr, "GetIOInfo: 0x%x\n", ret); return 1; }

  std::printf("inputs: %u\n", info->nInputSize);
  for (AX_U32 i = 0; i < info->nInputSize; ++i) {
    const auto& t = info->pInputs[i];
    std::string dims;
    for (AX_U8 d = 0; d < t.nShapeSize; ++d) dims += std::to_string(t.pShape[d]) + (d + 1 < t.nShapeSize ? "," : "");
    std::printf("  [%u] name=%s shape=[%s] bytes=%u dtype=%d\n", i, t.pName, dims.c_str(), t.nSize,
                static_cast<int>(t.eDataType));
  }
  std::printf("outputs: %u\n", info->nOutputSize);
  for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
    const auto& t = info->pOutputs[i];
    std::string dims;
    for (AX_U8 d = 0; d < t.nShapeSize; ++d) dims += std::to_string(t.pShape[d]) + (d + 1 < t.nShapeSize ? "," : "");
    std::printf("  [%u] name=%s shape=[%s] bytes=%u dtype=%d\n", i, t.pName, dims.c_str(), t.nSize,
                static_cast<int>(t.eDataType));
  }
  (void)WriteAll;
  AX_ENGINE_DestroyHandle(handle);
  AX_ENGINE_Deinit();
  AX_SYS_Deinit();
  return 0;
}

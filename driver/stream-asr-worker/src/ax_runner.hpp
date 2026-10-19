#pragma once

// Minimal named-IO AX Engine runner for the streaming SenseVoice graphs,
// adapted from driver/nemotron's ax_engine_runner: DISABLE-first init with
// BIG_LITTLE fallback (the device-global VNPU mode is shared with the
// pipeline and the diarization worker), best-effort affinity 0x1, and the
// cache flush/invalidate pair the runner needs for deterministic results.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ax_engine_api.h"
#include "ax_engine_type.h"
#include "ax_sys_api.h"

class AxRunner {
 public:
    explicit AxRunner(const std::string& model_path) {
        std::FILE* f = std::fopen(model_path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open axmodel: " + model_path);
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        model_.resize(static_cast<std::size_t>(size));
        if (std::fread(model_.data(), 1, model_.size(), f) != model_.size()) {
            std::fclose(f);
            throw std::runtime_error("short read on " + model_path);
        }
        std::fclose(f);
        InitShared();
        try {
            if (AX_ENGINE_CreateHandle(&handle_, model_.data(),
                                       static_cast<AX_U32>(model_.size())) != 0)
                throw std::runtime_error("AX_ENGINE_CreateHandle failed");
            if (AX_ENGINE_CreateContext(handle_) != 0)
                throw std::runtime_error("AX_ENGINE_CreateContext failed");
            if (AX_ENGINE_GetIOInfo(handle_, &info_) != 0)
                throw std::runtime_error("AX_ENGINE_GetIOInfo failed");
            io_.nInputSize = info_->nInputSize;
            io_.nOutputSize = info_->nOutputSize;
            io_.pInputs = new AX_ENGINE_IO_BUFFER_T[info_->nInputSize]();
            io_.pOutputs = new AX_ENGINE_IO_BUFFER_T[info_->nOutputSize]();
            for (AX_U32 i = 0; i < info_->nInputSize; ++i) {
                in_idx_[info_->pInputs[i].pName] = i;
                if (AX_SYS_MemAllocCached(&io_.pInputs[i].phyAddr, &io_.pInputs[i].pVirAddr,
                                          info_->pInputs[i].nSize, 128,
                                          reinterpret_cast<const AX_S8*>("npu")) != 0)
                    throw std::runtime_error("input alloc failed");
                io_.pInputs[i].nSize = info_->pInputs[i].nSize;
                std::memset(io_.pInputs[i].pVirAddr, 0, info_->pInputs[i].nSize);
            }
            for (AX_U32 i = 0; i < info_->nOutputSize; ++i) {
                if (AX_SYS_MemAllocCached(&io_.pOutputs[i].phyAddr, &io_.pOutputs[i].pVirAddr,
                                          info_->pOutputs[i].nSize, 128,
                                          reinterpret_cast<const AX_S8*>("npu")) != 0)
                    throw std::runtime_error("output alloc failed");
                io_.pOutputs[i].nSize = info_->pOutputs[i].nSize;
            }
            if (!AX_ENGINE_SetAffinity(handle_, 0x1))
                std::fprintf(stderr, "[asr-worker] affinity 0x1 bind failed (continuing)\n");
        } catch (...) {
            Destroy();
            throw;
        }
    }

    ~AxRunner() { Destroy(); }

    float* Input(const std::string& name) {
        auto it = in_idx_.find(name);
        if (it == in_idx_.end()) throw std::runtime_error("no input " + name);
        return static_cast<float*>(io_.pInputs[it->second].pVirAddr);
    }
    int32_t* InputInt(const std::string& name) {
        return reinterpret_cast<int32_t*>(Input(name));
    }
    const float* Output(const std::string& name) const {
        for (AX_U32 i = 0; i < info_->nOutputSize; ++i)
            if (std::strcmp(info_->pOutputs[i].pName, name.c_str()) == 0)
                return static_cast<const float*>(io_.pOutputs[i].pVirAddr);
        throw std::runtime_error("no output " + name);
    }

    void Run() {
        for (AX_U32 i = 0; i < io_.nInputSize; ++i)
            AX_SYS_MflushCache(io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr, io_.pInputs[i].nSize);
        if (AX_ENGINE_RunSync(handle_, &io_) != 0) throw std::runtime_error("AX_ENGINE_RunSync failed");
        for (AX_U32 i = 0; i < io_.nOutputSize; ++i)
            AX_SYS_MinvalidateCache(io_.pOutputs[i].phyAddr, io_.pOutputs[i].pVirAddr,
                                    io_.pOutputs[i].nSize);
    }

 private:
    static void InitShared() {
        static int count = 0;
        static bool inited = false;
        if (!inited) {
            if (AX_SYS_Init() != 0) throw std::runtime_error("AX_SYS_Init failed");
            AX_ENGINE_NPU_ATTR_T attr;
            std::memset(&attr, 0, sizeof(attr));
            attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
            if (AX_ENGINE_Init(&attr) != 0) {
                std::fprintf(stderr,
                             "[asr-worker] VNPU DISABLE init failed; retrying BIG_LITTLE\n");
                std::memset(&attr, 0, sizeof(attr));
                attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;
                if (AX_ENGINE_Init(&attr) != 0) throw std::runtime_error("AX_ENGINE_Init failed");
            }
            inited = true;
        }
        ++count;
    }

    void Destroy() {
        if (io_.pInputs) {
            for (AX_U32 i = 0; i < io_.nInputSize; ++i)
                if (io_.pInputs[i].pVirAddr)
                    AX_SYS_MemFree(io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr);
            delete[] io_.pInputs;
        }
        if (io_.pOutputs) {
            for (AX_U32 i = 0; i < io_.nOutputSize; ++i)
                if (io_.pOutputs[i].pVirAddr)
                    AX_SYS_MemFree(io_.pOutputs[i].phyAddr, io_.pOutputs[i].pVirAddr);
            delete[] io_.pOutputs;
        }
        io_.pInputs = nullptr;
        io_.pOutputs = nullptr;
        if (handle_) {
            AX_ENGINE_DestroyHandle(handle_);
            handle_ = nullptr;
        }
    }

    std::vector<char> model_;
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* info_ = nullptr;
    AX_ENGINE_IO_T io_{};
    std::unordered_map<std::string, AX_U32> in_idx_;
};

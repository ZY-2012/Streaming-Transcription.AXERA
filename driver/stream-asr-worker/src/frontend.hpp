#pragma once

// Streaming front-end for the dedicated streaming SenseVoice axmodel
// ([1,26,560] LFR features): kaldi-native-fbank (80 mel, 10 ms hop, hamming)
// -> LFR (window 7, shift 6, edge-replicated) -> CMVN (560-dim, constants
// from AXERA-TECH/ax_asr_api sensevoice.cpp) -> per-window feature packing.
//
// Frame indexing: mel frames are 10 ms; LFR frame j covers mel frames
// [j*6-3, j*6+3] clamped to [0, total_mel-1].  LFR frame j is computable
// once mel frame j*6+3 exists, i.e. total_mel >= j*6+4.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

class StreamFrontend {
 public:
    static constexpr int kMels = 80;
    static constexpr int kLfrWindow = 7;
    static constexpr int kLfrShift = 6;
    static constexpr int kFeatDim = kMels * kLfrWindow;  // 560
    static constexpr int kWindowFrames = 26;            // model input [1,26,560]
    static constexpr int kQueryNum = 4;                 // output [1,30,*]: 4 query + 26

    StreamFrontend() {
        knf::FbankOptions opts;
        opts.frame_opts.dither = 1.0f;
        opts.frame_opts.snip_edges = true;
        opts.frame_opts.samp_freq = 16000;
        opts.frame_opts.frame_shift_ms = 10;
        opts.frame_opts.frame_length_ms = 25;
        opts.frame_opts.remove_dc_offset = true;
        opts.frame_opts.window_type = "hamming";
        opts.mel_opts.num_bins = kMels;
        opts.mel_opts.high_freq = 0;
        opts.mel_opts.low_freq = 20;
        opts.mel_opts.is_librosa = false;
        opts_ = opts;
        fbank_ = std::make_unique<knf::OnlineFbank>(opts);
    }

    // Feed int16-scaled floats (16 kHz mono).
    void AcceptWaveform(const float* pcm, std::size_t n) {
        fbank_->AcceptWaveform(16000.0f, pcm, static_cast<int32_t>(n));
    }

    // No more audio (end of stream); flushes the final partial fbank frame.
    void InputFinished() { fbank_->InputFinished(); }

    // Fresh fbank for the next session (after InputFinished, knf accepts no
    // more waveform on the same instance).
    void Reset() { fbank_ = std::make_unique<knf::OnlineFbank>(opts_); }

    int NumMelFrames() const { return fbank_->NumFramesReady(); }

    // LFR frames computable from the mel frames seen so far.
    int NumLfrFrames(int total_mel) const {
        if (total_mel < 4) return 0;
        return (total_mel - 4) / kLfrShift + 1;
    }

    // Pack LFR frames [begin, begin+count) into `out` (count x 560), with
    // LFR mel context clamped to [0, total_mel-1] exactly like the upstream
    // apply_lfr_ (edge replication at both session ends).  CMVN applied.
    void PackWindow(int begin, int count, int total_mel, float* out) const {
        for (int j = 0; j < count; ++j) {
            const int lfr = begin + j;
            float* row = out + static_cast<std::size_t>(j) * kFeatDim;
            for (int w = 0; w < kLfrWindow; ++w) {
                int idx = lfr * kLfrShift + w - (kLfrWindow - 1) / 2;
                if (idx < 0) idx = 0;
                if (idx >= total_mel) idx = total_mel - 1;
                const float* mel = fbank_->GetFrame(idx);
                std::copy(mel, mel + kMels, row + w * kMels);
            }
            for (int d = 0; d < kFeatDim; ++d)
                row[d] = (row[d] + kNegMean[d]) * kInvStddev[d];
        }
    }

 private:
    knf::FbankOptions opts_;
    std::unique_ptr<knf::OnlineFbank> fbank_;

#include "cmvn_data.inc"
};

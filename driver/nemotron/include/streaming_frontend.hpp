#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mel_frontend.hpp"

// Rolling front-end for the streaming worker: incremental 16 kHz PCM in,
// absolute-indexed 10 ms log-mel frames out, bit-identical to file-mode
// MelFrontend::prepare + compute_range over the same audio.
//
// A mel frame f only reads pre-emphasized samples [f*hop - 200, f*hop + 200),
// so the sample buffer can drop everything before (keep_frame*hop - 200) once
// frames below keep_frame are never needed again, and the frame buffer can
// drop rows below keep_frame.  Memory therefore stays bounded for arbitrarily
// long sessions instead of growing with the audio.
//
// Two availability horizons exist:
//   - computable_frames(): frames whose full window is present.  While
//     streaming only these may be computed; their values can never change
//     again, because later audio only affects later frames.
//   - total_frames(): total_samples / hop, the file-mode frame count.  At
//     flush the tail frames in between are computed with zero-fill beyond
//     end-of-stream, exactly like a file that ends here.
class StreamingMel {
public:
    explicit StreamingMel(const MelFrontend& frontend) : fe_(frontend) {}

    // Append raw 16 kHz mono samples (float, |x| <= 1).
    void append(const float* samples, std::size_t n) {
        signal_.reserve(signal_.size() + n);
        for (std::size_t i = 0; i < n; ++i) {
            const float raw = samples[i];
            // First sample of the stream maps to itself, like prepare().
            signal_.push_back(have_last_ ? raw - 0.97f * last_ : raw);
            last_ = raw;
            have_last_ = true;
            ++total_;
        }
    }

    std::int64_t total_samples() const { return total_; }

    // Mel frames whose full 25 ms window is already present.  Frame f reads
    // absolute samples [f*hop - 200, f*hop + 200), so it needs
    // total_samples >= f*hop + 200.
    int computable_frames() const {
        const std::int64_t limit = total_ - HeadPad();
        return limit < 0 ? 0 : static_cast<int>(limit / MelFrontend::kHop);
    }

    // File-mode frame count for the audio seen so far.
    int total_frames() const { return static_cast<int>(total_ / MelFrontend::kHop); }

    // Compute frames [ready(), target).  With flush=false the target must not
    // exceed computable_frames(); with flush=true it must not exceed
    // total_frames() (tail frames use zero-fill beyond end of stream, like a
    // file ending here).
    void advance(int target, bool flush) {
        if (target <= ready_) return;
        const int limit = flush ? total_frames() : computable_frames();
        if (target > limit) {
            throw std::runtime_error("StreamingMel::advance: target beyond " +
                                     std::string(flush ? "total" : "computable") + " frames");
        }
        const std::int64_t from = ready_;
        const std::int64_t rows = static_cast<std::int64_t>(target) - from;
        feats_.resize(static_cast<std::size_t>(from - base_ + rows) * MelFrontend::kMels);
        fe_.compute_range_view(signal_.empty() ? nullptr : signal_.data(), static_cast<int>(signal_base_),
                               static_cast<int>(signal_.size()), static_cast<int>(from), target,
                               feats_.data() + static_cast<std::size_t>(from - base_) * MelFrontend::kMels);
        ready_ = target;
    }

    int ready() const { return ready_; }  // frames [base_, ready_) are available
    int base() const { return base_; }

    // Pointer to the 128-float row of an absolute frame, or nullptr when the
    // frame was trimmed away.
    const float* row(int frame) const {
        if (frame < base_ || frame >= ready_) return nullptr;
        return feats_.data() + static_cast<std::size_t>(frame - base_) * MelFrontend::kMels;
    }

    // Drop frames below keep_from and the samples they will never need again.
    // keep_from is clamped into [base_, ready_]; calling with keep_from <=
    // base() is a no-op.
    void trim(int keep_from) {
        keep_from = std::max(keep_from, base_);
        if (keep_from <= base_) return;
        if (keep_from > ready_) keep_from = ready_;
        feats_.erase(feats_.begin(),
                     feats_.begin() + static_cast<std::ptrdiff_t>(keep_from - base_) * MelFrontend::kMels);
        base_ = keep_from;
        // The earliest sample any retained or future frame reads is
        // keep_from*hop - 200; everything before that is dead.
        const std::int64_t sample_keep = std::max<std::int64_t>(0, static_cast<std::int64_t>(keep_from) * MelFrontend::kHop - HeadPad());
        if (sample_keep > signal_base_) {
            const std::int64_t drop = sample_keep - signal_base_;
            if (drop >= static_cast<std::int64_t>(signal_.size())) {
                signal_.clear();
            } else {
                signal_.erase(signal_.begin(), signal_.begin() + static_cast<std::ptrdiff_t>(drop));
            }
            signal_base_ = sample_keep;
        }
    }

private:
    static std::int64_t HeadPad() { return MelFrontend::kWin / 2; }  // 200 samples

    const MelFrontend& fe_;
    std::vector<float> signal_;   // pre-emphasized samples [signal_base_, total_)
    std::int64_t signal_base_ = 0;
    std::int64_t total_ = 0;      // absolute sample count appended so far
    float last_ = 0.0f;
    bool have_last_ = false;
    std::vector<float> feats_;   // frames [base_, ready_), 128 floats each
    int base_ = 0;
    int ready_ = 0;
};

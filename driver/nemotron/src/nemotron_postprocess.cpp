#include "nemotron_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
struct Segment {
    float start;
    float end;
};

std::vector<Segment> merge_overlap_segments(std::vector<Segment> segments) {
    if (segments.size() <= 1) return segments;
    std::sort(segments.begin(), segments.end(), [](const Segment& a, const Segment& b) { return a.start < b.start; });
    std::vector<Segment> merged;
    merged.push_back(segments[0]);
    for (std::size_t i = 1; i < segments.size(); ++i) {
        if (segments[i].start <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, segments[i].end);
        } else {
            merged.push_back(segments[i]);
        }
    }
    return merged;
}

// Hysteresis binarization (onset >= offset branch of NeMo's vectorized version).
std::vector<Segment> binarize(const float* sequence, int num_frames, const PostProcessingParams& params) {
    std::vector<Segment> segments;
    if (num_frames <= 0) return segments;

    std::vector<char> above(num_frames, 0);
    int last_event_position = 0;  // 0 means "no event yet" (state false)
    bool last_event_state = false;
    for (int t = 0; t < num_frames; ++t) {
        const float value = sequence[t];
        const bool force_on = value > params.onset;
        const bool force_off = value < params.offset;
        if (force_on || force_off) {
            last_event_position = t + 1;
            last_event_state = force_on;
        }
        above[t] = last_event_position > 0 ? last_event_state : false;
    }

    std::vector<int> starts, ends;
    if (above[0]) starts.push_back(0);
    for (int t = 1; t < num_frames; ++t) {
        if (!above[t - 1] && above[t]) starts.push_back(t);
        if (above[t - 1] && !above[t]) ends.push_back(t);
    }
    if (above[num_frames - 1]) ends.push_back(num_frames);

    const int count = static_cast<int>(std::min(starts.size(), ends.size()));
    for (int i = 0; i < count; ++i) {
        float start = static_cast<float>(starts[i]) * params.frame_length_in_sec - params.pad_onset;
        float end = static_cast<float>(ends[i]) * params.frame_length_in_sec + params.pad_offset;
        start = std::max(start, 0.0f);
        if (end > start) segments.push_back({start, end});
    }
    if (params.pad_onset > 0.0f || params.pad_offset > 0.0f) segments = merge_overlap_segments(segments);
    return segments;
}
}  // namespace

void preds_to_rttm_lines(const std::vector<float>& preds, int num_frames, int num_speakers, const std::string& uri,
                         const PostProcessingParams& params, std::vector<std::string>& lines_out) {
    for (int spk = 0; spk < num_speakers; ++spk) {
        std::vector<float> column(static_cast<std::size_t>(num_frames));
        for (int t = 0; t < num_frames; ++t) column[t] = preds[static_cast<std::size_t>(t) * num_speakers + spk];
        const std::vector<Segment> segments = merge_overlap_segments(binarize(column.data(), num_frames, params));
        for (const Segment& segment : segments) {
            const float duration = segment.end - segment.start;
            if (duration <= 0.0f) continue;
            char buffer[256];
            std::snprintf(buffer, sizeof(buffer), "SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%d <NA>", uri.c_str(),
                          segment.start, duration, spk);
            lines_out.emplace_back(buffer);
        }
    }
}

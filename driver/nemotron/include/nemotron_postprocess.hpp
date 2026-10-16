#pragma once

#include <string>
#include <vector>

// NeMo Nemotron-3-Diarization post-processing (10 ms output frames): frame probabilities -> RTTM lines.
// Mirrors predlist_to_timestamps + generate_diarization_output_lines with the
// default parameters (onset = offset = 0.5, no padding, no duration filters).
struct PostProcessingParams {
    float onset = 0.5f;
    float offset = 0.5f;
    float pad_onset = 0.0f;
    float pad_offset = 0.0f;
    float min_duration_on = 0.0f;
    float min_duration_off = 0.0f;
    float frame_length_in_sec = 0.01f;
};

// Appends "SPEAKER <uri> 1 <start> <dur> <NA> <NA> speaker_k <NA>" lines.
void preds_to_rttm_lines(const std::vector<float>& preds, int num_frames, int num_speakers, const std::string& uri,
                         const PostProcessingParams& params, std::vector<std::string>& lines_out);

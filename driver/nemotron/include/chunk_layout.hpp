#pragma once

#include <algorithm>
#include <vector>

#include "nemotron_state.hpp"

// Chunk layout shared by the file-mode diarizer (nemotron_diarizer.cpp) and
// the streaming worker (stream_worker.cpp).  Keeping the two paths through
// one definition is what makes the streaming output bit-comparable to the
// file-mode RTTM: given the same audio, both produce the same chunk
// boundaries, the same left/right context and therefore the same predictions.
//
// All frame indices are 10 ms mel frames (MelFrontend hop 160 @ 16 kHz).
struct DiarChunk {
    int start = 0;    // first mel frame of the chunk body
    int left = 0;    // left-context mel frames actually used
    int end = 0;     // one past the last mel frame of the body
    int right = 0;   // right-context mel frames actually used
};

// File-mode layout (upstream SortformerDiarizer::run): the full chunk list
// for a waveform with `raw_frames` mel frames.
inline std::vector<DiarChunk> BuildChunkLayout(int raw_frames, const SortformerConfig& config) {
    std::vector<DiarChunk> chunks;
    const int subsampling = config.subsampling_factor;
    int cursor = 0;
    while (cursor < raw_frames) {
        DiarChunk chunk;
        chunk.start = cursor;
        chunk.left = std::min(config.chunk_left_context * subsampling, cursor);
        chunk.end = std::min(cursor + config.chunk_len * subsampling, raw_frames);
        chunk.right = std::min(config.chunk_right_context * subsampling, raw_frames - chunk.end);
        chunks.push_back(chunk);
        cursor = chunk.end;
    }
    return chunks;
}

// Streaming layout: produce the next chunk given the mel frames available so
// far.  While streaming (flush=false) a chunk is only produced once its full
// body AND full right context are available, which makes it identical to the
// same chunk in file mode.  At flush (frames_available = final raw frame
// count) the tail is split exactly like BuildChunkLayout, including partial
// bodies and clipped right context.
//
// Latency note: with the low-latency profile (chunk_len 6, right context 4,
// 80 ms per frame) a chunk needs 0.48 s of new audio plus 0.32 s of look-ahead,
// i.e. labels for a moment t are available at about t + 0.8 s.
inline bool NextChunk(int cursor, int frames_available, const SortformerConfig& config, bool flush, DiarChunk* out) {
    const int subsampling = config.subsampling_factor;
    if (!flush) {
        const int body = config.chunk_len * subsampling;
        const int right = config.chunk_right_context * subsampling;
        if (frames_available < cursor + body + right) return false;
        out->start = cursor;
        out->left = std::min(config.chunk_left_context * subsampling, cursor);
        out->end = cursor + body;
        out->right = right;
        return true;
    }
    if (cursor >= frames_available) return false;
    out->start = cursor;
    out->left = std::min(config.chunk_left_context * subsampling, cursor);
    out->end = std::min(cursor + config.chunk_len * subsampling, frames_available);
    out->right = std::min(config.chunk_right_context * subsampling, frames_available - out->end);
    return true;
}

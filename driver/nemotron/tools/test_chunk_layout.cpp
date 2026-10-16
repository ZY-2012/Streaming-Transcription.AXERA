// Host-side parity test (no NPU): the streaming chunk sequence produced by
// NextChunk under arbitrary arrival patterns, finished by a flush, must equal
// the file-mode chunk list from BuildChunkLayout exactly - same order, same
// start/left/end/right.  This is the invariant that makes the streaming
// worker's predictions identical to nemotron_ax650's file-mode output.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "chunk_layout.hpp"

namespace {

int failures = 0;

void check_one(const char* name, const SortformerConfig& config, int raw_frames, std::uint32_t seed) {
    const std::vector<DiarChunk> file_chunks = BuildChunkLayout(raw_frames, config);

    std::mt19937 rng(seed);
    std::vector<DiarChunk> stream_chunks;
    int cursor = 0;
    int avail = 0;
    for (;;) {
        DiarChunk chunk;
        if (NextChunk(cursor, avail, config, false, &chunk)) {
            stream_chunks.push_back(chunk);
            cursor = chunk.end;
        } else if (avail < raw_frames) {
            // Simulate PCM arriving in arbitrary small jumps (1..137 frames,
            // i.e. 10 ms .. ~1.4 s), like a realtime feeder would.
            avail += 1 + static_cast<int>(rng() % 137);
            if (avail > raw_frames) avail = raw_frames;
        } else {
            break;  // input stream complete, flush the tail below
        }
    }
    DiarChunk chunk;
    while (NextChunk(cursor, raw_frames, config, true, &chunk)) {
        stream_chunks.push_back(chunk);
        cursor = chunk.end;
    }

    bool ok = stream_chunks.size() == file_chunks.size();
    if (ok) {
        for (std::size_t i = 0; i < file_chunks.size(); ++i) {
            if (stream_chunks[i].start != file_chunks[i].start ||
                stream_chunks[i].left != file_chunks[i].left ||
                stream_chunks[i].end != file_chunks[i].end ||
                stream_chunks[i].right != file_chunks[i].right) {
                ok = false;
                break;
            }
        }
    }
    if (!ok) {
        ++failures;
        std::printf("FAIL %s raw_frames=%d: file=%zu chunks, stream=%zu chunks\n", name, raw_frames,
                    file_chunks.size(), stream_chunks.size());
        const std::size_t show = std::min(file_chunks.size(), stream_chunks.size());
        for (std::size_t i = 0; i < show; ++i) {
            if (stream_chunks[i].start != file_chunks[i].start || stream_chunks[i].left != file_chunks[i].left ||
                stream_chunks[i].end != file_chunks[i].end || stream_chunks[i].right != file_chunks[i].right) {
                std::printf("  first mismatch at chunk %zu: file(%d,%d,%d,%d) stream(%d,%d,%d,%d)\n", i,
                            file_chunks[i].start, file_chunks[i].left, file_chunks[i].end, file_chunks[i].right,
                            stream_chunks[i].start, stream_chunks[i].left, stream_chunks[i].end,
                            stream_chunks[i].right);
                break;
            }
        }
    }
}

}  // namespace

int main() {
    SortformerConfig lowlat;
    lowlat.chunk_len = 6; lowlat.chunk_left_context = 0; lowlat.chunk_right_context = 4;
    lowlat.fifo_len = 264; lowlat.spkcache_len = 264; lowlat.spkcache_update_period = 222;

    SortformerConfig offline;  // library defaults
    SortformerConfig lefty;    // exercises non-zero left context + trimming
    lefty.chunk_len = 10; lefty.chunk_left_context = 2; lefty.chunk_right_context = 4;
    lefty.fifo_len = 40; lefty.spkcache_len = 264; lefty.spkcache_update_period = 300;

    const SortformerConfig* configs[] = {&lowlat, &offline, &lefty};
    const char* names[] = {"lowlat", "offline", "lefty"};

    const int raw_frames[] = {0, 1, 7, 47, 48, 79, 80, 81, 95, 96, 2719, 2720, 12345};
    for (int c = 0; c < 3; ++c) {
        for (int raw : raw_frames) {
            for (std::uint32_t seed = 0; seed < 4; ++seed)
                check_one(names[c], *configs[c], raw, seed * 7919 + c * 104729 + raw);
        }
        // long random sweeps
        std::mt19937 rng(1234 + c);
        for (int i = 0; i < 200; ++i)
            check_one(names[c], *configs[c], static_cast<int>(rng() % 50000), rng());
    }

    if (failures == 0) {
        std::printf("chunk layout parity: OK\n");
        return 0;
    }
    std::printf("chunk layout parity: %d FAILURES\n", failures);
    return 1;
}

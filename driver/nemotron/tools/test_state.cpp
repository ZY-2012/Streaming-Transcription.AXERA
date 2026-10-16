// Host-side streaming_update parity tool (no BSP).
//
// Input file layout (little-endian):
//   magic "SFS1" (4 bytes)
//   int32: n_spk, dim, chunk_frames, T_spk, T_fifo, n_sil
//   float32: spkcache[T_spk*dim], spkcache_preds[T_spk*n_spk],
//            fifo[T_fifo*dim], fifo_preds[T_fifo*n_spk], mean_sil[dim],
//            chunk[chunk_frames*dim],
//            preds[(T_spk+T_fifo+chunk_frames)*n_spk]
//   int32: lc, rc, spkcache_compressed
//
// Output file layout:
//   int32: n_spk, dim, chunk_len, T_spk_new, T_fifo_new, n_sil_new, compressed
//   float32: chunk_preds[chunk_len*n_spk], spkcache[..], spkcache_preds[..],
//            fifo[..], fifo_preds[..], mean_sil[dim]
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nemotron_state.hpp"
#include "npy_reader.hpp"

namespace {
template <typename T>
bool read(std::ifstream& in, T* data, std::size_t count) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<bool>(in);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_state in.bin out.bin [sil_emb.npy]\n");
        return 1;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "SFS1") { std::fprintf(stderr, "bad magic\n"); return 1; }

    int32_t header[11];
    if (!read(in, header, 11)) return 1;
    const int n_spk = header[0], dim = header[1], chunk_frames = header[2];
    const int t_spk = header[3], t_fifo = header[4];
    const int64_t n_sil = header[5];
    const int fifo_len = header[6], spkcache_len = header[7];
    const int update_period = header[8], sil_frames_per_spk = header[9];
    const int use_learnable_sil = header[10];

    std::vector<float> spkcache(t_spk * dim), spkcache_preds(t_spk * n_spk);
    std::vector<float> fifo(t_fifo * dim), fifo_preds(t_fifo * n_spk);
    std::vector<float> mean_sil(dim);
    std::vector<float> chunk(chunk_frames * dim);
    std::vector<float> preds((t_spk + t_fifo + chunk_frames) * n_spk);

    if (!read(in, spkcache.data(), spkcache.size()) || !read(in, spkcache_preds.data(), spkcache_preds.size()) ||
        !read(in, fifo.data(), fifo.size()) || !read(in, fifo_preds.data(), fifo_preds.size()) ||
        !read(in, mean_sil.data(), mean_sil.size()) || !read(in, chunk.data(), chunk.size()) ||
        !read(in, preds.data(), preds.size())) {
        std::fprintf(stderr, "truncated input\n");
        return 1;
    }
    int32_t tail[3];
    if (!read(in, tail, 3)) return 1;

    SortformerConfig config;
    config.num_speakers = n_spk;
    config.fc_d_model = dim;
    config.fifo_len = fifo_len;
    config.spkcache_len = spkcache_len;
    config.spkcache_update_period = update_period;
    config.spkcache_sil_frames_per_spk = sil_frames_per_spk;
    config.use_learnable_sil_emb = use_learnable_sil != 0;
    if (argc > 3) {
        std::vector<int> shape;
        if (!load_npy_float32(argv[3], config.learnable_sil_emb, shape) ||
            config.learnable_sil_emb.size() != static_cast<std::size_t>(dim)) {
            std::fprintf(stderr, "cannot load %s\n", argv[3]);
            return 1;
        }
        config.use_learnable_sil_emb = true;
    }
    StreamingState state;
    state.spkcache = spkcache;
    state.spkcache_preds = spkcache_preds;
    state.fifo = fifo;
    state.fifo_preds = fifo_preds;
    state.mean_sil_emb = mean_sil;
    state.n_sil_frames = n_sil;
    state.spkcache_compressed = tail[2] != 0;

    std::vector<float> chunk_preds;
    streaming_update(config, state, chunk, preds, tail[0], tail[1], chunk_preds);

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    int32_t out_header[7] = {n_spk,
                             dim,
                             static_cast<int32_t>(chunk_preds.size()) / n_spk,
                             state.spkcache_frames(),
                             state.fifo_frames(),
                             static_cast<int32_t>(state.n_sil_frames),
                             state.spkcache_compressed ? 1 : 0};
    out.write(reinterpret_cast<const char*>(out_header), sizeof(out_header));
    out.write(reinterpret_cast<const char*>(chunk_preds.data()), chunk_preds.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(state.spkcache.data()), state.spkcache.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(state.spkcache_preds.data()), state.spkcache_preds.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(state.fifo.data()), state.fifo.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(state.fifo_preds.data()), state.fifo_preds.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(state.mean_sil_emb.data()), state.mean_sil_emb.size() * sizeof(float));
    return 0;
}

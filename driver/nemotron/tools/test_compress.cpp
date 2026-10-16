// Host-side _compress_spkcache parity tool (no BSP).
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nemotron_state.hpp"
#include "npy_reader.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_compress in.bin out.bin [sil_emb.npy]\n");
        return 1;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot open input\n"); return 1; }
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "SFC1") { std::fprintf(stderr, "bad magic\n"); return 1; }

    int32_t header[6];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    const int n_spk = header[0], dim = header[1], n_frames = header[2];
    const int spkcache_len = header[3], sil_frames_per_spk = header[4];
    const int use_learnable_sil = header[5];

    std::vector<float> emb(static_cast<std::size_t>(n_frames) * dim), preds(static_cast<std::size_t>(n_frames) * n_spk), mean(dim);
    in.read(reinterpret_cast<char*>(emb.data()), static_cast<std::streamsize>(emb.size() * sizeof(float)));
    in.read(reinterpret_cast<char*>(preds.data()), static_cast<std::streamsize>(preds.size() * sizeof(float)));
    in.read(reinterpret_cast<char*>(mean.data()), static_cast<std::streamsize>(mean.size() * sizeof(float)));

    SortformerConfig config;
    config.num_speakers = n_spk;
    config.fc_d_model = dim;
    config.spkcache_len = spkcache_len;
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

    std::vector<float> spkcache, spkcache_preds, scores;
    std::vector<int> topk;
    std::vector<char> disabled;
    debug_compress(config, emb, preds, mean, spkcache, spkcache_preds, scores, topk, disabled);

    const int selected = static_cast<int>(topk.size());
    const int padded_frames = static_cast<int>(scores.size()) / n_spk;
    std::ofstream out(argv[2], std::ios::binary);
    int32_t out_header[4] = {selected, padded_frames, n_spk, dim};
    out.write(reinterpret_cast<const char*>(out_header), sizeof(out_header));
    out.write(reinterpret_cast<const char*>(scores.data()), scores.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(spkcache.data()), spkcache.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(spkcache_preds.data()), spkcache_preds.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(topk.data()), topk.size() * sizeof(int));
    out.write(reinterpret_cast<const char*>(disabled.data()), disabled.size() * sizeof(char));
    return 0;
}

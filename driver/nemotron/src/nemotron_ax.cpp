// Nemotron-3-Diarization streaming diarization on AX650N: wav -> RTTM.
//
//   ./nemotron_ax650 --preencode models/preencode.axmodel \
//                    --encoder   models/encoder.axmodel \
//                    --sil-emb   models/learnable_sil_emb.npy \
//                    --assets    models \
//                    --wav meeting.wav --rttm out.rttm --threads 8
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_reader.hpp"
#include "mel_frontend.hpp"
#include "npy_reader.hpp"
#include "nemotron_diarizer.hpp"
#include "nemotron_postprocess.hpp"

namespace {
void usage() {
    std::fprintf(stderr,
                 "usage: nemotron_ax650 --preencode <preencode.axmodel> --encoder <encoder.axmodel>\n"
                 "                      --wav <input.wav> --sil-emb <learnable_sil_emb.npy>\n"
                 "                      [--assets <dir with feature_window/mel_filterbank.npy>]\n"
                 "                      [--rttm <out.rttm>] [--dump-preds <out.bin>] [--threads N]\n"
                 "                      [--chunk-len N] [--chunk-left-context N] [--chunk-right-context N]\n"
                 "                      [--fifo-len N] [--spkcache-len N] [--spkcache-update-period N]\n");
}

std::string arg_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}
}  // namespace

int main(int argc, char** argv) {
    std::string preencode_path, encoder_path, wav_path, rttm_path, preds_path;
    std::string sil_emb_path, assets_dir;
    int threads = 8;
    SortformerConfig config;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--preencode") preencode_path = arg_value(i, argc, argv);
            else if (arg == "--encoder") encoder_path = arg_value(i, argc, argv);
            else if (arg == "--wav") wav_path = arg_value(i, argc, argv);
            else if (arg == "--rttm") rttm_path = arg_value(i, argc, argv);
            else if (arg == "--dump-preds") preds_path = arg_value(i, argc, argv);
            else if (arg == "--sil-emb") sil_emb_path = arg_value(i, argc, argv);
            else if (arg == "--assets") assets_dir = arg_value(i, argc, argv);
            else if (arg == "--threads") threads = std::atoi(arg_value(i, argc, argv).c_str());
            else if (arg == "--chunk-len") config.chunk_len = std::atoi(arg_value(i, argc, argv).c_str());
            else if (arg == "--chunk-left-context") config.chunk_left_context = std::atoi(arg_value(i, argc, argv).c_str());
            else if (arg == "--chunk-right-context") config.chunk_right_context = std::atoi(arg_value(i, argc, argv).c_str());
            else if (arg == "--fifo-len") config.fifo_len = std::atoi(arg_value(i, argc, argv).c_str());
            else if (arg == "--spkcache-len") config.spkcache_len = std::atoi(arg_value(i, argc, argv).c_str());
            else if (arg == "--spkcache-update-period") config.spkcache_update_period = std::atoi(arg_value(i, argc, argv).c_str());
            else if (arg == "-h" || arg == "--help") { usage(); return 0; }
            else throw std::runtime_error("unknown argument: " + arg);
        }
        if (preencode_path.empty() || encoder_path.empty() || wav_path.empty()) {
            usage();
            return 1;
        }

        // AOSC compression needs the learnable silence embedding; without it the
        // model falls back to the running silence profile and diverges from NeMo.
        if (!sil_emb_path.empty()) {
            std::vector<float> sil_emb;
            std::vector<int> shape;
            if (!load_npy_float32(sil_emb_path, sil_emb, shape) ||
                sil_emb.size() != static_cast<std::size_t>(config.fc_d_model)) {
                throw std::runtime_error("cannot load learnable silence embedding: " + sil_emb_path);
            }
            config.learnable_sil_emb = sil_emb;
            config.use_learnable_sil_emb = true;
        } else {
            std::fprintf(stderr, "warning: --sil-emb not given; using the running silence profile\n");
        }

        AudioData audio = read_wav_mono(wav_path);
        if (audio.sample_rate != MelFrontend::kSampleRate) {
            audio = resample_audio(audio, MelFrontend::kSampleRate);
        }

        SortformerDiarizer diarizer(preencode_path, encoder_path, config, threads, assets_dir);
        DiarizerReport report;
        const std::vector<float> preds = diarizer.run(audio.samples, report);

        const std::string uri = wav_path.substr(wav_path.find_last_of('/') + 1);
        const std::string stem = uri.substr(0, uri.find_last_of('.'));
        std::vector<std::string> lines;
        PostProcessingParams params;
        preds_to_rttm_lines(preds, report.frames, config.num_speakers, stem, params, lines);
        if (!rttm_path.empty()) {
            std::ofstream out(rttm_path.c_str());
            if (!out) throw std::runtime_error("cannot write " + rttm_path);
            for (const std::string& line : lines) out << line << "\n";
        }
        if (!preds_path.empty()) {
            std::ofstream out(preds_path.c_str(), std::ios::binary);
            if (!out) throw std::runtime_error("cannot write " + preds_path);
            out.write(reinterpret_cast<const char*>(preds.data()),
                      static_cast<std::streamsize>(preds.size() * sizeof(float)));
        }

        // pre-encode waits behind the encoder on the NPU, so only mel is separable.
        const double host_s = std::max(0.0, report.total_seconds - report.mel_seconds - report.enc_seconds);
        std::printf(
            "{\"steps\":%d,\"frames\":%d,\"segments\":%zu,\"audio_s\":%.3f,\"mel_s\":%.3f,\"pre_s\":%.3f,"
            "\"enc_s\":%.3f,\"host_s\":%.3f,\"total_s\":%.3f,\"rtf\":%.5f,\"ms_per_step\":%.3f,\"threads\":%d}\n",
            report.steps, report.frames, lines.size(), report.audio_seconds, report.mel_seconds, report.pre_seconds,
            report.enc_seconds, host_s, report.total_seconds,
            report.audio_seconds > 0 ? report.total_seconds / report.audio_seconds : 0.0,
            report.steps > 0 ? 1000.0 * report.total_seconds / report.steps : 0.0, report.threads);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}

#!/bin/bash
# Host-side checks (no BSP): mel front-end dump + streaming state test.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build_host
g++ -O3 -std=c++11 -pthread \
    -Iinclude -Ithird_party/kissfft \
    tools/check_mel.cpp src/audio_reader.cpp src/mel_frontend.cpp src/npy_reader.cpp \
    third_party/kissfft/kiss_fft.c \
    -o build_host/check_mel

g++ -O3 -std=c++11 -pthread \
    -Iinclude -Ithird_party/kissfft \
    tools/test_state.cpp src/nemotron_state.cpp src/npy_reader.cpp \
    -o build_host/test_state

g++ -O3 -std=c++11 -pthread \
    -Iinclude -Ithird_party/kissfft \
    tools/test_compress.cpp src/nemotron_state.cpp src/npy_reader.cpp \
    -o build_host/test_compress

g++ -O3 -std=c++11 -pthread \
    -Iinclude -Ithird_party/kissfft \
    tools/test_chunk_layout.cpp \
    -o build_host/test_chunk_layout

g++ -O3 -std=c++11 -pthread \
    -Iinclude -Ithird_party/kissfft \
    tools/test_stream_mel.cpp src/audio_reader.cpp src/mel_frontend.cpp src/npy_reader.cpp \
    third_party/kissfft/kiss_fft.c \
    -o build_host/test_stream_mel
echo "built: $PWD/build_host/check_mel $PWD/build_host/test_state $PWD/build_host/test_compress $PWD/build_host/test_chunk_layout $PWD/build_host/test_stream_mel"

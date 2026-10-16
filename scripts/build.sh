#!/usr/bin/env bash
# 整体编译 AX650 实时音频主程序及其运行时组件 (aarch64-linux)
# 目标平台配置见 config/aarch64-linux.ini
# 预编译库位于 driver/apm，链接时只指定库的搜索路径，
# 不设置 rpath，运行时在目标平台通过 LD_LIBRARY_PATH 指定库的路径。

set -euo pipefail

# SDK 根目录和产物目录（绝对路径）
SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${SDK_DIR}/build"

# 交叉编译工具链（与 config/aarch64-linux.ini 保持一致）
CXX="${CXX:-aarch64-linux-gnu-g++}"
CC="${CC:-${CXX%++}cc}"
CXX_PATH="$(command -v "${CXX}" || true)"
CC_PATH="$(command -v "${CC}" || true)"
if [[ -z "${CXX_PATH}" || -z "${CC_PATH}" ]]; then
    echo "Cross compiler not found: CC=${CC}, CXX=${CXX}" >&2
    exit 1
fi
CROSS_SYSROOT="${CROSS_SYSROOT:-}"
CROSS_FLAGS=()
CMAKE_SYSROOT_ARGS=()
if [[ -n "${CROSS_SYSROOT}" ]]; then
    if [[ ! -d "${CROSS_SYSROOT}" ]]; then
        echo "CROSS_SYSROOT is not a directory: ${CROSS_SYSROOT}" >&2
        exit 1
    fi
    CROSS_SYSROOT="$(cd "${CROSS_SYSROOT}" && pwd)"
    CROSS_FLAGS+=("--sysroot=${CROSS_SYSROOT}")
    CMAKE_SYSROOT_ARGS+=("-DCMAKE_SYSROOT=${CROSS_SYSROOT}")
fi

# 头文件与库路径
APM_DIR="${SDK_DIR}/driver/apm"
INCLUDE_DIRS=(
    "-I${APM_DIR}/include"
    "-I${APM_DIR}/include/webrtc-audio-processing-2"
)
LIB_DIR="${APM_DIR}/lib"
ALSA_SYSROOT="${ALSA_SYSROOT:-${CROSS_SYSROOT}}"
ALSA_INCLUDE_DIR="${ALSA_INCLUDE_DIR:-}"
ALSA_LIBRARY="${ALSA_LIBRARY:-}"
mkdir -p "${BUILD_DIR}"

AX650_SDK_ROOT="${AX650_SDK_ROOT:-}"

# Native C++ runtime for the target board.  It parses the yaml config and
# owns the complete audio/VAD/ASR/diarization/MP3 pipeline.
"${CXX}" "${CROSS_FLAGS[@]}" -std=c++17 -O2 \
    -I"${SDK_DIR}/framework" -I"${SDK_DIR}/framework/configui" \
    -I"${SDK_DIR}/driver/ns/include" -I"${SDK_DIR}/driver/vad/include" -I"${SDK_DIR}/driver/asr/sensevoice/include" \
    "${SDK_DIR}/framework/audio_pipeline.cpp" \
    "${SDK_DIR}/framework/configui/apm-config.cpp" \
    -ldl -pthread -Wl,--export-dynamic -Wl,--allow-shlib-undefined \
    -o "${BUILD_DIR}/audio-pipeline"
echo "Build done: ${BUILD_DIR}/audio-pipeline"

if [[ -n "${ALSA_SYSROOT}" && -d "${ALSA_SYSROOT}" ]]; then
    if [[ -z "${ALSA_INCLUDE_DIR}" ]]; then
        for candidate in "${ALSA_SYSROOT}/usr/include" "${ALSA_SYSROOT}/include"; do
            if [[ -r "${candidate}/alsa/asoundlib.h" ]]; then
                ALSA_INCLUDE_DIR="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${ALSA_LIBRARY}" ]]; then
        for candidate in \
            "${ALSA_SYSROOT}/usr/lib/aarch64-linux-gnu/libasound.so" \
            "${ALSA_SYSROOT}/lib/aarch64-linux-gnu/libasound.so" \
            "${ALSA_SYSROOT}/usr/lib/libasound.so" \
            "${ALSA_SYSROOT}/lib/libasound.so"; do
            if [[ -r "${candidate}" ]]; then
                ALSA_LIBRARY="${candidate}"
                break
            fi
        done
    fi
fi
ALSA_INCLUDE_DIR="${ALSA_INCLUDE_DIR:-/usr/include}"
ALSA_LIBRARY="${ALSA_LIBRARY:-/usr/lib/aarch64-linux-gnu/libasound.so}"
if [[ -r "${ALSA_INCLUDE_DIR}/alsa/asoundlib.h" && -r "${ALSA_LIBRARY}" ]]; then
    "${CXX}" "${CROSS_FLAGS[@]}" -std=c++17 -O2 \
        "${INCLUDE_DIRS[@]}" -I"${ALSA_INCLUDE_DIR}" \
        -I"${SDK_DIR}/driver/alsa_briage" -I"${SDK_DIR}/framework" -I"${SDK_DIR}/framework/configui" \
        -DWEBRTC_LIBRARY_IMPL -DWEBRTC_POSIX \
        "${SDK_DIR}/driver/alsa_briage/run-alsa-apm.cpp" \
        "${SDK_DIR}/driver/alsa_briage/alsa-apm-backend.cpp" \
        "${SDK_DIR}/framework/configui/apm-config.cpp" \
        -L"${LIB_DIR}" -lwebrtc-audio-processing-2 "${ALSA_LIBRARY}" -lrt -ldl -pthread \
        -Wl,--allow-shlib-undefined \
        -o "${BUILD_DIR}/run-alsa-apm"
    echo "Build done: ${BUILD_DIR}/run-alsa-apm"
else
    echo "ALSA headers/library not found; skipped run-alsa-apm." >&2
    echo "Set ALSA_INCLUDE_DIR and ALSA_LIBRARY, or ALSA_SYSROOT." >&2
fi

if [[ -d "${AX650_SDK_ROOT}/include" && -d "${AX650_SDK_ROOT}/lib" ]]; then
    # Speaker diarization runs as a persistent worker; the eight axmodels are
    # loaded once at pipeline start.  The vendored engine (MIT) lives under
    # driver/pyannote and is serialized with the NPU2 Cell lock at runtime.
    DIAR_DIR="${SDK_DIR}/driver/pyannote"
    "${CC}" "${CROSS_FLAGS[@]}" -O2 \
        -c "${DIAR_DIR}/third_party/kissfft/kiss_fft.c" \
        -o "${BUILD_DIR}/diar-kiss_fft.o"
    "${CC}" "${CROSS_FLAGS[@]}" -O2 \
        -c "${DIAR_DIR}/third_party/kissfft/kiss_fftr.c" \
        -o "${BUILD_DIR}/diar-kiss_fftr.o"
    "${CXX}" "${CROSS_FLAGS[@]}" -std=c++17 -O2 -fopenmp -DAX650 \
        -I"${DIAR_DIR}/src" -I"${DIAR_DIR}/third_party/kissfft" \
        -I"${AX650_SDK_ROOT}/include" \
        "${DIAR_DIR}/src/diar_worker.cpp" \
        "${DIAR_DIR}/src/diar_engine.cpp" \
        "${DIAR_DIR}/src/EngineWrapper_axera.cpp" \
        "${DIAR_DIR}/src/speaker_table.cpp" \
        "${BUILD_DIR}/diar-kiss_fft.o" "${BUILD_DIR}/diar-kiss_fftr.o" \
        -L"${AX650_SDK_ROOT}/lib" -lax_sys -lax_engine -lax_interpreter -pthread \
        -o "${BUILD_DIR}/diar-worker"
    echo "Build done: ${BUILD_DIR}/diar-worker"

    # Nemotron-3 end-to-end streaming diarization (true-streaming engine).
    # nemotron_ax650 is the upstream file-mode binary, kept as the parity
    # oracle for stream-diar-worker (same wav must produce identical
    # --dump-preds output); the worker adds the resident streaming protocol.
    NEMO_DIR="${SDK_DIR}/driver/nemotron"
    "${CC}" "${CROSS_FLAGS[@]}" -O2 \
        -c "${NEMO_DIR}/third_party/kissfft/kiss_fft.c" \
        -o "${BUILD_DIR}/nemotron-kiss_fft.o"
    NEMO_CXXFLAGS=(-std=c++17 -O2
        -I"${NEMO_DIR}/include" -I"${NEMO_DIR}/third_party/kissfft"
        -I"${AX650_SDK_ROOT}/include")
    NEMO_LIBS=(-L"${AX650_SDK_ROOT}/lib" -lax_sys -lax_engine -lax_interpreter -pthread -ldl)
    NEMO_SRCS=("${NEMO_DIR}/src/nemotron_ax.cpp"
        "${NEMO_DIR}/src/nemotron_diarizer.cpp"
        "${NEMO_DIR}/src/ax_engine_runner.cpp"
        "${NEMO_DIR}/src/mel_frontend.cpp"
        "${NEMO_DIR}/src/nemotron_state.cpp"
        "${NEMO_DIR}/src/nemotron_postprocess.cpp"
        "${NEMO_DIR}/src/audio_reader.cpp"
        "${NEMO_DIR}/src/npy_reader.cpp"
        "${BUILD_DIR}/nemotron-kiss_fft.o")
    "${CXX}" "${CROSS_FLAGS[@]}" "${NEMO_CXXFLAGS[@]}" "${NEMO_SRCS[@]}" "${NEMO_LIBS[@]}" \
        -o "${BUILD_DIR}/nemotron_ax650"
    echo "Build done: ${BUILD_DIR}/nemotron_ax650"

    "${CXX}" "${CROSS_FLAGS[@]}" "${NEMO_CXXFLAGS[@]}" \
        "${NEMO_DIR}/src/stream_worker.cpp" \
        "${NEMO_DIR}/src/ax_engine_runner.cpp" \
        "${NEMO_DIR}/src/mel_frontend.cpp" \
        "${NEMO_DIR}/src/nemotron_state.cpp" \
        "${NEMO_DIR}/src/nemotron_postprocess.cpp" \
        "${NEMO_DIR}/src/audio_reader.cpp" \
        "${NEMO_DIR}/src/npy_reader.cpp" \
        "${BUILD_DIR}/nemotron-kiss_fft.o" \
        "${NEMO_LIBS[@]}" \
        -o "${BUILD_DIR}/stream-diar-worker"
    echo "Build done: ${BUILD_DIR}/stream-diar-worker"

    AX650_SDK_ROOT="${AX650_SDK_ROOT}" CROSS_SYSROOT="${CROSS_SYSROOT}" CXX="${CXX}" \
        "${SDK_DIR}/driver/ns/build-fastenhance.sh"
else
    echo "AX650_SDK_ROOT is unavailable; skipped AX650 runtime components." >&2
fi
echo "在目标平台运行时请指定库的路径，例如："
echo "  ./scripts/run-target.sh"

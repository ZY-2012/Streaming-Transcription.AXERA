#!/usr/bin/env bash
# Launch the true-streaming transcription pipeline on the AX650 board.
#   CONFIG=./config/streaming-meeting.yaml ./scripts/run-target.sh
set -euo pipefail
SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SDK_DIR}"
export LD_LIBRARY_PATH="${SDK_DIR}/build:${SDK_DIR}/driver/apm/lib:${SDK_DIR}/driver/vad/lib:${SDK_DIR}/driver/asr/sensevoice/lib:${SDK_DIR}/driver/asr/lib:/soc/lib:/opt/lib:${LD_LIBRARY_PATH:-}"
# Host-side clustering in the pyannote fallback links OpenMP; keep one team so
# it does not compete with the realtime APM/VAD threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# SenseVoice resources and its AX650 front-end dependencies live together.
export AX_SENSEVOICE_SHARE_DIR="${AX_SENSEVOICE_SHARE_DIR:-${SDK_DIR}/driver/asr/sensevoice}"
CONFIG_PATH="${CONFIG:-${SDK_DIR}/config/streaming-meeting.yaml}"
echo "Starting AX650 true-streaming transcription pipeline: ${CONFIG_PATH}" >&2
exec build/audio-pipeline --config "${CONFIG_PATH}" "$@"

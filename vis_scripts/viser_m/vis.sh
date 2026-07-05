#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sequence_name>" >&2
  exit 1
fi

SEQ="$1"

HMR_TYPE="${HMR_TYPE:-gv}"
SCENE_FILE="${SCENE_FILE:-../../results/output/scene/${SEQ}_${HMR_TYPE}_sgd_cvd_hr.npz}"
SCENE_PRIOR_BASE_PATH="${SCENE_PRIOR_BASE_PATH:-}"
SCENE_CAMERA_ROOT="${SCENE_CAMERA_ROOT:-}"
SCENE_OUTPUT_DIR="${SCENE_OUTPUT_DIR:-}"
HMR_RESULTS_ROOT="${HMR_RESULTS_ROOT:-}"
PORT="${PORT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEGMENT_MIN_FRAMES="${SEGMENT_MIN_FRAMES:-}"
FUSION_INTERVAL="${FUSION_INTERVAL:-}"
SEGMENT_MODE="${SEGMENT_MODE:-}"

# default OFF
SAVE_MODE="${SAVE_MODE:-off}"
USE_CONTACT="${USE_CONTACT:-off}"
SAVE_CLUSTERING="${SAVE_CLUSTERING:-on}"

case "${SAVE_MODE,,}" in
  on|true|1|yes|y)
    SAVE_FLAG="--save_mode"
    ;;
  off|false|0|no|n|"")
    SAVE_FLAG="--no-save_mode"
    ;;
  *)
    echo "Invalid SAVE_MODE='$SAVE_MODE' (use on/off or true/false)" >&2
    exit 2
    ;;
esac

case "${USE_CONTACT,,}" in
  on|true|1|yes|y)
    CONTACT_FLAG="--use_contact"
    ;;
  off|false|0|no|n|"")
    CONTACT_FLAG="--no-use_contact"
    ;;
  *)
    echo "Invalid USE_CONTACT='$USE_CONTACT' (use on/off or true/false)" >&2
    exit 2
    ;;
esac

case "${SAVE_CLUSTERING,,}" in
  on|true|1|yes|y)
    CLUSTERING_FLAG="--save-clustering"
    ;;
  off|false|0|no|n|"")
    CLUSTERING_FLAG="--no-save-clustering"
    ;;
  *)
    echo "Invalid SAVE_CLUSTERING='$SAVE_CLUSTERING' (use on/off or true/false)" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cmd=(
  "$PYTHON_BIN" visualizer_megasam.py
  --data "$SCENE_FILE"
  --hmr-type "$HMR_TYPE"
  --sequence-name "$SEQ"
  "$SAVE_FLAG"
  "$CONTACT_FLAG"
  "$CLUSTERING_FLAG"
)
if [[ -n "$SCENE_PRIOR_BASE_PATH" ]]; then
  cmd+=(--scene-prior-base-path "$SCENE_PRIOR_BASE_PATH")
fi
if [[ -n "$SCENE_CAMERA_ROOT" ]]; then
  cmd+=(--scene-camera-root "$SCENE_CAMERA_ROOT")
fi
if [[ -n "$SCENE_OUTPUT_DIR" ]]; then
  cmd+=(--scene-output-dir "$SCENE_OUTPUT_DIR")
fi
if [[ -n "$HMR_RESULTS_ROOT" ]]; then
  cmd+=(--hmr-results-root "$HMR_RESULTS_ROOT")
fi
if [[ -n "$PORT" ]]; then
  cmd+=(--port "$PORT")
fi
if [[ -n "$SEGMENT_MIN_FRAMES" ]]; then
  cmd+=(--segment-min-frames "$SEGMENT_MIN_FRAMES")
fi
if [[ -n "$FUSION_INTERVAL" ]]; then
  cmd+=(--fusion-interval "$FUSION_INTERVAL")
fi
if [[ -n "$SEGMENT_MODE" ]]; then
  cmd+=(--segment-mode "$SEGMENT_MODE")
fi

"${cmd[@]}"

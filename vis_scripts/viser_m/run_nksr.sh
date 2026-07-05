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
NKSR_ENV="${NKSR_ENV:-crisp_nksr}"
NKSR_DETAIL_LEVEL="${NKSR_DETAIL_LEVEL:-0.1}"
NKSR_MISE_ITER="${NKSR_MISE_ITER:-1}"
NKSR_MAX_INPUT_POINTS="${NKSR_MAX_INPUT_POINTS:--1}"
NKSR_CHUNK_SIZE="${NKSR_CHUNK_SIZE:--1}"
NKSR_DEVICE="${NKSR_DEVICE:-cuda:0}"
SCENE_ROOT="${SCENE_OUTPUT_DIR:-${SCENE_ARTIFACT_OUTPUT_DIR:-}}"
NKSR_INPUT_NPZ="${NKSR_INPUT_NPZ:-}"
NKSR_OUTPUT_DIR="${NKSR_OUTPUT_DIR:-}"

cmd=(
  conda run -n "$NKSR_ENV" python run_nksr.py
  --sequence-name "$SEQ" \
  --hmr-type "$HMR_TYPE" \
  --detail-level "$NKSR_DETAIL_LEVEL" \
  --mise-iter "$NKSR_MISE_ITER" \
  --max-input-points "$NKSR_MAX_INPUT_POINTS" \
  --chunk-size "$NKSR_CHUNK_SIZE" \
  --device "$NKSR_DEVICE"
)

if [[ -n "$NKSR_INPUT_NPZ" ]]; then
  cmd+=(--input-npz "$NKSR_INPUT_NPZ")
  if [[ -z "$NKSR_OUTPUT_DIR" ]]; then
    input_dir="$(dirname "$NKSR_INPUT_NPZ")"
    if [[ -d "$input_dir/.." ]]; then
      NKSR_OUTPUT_DIR="$(cd "$input_dir/.." && pwd)/nksr"
    fi
  fi
elif [[ -n "$SCENE_ROOT" ]]; then
  cmd+=(--input-npz "$SCENE_ROOT/$SEQ/$HMR_TYPE/nksr_input/pointcloud_world.npz")
  NKSR_OUTPUT_DIR="${NKSR_OUTPUT_DIR:-$SCENE_ROOT/$SEQ/$HMR_TYPE/nksr}"
fi

if [[ -n "$NKSR_OUTPUT_DIR" ]]; then
  cmd+=(--output-dir "$NKSR_OUTPUT_DIR")
fi

exec "${cmd[@]}"

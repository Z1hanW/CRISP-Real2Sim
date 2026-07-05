#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/3_vggt_omega.sh <split_or_path>" >&2
  exit 1
fi

ROOT="${1%/}"
DATA_VIDEOS="${ROOT}_videos"
DATA_IMAGES="${ROOT}_img"

PYTHON_BIN="${VGGT_OMEGA_PYTHON:-${PYTHON_BIN:-python}}"
CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-}"
VGGT_REPO="${VGGT_OMEGA_REPO:-}"
if [[ -z "$VGGT_REPO" && -d "$REPO_ROOT/prep/vggt-omega" ]]; then
  VGGT_REPO="$REPO_ROOT/prep/vggt-omega"
fi
IMAGE_RESOLUTION="${VGGT_OMEGA_IMAGE_RESOLUTION:-512}"
IMAGE_MODE="${VGGT_OMEGA_IMAGE_MODE:-balanced}"
FRAME_STRIDE="${VGGT_OMEGA_FRAME_STRIDE:-1}"
MAX_FRAMES="${VGGT_OMEGA_MAX_FRAMES:-0}"
ENABLE_ALIGNMENT="${VGGT_OMEGA_ENABLE_ALIGNMENT:-0}"
RAW_PRIORS_ROOT="${VGGT_OMEGA_RAW_PRIORS_ROOT:-$REPO_ROOT/results/init/vslam/raw_vggt_omega_priors}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "[vggt_omega] VGGT_OMEGA_CHECKPOINT is required." >&2
  echo "[vggt_omega] Request/download a VGGT-Omega checkpoint, then export VGGT_OMEGA_CHECKPOINT=/path/to/model.pt" >&2
  exit 2
fi

shopt -s nullglob
inputs=("$DATA_VIDEOS"/*.mp4 "$DATA_VIDEOS"/*.avi "$DATA_VIDEOS"/*.mov "$DATA_VIDEOS"/*.mkv "$DATA_VIDEOS"/*.webm)
if (( ${#inputs[@]} == 0 )); then
  inputs=("$DATA_IMAGES"/*/)
fi
shopt -u nullglob

if (( ${#inputs[@]} == 0 )); then
  echo "[vggt_omega] no inputs found under $DATA_VIDEOS or $DATA_IMAGES" >&2
  exit 3
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l)"
  if [[ "$GPU_COUNT" -gt 0 ]]; then
    mapfile -t GPU_IDS < <(seq 0 $((GPU_COUNT - 1)))
  else
    GPU_IDS=("cpu")
  fi
else
  GPU_IDS=("cpu")
fi

GPU_COUNT="${#GPU_IDS[@]}"
mkdir -p "$RAW_PRIORS_ROOT"

echo "[vggt_omega] inputs         : ${#inputs[@]}"
echo "[vggt_omega] checkpoint     : $CHECKPOINT"
echo "[vggt_omega] repo           : ${VGGT_REPO:-<installed package>}"
echo "[vggt_omega] raw priors root: $RAW_PRIORS_ROOT"
echo "[vggt_omega] image settings : resolution=$IMAGE_RESOLUTION mode=$IMAGE_MODE"

run_one() {
  local gpu_id="$1"
  local input_path="$2"
  local device="cuda"
  local -a env_prefix=()
  if [[ "$gpu_id" == "cpu" ]]; then
    device="cpu"
  else
    env_prefix=(env "CUDA_VISIBLE_DEVICES=$gpu_id")
  fi

  local -a cmd=(
    "$PYTHON_BIN"
    "$REPO_ROOT/prep/VGGT_Omega/run_vggt_omega_prior.py"
    --input-path "$input_path"
    --checkpoint "$CHECKPOINT"
    --raw-priors-root "$RAW_PRIORS_ROOT"
    --device "$device"
    --image-resolution "$IMAGE_RESOLUTION"
    --image-mode "$IMAGE_MODE"
    --frame-stride "$FRAME_STRIDE"
    --max-frames "$MAX_FRAMES"
  )
  if [[ -n "$VGGT_REPO" ]]; then
    cmd+=(--repo-path "$VGGT_REPO")
  fi
  case "${ENABLE_ALIGNMENT,,}" in
    1|true|yes|on) cmd+=(--enable-alignment) ;;
  esac

  echo "-> VGGT-Omega GPU $gpu_id | $input_path"
  "${env_prefix[@]}" "${cmd[@]}"
}

worker() {
  local gpu_id="$1"
  shift
  local item
  for item in "$@"; do
    run_one "$gpu_id" "$item"
  done
}

pids=()
for logical_idx in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$logical_idx]}"
  assigned=()
  for (( idx=logical_idx; idx<${#inputs[@]}; idx+=GPU_COUNT )); do
    assigned+=("${inputs[idx]}")
  done
  if (( ${#assigned[@]} > 0 )); then
    worker "$gpu_id" "${assigned[@]}" &
    pids+=("$!")
  fi
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "[vggt_omega] one or more workers failed" >&2
  exit 1
fi

echo "[vggt_omega] all jobs finished."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/crisp/bin/python}"
HMR_TYPE="${HMR_TYPE:-gv}"
DISPLAY_ROOT="${DISPLAY_ROOT:-$REPO_ROOT/results/output/scene_vggt_omega_consistent_camera_min1}"
GMR_INPUT_ROOT="${GMR_INPUT_ROOT:-$REPO_ROOT/results/output/gmr_scene_inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/output/retargeting_gmr_vggt_omega_scene_hmr}"
LOG_DIR="${LOG_DIR:-/tmp/stairs_vggt_omega_scene_gmr}"
GMR_JOBS="${GMR_JOBS:-4}"
GPU_IDS_OVERRIDE="${GPU_IDS_OVERRIDE:-0}"
FORCE_GMR="${FORCE_GMR:-off}"
FORCE_PREP="${FORCE_PREP:-off}"

mkdir -p "$LOG_DIR" "$GMR_INPUT_ROOT" "$OUTPUT_ROOT"

IFS=',' read -r -a GPU_IDS <<<"$GPU_IDS_OVERRIDE"
if (( ${#GPU_IDS[@]} == 0 )); then
  GPU_IDS=(0)
fi

qpos_path() {
  local seq="$1"
  printf '%s\n' "$OUTPUT_ROOT/gmr/$seq/unitree_g1/${seq}_unitree_g1_qpos.npz"
}

input_path() {
  local seq="$1"
  printf '%s\n' "$GMR_INPUT_ROOT/vggt_omega/$seq/joint_frames_scene.npz"
}

run_one() {
  local seq="$1"
  local gpu="$2"
  local logfile="$LOG_DIR/$seq.log"
  local in_npz
  in_npz="$(input_path "$seq")"

  {
    echo "===== $(date +'%F %T') scene-gmr $seq gpu=$gpu ====="
    if [[ "$FORCE_PREP" == "on" || ! -f "$in_npz" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$REPO_ROOT/vis_scripts/viser_m/prepare_scene_frame_gmr_inputs.py" \
        --out-root "$GMR_INPUT_ROOT" \
        --sequences "$seq" \
        --sources vggt_omega \
        --formats smplx_npz joint_frames_npz \
        --display-root "$DISPLAY_ROOT"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$REPO_ROOT/vis_scripts/viser_m/run_retargeting_backend.py" \
      --backend gmr \
      --seq-name "$seq" \
      --robot g1 \
      --gmr-input-file "$in_npz" \
      --gmr-input-format joint_frames_npz \
      --output-root "$OUTPUT_ROOT"
  } >"$logfile" 2>&1
}

mapfile -t SEQS < <(
  find -L "$DISPLAY_ROOT" -mindepth 3 -maxdepth 3 -type d -path "*/$HMR_TYPE/scene_mesh_sqs" \
    -printf '%h\n' \
    | sed "s#/$HMR_TYPE\$##" \
    | xargs -r -n1 basename \
    | sort
)

if (( ${#SEQS[@]} == 0 )); then
  echo "[scene-gmr] no sequences found under $DISPLAY_ROOT" >&2
  exit 1
fi

echo "[scene-gmr] sequences=${#SEQS[@]} jobs=$GMR_JOBS gpus=${GPU_IDS[*]}"
fail=0
launched=0

for idx in "${!SEQS[@]}"; do
  seq="${SEQS[$idx]}"
  if [[ "$FORCE_GMR" != "on" ]] && [[ -f "$(qpos_path "$seq")" ]]; then
    echo "[scene-gmr] skip existing $seq"
    continue
  fi
  gpu="${GPU_IDS[$((launched % ${#GPU_IDS[@]}))]}"
  run_one "$seq" "$gpu" &
  launched=$((launched + 1))
  while (( $(jobs -rp | wc -l) >= GMR_JOBS )); do
    if ! wait -n; then
      fail=1
    fi
  done
done

while (( $(jobs -rp | wc -l) > 0 )); do
  if ! wait -n; then
    fail=1
  fi
done

if (( fail != 0 )); then
  echo "[scene-gmr] one or more sequences failed. Logs: $LOG_DIR" >&2
  exit 1
fi

echo "[scene-gmr] done. Logs: $LOG_DIR"

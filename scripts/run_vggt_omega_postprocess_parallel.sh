#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/crisp/bin/python}"
HMR_TYPE="${HMR_TYPE:-gv}"
SCENE_ARTIFACT_OUTPUT_DIR="${SCENE_ARTIFACT_OUTPUT_DIR:-$REPO_ROOT/results/output/scene_vggt_omega_consistent_camera_min1}"
POST_SCENE_OUTPUT_DIR="${POST_SCENE_OUTPUT_DIR:-$REPO_ROOT/results/output/post_scene_vggt_omega}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
SCENE_NPZ_TEMPLATE="${SCENE_NPZ_TEMPLATE:-}"
LOG_DIR="${LOG_DIR:-/tmp/stairs_vggt_omega_postprocess}"
POST_JOBS="${POST_JOBS:-2}"
GPU_IDS_OVERRIDE="${GPU_IDS_OVERRIDE:-0}"
FORCE_POST="${FORCE_POST:-off}"
ROT_BABY_EXTRA_ARGS="${ROT_BABY_EXTRA_ARGS:---debug-stride 1000000 --no-optimize-penetration}"

if [[ -z "$SCENE_NPZ_TEMPLATE" ]]; then
  SCENE_NPZ_TEMPLATE="$REPO_ROOT/results/output/scene/{seq}_vggt_omega_${HMR_TYPE}_sgd_cvd_hr.npz"
fi

mkdir -p "$LOG_DIR" "$POST_SCENE_OUTPUT_DIR"

IFS=',' read -r -a GPU_IDS <<<"$GPU_IDS_OVERRIDE"
if (( ${#GPU_IDS[@]} == 0 )); then
  GPU_IDS=(0)
fi

render_scene_npz() {
  local seq="$1"
  local path="$SCENE_NPZ_TEMPLATE"
  path="${path//\{seq\}/$seq}"
  path="${path//\{seq_name\}/$seq}"
  path="${path//\{hmr_type\}/$HMR_TYPE}"
  printf '%s\n' "$path"
}

has_outputs() {
  local seq="$1"
  local root="$POST_SCENE_OUTPUT_DIR/$seq/$HMR_TYPE"
  [[ -f "$root/world_rotation.npy" \
    && -f "$root/hmr/human_motion.npz" \
    && -f "$root/hmr/hps_track.npy" \
    && -f "$root/scene_mesh_sqs/scene_mesh_sqs.obj" \
    && -f "$root/scene_mesh_sqs/sqs_params.npz" ]]
}

run_one() {
  local seq="$1"
  local gpu="$2"
  local logfile="$LOG_DIR/$seq.log"
  local scene_npz
  scene_npz="$(render_scene_npz "$seq")"

  {
    echo "===== $(date +'%F %T') postprocess $seq gpu=$gpu ====="
    echo "scene_npz=$scene_npz"
    if [[ ! -f "$scene_npz" ]]; then
      echo "Missing scene npz: $scene_npz" >&2
      exit 2
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$REPO_ROOT/vis_scripts/viser_m/rot_baby.py" \
      --seq-names "$seq" \
      --hmr-type "$HMR_TYPE" \
      --input-root "$SCENE_ARTIFACT_OUTPUT_DIR" \
      --output-root "$POST_SCENE_OUTPUT_DIR" \
      --camera-npz "$scene_npz" \
      --data-root "$DATA_ROOT" \
      $ROT_BABY_EXTRA_ARGS

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$REPO_ROOT/vis_scripts/viser_m/rotate_scene_sqs_only.py" \
      --sequence-name "$seq" \
      --hmr-type "$HMR_TYPE" \
      --input-root "$SCENE_ARTIFACT_OUTPUT_DIR" \
      --output-root "$POST_SCENE_OUTPUT_DIR"
  } >"$logfile" 2>&1
}

mapfile -t SEQS < <(
  find -L "$SCENE_ARTIFACT_OUTPUT_DIR" -mindepth 3 -maxdepth 3 -type d -path "*/$HMR_TYPE/scene_mesh_sqs" \
    -printf '%h\n' \
    | sed "s#/$HMR_TYPE\$##" \
    | xargs -r -n1 basename \
    | sort
)

if (( ${#SEQS[@]} == 0 )); then
  echo "[postprocess] no sequences found under $SCENE_ARTIFACT_OUTPUT_DIR" >&2
  exit 1
fi

echo "[postprocess] sequences=${#SEQS[@]} jobs=$POST_JOBS gpus=${GPU_IDS[*]}"
fail=0
launched=0

for idx in "${!SEQS[@]}"; do
  seq="${SEQS[$idx]}"
  if [[ "$FORCE_POST" != "on" ]] && has_outputs "$seq"; then
    echo "[postprocess] skip existing $seq"
    continue
  fi
  gpu="${GPU_IDS[$((launched % ${#GPU_IDS[@]}))]}"
  run_one "$seq" "$gpu" &
  launched=$((launched + 1))
  while (( $(jobs -rp | wc -l) >= POST_JOBS )); do
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
  echo "[postprocess] one or more sequences failed. Logs: $LOG_DIR" >&2
  exit 1
fi

echo "[postprocess] done. Logs: $LOG_DIR"

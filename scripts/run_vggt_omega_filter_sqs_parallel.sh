#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/crisp/bin/python}"
HMR_TYPE="${HMR_TYPE:-gv}"
SOURCE_POST_ROOT="${SOURCE_POST_ROOT:-$REPO_ROOT/results/output/post_scene_vggt_omega}"
RAW_ROOT="${RAW_ROOT:-$REPO_ROOT/results/output/scene_vggt_omega_consistent_camera_min1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/output/post_scene_vggt_omega_filtered}"
LOG_DIR="${LOG_DIR:-/tmp/stairs_vggt_omega_filter_sqs}"
FILTER_JOBS="${FILTER_JOBS:-6}"
FORCE_FILTER="${FORCE_FILTER:-on}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

mapfile -t SEQS < <(
  find -L "$SOURCE_POST_ROOT" -mindepth 3 -maxdepth 3 -type d -path "*/$HMR_TYPE/scene_mesh_sqs" \
    -printf '%h\n' \
    | sed "s#/$HMR_TYPE\$##" \
    | xargs -r -n1 basename \
    | sort
)

if (( ${#SEQS[@]} == 0 )); then
  echo "[filter-sqs] no sequences found under $SOURCE_POST_ROOT" >&2
  exit 1
fi

run_one() {
  local seq="$1"
  local logfile="$LOG_DIR/$seq.log"
  local force_args=()
  if [[ "$FORCE_FILTER" == "on" ]]; then
    force_args=(--force)
  fi
  "$PYTHON_BIN" "$REPO_ROOT/vis_scripts/viser_m/filter_sqs_primitives_by_points.py" \
    --source-post-root "$SOURCE_POST_ROOT" \
    --raw-root "$RAW_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --hmr-type "$HMR_TYPE" \
    --sequences "$seq" \
    "${force_args[@]}" \
    >"$logfile" 2>&1
}

echo "[filter-sqs] sequences=${#SEQS[@]} jobs=$FILTER_JOBS output=$OUTPUT_ROOT"
fail=0
for seq in "${SEQS[@]}"; do
  run_one "$seq" &
  while (( $(jobs -rp | wc -l) >= FILTER_JOBS )); do
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
  echo "[filter-sqs] one or more sequences failed. Logs: $LOG_DIR" >&2
  exit 1
fi

echo "[filter-sqs] done. Logs: $LOG_DIR"

#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VIS_SCRIPT="$SCRIPT_DIR/vis_c.sh"
DATA_ROOT="$REPO_ROOT/humanoid/hybrid-imitation-parkour"

if [[ ! -f "$VIS_SCRIPT" ]]; then
  echo "vis.sh not found at $VIS_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Humanoid data directory missing: $DATA_ROOT" >&2
  exit 1
fi

MAX_CONCURRENT=${MAX_CONCURRENT:-2}

if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a GPU_LIST <<< "${GPU_IDS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk 'NF')
else
  GPU_LIST=()
fi

if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs detected; using ${MAX_CONCURRENT} CPU slots."
  for ((i = 0; i < MAX_CONCURRENT; i++)); do
    GPU_LIST+=("cpu_${i}")
  done
else
  if [[ ${#GPU_LIST[@]} -gt MAX_CONCURRENT ]]; then
    GPU_LIST=("${GPU_LIST[@]:0:MAX_CONCURRENT}")
  fi
fi

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/render_logs}"
mkdir -p "$LOG_DIR"

declare -a TASKS=()
for exp_dir in "$DATA_ROOT"/post_asset_ours_1*; do
  [[ -d "$exp_dir" ]] || continue
  parent_name="${exp_dir##*_}"
  for seq_dir in "$exp_dir"/*; do
    [[ -d "$seq_dir" ]] || continue
    seq_name="$(basename "$seq_dir")"
    if [[ ! -f "$seq_dir/rigid_bodies_0.npz" ]]; then
      echo "Skipping ${parent_name}/${seq_name} (missing rigid_bodies_0.npz)" >&2
      continue
    fi
    TASKS+=("${parent_name}_${seq_name}_ours")
  done
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "No sequences found under $DATA_ROOT/post_asset_ours_1*" >&2
  exit 1
fi

mapfile -t TASKS < <(printf '%s\n' "${TASKS[@]}" | sort)

declare -A GPU_PIDS=()
declare -A PID_LABELS=()
declare -A PID_LOGS=()

cleanup() {
  for gpu in "${GPU_LIST[@]}"; do
    pid="${GPU_PIDS[$gpu]-}"
    [[ -n "${pid:-}" ]] || continue
    echo "Stopping job ${PID_LABELS[$pid]} on slot $gpu (pid $pid)" >&2
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  exit 1
}
trap cleanup INT TERM

finalize_gpu() {
  local gpu="$1"
  local pid="${GPU_PIDS[$gpu]-}"
  [[ -n "${pid:-}" ]] || return
  if wait "$pid"; then
    echo "[Slot $gpu] ${PID_LABELS[$pid]} finished (log: ${PID_LOGS[$pid]})"
  else
    echo "[Slot $gpu] ${PID_LABELS[$pid]} failed (log: ${PID_LOGS[$pid]})" >&2
  fi
  unset GPU_PIDS["$gpu"]
  unset PID_LABELS["$pid"]
  unset PID_LOGS["$pid"]
}

wait_for_slot() {
  local selected=""
  while true; do
    for gpu in "${GPU_LIST[@]}"; do
      local pid="${GPU_PIDS[$gpu]-}"
      if [[ -z "${pid:-}" ]]; then
        selected="$gpu"
        break
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        finalize_gpu "$gpu"
        selected="$gpu"
        break
      fi
    done
    [[ -n "$selected" ]] && break
    sleep 2
  done
  echo "$selected"
}

total_tasks=${#TASKS[@]}
echo "Scheduling ${total_tasks} sequences across slots: ${GPU_LIST[*]}"

BATCH_LIMIT=${BATCH_LIMIT:-2}
num_batches=$(( (total_tasks + BATCH_LIMIT - 1) / BATCH_LIMIT ))

for ((batch_idx = 0; batch_idx < num_batches; batch_idx++)); do
  start=$((batch_idx * BATCH_LIMIT))
  remaining=$((total_tasks - start))
  batch_len=$(( remaining < BATCH_LIMIT ? remaining : BATCH_LIMIT ))
  BATCH=("${TASKS[@]:start:batch_len}")
  echo "Processing batch $((batch_idx + 1))/${num_batches} (${batch_len} sequences)"

  for task in "${BATCH[@]}"; do
    gpu="$(wait_for_slot)"
    log_file="$LOG_DIR/${task}.log"
    echo "[Slot $gpu] Starting $task (log: $log_file)"
    (
      if [[ "$gpu" == cpu_* ]]; then
        unset CUDA_VISIBLE_DEVICES
      else
        export CUDA_VISIBLE_DEVICES="$gpu"
      fi
      set -x
      sh "$VIS_SCRIPT" "$task" --save_mode
    ) &> "$log_file" &
    pid=$!
    GPU_PIDS["$gpu"]=$pid
    PID_LABELS["$pid"]="$task"
    PID_LOGS["$pid"]="$log_file"
  done

  for gpu in "${GPU_LIST[@]}"; do
    finalize_gpu "$gpu"
  done
done

echo "All sequences processed."

#!/usr/bin/env bash
set -euo pipefail
#eval "$(conda shell.bash hook)"
# conda activate crisp

cd ../prep/MogeSAM

ROOT="$1"
DATA_PATH="${ROOT%/}_videos"  # Append "_video" suffix

shopt -s nullglob
DIRS=("$DATA_PATH"/*)
shopt -u nullglob
NUM_DIRS=${#DIRS[@]}
if (( NUM_DIRS == 0 )); then
    echo "[ERR] no input frame directories under ${DATA_PATH}" >&2
    exit 2
fi

MEGASAM_TMPDIR="${MEGASAM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "${MEGASAM_TMPDIR}"
export TMPDIR="${MEGASAM_TMPDIR}"
echo "[MegaSAM] TMPDIR=${TMPDIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_IDS=($(seq 0 $((GPU_COUNT-1))))
fi
GPU_COUNT=${#GPU_IDS[@]}

worker() {
    local gpu_id="$1"
    shift
    local folders=("$@")

    for cam_folder in "${folders[@]}"; do
        echo "→ GPU $gpu_id │ $cam_folder"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        python inference.py \
            --input_path "$cam_folder" \
            --checkpoint checkpoints/tapip3d_final.pth \
            --resolution_factor 1
    done
}

pids=()
for logical_idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$logical_idx]}"
    gpu_dirs=()
    for (( idx=logical_idx; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    worker "$gpu_id" "${gpu_dirs[@]}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "[ERR] one or more MegaSAM workers failed" >&2
    exit 1
fi
# ModuleNotFoundError: No module named 'timm.layers'
echo "🏁  All jobs finished."

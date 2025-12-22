#!/usr/bin/env bash
#set -eo pipefail 
#eval "$(conda shell.bash hook)"
# conda activate crisp

cd ../prep/MogeSAM

ROOT="$1"
DATA_PATH="${ROOT%/}_videos"  # Append "_video" suffix

DIRS=("$DATA_PATH"/*)
NUM_DIRS=${#DIRS[@]}

GPU_COUNT=$(nvidia-smi -L | wc -l)
GPU_IDS=($(seq 0 $((GPU_COUNT-1))))

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

for gpu_id in "${GPU_IDS[@]}"; do
    gpu_dirs=()
    for (( idx=gpu_id; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    worker "$gpu_id" "${gpu_dirs[@]}" &
done

wait
echo "🏁  All jobs finished."

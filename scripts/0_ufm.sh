set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/prep/MogeSAM"

###############################################################################
# 2) Paths
###############################################################################
ROOT="$1"
if [[ "$ROOT" = /* ]]; then
    DATA_ROOT="${ROOT%/}"
else
    DATA_ROOT="$REPO_ROOT/${ROOT%/}"
fi
DATA_PATH="${DATA_ROOT}_img"              # append “_img” if not already
[[ -d "$DATA_PATH" ]] || { echo "❌  '$DATA_PATH' not found"; exit 1; }
export DINOV2_TORCH_HUB_REPO="${DINOV2_TORCH_HUB_REPO:-/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
UFM_STRIDE="${UFM_STRIDE:-7}"
UFM_MODE="${UFM_MODE:-window}"
UFM_WINDOW="${UFM_WINDOW:-4}"
UFM_SAVE_VIS="${UFM_SAVE_VIS:-off}"

###############################################################################
# 3) GPUs
###############################################################################
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_IDS=($(seq 0 $((GPU_COUNT-1))))
fi
GPU_COUNT=${#GPU_IDS[@]}

echo "🖥️  Found $GPU_COUNT GPUs → ${GPU_IDS[*]}"
echo "📂  Scanning '$DATA_PATH' …"

###############################################################################
# 4) List all immediate sub‑folders (one job per folder)
###############################################################################
mapfile -d '' DIRS < <(find "$DATA_PATH" -mindepth 1 -maxdepth 1 -type d -print0)
NUM_DIRS=${#DIRS[@]}
echo "📄  ${NUM_DIRS} folders to process"
if (( NUM_DIRS == 0 )); then
    echo "❌  no image folders found under '$DATA_PATH'" >&2
    exit 2
fi

###############################################################################
# 5) Define worker (runs on a *single* GPU)
###############################################################################
worker() {
    local gpu_id="$1"
    shift
    local folders=("$@")

    for cam_folder in "${folders[@]}"; do
        seq=$(basename "$cam_folder")            # e.g. cam_06
        parent_dir=$(dirname  "$cam_folder")     # e.g. …/rich_07_img
        video_dir="${parent_dir}"             

        echo "→ GPU $gpu_id │ $seq"
        cmd=(
            "$PYTHON_BIN" ufm.py
            --images "$cam_folder"
            --stride "$UFM_STRIDE"
            --out "../../results/init/flows/$seq"
            --mode "$UFM_MODE"
            --window "$UFM_WINDOW"
        )
        case "${UFM_SAVE_VIS,,}" in
            on|true|1|yes|y) cmd+=(--save-vis) ;;
        esac
        CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
    done
}

###############################################################################
# 6) Dispatch jobs: split DIRS array round‑robin by modulo GPU_COUNT
###############################################################################
pids=()
for logical_idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$logical_idx]}"
    # build slice for this GPU
    gpu_dirs=()
    for (( idx=logical_idx; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    # start worker in background
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
    echo "❌  one or more UFM workers failed" >&2
    exit 1
fi
echo "🏁  All jobs finished."

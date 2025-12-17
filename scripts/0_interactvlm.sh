
cd ../prep/Contact-Predictor
###############################################################################
# 2) Paths
###############################################################################
ROOT="$1"
OBJ='stairs'
DATA_PATH="${ROOT%/}_img"              # append “_img” if not already
[[ -d "$DATA_PATH" ]] || { echo "❌  '$DATA_PATH' not found"; exit 1; }

###############################################################################
# 3) GPUs
###############################################################################
GPU_COUNT=$(nvidia-smi -L | wc -l)
GPU_IDS=($(seq 0 $((GPU_COUNT-1))))

echo "🖥️  Found $GPU_COUNT GPUs → ${GPU_IDS[*]}"
echo "📂  Scanning '$DATA_PATH' …"

###############################################################################
# 4) List all immediate sub‑folders (one job per folder)
###############################################################################
mapfile -d '' DIRS < <(find "$DATA_PATH" -mindepth 1 -maxdepth 1 -type d -print0)
NUM_DIRS=${#DIRS[@]}
echo "📄  ${NUM_DIRS} folders to process"

###############################################################################
# 5) Define worker (runs on a *single* GPU)
###############################################################################
worker() {
    local gpu_id="$1"
    shift
    local folders=("$@")

    # 让调用总能找到同目录的 process.sh（而不是依赖当前工作目录）
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

    for cam_folder in "${folders[@]}"; do
        local seq
        seq="$(basename "$cam_folder")"
        local parent_dir
        parent_dir="$(dirname "$cam_folder")"

        echo "→ GPU $gpu_id │ $seq"

        # ① 不要在反斜杠后面留空格
        # ② 最后一行不要加反斜杠
        # ③ 给所有参数加引号
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        bash "process.sh" \
          "$parent_dir" \
          "$seq" \
          "$OBJ"
    done
}


###############################################################################
# 6) Dispatch jobs: split DIRS array round‑robin by modulo GPU_COUNT
###############################################################################
for gpu_id in "${GPU_IDS[@]}"; do
    # build slice for this GPU
    gpu_dirs=()
    for (( idx=gpu_id; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    # start worker in background
    worker "$gpu_id" "${gpu_dirs[@]}" &
done

wait
echo "🏁  All jobs finished."

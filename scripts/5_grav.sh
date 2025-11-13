#!/bin/bash

cd ../prep/HMR
#!/bin/bash

VIDEO_DIR="${1}_videos"


# Get total number of GPUs available
GPU_COUNT=$(nvidia-smi -L | wc -l)
GPU_IDS=($(seq 0 $((GPU_COUNT - 1))))

# Find all *.mp4 files in the directory
VIDEO_FILES=($(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4"))
TOTAL_FILES=${#VIDEO_FILES[@]}
echo $pwd
# Function to process a video on a specific GPU
process_video() {
    local video_file="$1"
    local gpu_id="$2"
    echo "Processing $video_file on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id sh demo.sh "$video_file"
}


# Track how many tasks are on each GPU
declare -A GPU_TASKS
for GPU_ID in "${GPU_IDS[@]}"; do
    GPU_TASKS[$GPU_ID]=0
done

# Distribute videos across GPUs
for ((i=0; i < TOTAL_FILES; i++)); do
    # Identify the GPU with the fewest tasks
    MIN_GPU=$(for GPU_ID in "${GPU_IDS[@]}"; do echo "$GPU_ID ${GPU_TASKS[$GPU_ID]}"; done \
              | sort -nk2 \
              | head -n1 \
              | awk '{print $1}')

    video_file="${VIDEO_FILES[$i]}"

    # Run the job in the background
    process_video "$video_file" "$MIN_GPU" &

    # Update the task count
    GPU_TASKS[$MIN_GPU]=$((GPU_TASKS[$MIN_GPU] + 1))

    # Display current distribution
    echo "GPU Task Distribution: ${GPU_TASKS[@]}"
done

# Wait for all background jobs
wait
echo "All processing complete."

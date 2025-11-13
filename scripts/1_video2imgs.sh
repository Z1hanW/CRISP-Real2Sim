#!/bin/bash

cd ../prep/AutoMask/preprocess #/data3/zihanwa3/_Robotics/_vision/tram

VIDEO_DIR="${1}_videos"
IMG_DIR="${1}_img"
RESIZE=$2

# Check if any .mp4 file exists in VIDEO_DIR
shopt -s nullglob
mp4_files=("$VIDEO_DIR"/*.mp4)
shopt -u nullglob

if [ ${#mp4_files[@]} -gt 0 ]; then
  echo "Found video files. Running video2frames.py..."
  for video_file in "${mp4_files[@]}"; do
    python video2frames.py \
      --video_path "$video_file" \
      ${RESIZE:+--resize $RESIZE}
  done
else
  echo "No video files found. Assuming images exist. Running frames2video.py..."
  for folder in "$IMG_DIR"/*; do
    if [ -d "$folder" ]; then  # ensure it's a directory
      python frames2video.py \
        --video_path "$folder" \
        ${RESIZE:+--resize $RESIZE}
    fi
  done
fi
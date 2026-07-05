#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SPLIT_ROOT="${SPLIT_ROOT:-/home/ubuntu/FAR/_stairs}"
VIDEO_ROOT="${SPLIT_ROOT}_videos"
IMG_ROOT="${SPLIT_ROOT}_img"

CRISP_PYTHON="${CRISP_PYTHON:-/home/ubuntu/miniconda3/envs/crisp/bin/python}"
VGGT_PYTHON="${VGGT_PYTHON:-/home/ubuntu/miniconda3/envs/vggt_omega/bin/python}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-/home/ubuntu/FAR/models/vggt-omega/vggt_omega_1b_512.pt}"
VGGT_REPO="${VGGT_REPO:-$REPO_ROOT/prep/vggt-omega}"

RAW_PRIORS_ROOT="${RAW_PRIORS_ROOT:-$REPO_ROOT/results/init/vslam/raw_vggt_omega_priors}"
VGGT_CAMERA_ROOT="${VGGT_CAMERA_ROOT:-$REPO_ROOT/results/init/vslam/vggt_omega_cam}"
HMR_OUTPUT_ROOT="${HMR_OUTPUT_ROOT:-$REPO_ROOT/results/init/hmr_vggt_omega}"
SCENE_ARTIFACT_OUTPUT_DIR="${SCENE_ARTIFACT_OUTPUT_DIR:-$REPO_ROOT/results/output/scene_vggt_omega_consistent_camera_min1}"
LOG_DIR="${LOG_DIR:-/tmp/stairs_vggt_omega_batch}"
STAGES="${STAGES:-frames,masks,vggt_raw,vggt_camera,hmr,ufm,align,sqs}"

mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/runtime_shims${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/home/ubuntu/miniconda3/envs/crisp/bin:$PATH"

if [[ ! -d "$VIDEO_ROOT" ]]; then
  echo "[stairs-vggt] missing video root: $VIDEO_ROOT" >&2
  exit 2
fi
if [[ ! -f "$VGGT_CHECKPOINT" ]]; then
  echo "[stairs-vggt] missing VGGT checkpoint: $VGGT_CHECKPOINT" >&2
  exit 3
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l)"
  mapfile -t GPU_IDS < <(seq 0 $((GPU_COUNT - 1)))
else
  GPU_IDS=("cpu")
fi
if [[ -n "${GPU_IDS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_OVERRIDE"
fi
GPU_COUNT="${#GPU_IDS[@]}"

has_stage() {
  local needle="$1"
  [[ ",$STAGES," == *",$needle,"* ]]
}

video_frame_count() {
  "$CRISP_PYTHON" - "$1" <<'PY'
import cv2, sys
cap = cv2.VideoCapture(sys.argv[1])
count = 0
while True:
    ok, _ = cap.read()
    if not ok:
        break
    count += 1
print(count)
cap.release()
PY
}

collect_videos() {
  find "$VIDEO_ROOT" -maxdepth 1 -type f -name '*.mp4' -print0 | sort -z
}

collect_img_dirs() {
  find "$IMG_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
}

print_counts() {
  local videos imgs masks raw hmr scene sqs
  videos="$(find "$VIDEO_ROOT" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l || true)"
  imgs="$([[ -d "$IMG_ROOT" ]] && find "$IMG_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l || true)"
  masks="$([[ -d "$REPO_ROOT/results/init/dyn_mask" ]] && find "$REPO_ROOT/results/init/dyn_mask" -mindepth 1 -maxdepth 1 -type d -name 'stair_*' 2>/dev/null | wc -l || true)"
  raw="$([[ -d "$RAW_PRIORS_ROOT" ]] && find "$RAW_PRIORS_ROOT" -maxdepth 1 -type f -name 'stair_*.npz' 2>/dev/null | wc -l || true)"
  hmr="$([[ -d "$HMR_OUTPUT_ROOT" ]] && find -L "$HMR_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'stair_*' 2>/dev/null | wc -l || true)"
  scene="$([[ -d "$REPO_ROOT/results/output/scene" ]] && find -L "$REPO_ROOT/results/output/scene" -maxdepth 1 -type f -name 'stair_*_vggt_omega_gv_sgd_cvd_hr.npz' 2>/dev/null | wc -l || true)"
  sqs="$([[ -d "$SCENE_ARTIFACT_OUTPUT_DIR" ]] && find -L "$SCENE_ARTIFACT_OUTPUT_DIR" -path '*/gv/scene_mesh_sqs/scene_mesh_sqs.obj' -name 'scene_mesh_sqs.obj' 2>/dev/null | grep -c '/stair_' || true)"
  imgs="${imgs:-0}"
  masks="${masks:-0}"
  raw="${raw:-0}"
  hmr="${hmr:-0}"
  scene="${scene:-0}"
  sqs="${sqs:-0}"
  echo "[stairs-vggt] counts videos=$videos imgs=$imgs masks=$masks raw=$raw hmr=$hmr scene_npz=$scene sqs=$sqs"
}

run_frames() {
  echo "[stairs-vggt] stage frames"
  mkdir -p "$IMG_ROOT"
  local video seq out_dir frame_count image_count
  while IFS= read -r -d '' video; do
    seq="$(basename "${video%.mp4}")"
    out_dir="$IMG_ROOT/$seq"
    frame_count="$(video_frame_count "$video")"
    image_count="$(find "$out_dir" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l || true)"
    if (( image_count >= frame_count && frame_count > 0 )); then
      echo "[frames] skip $seq ($image_count/$frame_count)"
      continue
    fi
    echo "[frames] $seq ($image_count/$frame_count)"
    "$CRISP_PYTHON" "$REPO_ROOT/prep/AutoMask/preprocess/video2frames.py" --video_path "$video"
  done < <(collect_videos)
}

run_masks() {
  echo "[stairs-vggt] stage masks on GPUs: ${GPU_IDS[*]}"
  mapfile -d '' DIRS < <(collect_img_dirs)
  local num_dirs="${#DIRS[@]}"
  if (( num_dirs == 0 )); then
    echo "[masks] no image dirs under $IMG_ROOT" >&2
    exit 4
  fi

  mask_worker() {
    local gpu_id="$1"
    shift
    local folder seq frame_count mask_count
    for folder in "$@"; do
      seq="$(basename "$folder")"
      frame_count="$(find "$folder" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
      mask_count="$(find "$REPO_ROOT/results/init/dyn_mask/$seq/person" -maxdepth 1 -type f -name 'dyn_mask_*.npz' 2>/dev/null | wc -l)"
      if (( mask_count >= frame_count && frame_count > 0 )); then
        echo "[masks] GPU $gpu_id skip $seq ($mask_count/$frame_count)"
        continue
      fi
      echo "[masks] GPU $gpu_id $seq ($mask_count/$frame_count)"
      (
        cd "$REPO_ROOT/prep/AutoMask"
        TEXT_THRESHOLDS="${TEXT_THRESHOLDS:-0.3,0.1,0.05,0.02}" \
          MASK_ANCHOR_STRIDE="${MASK_ANCHOR_STRIDE:-15}" \
          ALLOW_EMPTY_MASK="${ALLOW_EMPTY_MASK:-1}" \
          CUDA_VISIBLE_DEVICES="$gpu_id" "$CRISP_PYTHON" custom_mask.py \
          --seq "$seq" \
          --text_prompt "person" \
          --video_dir "$IMG_ROOT" \
          --save_dir "$REPO_ROOT/results/init/dyn_mask"
      )
    done
  }

  local pids=()
  for logical_idx in "${!GPU_IDS[@]}"; do
    local gpu_id="${GPU_IDS[$logical_idx]}"
    local assigned=()
    for (( idx=logical_idx; idx<num_dirs; idx+=GPU_COUNT )); do
      assigned+=("${DIRS[idx]}")
    done
    if (( ${#assigned[@]} > 0 )); then
      mask_worker "$gpu_id" "${assigned[@]}" >"$LOG_DIR/masks_gpu${gpu_id}.log" 2>&1 &
      pids+=("$!")
    fi
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "[masks] one or more workers failed" >&2
    exit 5
  fi

  local missing=0
  for folder in "${DIRS[@]}"; do
    seq="$(basename "$folder")"
    frame_count="$(find "$folder" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
    mask_count="$(find "$REPO_ROOT/results/init/dyn_mask/$seq/person" -maxdepth 1 -type f -name 'dyn_mask_*.npz' 2>/dev/null | wc -l)"
    if (( frame_count == 0 || mask_count < frame_count )); then
      echo "[masks] incomplete $seq ($mask_count/$frame_count)" >&2
      missing=1
    fi
  done
  if (( missing != 0 )); then
    echo "[masks] incomplete masks remain after mask stage" >&2
    exit 5
  fi
}

run_vggt_raw() {
  echo "[stairs-vggt] stage vggt_raw on GPUs: ${GPU_IDS[*]}"
  mkdir -p "$RAW_PRIORS_ROOT"
  mapfile -d '' VIDEOS < <(collect_videos)
  local num_videos="${#VIDEOS[@]}"

  vggt_worker() {
    local gpu_id="$1"
    shift
    local video seq out_path device max_frames retry_max_frames
    run_vggt_one() {
      local max_frames="$1"
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        CUDA_VISIBLE_DEVICES="$gpu_id" "$VGGT_PYTHON" "$REPO_ROOT/prep/VGGT_Omega/run_vggt_omega_prior.py" \
        --input-path "$video" \
        --checkpoint "$VGGT_CHECKPOINT" \
        --repo-path "$VGGT_REPO" \
        --raw-priors-root "$RAW_PRIORS_ROOT" \
        --device "$device" \
        --image-resolution "${VGGT_OMEGA_IMAGE_RESOLUTION:-512}" \
        --image-mode "${VGGT_OMEGA_IMAGE_MODE:-balanced}" \
        --frame-stride 1 \
        --max-frames "$max_frames"
    }
    for video in "$@"; do
      seq="$(basename "${video%.mp4}")"
      out_path="$RAW_PRIORS_ROOT/$seq.npz"
      if [[ -f "$out_path" ]]; then
        echo "[vggt_raw] GPU $gpu_id skip $seq"
        continue
      fi
      device="cuda"
      [[ "$gpu_id" == "cpu" ]] && device="cpu"
      echo "[vggt_raw] GPU $gpu_id $seq"
      max_frames="${VGGT_OMEGA_MAX_FRAMES:-0}"
      if ! run_vggt_one "$max_frames"; then
        retry_max_frames="${VGGT_OMEGA_RETRY_MAX_FRAMES:-360}"
        if [[ "$retry_max_frames" != "0" && "$retry_max_frames" != "$max_frames" ]]; then
          echo "[vggt_raw] GPU $gpu_id retry $seq max_frames=$retry_max_frames"
          rm -f "$out_path"
          if ! run_vggt_one "$retry_max_frames"; then
            return 1
          fi
        else
          return 1
        fi
      fi
    done
  }

  local pids=()
  for logical_idx in "${!GPU_IDS[@]}"; do
    local gpu_id="${GPU_IDS[$logical_idx]}"
    local assigned=()
    for (( idx=logical_idx; idx<num_videos; idx+=GPU_COUNT )); do
      assigned+=("${VIDEOS[idx]}")
    done
    if (( ${#assigned[@]} > 0 )); then
      vggt_worker "$gpu_id" "${assigned[@]}" >"$LOG_DIR/vggt_raw_gpu${gpu_id}.log" 2>&1 &
      pids+=("$!")
    fi
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "[vggt_raw] one or more workers failed" >&2
    exit 6
  fi

  local missing=0
  for video in "${VIDEOS[@]}"; do
    seq="$(basename "${video%.mp4}")"
    out_path="$RAW_PRIORS_ROOT/$seq.npz"
    if [[ ! -f "$out_path" ]]; then
      echo "[vggt_raw] missing $seq: $out_path" >&2
      missing=1
    fi
  done
  if (( missing != 0 )); then
    echo "[vggt_raw] missing raw priors remain after raw stage" >&2
    exit 6
  fi
}

run_vggt_camera() {
  echo "[stairs-vggt] stage vggt_camera"
  "$CRISP_PYTHON" "$SCRIPT_DIR/export_vggt_omega_cameras.py" \
    --split-root "$SPLIT_ROOT" \
    --raw-priors-root "$RAW_PRIORS_ROOT" \
    --camera-output-root "$VGGT_CAMERA_ROOT" \
    --pattern "stair_*.npz"
}

run_hmr() {
  echo "[stairs-vggt] stage hmr on GPUs: ${GPU_IDS[*]}"
  mkdir -p "$HMR_OUTPUT_ROOT"
  mapfile -d '' VIDEOS < <(collect_videos)
  local num_videos="${#VIDEOS[@]}"

  hmr_worker() {
    local gpu_id="$1"
    shift
    local video seq frame_count depth_count
    for video in "$@"; do
      seq="$(basename "${video%.mp4}")"
      frame_count="$(video_frame_count "$video")"
      if [[ -d "$HMR_OUTPUT_ROOT/$seq/depth_out" ]]; then
        depth_count="$(find "$HMR_OUTPUT_ROOT/$seq/depth_out" -maxdepth 1 -type f -name 'mesh_depth_*.npy' | wc -l)"
      else
        depth_count=0
      fi
      if [[ -f "$HMR_OUTPUT_ROOT/$seq/hmr4d_results.pt" && "$depth_count" -ge "$frame_count" ]]; then
        echo "[hmr] GPU $gpu_id skip $seq ($depth_count/$frame_count)"
        continue
      fi
      echo "[hmr] GPU $gpu_id $seq ($depth_count/$frame_count)"
      (
        cd "$REPO_ROOT/prep/HMR"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
          HMR_OUTPUT_ROOT="$HMR_OUTPUT_ROOT" \
          HMR_CAMERA_ROOT="$VGGT_CAMERA_ROOT" \
          HMR_SKIP_VIS_VIDEO="${HMR_SKIP_VIS_VIDEO:-1}" \
          "$CRISP_PYTHON" tools/demo/demo.py --video="$video" --output_root="$HMR_OUTPUT_ROOT" --camera_root="$VGGT_CAMERA_ROOT"
      )
    done
  }

  local pids=()
  for logical_idx in "${!GPU_IDS[@]}"; do
    local gpu_id="${GPU_IDS[$logical_idx]}"
    local assigned=()
    for (( idx=logical_idx; idx<num_videos; idx+=GPU_COUNT )); do
      assigned+=("${VIDEOS[idx]}")
    done
    if (( ${#assigned[@]} > 0 )); then
      hmr_worker "$gpu_id" "${assigned[@]}" >"$LOG_DIR/hmr_gpu${gpu_id}.log" 2>&1 &
      pids+=("$!")
    fi
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "[hmr] one or more workers failed" >&2
    exit 7
  fi
}

run_ufm() {
  echo "[stairs-vggt] stage ufm"
  local cuda_devices
  cuda_devices="$(IFS=,; echo "${GPU_IDS[*]}")"
  PYTHON_BIN="$CRISP_PYTHON" \
    CUDA_VISIBLE_DEVICES="$cuda_devices" \
    UFM_STRIDE="${UFM_STRIDE:-7}" \
    UFM_MODE="${UFM_MODE:-window}" \
    UFM_WINDOW="${UFM_WINDOW:-4}" \
    UFM_SAVE_VIS="${UFM_SAVE_VIS:-off}" \
    bash "$SCRIPT_DIR/0_ufm.sh" "$SPLIT_ROOT"
}

run_align() {
  echo "[stairs-vggt] stage align"
  mapfile -d '' DIRS < <(collect_img_dirs)
  local num_dirs="${#DIRS[@]}"
  if (( num_dirs == 0 )); then
    echo "[align] no image dirs under $IMG_ROOT" >&2
    exit 8
  fi

  local scene_output_dir="${SCENE_OUTPUT_DIR:-$REPO_ROOT/results/output/scene}"
  local align_jobs="${ALIGN_JOBS:-4}"
  if (( align_jobs < 1 )); then align_jobs=1; fi
  if (( align_jobs > num_dirs )); then align_jobs="$num_dirs"; fi
  mkdir -p "$scene_output_dir"
  echo "[align] jobs             : $align_jobs"
  echo "[align] scene output dir : $scene_output_dir"

  align_worker() {
    local worker_id="$1"
    shift
    local folder seq output_path
    for folder in "$@"; do
      seq="$(basename "${folder%/}")"
      output_path="$scene_output_dir/${seq}_vggt_omega_gv_sgd_cvd_hr.npz"
      if [[ -s "$output_path" && "${FORCE_ALIGN:-off}" != "on" ]]; then
        echo "[align] worker $worker_id skip $seq"
        continue
      fi
      echo "[align] worker $worker_id $seq -> $output_path"
      (
        cd "$REPO_ROOT/prep/MogeSAM"
        "$CRISP_PYTHON" post_process.py \
          --output_dir "$RAW_PRIORS_ROOT" \
          --scene_name "$seq" \
          --hmr_type gv \
          --scene_output_dir "$scene_output_dir" \
          --output_path "$output_path" \
          --backend vggt_omega \
          --hmr_root "$HMR_OUTPUT_ROOT"
      )
    done
  }

  local pids=()
  for (( worker_idx=0; worker_idx<align_jobs; worker_idx++ )); do
    local assigned=()
    for (( idx=worker_idx; idx<num_dirs; idx+=align_jobs )); do
      assigned+=("${DIRS[idx]}")
    done
    if (( ${#assigned[@]} > 0 )); then
      align_worker "$worker_idx" "${assigned[@]}" >"$LOG_DIR/align_worker${worker_idx}.log" 2>&1 &
      pids+=("$!")
    fi
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "[align] one or more workers failed" >&2
    exit 8
  fi

  local missing=0
  for folder in "${DIRS[@]}"; do
    seq="$(basename "${folder%/}")"
    output_path="$scene_output_dir/${seq}_vggt_omega_gv_sgd_cvd_hr.npz"
    if [[ ! -s "$output_path" ]]; then
      echo "[align] missing $seq: $output_path" >&2
      missing=1
    fi
  done
  if (( missing != 0 )); then
    echo "[align] missing scene npz outputs remain after align stage" >&2
    exit 8
  fi
}

run_sqs() {
  echo "[stairs-vggt] stage sqs"
  local cuda_devices
  cuda_devices="$(IFS=,; echo "${GPU_IDS[*]}")"
  SCENE_RECON_BACKEND=vggt_omega \
    SCENE_PRIOR_BASE_PATH="$RAW_PRIORS_ROOT" \
    SCENE_CAMERA_ROOT="$VGGT_CAMERA_ROOT" \
    SCENE_ARTIFACT_OUTPUT_DIR="$SCENE_ARTIFACT_OUTPUT_DIR" \
    HMR_RESULTS_ROOT="$HMR_OUTPUT_ROOT" \
    PYTHON_BIN="$CRISP_PYTHON" \
    CUDA_VISIBLE_DEVICES="$cuda_devices" \
    FORCE_FUSE="${FORCE_FUSE:-on}" \
    FUSION_INTERVAL="${FUSION_INTERVAL:-7}" \
    SEGMENT_MIN_FRAMES="${SEGMENT_MIN_FRAMES:-2}" \
    RUN_NKSR="${RUN_NKSR:-off}" \
    LOG_DIR="${SQS_LOG_DIR:-/tmp/vis_stairs_vggt_omega_planar_logs}" \
    bash "$SCRIPT_DIR/7_vggt_omega_planar.sh" "$SPLIT_ROOT" gv
}

echo "[stairs-vggt] split root       : $SPLIT_ROOT"
echo "[stairs-vggt] video root       : $VIDEO_ROOT"
echo "[stairs-vggt] image root       : $IMG_ROOT"
echo "[stairs-vggt] stages           : $STAGES"
echo "[stairs-vggt] GPUs             : ${GPU_IDS[*]}"
echo "[stairs-vggt] VGGT checkpoint  : $VGGT_CHECKPOINT"
echo "[stairs-vggt] raw priors root  : $RAW_PRIORS_ROOT"
echo "[stairs-vggt] VGGT camera root : $VGGT_CAMERA_ROOT"
echo "[stairs-vggt] HMR output root  : $HMR_OUTPUT_ROOT"
echo "[stairs-vggt] SQS output root  : $SCENE_ARTIFACT_OUTPUT_DIR"
print_counts

if has_stage frames; then run_frames; print_counts; fi
if has_stage masks; then run_masks; print_counts; fi
if has_stage vggt_raw; then run_vggt_raw; print_counts; fi
if has_stage vggt_camera; then run_vggt_camera; print_counts; fi
if has_stage hmr; then run_hmr; print_counts; fi
if has_stage ufm; then run_ufm; print_counts; fi
if has_stage align; then run_align; print_counts; fi
if has_stage sqs; then run_sqs; print_counts; fi

echo "[stairs-vggt] done"

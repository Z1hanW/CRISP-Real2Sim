#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SPLIT_ROOT="${SPLIT_ROOT:-/home/ubuntu/FAR/_stairs}"
IMG_ROOT="${IMG_ROOT:-${SPLIT_ROOT}_img}"
CRISP_PYTHON="${CRISP_PYTHON:-/home/ubuntu/miniconda3/envs/crisp/bin/python}"
SQS_LOG_DIR="${SQS_LOG_DIR:-/tmp/vis_stairs_vggt_omega_planar_logs}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/tmp/stairs_vggt_omega_batch}"
mkdir -p "$SQS_LOG_DIR" "$RUN_LOG_DIR"

if [[ ! -d "$IMG_ROOT" ]]; then
  echo "[sqs-parallel] missing image root: $IMG_ROOT" >&2
  exit 2
fi

if [[ -n "${GPU_IDS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_OVERRIDE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_IDS < <(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 )))
else
  GPU_IDS=("cpu")
fi

SQS_JOBS="${SQS_JOBS:-${#GPU_IDS[@]}}"
if (( SQS_JOBS < 1 )); then SQS_JOBS=1; fi
if (( SQS_JOBS > ${#GPU_IDS[@]} )); then SQS_JOBS="${#GPU_IDS[@]}"; fi

SUBSET_ROOT="${SQS_SUBSET_ROOT:-$(mktemp -d /tmp/far_stairs_sqs_subsets_XXXXXX)}"
mkdir -p "$SUBSET_ROOT"
for (( i=0; i<SQS_JOBS; i++ )); do
  mkdir -p "$SUBSET_ROOT/subset_$i"
done

mapfile -t SEQ_DIRS < <(find "$IMG_ROOT" -mindepth 1 -maxdepth 1 -type d | sort -V)
if (( ${#SEQ_DIRS[@]} == 0 )); then
  echo "[sqs-parallel] no image dirs under $IMG_ROOT" >&2
  exit 3
fi

idx=0
for dir in "${SEQ_DIRS[@]}"; do
  worker=$((idx % SQS_JOBS))
  ln -s "$dir" "$SUBSET_ROOT/subset_$worker/$(basename "$dir")"
  idx=$((idx + 1))
done

echo "[sqs-parallel] subset root : $SUBSET_ROOT"
echo "[sqs-parallel] jobs        : $SQS_JOBS"
echo "[sqs-parallel] GPUs        : ${GPU_IDS[*]}"
for (( i=0; i<SQS_JOBS; i++ )); do
  count="$(find "$SUBSET_ROOT/subset_$i" -mindepth 1 -maxdepth 1 -type l | wc -l)"
  echo "[sqs-parallel] subset_$i   : $count"
done

pids=()
for (( i=0; i<SQS_JOBS; i++ )); do
  gpu="${GPU_IDS[$i]}"
  (
    cd "$REPO_ROOT"
    if [[ "$gpu" == "cpu" ]]; then
      unset CUDA_VISIBLE_DEVICES
    else
      export CUDA_VISIBLE_DEVICES="$gpu"
    fi
    SCENE_RECON_BACKEND=vggt_omega \
      PYTHON_BIN="$CRISP_PYTHON" \
      FORCE_FUSE="${FORCE_FUSE:-off}" \
      FUSION_INTERVAL="${FUSION_INTERVAL:-7}" \
      SEGMENT_MIN_FRAMES="${SEGMENT_MIN_FRAMES:-2}" \
      RUN_NKSR="${RUN_NKSR:-off}" \
      LOG_DIR="$SQS_LOG_DIR/gpu_${gpu}" \
      bash "$SCRIPT_DIR/7_vggt_omega_planar.sh" "$SUBSET_ROOT/subset_$i" gv
  ) >"$RUN_LOG_DIR/sqs_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "[sqs-parallel] one or more workers failed" >&2
  exit 4
fi

echo "[sqs-parallel] done"

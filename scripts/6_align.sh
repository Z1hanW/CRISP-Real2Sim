#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/6_align.sh <split_or_path> [hmr_type]" >&2
  exit 1
fi

ROOT="${1%/}"
HMR_TYPE="${2:-gv}"

if [[ "$ROOT" == *_img ]]; then
  DATA_PATH="$ROOT"
else
  DATA_PATH="${ROOT}_img"
fi

BACKEND_RAW="${SCENE_RECON_BACKEND:-megasam}"
BACKEND_RAW="${BACKEND_RAW,,}"
case "$BACKEND_RAW" in
  megasam|moge|tapip3d)
    BACKEND="megasam"
    DEFAULT_PRIORS_ROOT="$REPO_ROOT/results/init/vslam/raw_mega_priors"
    DEFAULT_HMR_ROOT="$REPO_ROOT/results/init/hmr"
    ;;
  vggt_omega|vggt-omega|vggt)
    BACKEND="vggt_omega"
    DEFAULT_PRIORS_ROOT="$REPO_ROOT/results/init/vslam/raw_vggt_omega_priors"
    DEFAULT_HMR_ROOT="$REPO_ROOT/results/init/hmr_vggt_omega"
    ;;
  *)
    echo "[6_align] unknown SCENE_RECON_BACKEND='$SCENE_RECON_BACKEND'" >&2
    echo "[6_align] supported: megasam, vggt_omega" >&2
    exit 2
    ;;
esac

SCENE_RAW_PRIORS_ROOT="${SCENE_RAW_PRIORS_ROOT:-$DEFAULT_PRIORS_ROOT}"
SCENE_OUTPUT_DIR="${SCENE_OUTPUT_DIR:-$REPO_ROOT/results/output/scene}"
POST_PROCESS_SCRIPT="${POST_PROCESS_SCRIPT:-post_process.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OBJ_MASK_NAME="${OBJ_MASK_NAME:-}"
OBJ_MESH="${OBJ_MESH:-}"
HMR_ROOT="${HMR_ROOT:-$DEFAULT_HMR_ROOT}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[6_align] missing data directory: $DATA_PATH" >&2
  exit 3
fi

pushd "$REPO_ROOT/prep/MogeSAM" >/dev/null

shopt -s nullglob
seq_dirs=("$DATA_PATH"/*/)
shopt -u nullglob

if (( ${#seq_dirs[@]} == 0 )); then
  echo "[6_align] no sequence folders found under $DATA_PATH" >&2
  popd >/dev/null
  exit 4
fi

echo "[6_align] backend          : $BACKEND"
echo "[6_align] raw priors root  : $SCENE_RAW_PRIORS_ROOT"
echo "[6_align] scene output dir : $SCENE_OUTPUT_DIR"
echo "[6_align] hmr root         : $HMR_ROOT"
echo "[6_align] hmr type         : $HMR_TYPE"

for folder in "${seq_dirs[@]}"; do
  seq="$(basename "${folder%/}")"
  if [[ "$BACKEND" == "megasam" ]]; then
    output_path="$SCENE_OUTPUT_DIR/${seq}_${HMR_TYPE}_sgd_cvd_hr.npz"
  else
    output_path="$SCENE_OUTPUT_DIR/${seq}_${BACKEND}_${HMR_TYPE}_sgd_cvd_hr.npz"
  fi
  if [[ -s "$output_path" && "${FORCE_ALIGN:-off}" != "on" ]]; then
    echo "[6_align] skip ${seq}: ${output_path}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" "$POST_PROCESS_SCRIPT"
    --output_dir "$SCENE_RAW_PRIORS_ROOT"
    --scene_name "$seq"
    --hmr_type "$HMR_TYPE"
    --scene_output_dir "$SCENE_OUTPUT_DIR"
    --output_path "$output_path"
    --backend "$BACKEND"
    --hmr_root "$HMR_ROOT"
  )
  if [[ -n "$OBJ_MASK_NAME" ]]; then
    cmd+=(--obj_mask_name "$OBJ_MASK_NAME")
  fi
  if [[ -n "$OBJ_MESH" ]]; then
    cmd+=(--obj_mesh "$OBJ_MESH")
  fi

  echo "[6_align] ${seq} -> ${output_path}"
  "${cmd[@]}"
done

popd >/dev/null

echo "All demos completed successfully."

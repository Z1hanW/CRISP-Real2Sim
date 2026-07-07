#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/all_gv.sh <split_or_path>" >&2
  exit 1
fi

ROOT_DIR="${1%/}"
HMR_TYPE="${HMR_TYPE:-gv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BACKEND_RAW="${SCENE_RECON_BACKEND:-megasam}"
BACKEND_RAW="${BACKEND_RAW,,}"
case "$BACKEND_RAW" in
  megasam|moge|tapip3d)
    BACKEND="megasam"
    ;;
  vggt_omega|vggt-omega|vggt)
    BACKEND="vggt_omega"
    ;;
  *)
    echo "[all_gv] unknown SCENE_RECON_BACKEND='$SCENE_RECON_BACKEND'" >&2
    echo "[all_gv] supported: megasam, vggt_omega" >&2
    exit 2
    ;;
esac

bash "$SCRIPT_DIR/1_video2imgs.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/2_get_mask.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/3_scene_reconstruction.sh" "$ROOT_DIR"

if [[ "$BACKEND" == "vggt_omega" ]]; then
  VGGT_RAW_PRIORS_ROOT="${VGGT_OMEGA_RAW_PRIORS_ROOT:-$REPO_ROOT/results/init/vslam/raw_vggt_omega_priors}"
  VGGT_CAMERA_ROOT="${VGGT_CAMERA_ROOT:-$REPO_ROOT/results/init/vslam/vggt_omega_cam}"
  "$PYTHON_BIN" "$SCRIPT_DIR/export_vggt_omega_cameras.py" \
    --split-root "$ROOT_DIR" \
    --raw-priors-root "$VGGT_RAW_PRIORS_ROOT" \
    --camera-output-root "$VGGT_CAMERA_ROOT"
  export SCENE_CAMERA_ROOT="${SCENE_CAMERA_ROOT:-$VGGT_CAMERA_ROOT}"
  export HMR_CAMERA_ROOT="${HMR_CAMERA_ROOT:-$SCENE_CAMERA_ROOT}"
else
  bash "$SCRIPT_DIR/4_post_camera.sh" "$ROOT_DIR"
fi

bash "$SCRIPT_DIR/5_grav.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/0_ufm.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/6_align.sh" "$ROOT_DIR" "$HMR_TYPE"

if [[ "$BACKEND" == "vggt_omega" ]]; then
  export HMR_RESULTS_ROOT="${HMR_RESULTS_ROOT:-$REPO_ROOT/results/init/hmr}"
  bash "$SCRIPT_DIR/7_vggt_omega_planar.sh" "$ROOT_DIR" "$HMR_TYPE"
  export POST_SEQ_ROOT="${POST_SEQ_ROOT:-$ROOT_DIR}"
  bash "$SCRIPT_DIR/run_vggt_omega_postprocess_parallel.sh"
else
  bash "$SCRIPT_DIR/7_glue_sqs.sh" "$ROOT_DIR" "$HMR_TYPE"
  bash "$SCRIPT_DIR/8_postprocessing.sh" "$ROOT_DIR" "$HMR_TYPE"
fi

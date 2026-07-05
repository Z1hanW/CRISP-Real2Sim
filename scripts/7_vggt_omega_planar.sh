#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage: bash scripts/7_vggt_omega_planar.sh <split_or_path> [hmr_type]

Example:
  bash scripts/7_vggt_omega_planar.sh anything-chair gv

Useful overrides:
  SCENE_ARTIFACT_OUTPUT_DIR=...  # default: results/output/scene_vggt_omega_consistent_camera_min1
  FUSION_INTERVAL=7              # default VGGT-Omega frame interval for point-cloud fusion
  SEGMENT_MIN_FRAMES=2           # default for VGGT-Omega planar extraction
  RUN_NKSR=on                    # optional NKSR surface reconstruction from fused point cloud
EOF
  exit 1
fi

export SCENE_RECON_BACKEND="${SCENE_RECON_BACKEND:-vggt_omega}"
export SCENE_PRIOR_BASE_PATH="${SCENE_PRIOR_BASE_PATH:-$REPO_ROOT/results/init/vslam/raw_vggt_omega_priors}"
export SCENE_CAMERA_ROOT="${SCENE_CAMERA_ROOT:-$REPO_ROOT/results/init/vslam/vggt_omega_cam}"
export SCENE_ARTIFACT_OUTPUT_DIR="${SCENE_ARTIFACT_OUTPUT_DIR:-$REPO_ROOT/results/output/scene_vggt_omega_consistent_camera_min1}"
export SCENE_NPZ_DIR="${SCENE_NPZ_DIR:-$REPO_ROOT/results/output/scene}"
export HMR_RESULTS_ROOT="${HMR_RESULTS_ROOT:-$REPO_ROOT/results/init/hmr_vggt_omega}"
export LOG_DIR="${LOG_DIR:-/tmp/vis_vggt_omega_planar_logs}"
export PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/crisp/bin/python}"
export SAVE_CLUSTERING="${SAVE_CLUSTERING:-on}"
export FUSION_INTERVAL="${FUSION_INTERVAL:-7}"
export SEGMENT_MIN_FRAMES="${SEGMENT_MIN_FRAMES:-2}"
export RUN_NKSR="${RUN_NKSR:-off}"

exec bash "$SCRIPT_DIR/7_glue_sqs.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISER_DIR="$REPO_ROOT/vis_scripts/viser_m"
VIS_SCRIPT="$VISER_DIR/vis.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/vis_scene_backend_compare.sh <sequence_name> [hmr_type]" >&2
  echo "Example: bash scripts/vis_scene_backend_compare.sh anything-chair gv" >&2
  exit 1
fi

SEQ="$1"
HMR_TYPE="${2:-gv}"

MEGASAM_PORT="${MEGASAM_PORT:-9140}"
VGGT_OMEGA_PORT="${VGGT_OMEGA_PORT:-9141}"
LOG_DIR="${LOG_DIR:-/tmp/far_viser}"
VIEW_OUTPUT_ROOT="${SCENE_VIEW_OUTPUT_ROOT:-/tmp/far_crisp_view_outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MEGASAM_PRIORS_ROOT="${MEGASAM_PRIORS_ROOT:-$REPO_ROOT/results/init/vslam/raw_mega_priors}"
VGGT_OMEGA_PRIORS_ROOT="${VGGT_OMEGA_PRIORS_ROOT:-$REPO_ROOT/results/init/vslam/raw_vggt_omega_priors}"
MEGASAM_HMR_ROOT="${MEGASAM_HMR_ROOT:-$REPO_ROOT/results/init/hmr}"
VGGT_OMEGA_HMR_ROOT="${VGGT_OMEGA_HMR_ROOT:-$REPO_ROOT/results/init/hmr_vggt_omega}"
MEGASAM_SCENE_FILE="${MEGASAM_SCENE_FILE:-$REPO_ROOT/results/output/scene/${SEQ}_${HMR_TYPE}_sgd_cvd_hr.npz}"
VGGT_OMEGA_SCENE_FILE="${VGGT_OMEGA_SCENE_FILE:-$REPO_ROOT/results/output/scene/${SEQ}_vggt_omega_${HMR_TYPE}_sgd_cvd_hr.npz}"

mkdir -p "$LOG_DIR" "$VIEW_OUTPUT_ROOT"

check_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[vis_scene_backend_compare] missing $label: $path" >&2
    exit 2
  fi
}

check_port_free() {
  local port="$1"
  if ss -ltn | awk '{print $4}' | grep -Eq "(:|\\.)${port}$"; then
    echo "[vis_scene_backend_compare] port $port is already in use" >&2
    exit 3
  fi
}

launch_viewer() {
  local backend="$1"
  local scene_file="$2"
  local priors_root="$3"
  local hmr_root="$4"
  local port="$5"
  local out_dir="$6"
  local log_file="$7"

  echo "[vis_scene_backend_compare] launching $backend on http://localhost:$port"
  setsid env \
    HMR_TYPE="$HMR_TYPE" \
    SAVE_MODE=off \
    SAVE_CLUSTERING=off \
    USE_CONTACT=off \
    SCENE_FILE="$scene_file" \
    SCENE_PRIOR_BASE_PATH="$priors_root" \
    HMR_RESULTS_ROOT="$hmr_root" \
    SCENE_OUTPUT_DIR="$out_dir" \
    PORT="$port" \
    PYTHON_BIN="$PYTHON_BIN" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-64}" \
    bash "$VIS_SCRIPT" "$SEQ" >"$log_file" 2>&1 &
  echo "$!" >"${log_file%.log}.pid"
}

check_file "$MEGASAM_SCENE_FILE" "MegaSAM scene npz"
check_file "$VGGT_OMEGA_SCENE_FILE" "VGGT-Omega scene npz"
check_file "$MEGASAM_PRIORS_ROOT/$SEQ.npz" "MegaSAM raw prior"
check_file "$VGGT_OMEGA_PRIORS_ROOT/$SEQ.npz" "VGGT-Omega raw prior"
check_file "$MEGASAM_HMR_ROOT/$SEQ/hmr4d_results.pt" "MegaSAM HMR results"
check_file "$VGGT_OMEGA_HMR_ROOT/$SEQ/hmr4d_results.pt" "VGGT-Omega HMR results"
check_port_free "$MEGASAM_PORT"
check_port_free "$VGGT_OMEGA_PORT"

launch_viewer \
  "megasam" \
  "$MEGASAM_SCENE_FILE" \
  "$MEGASAM_PRIORS_ROOT" \
  "$MEGASAM_HMR_ROOT" \
  "$MEGASAM_PORT" \
  "$VIEW_OUTPUT_ROOT/megasam" \
  "$LOG_DIR/crisp_${SEQ}_megasam_bgptc_${MEGASAM_PORT}.log"

launch_viewer \
  "vggt_omega" \
  "$VGGT_OMEGA_SCENE_FILE" \
  "$VGGT_OMEGA_PRIORS_ROOT" \
  "$VGGT_OMEGA_HMR_ROOT" \
  "$VGGT_OMEGA_PORT" \
  "$VIEW_OUTPUT_ROOT/vggtomega" \
  "$LOG_DIR/crisp_${SEQ}_vggtomega_bgptc_${VGGT_OMEGA_PORT}.log"

echo "[vis_scene_backend_compare] MegaSAM    http://localhost:$MEGASAM_PORT"
echo "[vis_scene_backend_compare] VGGT-Omega http://localhost:$VGGT_OMEGA_PORT"

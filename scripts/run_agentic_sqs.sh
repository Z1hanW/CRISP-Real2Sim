#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage: bash scripts/run_agentic_sqs.sh <sequence> [hmr_type]

Environment overrides:
  SCENE_ARTIFACT_OUTPUT_DIR   Root containing <sequence>/<hmr_type>
  IMAGE_SPLIT_ROOT            Root containing the sequence's RGB frames
  CRISP_AGENTIC_OUTPUT_ROOT   Output root (default is under /data)
  CRISP_AGENTIC_CLUSTER_ROOT  Per-frame cluster evidence root
  AGENTIC_MAX_ITERATIONS      Planner iterations, default 3
  PYTHON_BIN                  CRISP Python executable
EOF
  exit 1
fi

SEQUENCE="$1"
HMR_TYPE="${2:-gv}"
PYTHON_BIN="${PYTHON_BIN:-/data/ubuntu/envs/crisp-v2/bin/python}"
SCENE_ARTIFACT_OUTPUT_DIR="${SCENE_ARTIFACT_OUTPUT_DIR:-$REPO_ROOT/results/output/scene_vggt_omega_consistent_camera_min1}"
IMAGE_SPLIT_ROOT="${IMAGE_SPLIT_ROOT:-$REPO_ROOT/data/agentic_demo_img}"
CRISP_AGENTIC_OUTPUT_ROOT="${CRISP_AGENTIC_OUTPUT_ROOT:-/data/ubuntu/artifacts/crisp-agentic}"
CRISP_AGENTIC_CLUSTER_ROOT="${CRISP_AGENTIC_CLUSTER_ROOT:-$CRISP_AGENTIC_OUTPUT_ROOT/clusters}"
AGENTIC_MAX_ITERATIONS="${AGENTIC_MAX_ITERATIONS:-3}"
CODEX_BIN="${CODEX_BIN:-/usr/local/bin/codex}"
LOG_ROOT="${CRISP_AGENTIC_LOG_ROOT:-/data/ubuntu/logs/crisp-agentic}"

SEQ_ROOT="$SCENE_ARTIFACT_OUTPUT_DIR/$SEQUENCE/$HMR_TYPE"
POINTCLOUD="$SEQ_ROOT/nksr_input/pointcloud_world.npz"
BASELINE="$SEQ_ROOT/scene_mesh_sqs/sqs_params.npz"
if [[ ! -f "$BASELINE" ]]; then
  BASELINE="$SEQ_ROOT/scene_mesh_sqs/sqs_params.npy"
fi
IMAGE_ROOT="$IMAGE_SPLIT_ROOT/$SEQUENCE"
CLUSTER_ROOT="$CRISP_AGENTIC_CLUSTER_ROOT/$SEQUENCE/$HMR_TYPE"
if [[ ! -d "$CLUSTER_ROOT" ]]; then
  CLUSTER_ROOT="$REPO_ROOT/vis/$SEQUENCE/$HMR_TYPE"
fi
if [[ ! -d "$CLUSTER_ROOT" ]]; then
  CLUSTER_ROOT="$REPO_ROOT/vis_scripts/viser_m/vis/$SEQUENCE/$HMR_TYPE"
fi
OUTPUT_DIR="$CRISP_AGENTIC_OUTPUT_ROOT/$SEQUENCE"

if [[ ! -f "$POINTCLOUD" ]]; then
  echo "Missing point cloud: $POINTCLOUD" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"
command=(
  "$PYTHON_BIN" -m agentic_fitting.run
  --pointcloud "$POINTCLOUD"
  --output-dir "$OUTPUT_DIR"
  --repo-root "$REPO_ROOT"
  --python-bin "$PYTHON_BIN"
  --codex-bin "$CODEX_BIN"
  --image-root "$IMAGE_ROOT"
  --cluster-root "$CLUSTER_ROOT"
  --max-iterations "$AGENTIC_MAX_ITERATIONS"
  --force
)
if [[ -f "$BASELINE" ]]; then
  command+=(--baseline-params "$BASELINE")
fi

cd "$REPO_ROOT"
"${command[@]}" 2>&1 | tee "$LOG_ROOT/${SEQUENCE}-agentic.log"

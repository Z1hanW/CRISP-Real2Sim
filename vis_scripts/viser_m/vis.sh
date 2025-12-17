#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sequence_name> [detailed_planes] [static_camera] [segment_mode] [sq_loss_threshold] [save_mode] [save_clustering]" >&2
    exit 1
fi

HMR_TYPE="${HMR_TYPE:-gv}"

# Optional knob: set DETAILED_PLANES=1 or pass "on" as the second argument to
# enable aggressive planar fitting that keeps very small regions.
DETAIL_FLAG_INPUT="${2:-${DETAILED_PLANES:-off}}"
DETAIL_FLAG_NORMALIZED="$(echo "${DETAIL_FLAG_INPUT}" | tr '[:upper:]' '[:lower:]')"

# Optional knob: set STATIC_CAMERA=1 or pass "on" as the third argument to
# assume a static camera and average depth maps across frames.
STATIC_FLAG_INPUT="${3:-${STATIC_CAMERA:-off}}"
STATIC_FLAG_NORMALIZED="$(echo "${STATIC_FLAG_INPUT}" | tr '[:upper:]' '[:lower:]')"

# Optional knob: set SEGMENT_MODE (frame_union|cluster_3d) or pass as the fourth argument.
SEGMENT_MODE_INPUT="${4:-${SEGMENT_MODE:-frame_union}}"
SEGMENT_MODE_NORMALIZED="$(echo "${SEGMENT_MODE_INPUT}" | tr '[:upper:]' '[:lower:]')"

# Optional knob: set SQ_LOSS_THRESHOLD or pass as the fifth argument to drop high-loss planes.
SQ_LOSS_THRESHOLD_INPUT="${5:-${SQ_LOSS_THRESHOLD:-}}"

# Optional knob: set SAVE_MODE=1 or pass as the sixth argument to save visualizations.
SAVE_MODE_INPUT="${6:-${SAVE_MODE:-on}}"
SAVE_MODE_NORMALIZED="$(echo "${SAVE_MODE_INPUT}" | tr '[:upper:]' '[:lower:]')"

# Optional knob: set SAVE_CLUSTERING=1 or pass as the seventh argument to dump clustering artifacts.
SAVE_CLUSTERING_INPUT="${7:-${SAVE_CLUSTERING:-off}}"
SAVE_CLUSTERING_NORMALIZED="$(echo "${SAVE_CLUSTERING_INPUT}" | tr '[:upper:]' '[:lower:]')"

ROOT_DIR="../../results/output/scene/${1}_${HMR_TYPE}_sgd_cvd_hr.npz"

EXTRA_ARGS=()
if [[ "${DETAIL_FLAG_NORMALIZED}" == "1" || \
      "${DETAIL_FLAG_NORMALIZED}" == "true" || \
      "${DETAIL_FLAG_NORMALIZED}" == "on" || \
      "${DETAIL_FLAG_NORMALIZED}" == "detail" || \
      "${DETAIL_FLAG_NORMALIZED}" == "detailed" ]]; then
    EXTRA_ARGS+=("--detailed-planes")
fi

if [[ "${STATIC_FLAG_NORMALIZED}" == "1" || \
      "${STATIC_FLAG_NORMALIZED}" == "true" || \
      "${STATIC_FLAG_NORMALIZED}" == "on" || \
      "${STATIC_FLAG_NORMALIZED}" == "stat" || \
      "${STATIC_FLAG_NORMALIZED}" == "static" ]]; then
    EXTRA_ARGS+=("--static-camera")
fi

if [[ "${SEGMENT_MODE_NORMALIZED}" == "cluster_3d" || \
      "${SEGMENT_MODE_NORMALIZED}" == "cluster" || \
      "${SEGMENT_MODE_NORMALIZED}" == "3d" ]]; then
    EXTRA_ARGS+=("--segment-mode" "cluster_3d")
else
    EXTRA_ARGS+=("--segment-mode" "frame_union")
fi

if [[ -n "${SQ_LOSS_THRESHOLD_INPUT}" ]]; then
    EXTRA_ARGS+=("--sq-loss-threshold" "${SQ_LOSS_THRESHOLD_INPUT}")
fi

if [[ "${SAVE_MODE_NORMALIZED}" == "1" || \
      "${SAVE_MODE_NORMALIZED}" == "true" || \
      "${SAVE_MODE_NORMALIZED}" == "on" || \
      "${SAVE_MODE_NORMALIZED}" == "yes" || \
      "${SAVE_MODE_NORMALIZED}" == "save" ]]; then
    EXTRA_ARGS+=("--save-mode")
fi

if [[ "${SAVE_CLUSTERING_NORMALIZED}" == "1" || \
      "${SAVE_CLUSTERING_NORMALIZED}" == "true" || \
      "${SAVE_CLUSTERING_NORMALIZED}" == "on" || \
      "${SAVE_CLUSTERING_NORMALIZED}" == "yes" || \
      "${SAVE_CLUSTERING_NORMALIZED}" == "dump" || \
      "${SAVE_CLUSTERING_NORMALIZED}" == "save" ]]; then
    EXTRA_ARGS+=("--save-clustering")
fi

python visualizer_megasam.py --data "$ROOT_DIR" --hmr_type "${HMR_TYPE}" "${EXTRA_ARGS[@]}"

# video_777_777
# video_1533_1599
#
# 19_indoor_walk_off_mvs

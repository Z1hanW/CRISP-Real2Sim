#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PYTHON_SHIM_DIR="$REPO_ROOT/runtime_shims"

usage() {
    cat <<'EOF'
Usage:
  bash run_crisp_video.sh --demo
  bash run_crisp_video.sh /abs/path/to/data_split_root

Examples:
  bash run_crisp_video.sh --demo
  bash run_crisp_video.sh /home/ubuntu/FAR/CRISP-Real2Sim/data/demo

The input should match the repository convention:
  <root>_videos/*.mp4
or
  <root>_img/<sequence>/*
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

ROOT_INPUT="${1:-}"
if [[ "$ROOT_INPUT" == "--help" || "$ROOT_INPUT" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$ROOT_INPUT" == "--demo" ]]; then
    DEMO_ROOT="$REPO_ROOT/data/demo"
    DEMO_SRC="${DEMO_ROOT}_videos/wall-kicking.mp4"

    if [[ ! -f "$DEMO_SRC" ]]; then
        echo "Demo video not found: $DEMO_SRC" >&2
        echo "Download demo assets first, or pass your own split root." >&2
        exit 1
    fi

    ROOT_DIR="$DEMO_ROOT"
else
    ROOT_DIR="${ROOT_INPUT%/}"
fi

export PYTHONPATH="$PYTHON_SHIM_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export PYTHONUNBUFFERED=1
CRISP_BIN_DIR="${CRISP_BIN_DIR:-/home/ubuntu/miniconda3/envs/crisp/bin}"
if [[ -d "$CRISP_BIN_DIR" ]]; then
    export PATH="$CRISP_BIN_DIR:$PATH"
fi

cd "$REPO_ROOT/scripts"
exec bash all_gv.sh "$ROOT_DIR"

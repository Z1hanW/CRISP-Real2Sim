#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/test_scene_backend.sh <split_or_path> [hmr_type] [backend]" >&2
  echo "Example: bash scripts/test_scene_backend.sh data/anything-chair gv vggt_omega" >&2
  exit 1
fi

ROOT="${1%/}"
HMR_TYPE="${2:-gv}"
BACKEND_RAW="${3:-${SCENE_RECON_BACKEND:-megasam}}"
BACKEND_RAW="${BACKEND_RAW,,}"

case "$BACKEND_RAW" in
  megasam|moge|tapip3d)
    BACKEND="megasam"
    RAW_PRIORS_ROOT="${SCENE_RAW_PRIORS_ROOT:-$REPO_ROOT/results/init/vslam/raw_mega_priors}"
    HMR_ROOT_DEFAULT="$REPO_ROOT/results/init/hmr"
    ;;
  vggt_omega|vggt-omega|vggt)
    BACKEND="vggt_omega"
    RAW_PRIORS_ROOT="${SCENE_RAW_PRIORS_ROOT:-$REPO_ROOT/results/init/vslam/raw_vggt_omega_priors}"
    HMR_ROOT_DEFAULT="$REPO_ROOT/results/init/hmr_vggt_omega"
    ;;
  *)
    echo "[test_scene_backend] unknown backend '$BACKEND_RAW'" >&2
    exit 2
    ;;
esac

SCENE_OUTPUT_DIR="${SCENE_OUTPUT_DIR:-$REPO_ROOT/results/output/scene}"
HMR_ROOT="${HMR_ROOT:-$HMR_ROOT_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCENE_RECON_BACKEND="$BACKEND" \
SCENE_RAW_PRIORS_ROOT="$RAW_PRIORS_ROOT" \
SCENE_OUTPUT_DIR="$SCENE_OUTPUT_DIR" \
HMR_ROOT="$HMR_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$SCRIPT_DIR/6_align.sh" "$ROOT" "$HMR_TYPE"

if [[ "$ROOT" == *_img ]]; then
  DATA_PATH="$ROOT"
else
  DATA_PATH="${ROOT}_img"
fi

"$PYTHON_BIN" - "$DATA_PATH" "$HMR_TYPE" "$BACKEND" "$RAW_PRIORS_ROOT" "$SCENE_OUTPUT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

data_path = Path(sys.argv[1])
hmr_type = sys.argv[2]
backend = sys.argv[3]
raw_root = Path(sys.argv[4])
scene_output_dir = Path(sys.argv[5])

summaries = []
for seq_dir in sorted(p for p in data_path.iterdir() if p.is_dir()):
    seq = seq_dir.name
    raw_path = raw_root / f"{seq}.npz"
    if backend == "megasam":
        scene_path = scene_output_dir / f"{seq}_{hmr_type}_sgd_cvd_hr.npz"
    else:
        scene_path = scene_output_dir / f"{seq}_{backend}_{hmr_type}_sgd_cvd_hr.npz"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw prior: {raw_path}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing aligned scene: {scene_path}")

    raw = np.load(raw_path)
    scene = np.load(scene_path)
    required = ("scale", "images", "depths", "intrinsic", "cam_c2w", "valid_frame_indices")
    missing = [key for key in required if key not in scene]
    if missing:
        raise KeyError(f"{scene_path} missing keys: {missing}")

    num_frames = int(scene["images"].shape[0])
    if num_frames == 0:
        raise ValueError(f"{scene_path} has no valid frames")
    if scene["depths"].shape[0] != num_frames or scene["cam_c2w"].shape[0] != num_frames:
        raise ValueError(f"{scene_path} has inconsistent frame counts")

    backend_value = str(
        np.asarray(
            scene["scene_reconstruction_backend"]
            if "scene_reconstruction_backend" in scene
            else backend
        ).reshape(-1)[0]
    )
    if backend_value != backend:
        raise ValueError(f"{scene_path} backend={backend_value!r}, expected {backend!r}")

    summaries.append(
        {
            "sequence": seq,
            "backend": backend_value,
            "raw_prior": str(raw_path),
            "scene": str(scene_path),
            "scale": float(np.asarray(scene["scale"]).reshape(-1)[0]),
            "raw_frames": int(raw["images"].shape[0]),
            "valid_frames": num_frames,
            "image_shape": list(scene["images"].shape[1:]),
            "depth_shape": list(scene["depths"].shape[1:]),
            "has_obj_masks": "obj_masks" in scene,
        }
    )

print(json.dumps(summaries, indent=2))
PY

echo "[test_scene_backend] OK"

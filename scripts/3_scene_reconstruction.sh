#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${SCENE_RECON_BACKEND:-megasam}"
BACKEND="${BACKEND,,}"

case "$BACKEND" in
  megasam|moge|tapip3d)
    exec bash "$SCRIPT_DIR/3_megasam.sh" "$@"
    ;;
  vggt_omega|vggt-omega|vggt)
    exec bash "$SCRIPT_DIR/3_vggt_omega.sh" "$@"
    ;;
  *)
    echo "[scene_reconstruction] unknown SCENE_RECON_BACKEND='$SCENE_RECON_BACKEND'" >&2
    echo "[scene_reconstruction] supported: megasam, vggt_omega" >&2
    exit 2
    ;;
esac

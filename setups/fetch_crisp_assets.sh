#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
TORCH_HUB_CHECKPOINTS="$TORCH_HUB_DIR/checkpoints"

mkdir -p "$TORCH_HUB_DIR" "$TORCH_HUB_CHECKPOINTS"

echo "[1/7] fetch AutoMask checkpoints"
if [[ ! -f "$ROOT/prep/AutoMask/checkpoints/sam2_hiera_large.pt" ]]; then
    (cd "$ROOT/prep/AutoMask/checkpoints" && sh ./download_ckpts.sh)
fi

echo "[2/7] fetch MogeSAM checkpoints"
if [[ ! -f "$ROOT/prep/MogeSAM/checkpoints/tapip3d_final.pth" ]]; then
    mkdir -p "$ROOT/prep/MogeSAM/checkpoints"
    wget -q -O "$ROOT/prep/MogeSAM/checkpoints/tapip3d_final.pth" \
        https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
fi
if [[ ! -f "$ROOT/prep/MogeSAM/third_party/megasam/Depth-Anything/checkpoints/depth_anything_vitl14.pth" ]]; then
    mkdir -p "$ROOT/prep/MogeSAM/third_party/megasam/Depth-Anything/checkpoints"
    curl -L \
        -o "$ROOT/prep/MogeSAM/third_party/megasam/Depth-Anything/checkpoints/depth_anything_vitl14.pth" \
        "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
fi
if [[ ! -f "$ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/raft-things.pth" ]]; then
    mkdir -p "$ROOT/prep/MogeSAM/third_party/megasam/cvd_opt"
    python -m gdown --folder --fuzzy \
        "https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT" \
        -O "$ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/" \
        --remaining-ok --continue
    if [[ -f "$ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/models/raft-things.pth" ]]; then
        mv -f \
            "$ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/models/raft-things.pth" \
            "$ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/raft-things.pth"
    fi
fi

echo "[3/7] fetch HMR checkpoints"
if [[ ! -f "$ROOT/prep/HMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" ]]; then
    mkdir -p "$ROOT/prep/HMR/inputs" "$ROOT/prep/HMR/outputs"
    python -m gdown --folder \
        "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD" \
        -O "$ROOT/prep/HMR/inputs/" \
        --remaining-ok --continue
fi

echo "[4/7] fetch SMPL neutral"
if [[ ! -f "$ROOT/prep/HMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" ]]; then
    mkdir -p "$ROOT/prep/HMR/inputs/checkpoints/body_models/smpl" "$ROOT/prep/data/smpl"
    python -m gdown \
        "https://drive.google.com/uc?id=1bcj7zu3K_u2c_g1ed5FchqZQP6xTJPtN" \
        -O "$ROOT/prep/HMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" \
        --continue
fi
cp -f \
    "$ROOT/prep/HMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" \
    "$ROOT/prep/data/smpl/SMPL_NEUTRAL.pkl"
if [[ -f "$ROOT/prep/data/smpl/J_regressor_extra.npy" ]]; then
    cp -f \
        "$ROOT/prep/data/smpl/J_regressor_extra.npy" \
        "$ROOT/prep/HMR/inputs/checkpoints/body_models/smpl/J_regressor_extra.npy"
else
    echo "Warning: prep/data/smpl/J_regressor_extra.npy not found; postprocess SMPL helpers may fail until it is staged."
fi

echo "[5/7] fetch torch hub repos"
if [[ ! -d "$TORCH_HUB_DIR/facebookresearch_co-tracker_main/.git" ]]; then
    rm -rf "$TORCH_HUB_DIR/facebookresearch_co-tracker_main"
    git clone --depth 1 https://github.com/facebookresearch/co-tracker.git \
        "$TORCH_HUB_DIR/facebookresearch_co-tracker_main"
fi
if [[ ! -d "$TORCH_HUB_DIR/facebookresearch_dinov2_main/.git" ]]; then
    rm -rf "$TORCH_HUB_DIR/facebookresearch_dinov2_main"
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git \
        "$TORCH_HUB_DIR/facebookresearch_dinov2_main"
fi

echo "[6/7] fetch torch hub checkpoints"
if [[ ! -f "$TORCH_HUB_CHECKPOINTS/scaled_offline.pth" ]]; then
    wget -q -O "$TORCH_HUB_CHECKPOINTS/scaled_offline.pth" \
        https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
fi

echo "[7/7] done"

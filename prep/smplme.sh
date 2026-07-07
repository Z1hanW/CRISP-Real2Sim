#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="$SCRIPT_DIR/data"
SMPL_DIR="$DATA_DIR/smpl"
SMPLX_DIR="$DATA_DIR/smplx"
SMPLIFY_EXTRACT_DIR="$SMPL_DIR/smplify"
SMPL_ZIP="$DATA_DIR/smpl.zip"
SMPLIFY_ZIP="$DATA_DIR/smplify.zip"
SMPLX_ZIP="$DATA_DIR/smplx.zip"

mkdir -p "$SMPL_DIR"
mkdir -p "$SMPLX_DIR"
mkdir -p HMR/inputs/checkpoints/body_models/smpl
mkdir -p HMR/inputs/checkpoints/body_models/smplx

download_zip_if_missing() {
    local zip_path="$1"
    local url="$2"
    local label="$3"

    if [[ -f "$zip_path" ]]; then
        echo "Using existing $label zip: $zip_path"
        return 0
    fi

    USERNAME="${SMPL_USERNAME:-}"
    PASSWORD="${SMPL_PASSWORD:-}"

    if [[ -z "$USERNAME" ]]; then
        echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
        read -r -p "Username (SMPL):" USERNAME
    fi

    if [[ -z "$PASSWORD" ]]; then
        read -r -s -p "Password (SMPL):" PASSWORD
        echo
    fi

    wget --post-data "username=$USERNAME&password=$PASSWORD" "$url" -O "$zip_path" --no-check-certificate --continue
}

download_zip_if_missing "$SMPL_ZIP" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' "SMPL"
download_zip_if_missing "$SMPLIFY_ZIP" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' "SMPLify"
download_zip_if_missing "$SMPLX_ZIP" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' "SMPL-X"

unzip -tq "$SMPL_ZIP" >/dev/null
unzip -tq "$SMPLIFY_ZIP" >/dev/null
unzip -tq "$SMPLX_ZIP" >/dev/null

rm -rf "$SMPL_DIR/smpl" "$SMPLIFY_EXTRACT_DIR" "$SMPLX_DIR/models"

unzip -o "$SMPL_ZIP" -d "$SMPL_DIR"
unzip -o "$SMPLIFY_ZIP" -d "$SMPLIFY_EXTRACT_DIR"
unzip -o "$SMPLX_ZIP" -d "$SMPLX_DIR"

cp "$SMPL_DIR/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl" "$SMPL_DIR/SMPL_FEMALE.pkl"
cp "$SMPL_DIR/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl" "$SMPL_DIR/SMPL_MALE.pkl"
cp "$SMPL_DIR/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl" HMR/inputs/checkpoints/body_models/smpl/SMPL_FEMALE.pkl
cp "$SMPL_DIR/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl" HMR/inputs/checkpoints/body_models/smpl/SMPL_MALE.pkl

mv "$SMPLIFY_EXTRACT_DIR/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" "$SMPL_DIR/SMPL_NEUTRAL.pkl"
cp "$SMPL_DIR/SMPL_NEUTRAL.pkl" HMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
if [[ -f "$SMPL_DIR/J_regressor_extra.npy" ]]; then
    cp "$SMPL_DIR/J_regressor_extra.npy" HMR/inputs/checkpoints/body_models/smpl/J_regressor_extra.npy
else
    echo "Warning: $SMPL_DIR/J_regressor_extra.npy not found; postprocess SMPL helpers may fail until it is staged."
fi
rm -rf "$SMPLIFY_EXTRACT_DIR"

cp "$SMPLX_DIR/models/smplx/SMPLX_FEMALE.npz" HMR/inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz
cp "$SMPLX_DIR/models/smplx/SMPLX_MALE.npz" HMR/inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz
cp "$SMPLX_DIR/models/smplx/SMPLX_NEUTRAL.npz" HMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz

echo "SMPL / SMPLify / SMPL-X zip files staged successfully."

#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL Male and Female model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p HMR/inputs/checkpoints/body_models
mkdir -p HMR/inputs/checkpoints/body_models/smpl
mkdir -p HMR/inputs/checkpoints/body_models/smplx


wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './data/smpl/smpl.zip' --no-check-certificate --continue
unzip data/smpl/smpl.zip -d data/smpl/smpl
mv data/smpl/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl HMR/inputs/checkpoints/body_models/smpl/SMPL_FEMALE.pkl
mv data/smpl/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl HMR/inputs/checkpoints/body_models/smpl/SMPL_MALE.pkl
rm -rf data/smpl/smpl
rm -rf data/smpl/smpl.zip

wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip" -O './data/smplx/smplx.zip' --no-check-certificate --continue
unzip data/smplx/smplx.zip -d data/smplx

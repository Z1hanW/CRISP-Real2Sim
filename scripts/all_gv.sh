
#!/usr/bin/env bash
#set -euo pipefail

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
# eval "$(conda shell.bash hook)"


ROOT_DIR="$1"
HMR_TYPE="gv"



bash 1_video2imgs.sh "$ROOT_DIR"
bash 2_get_mask.sh "$ROOT_DIR"
bash 3_megasam.sh "$ROOT_DIR" 
bash 4_post_camera.sh "$ROOT_DIR"
bash 5_grav.sh "$ROOT_DIR"
bash 0_ufm.sh "$ROOT_DIR"
bash 6_align.sh "$ROOT_DIR" "$HMR_TYPE" 
# bash 7_glue_sqs.sh "$ROOT_DIR" "$HMR_TYPE" 


# sh re_glue_sqs.sh "$ROOT_DIR" "$HMR_TYPE" 
# sh 8_post_scene_process.sh "$ROOT_DIR"
# sh 9_train_my_agent.sh "$ROOT_DIR"

# sh 1_video2imgs.sh "$ROOT_DIR"
# bash 2_get_mask.sh "$ROOT_DIR"
# sh 3_megasam.sh "$ROOT_DIR" 
# sh 4_post_camera.sh "$ROOT_DIR"
#sh 5_grav.sh "$ROOT_DIR"
# sh 6_align.sh "$ROOT_DIR" "$HMR_TYPE" 
#sh re_glue_sqs.sh "$ROOT_DIR" "$HMR_TYPE" 


#!/usr/bin/env bash


################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":

ROOT_DIR="$1"
HMR_TYPE="gv"



bash 7_glue_sqs.sh "$ROOT_DIR" "$HMR_TYPE" 
# sh 8_post_scene_process.sh "$ROOT_DIR" "$HMR_TYPE" 
# sh 9_train_my_agent.sh "$ROOT_DIR"





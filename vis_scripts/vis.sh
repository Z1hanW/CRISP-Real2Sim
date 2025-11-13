
#!/usr/bin/env bash

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
eval "$(conda shell.bash hook)"


conda activate sqs
HMR_TYPE='gv'
ROOT_DIR="/data3/zihanwa3/_Robotics/_vision/mega-sam/postprocess/${1}_${HMR_TYPE}_sgd_cvd_hr.npz"

python viser_m/visualizer_megasam.py --data "$ROOT_DIR" --hmr_type "${HMR_TYPE}"

# video_777_777
# video_1533_1599
#
# 19_indoor_walk_off_mvs





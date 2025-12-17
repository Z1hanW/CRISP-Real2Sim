
#DATA_PATH='/data3/zihanwa3/_Robotics/_data/door_push'
DATA_PATH="${1}_img"
HTM_TYPE="${2:gv}"


for folder in "$DATA_PATH"/*/
do
    seq=$(basename "$folder")
    python post_process.py --scene_name 56_outdoor_stairs_up_down --hmr_type "$HTM_TYPE"
done


echo "All demos completed successfully."

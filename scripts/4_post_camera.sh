cd ../prep/MogeSAM
  

#DATA_PATH='/data3/zihanwa3/_Robotics/_data/door_push'
DATA_PATH="${1}_img"



for folder in "$DATA_PATH"/*/
do
    python post_cam.py --input_dir "$DATA_PATH" --scene_name "$(basename "$folder")"
done
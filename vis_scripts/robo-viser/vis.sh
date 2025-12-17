name="${1:?Give base name like 0831_19_indoor_walk_off_mvs_ours}"
DATE="${name%%_*}"          # -> 0831
without_first="${name#*_}"   # -> 19_indoor_walk_off_mvs_ours
SCENE="${without_first%_*}" # -> 19_indoor_walk_off_mvs
METHOD="${name##*_}"           # -> ours





# python vis_c_dual.py 0917_pkr_5_ours 0917_pkr_5_ours
python visualize_viser_robot_z.py --scene $SCENE --types $METHOD --parent_name $DATE --method $METHOD
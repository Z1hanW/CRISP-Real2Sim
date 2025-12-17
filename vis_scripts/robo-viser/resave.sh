#!/usr/bin/env bash
set -euo pipefail


if [ $# -lt 1 ]; then
  echo "Usage: $0 <name> [options]"
  echo "Example: $0 0831_19_indoor_walk_off_mvs_ours"
  echo "         $0 0831_19_indoor_walk_off_mvs_ours_trimesh"
  exit 1
fi

name="$1"
shift

# 0831
DATE="${name%%_*}"

# 19_indoor_walk_off_mvs_ours[_trimesh]
without_first="${name#*_}"

if [[ "$name" == *_trimesh ]]; then
  # METHOD = 最后两段（如：ours_trimesh）
  last="${without_first##*_}"          # trimesh
  without_last="${without_first%_*}"   # 去掉最后一段
  prev="${without_last##*_}"           # ours
  METHOD="${prev}_${last}"             # ours_trimesh

  # SCENE = 去掉最后两段
  SCENE="${without_first%_*_*}"        # 19_indoor_walk_off_mvs
else
  # METHOD = 最后一段（如：ours）
  METHOD="${without_first##*_}"

  # SCENE = 去掉最后一段
  SCENE="${without_first%_*}"          # 19_indoor_walk_off_mvs
fi

echo "DATE=$DATE"
echo "SCENE=$SCENE"
echo "METHOD=$METHOD"

python resave_unified_data.py \
  --scene "$SCENE" \
  --types "$METHOD" \
  --parent_name "$DATE" \
  --method "$METHOD" \
  --run_name "$name" \
  --save_mode \
  "$@"

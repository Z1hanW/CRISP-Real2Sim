#!/usr/bin/env bash
set -euo pipefail

# Check if two names are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <name_1> <name_2> [options]"
    echo "Example: $0 0831_19_indoor_walk_off_mvs_ours 0831_19_indoor_walk_off_mvs_baseline"
    echo "         $0 0831_19_indoor_walk_off_mvs_ours_trimesh 0831_19_indoor_walk_off_mvs_baseline"
    exit 1
fi

name_1="${1}"
name_2="${2}"

# Shift to get remaining arguments
shift 2

# Display what will be compared
echo "==========================================="
echo "Dual Robot Visualization"
echo "==========================================="
echo "Robot 1: $name_1"
echo "Robot 2: $name_2"
echo "==========================================="

# Run the Python script with both names
python vis_c_dual.py "$name_1" "$name_2" "$@"
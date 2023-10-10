#!/bin/bash

# Get command line arguments
DIRECTORY="$1"
ITERATIONS="$2"
THRESHOLD="$3"

# Check if all arguments are provided
if [ -z "$DIRECTORY" ] || [ -z "$ITERATIONS" ] || [ -z "$THRESHOLD" ]; then
    echo "Usage: ./run_segmentation.sh <directory> <iterations> <threshold>"
    exit 1
fi

# Loop over all .pcd files in the directory
for file in "$DIRECTORY"/*.ply; do
    python3 /home/scpmaotj/Github/mrdja/mrdja/experiments/cli_for_segmenting_with_open3d.py --file "$file" --iterations "$ITERATIONS" --threshold "$THRESHOLD"
done

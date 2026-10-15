#!/bin/bash

### Queue
#BSUB -q c02516

### GPU
#BSUB -gpu "num=1:mode=exclusive_process"

### Job name
#BSUB -J per_frame_2D

### CPU cores
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### Memory
#BSUB -R "rusage[mem=20GB]"

### Maximum runtime
#BSUB -W 12:00

### Output files
#BSUB -o per_frame_2D/results/per_frame_2D_%J.out
#BSUB -e per_frame_2D/results/per_frame_2D_%J.err


# Go to project directory
cd ~/IDLCV_02

# Activate virtual environment
source .venv/bin/activate

# Run late fusion experiment
# python -m per_frame_2D.per_frame_2D_main
python -m per_frame_2D.per_frame_2D_evaluate
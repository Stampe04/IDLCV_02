#!/bin/bash

### Queue
#BSUB -q c02516

### GPU
#BSUB -gpu "num=1:mode=exclusive_process"

### Job name
#BSUB -J late_fusion_2D

### CPU cores
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### Memory
#BSUB -R "rusage[mem=20GB]"

### Maximum runtime
#BSUB -W 12:00

### Output files
#BSUB -o late_fusion_2D/results/late_fusion_%J.out
#BSUB -e late_fusion_2D/results/late_fusion_%J.err


# Go to project directory
cd ~/IDLCV_02

# Activate virtual environment
source .venv/bin/activate

# Run late fusion experiment
python -m late_fusion_2D.late_fusion_2D_main
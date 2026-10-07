#!/bin/bash

### Queue
#BSUB -q c02516

### GPU
#BSUB -gpu "num=1:mode=exclusive_process"

### Job name
#BSUB -J early_fusion_2D

### CPU cores
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### Memory
#BSUB -R "rusage[mem=20GB]"

### Maximum runtime
#BSUB -W 12:00

### Output files
#BSUB -o early_fusion_2D/results/early_fusion_%J.out
#BSUB -e early_fusion_2D/results/early_fusion_%J.err


# Go to project directory
cd ~/IDLCV_02

# Activate virtual environment
source .venv/bin/activate

# Run early fusion experiment
python -m early_fusion_2D.early_fusion_2D_main
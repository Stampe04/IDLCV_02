#!/bin/bash

### Queue
#BSUB -q c02516

### GPU
#BSUB -gpu "num=1:mode=exclusive_process"

### Job name
#BSUB -J ResNet_3D

### CPU cores
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### Memory
#BSUB -R "rusage[mem=20GB]"

### Maximum runtime
#BSUB -W 12:00

### Output files
#BSUB -o ResNet_3D_%J.out
#BSUB -e ResNet_3D_%J.err


# Go to project directory
cd ~/IDLCV_02

# Activate virtual environment
source .venv/bin/activate

# Run early fusion experiment
python -m resnet_3D.main
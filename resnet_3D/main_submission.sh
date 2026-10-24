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


# LSF may execute a submitted stdin script from its .lsbatch directory, so
# do not derive these paths from BASH_SOURCE.
PROJECT_ROOT="${IDLCV_PROJECT_ROOT:-/zhome/99/3/215784/Desktop/archive/IDLCV_02}"
SCRIPT_DIR="${PROJECT_ROOT}/resnet_3D"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# Experiment configuration.
export RESNET_3D_DATA_ROOT="/dtu/datasets1/02516/ufc10"
export RESNET_3D_BATCH_SIZE=16
export RESNET_3D_NUM_WORKERS=4
export RESNET_3D_NUM_EPOCHS=25
export RESNET_3D_LAYERS="2,2,2,2"
export RESNET_3D_BASE_CHANNELS=64
export RESNET_3D_DROPOUT=0
export RESNET_3D_LEARNING_RATE=3e-4
export RESNET_3D_SCHEDULER_FACTOR=0.5
export RESNET_3D_SCHEDULER_PATIENCE=5
export RESNET_3D_RESULTS_DIR="${SCRIPT_DIR}/results_leakage_normal_3_imagenette-weights"

# Run without relying on the caller's current working directory.
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" "${SCRIPT_DIR}/main.py"
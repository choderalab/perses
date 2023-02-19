#!/bin/bash
#BSUB -P "asap"
#BSUB -J "mainseries[1-33]"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 12 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 24:00
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1

echo "changing directory to ${LS_SUBCWD}"
cd "$LS_SUBCWD"
conda activate perses-0.11.0-preview

# export environment information to file
conda list > environment.txt

# Report node in use
hostname

# Open eye license activation/env
export OE_LICENSE=~/.openeye/oe_license.txt

# Report CUDA info
env | sort | grep 'CUDA'

# Report GPU info
nvidia-smi -L
nvidia-smi --query-gpu=name --format=csv

# Generate verbose logs
export LOGLEVEL=INFO

python run-perses.py run --index $(( $LSB_JOBINDEX - 1 ))

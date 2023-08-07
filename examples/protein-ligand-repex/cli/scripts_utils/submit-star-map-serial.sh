#!/bin/bash
#BSUB -P "perses-tyk2"
#BSUB -J "perses-benchmark-tyk2-5ns-[1-14]"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 5:59
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

set -xeuo pipefail

source ~/.bashrc
OPENMM_CPU_THREADS=1

echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD
conda activate perses-dev

# Report node in use
hostname

# Open eye license activation/env
export OE_LICENSE=~/.openeye/oe_license.txt

# Report CUDA info
env | sort | grep 'CUDA'

# Report GPU info
nvidia-smi -L
nvidia-smi --query-gpu=name --format=csv

# launching a benchmark pair (target, edge) per job (0-based thus substract 1)
# python run_benchmarks.py --target tyk2 --edge $(( $LSB_JOBINDEX - 1 ))
target_ligand=${LSB_JOBINDEX}
perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:${target_ligand} --override n_cycles:5000 --override trajectory_directory:5ns_lig0to${target_ligand}

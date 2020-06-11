#!/bin/bash
#BSUB -P "testing"
#BSUB -J "throm[1-110]"
#BSUB -n 1
#BSUB -R rusage[mem=16]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W  03:00
#BSUB -m "ls-gpu lg-gpu lt-gpu lp-gpu lg-gpu lu-gpu ld-gpu"
#BSUB -o out_%I.stdout 
##BSUB -cwd "/scratch/%U/%J"
#BSUB -eo out_%I.stderr
#BSUB -L /bin/bash

# quit on first error
set -e

OPENMM_CPU_THREADS=1

cd $LS_SUBCWD

# Launch my program.
module load cuda/10.1
env | sort | grep 'CUDA'
python run.py $LSB_JOBINDEX 

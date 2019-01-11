#!/bin/bash
#BSUB -J "neqeq[1-42]"
#BSUB -n 1
#BSUB -R rusage[mem=16]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W  6:00
#BSUB -We 5:30
#BSUB -m "ls-gpu lt-gpu lp-gpu lg-gpu"
#BSUB -o %J.stdout
##BSUB -cwd "/scratch/%U/%J"
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

# quit on first error
set -e

cd $LS_SUBCWD

# Launch my program.
module load cuda/9.2
python run_equilibrium.py input_options.yaml $LSB_JOBINDEX

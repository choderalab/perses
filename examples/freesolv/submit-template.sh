#!/bin/bash
#BSUB -J rj-hydration
#BSUB -n 1
#BSUB -R rusage[mem=16]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W  23:00
#BSUB -We 22:30
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

# quit on first error
set -e

# Change to working directory used for job submission
cd $LS_SUBCWD

# Launch my program.
module load cuda/9.2
python /home/pgrinaway/perses/examples/freesolv/freesolv_testsystem.py {}

#!/bin/bash
#BSUB -J "cdk2[1-10]"
#BSUB -n 1
#BSUB -R rusage[mem=16]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W  1:00
#BSUB -We 0:30
#BSUB -m "ls-gpu lt-gpu lp-gpu lg-gpu"
#BSUB -o %J.stdout
##BSUB -cwd "/scratch/%U/%J"
#BSUB -eo %J.stderr
#BSUB -L /bin/bash


# quit on first error
set -e

cd $LS_SUBCWD
module load cuda/9.2


run () {

inputfile="cdk2_sams"$1$2".yaml"
output_dir="ligs"$1"-"$2
cp cdk2_sams.yaml $inputfile 
sed -i "s/LIGANDA_INDEX/$1/g" $inputfile #replace liga 
sed -i "s/LIGANDB_INDEX/$2/g" $inputfile #replace ligb 
sed -i "s/OUTPUT_DIRECTORY/$output_dir/g" $inputfile #replace outputdirectory 
python ../../scripts/setup_relative_calculation.py $inputfile 
rm $inputfile

}

COUNTER=1 # tracks which ligand pair this job is performing
set -- {0..4} 
for liga; do
    shift
    for ligb; do
        if [ $COUNTER -eq $LSB_JOBINDEX ] 
        then
          # FORWARDS
          run $liga $ligb 
          # BACKWARDS 
          run $ligb $liga
        fi
        let COUNTER=COUNTER+1
    done
done

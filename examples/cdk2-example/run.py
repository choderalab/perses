import yaml 
import sys
import itertools
import os

def run_relative_perturbation(ligA, ligB,tidy=True):
    print('Starting relative calcluation of ligand {} to {}'.format(ligA,ligB))
    trajectory_directory = 'lig{}to{}'.format(ligA,ligB) 
    new_yaml = 'cdk2_repex_{}to{}.yaml'.format(ligA,ligB) 
    
    # rewrite yaml file
    with open('cdk2_repex.yaml', "r") as yaml_file:
        options = yaml.load(yaml_file)
    options['old_ligand_index'] = ligA
    options['new_ligand_index'] = ligB
    options['trajectory_directory'] = trajectory_directory
    with open(new_yaml, 'w') as outfile:
        yaml.dump(options, outfile)
    
    # run the simulation
    # change this to the path where your perses installation is
    os.system(f'python $PERSESHOME/perses/app/setup_relative_calculation.py {format(new_yaml)}')

    print('Relative calcluation of ligand {} to {} complete'.format(ligA,ligB))

    if tidy:
        os.remove(new_yaml)

    return

# work out which ligand pair to run
#!!! change this to the list of ligand pairs you want to compare
# the first ligand in the .sdf file will be 0 
ligand_pairs = [(1,10),(2,13),(7,10),(8,10)]
ligand1, ligand2 = ligand_pairs[int(sys.argv[1])-1] # jobarray starts at 1 

run_relative_perturbation(ligand1, ligand2)

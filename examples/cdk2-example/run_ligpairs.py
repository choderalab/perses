import yaml 
import sys
import itertools
import os

def run_relative_perturbation(ligA, ligB,tidy=True):
    print('Starting relative calcluation of ligand {} to {}'.format(ligA,ligB))
    trajectory_directory = 'lig{}to{}'.format(ligA,ligB) 
    new_yaml = 'cdk2_{}to{}sams.yaml'.format(ligA,ligB) 
    
    # rewrite yaml file
    with open('cdk2_sams.yaml', "r") as yaml_file:
        options = yaml.load(yaml_file)
    options['old_ligand_index'] = ligA
    options['new_ligand_index'] = ligB
    options['trajectory_directory'] = trajectory_directory
    with open(new_yaml, 'w') as outfile:
        yaml.dump(options, outfile)
    
    # run the simulation
    os.system('python ../../scripts/setup_relative_calculation.py {}'.format(new_yaml))

    print('Relative calcluation of ligand {} to {} complete'.format(ligA,ligB))

    if tidy:
        os.remove(new_yaml)

    return

# work out which ligand pair to run
n_ligands = range(0,16) # 16 ligands
ligand_pairs = list(itertools.combinations(n_ligands,2))
ligand1, ligand2 = ligand_pairs[int(sys.argv[1])-1] # jobarray starts at 1 

#running forwards
run_relative_perturbation(ligand1, ligand2)
run_relative_perturbation(ligand2, ligand1)

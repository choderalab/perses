import yaml 
import sys
import itertools
import os

def run_relative_perturbation(ligA, ligB,tidy=True):
    print(f'Starting relative calcluation of ligand {ligA} to {ligB}')
    trajectory_directory = f'lig{ligA}to{ligB}'
    new_yaml = f'constraints_{ligA}to{ligB}.yaml'
    
    # rewrite yaml file
    with open(f'constraints.yaml', "r") as yaml_file:
        options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    options['old_ligand_index'] = ligA
    options['new_ligand_index'] = ligB
    options['trajectory_directory'] = f'{trajectory_directory}'
    with open(new_yaml, 'w') as outfile:
        yaml.dump(options, outfile)
    
    # run the simulation
    os.system(f'perses-relative {new_yaml}')

    print(f'Relative calcluation of ligand {ligA} to {ligB} complete')

    if tidy:
        os.remove(new_yaml)

    return

# work out which ligand pair to run
ligand_pairs = [(0, 3), (0, 5), (0, 8), (0, 6), (0, 11), (14, 11), (14, 4), (14, 15), (5, 4), (9, 4), (9, 14), (8, 14), (12, 0), (12, 4), (7, 0), (7, 2), (2, 14), (4, 15), (13, 4), (13, 3), (13, 10), (13, 1), (6, 10), (6, 1)]
ligand1, ligand2 = ligand_pairs[int(sys.argv[1])-1] # jobarray starts at 1 

#running forwards
run_relative_perturbation(ligand2, ligand1)

import yaml 
import sys
import itertools
import os

def run_relative_perturbation(ligA, ligB, index, tidy=True):
    print('Starting relative calcluation of ligand {} to {}'.format(ligA,ligB))
    trajectory_directory = 'lig{}to{}'.format(ligA,ligB) 
    new_yaml = 'fah_{}to{}.yaml'.format(ligA,ligB) 
    
    # rewrite yaml file
    with open('fah_all.yaml', "r") as yaml_file:
        options = yaml.load(yaml_file)
    options['old_ligand_index'] = ligA
    options['new_ligand_index'] = ligB
    options['trajectory_directory'] = trajectory_directory
    with open(new_yaml, 'w') as outfile:
        yaml.dump(options, outfile)
    
    # run the simulation
    os.system(f'perses-fah {new_yaml} {index}')

    print('Relative calcluation of ligand {} to {} complete'.format(ligA,ligB))

    if tidy:
        os.remove(new_yaml)

    return

# work out which ligand pair to run
index = int(sys.argv[1])-1
ligands = [i for i in range(11)]
ligand_pairs = [pair for pair in itertools.combinations(ligands,2)]
backwards = [(j,i) for i,j in ligand_pairs]
ligand_pairs = ligand_pairs + backwards
ligand1, ligand2 = ligand_pairs[index] # jobarray starts at 1 

#running forwards
run_relative_perturbation(ligand1, ligand2,index)

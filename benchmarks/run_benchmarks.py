#!/usr/bin/env python

"""
CLI utility to automatically run benchmarks using data from the open force field protein-ligand benchmark at
https://github.com/openforcefield/protein-ligand-benchmark

It requires internet connection to function properly, by connecting to the mentioned repository.
"""
# TODO: Use plbenchmarks when conda package is available.

import argparse
import logging
import os
import yaml

from perses.app.setup_relative_calculation import run
from perses.utils.url_utils import retrieve_file_url
from perses.utils.url_utils import fetch_url_contents

# Setting logging level config
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=LOGLEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger()
_logger.setLevel(LOGLEVEL)

# global variables
base_repo_url = "https://github.com/openforcefield/protein-ligand-benchmark"


def concatenate_files(input_files, output_file):
    """
    Concatenate files given in input_files iterator into output_file.
    """
    input_files = list(input_files)  # handles both single or multiple files
    with open(output_file, 'w') as outfile:
        for filename in input_files:
            with open(filename) as infile:
                for line in infile:
                    outfile.write(line)


def get_target_dir(target_name):
    """
    Retrieves the target subdirectory in upstream repo structure given the target name.
    """
    # Targets information on branch
    targets_url = f"{base_repo_url}/raw/{branch}/data/targets.yml"
    with fetch_url_contents(targets_url) as response:
        targets_dict = yaml.safe_load(response.read())
    target_dir = targets_dict[target]['dir']
    return target_dir


def get_ligands_information(target, branch="main"):
    """
    Retrieves the ligands information in a dictionary given the target name,
    """
    # TODO: This part should be done using plbenchmarks API - once there is a conda pkg
    target_dir = get_target_dir(target)
    ligands_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/00_data/ligands.yml"
    with fetch_url_contents(ligands_url) as response:
        ligands_dict = yaml.safe_load(response.read())
    return ligands_dict


def generate_ligands_sdf(ligands_dict, target, branch='main'):
    """
    Generates input ligands.sdf file for running the simulation given a dictionary
    with the ligands information.

    Can handle previous and new structure of upstream repo to date (26-Oct-2022).
    """
    from urllib.error import HTTPError
    # Get target dir where to get ligands sdf files
    target_dir = get_target_dir(target)
    # Fetch ligands sdf files and concatenate them in one
    # TODO: This part should be done using plbenchmarks API - once there is a conda pkg
    try:
        ligands_file_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/02_ligands/ligands.sdf"
        ligand_files = retrieve_file_url(ligands_file_url)
        new_repo = True
    except HTTPError:
        _logger.info(f'Must be previous old repository structure. Trying downloading individual ligand files.')
        ligand_files = []
        for ligand in ligands_dict.keys():
            ligand_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/02_ligands/{ligand}/crd/{ligand}.sdf"
            ligand_file = retrieve_file_url(ligand_url)
            ligand_files.append(ligand_file)
        new_repo = False
    # concatenate sdf files as needed
    if new_repo:
        # TODO: Just retrieve the ligands file to the working directory without need to concatenate
        concatenate_files((ligand_files,), "ligands.sdf")  # "concatenate" single file
    else:
        concatenate_files(ligand_files, 'ligands.sdf')  # concatenate multiple files

def get_ligand_indices_from_edge(edge_index, target_name, branch="main"):
    """
    Returns the ligand indices from the edge index according to edges.yaml file in repo.

    Allows specifying a branch/revision in repo.
    """
    # fetch edges information
    # TODO: This part should be done using plbenchmarks API - once there is a conda pkg
    target_dir = get_target_dir(target_name)
    ligands_dict = get_ligands_information(target, branch=branch)
    edges_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/00_data/edges.yml"
    with fetch_url_contents(edges_url) as response:
        edges_dict = yaml.safe_load(response.read())
    edges_list = list(edges_dict.values())  # suscriptable edges object - note dicts are ordered for py>=3.7

    edge = edges_list[edge_index]
    ligand_a_name = edge['ligand_a']
    ligand_b_name = edge['ligand_b']
    # ligands list to get indices -- preserving same order as upstream yaml file
    ligands_list = list(ligands_dict.keys())
    lig_a_index = ligands_list.index(ligand_a_name)
    lig_b_index = ligands_list.index(ligand_b_name)
    return lig_a_index, lig_b_index


def run_relative_perturbation(lig_a_idx, lig_b_idx, reverse=False, tidy=True):
    """
    Perform relative free energy simulation using perses CLI.

    Parameters
    ----------
        lig_a_idx : int
            Index for first ligand (ligand A)
        lig_b_idx : int
            Index for second ligand (ligand B)
        reverse: bool
            Run the edge in reverse direction. Swaps the ligands.
        tidy : bool, optional
            remove auto-generated yaml files.

    Expects the target/protein pdb file in the same directory to be called 'target.pdb', and ligands file
    to be called 'ligands.sdf'.
    """
    _logger.info(f'Starting relative calculation of ligand {lig_a_idx} to {lig_b_idx}')
    trajectory_directory = f'out_{lig_a_idx}_{lig_b_idx}'
    new_yaml = f'relative_{lig_a_idx}_{lig_b_idx}.yaml'

    # read base template yaml file
    # TODO: template.yaml file is configured for Tyk2, check if the same options work for others.
    with open(f'template.yaml', "r") as yaml_file:
        options = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # TODO: add a step to perform some minimization - should help with NaNs
    # generate yaml file from template
    options['protein_pdb'] = 'target.pdb'
    options['ligand_file'] = 'ligands.sdf'
    if reverse:
        # Do the other direction of ligands
        options['old_ligand_index'] = lig_b_idx
        options['new_ligand_index'] = lig_a_idx
        # mark the output directory with reversed
        trajectory_directory = f'{trajectory_directory}_reversed'
        # mark new yaml file with reversed
        temp_path = new_yaml.split('.')
        new_yaml = f'{temp_path[0]}_reversed.{temp_path[1]}'
    else:
        options['old_ligand_index'] = lig_a_idx
        options['new_ligand_index'] = lig_b_idx
    options['trajectory_directory'] = f'{trajectory_directory}'
    with open(new_yaml, 'w') as outfile:
        yaml.dump(options, outfile)

    # run the simulation - using API point to respect logging level
    run(new_yaml)

    _logger.info(f'Relative calculation of ligand {lig_a_idx} to {lig_b_idx} complete')

    if tidy:
        os.remove(new_yaml)


# Defining command line arguments
# fetching targets from github repo
# TODO: This part should be done using plbenchmarks API - once there is a conda pkg
targets_url = f"{base_repo_url}/raw/main/data/targets.yml"
with fetch_url_contents(targets_url) as response:
    targets_dict = yaml.safe_load(response.read())
# get the possible choices from targets yaml file
target_choices = targets_dict.keys()

arg_parser = argparse.ArgumentParser(description='CLI tool for running perses protein-ligand benchmarks.')
arg_parser.add_argument(
    "--target",
    type=str,
    help="Target biomolecule, use openff's plbenchmark names.",
    choices=target_choices,
    required=True
)
arg_parser.add_argument(
    "--edge",
    type=int,
    help="Edge index (0-based) according to edges yaml file in dataset. Ex. --edge 5 (for sixth edge)",
    required=True
)
arg_parser.add_argument(
    "--reversed",
    action='store_true',
    help="Whether to run the edge in reverse direction. Helpful for consistency checks."
)
arg_parser.add_argument(
    "--revision",
    type=str,
    default="0.2.1",
    help="Specify revision, release or branch in upstream repo to use. Defaults to using the 0.2.1 release branch."
)
arg_parser.add_argument(
    "--local",
    action='store_true',
    help="Run simulation with local data. It expects template.yaml, "
         "ligands.sdf and target.pdb in the same directory as this script."
)
arg_parser.add_argument(
    "--ligand-a",

)
args = arg_parser.parse_args()
target = args.target
edge_index = args.edge
is_reversed = args.reversed
branch = args.revision
local_run = args.local

if local_run:
    get_ligand_indices_from_edge(edge_index, target, branch=branch)
    lig_a_index, lig_b_index = get_ligand_indices_from_edge(edge_index, target, branch=branch)
    run_relative_perturbation(lig_a_index, lig_b_index, reverse=is_reversed)
else:
    # get target information
    target_dir = get_target_dir(target)
    pdb_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/01_protein/crd/protein.pdb"
    pdb_file = retrieve_file_url(pdb_url)

    # Fetch cofactors crystalwater pdb file
    # TODO: This part should be done using plbenchmarks API - once there is a conda pkg
    # TODO: No cofactors now in fix_data branch
    #cofactors_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/01_protein/crd/cofactors_crystalwater.pdb"
    #cofactors_file = retrieve_file_url(cofactors_url)

    # Concatenate protein with cofactors pdbs
    # TODO: No need to concatenate now that there are no cofactors
    concatenate_files((pdb_file,), 'target.pdb')

    # Fetch ligands sdf files and concatenate them in one
    ligands_dict = get_ligands_information(target, branch=branch)
    generate_ligands_sdf(ligands_dict, target, branch=branch)

    # get ligand indices
    lig_a_index, lig_b_index = get_ligand_indices_from_edge(edge_index, target, branch=branch)

    # run simulation
    run_relative_perturbation(lig_a_index, lig_b_index, reverse=is_reversed)

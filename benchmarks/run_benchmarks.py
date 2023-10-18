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


def get_target_dir(target_name, branch="0.2.1"):
    """
    Retrieves the target subdirectory in upstream repo structure given the target name.
    """
    # Targets information on branch
    targets_url = f"{base_repo_url}/raw/{branch}/data/targets.yml"
    with fetch_url_contents(targets_url) as response:
        targets_dict = yaml.safe_load(response.read())
    target_dir = targets_dict[target_name]['dir']
    return target_dir


def get_ligands_information(target_name, branch="0.2.1"):
    """
    Retrieves the ligands information in a dictionary given the target name,
    """
    # TODO: This part should be done using plbenchmarks API - once there is a conda pkg
    target_dir = get_target_dir(target_name, branch=branch)
    ligands_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/00_data/ligands.yml"
    with fetch_url_contents(ligands_url) as response:
        ligands_dict = yaml.safe_load(response.read())
    return ligands_dict


def generate_ligands_sdf(ligands_dict, target, branch='0.2.1'):
    """
    Generates input ligands.sdf file for running the simulation given a dictionary
    with the ligands information.

    Can handle previous and new structure of upstream repo to date (26-Oct-2022).
    """
    from urllib.error import HTTPError
    # Get target dir where to get ligands sdf files
    target_dir = get_target_dir(target, branch=branch)
    # Fetch ligands sdf files and concatenate them in one
    # TODO: This part should be done using plbenchmarks API - once there is a conda pkg
    try:
        # _logger.info(f'Must be previous old repository structure. Trying downloading individual ligand files.')
        ligand_files = []
        for ligand in ligands_dict.keys():
            ligand_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/02_ligands/{ligand}/crd/{ligand}.sdf"
            ligand_file = retrieve_file_url(ligand_url)
            ligand_files.append(ligand_file)
        new_repo = False
    except HTTPError:
        _logger.info(f'Must be a newer revision and repository structure. Trying to download single ligands file.')
        ligands_file_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/02_ligands/ligands.sdf"
        ligand_files = retrieve_file_url(ligands_file_url)
        new_repo = True

    # concatenate sdf files as needed
    if new_repo:
        # TODO: Just retrieve the ligands file to the working directory without need to concatenate
        concatenate_files((ligand_files,), "ligands.sdf")  # "concatenate" single file
    else:
        concatenate_files(ligand_files, 'ligands.sdf')  # concatenate multiple files


def get_edges_data_from_repo(target_name, revision="0.2.1", edges_file_name="edges.yml",
                             base_url="https://github.com/openforcefield/protein-ligand-benchmark"):
    """
    Gets the content of edges file according to branch and edges file name specified.

    Meant to be used with https://github.com/openforcefield/protein-ligand-benchmark/

    Legacy revision/branch (0.2.1) only has one edge file. Newer revisions could have multiple
    edges files.

    Parameters
    ----------
    target_name: str
        Target name from the dataset. E.g. "tyk2".
    revision: str
        Revision/branch to use when looking up information in the repo.
    edges_file_name: str
        File name for the edges file. Useful to specify edges file in repo when there are multiple
        options. Defaults to legacy "edges.yml".
    base_url: str
        URL to repository where to get the data from.

    Returns
    -------
    edges_dict: dict
        Dictionary with edges information

    """
    # TODO: This part should be done using plbenchmarks API - once there is a conda pkg
    target_dir = get_target_dir(target_name, branch=revision)
    if revision == "0.2.1":
        edges_url = f"{base_url}/raw/{branch}/data/{target_dir}/00_data/{edges_file_name}"
    else:
        edges_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/03_edges/{edges_file_name}"
    with fetch_url_contents(edges_url) as response:
        data = yaml.safe_load(response.read())
        try:
            edges_dict = data["edges"]
        except KeyError:  # "edges" key doesn't exist for legacy revision (0.2.1)
            edges_dict = data
    return edges_dict


def get_ligand_names_from_edge(edge_index, target_name, branch="0.2.1", edges_file_name="edges.yml"):
    """
    Retrieve the names of ligands associated with a specific edge from a protein-ligand benchmark dataset.

    Parameters:
        edge_index (int): The index of the edge for which to retrieve ligand names.
        target_name (str): The name of the target protein in the benchmark dataset.
        branch (str, optional): The revision or branch of the dataset (default is "0.2.1").
        edges_file_name (str, optional): The name of the edges file in the repository (default is "edges.yml").

    Returns:
        tuple: A tuple containing the names of the ligands connected by the specified edge.
            - ligand_a_name (str): Name of the first ligand (ligand A).
            - ligand_b_name (str): Name of the second ligand (ligand B).

    Note:
        This function retrieves the benchmark dataset information from a GitHub repository and extracts
        ligand names based on the provided edge index and target protein name. It relies on the
        'get_edges_data_from_repo' function to access the dataset.

    Example:
        ligand_a, ligand_b = get_ligand_names_from_edge(0, "tyk2, branch="0.2.1")
        print(f"Ligand A name: {ligand_a}, Ligand B name: {ligand_b}")
    """
    # fetch edges information
    edges_dict = get_edges_data_from_repo(target_name, revision=branch, edges_file_name=edges_file_name)
    edges_list = list(edges_dict.values())  # suscriptable edges object - note dicts are ordered for py>=3.7

    edge = edges_list[edge_index]
    ligand_a_name = edge['ligand_a']
    ligand_b_name = edge['ligand_b']

    return ligand_a_name, ligand_b_name


def get_ligand_index_from_file(sdf_file, ligand_name):
    """
    Extract index for ligand in an sdf file given its name.

    Parameters
    ----------
    sdf_file : str
        Path to sdf file with ligands information.
    ligand_name : str
        Name of the ligand.

    Returns
    -------
    index : int
        Index of the ligand in the sdf file
    """
    from openff.toolkit.topology import Molecule

    # Reading with off-toolkit should preserve the order
    molecule_objects = Molecule.from_file(sdf_file)
    names = []
    try:
        for molecule in molecule_objects:
            names.append(molecule.name)
        index = names.index(ligand_name)
    except TypeError:  # assuming it must be a single molecule sdf file
        index = 0

    return index


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
    "--edges-file-name",
    type=str,
    default="edges.yml",
    help="Specify the name of the file for the edges in the repo. "
         "Useful when we have multiple edges files for the same system.",
)
args = arg_parser.parse_args()
target = args.target
edge_index = args.edge
is_reversed = args.reversed
branch = args.revision
local_run = args.local
edges_file_name = args.edges_file_name

if local_run:
    # FIXME: This isn't working we need something that takes a local edges.yaml file
    lig_a_name, lig_b_name = get_ligand_names_from_edge(edge_index, target, branch=branch,
                                                        edges_file_name=edges_file_name)
    # get ligand indices from names -- expects ligands.sdf file in the same dir
    lig_a_index = get_ligand_index_from_file("ligands.sdf", lig_a_name)
    lig_b_index = get_ligand_index_from_file("ligands.sdf", lig_b_name)
    run_relative_perturbation(lig_a_index, lig_b_index, reverse=is_reversed)
else:
    # get target information
    target_dir = get_target_dir(target, branch=branch)
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

    # get ligand names
    lig_a_name, lig_b_name = get_ligand_names_from_edge(edge_index, target, branch=branch,
                                                        edges_file_name=edges_file_name)
    # get ligand indices from names -- expects ligands.sdf file in the same dir
    lig_a_index = get_ligand_index_from_file("ligands.sdf", lig_a_name)
    lig_b_index = get_ligand_index_from_file("ligands.sdf", lig_b_name)

    # run simulation
    run_relative_perturbation(lig_a_index, lig_b_index, reverse=is_reversed)

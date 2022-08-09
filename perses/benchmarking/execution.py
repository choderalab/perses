#!/usr/bin/env python

"""
Benchmarking module for setting up and running the Free Energy simulations based on the data in public datasets on
remote git repositories.

.. warning:: Only tested with data in https://github.com/openforcefield/protein-ligand-benchmark to date.
"""

import argparse
import os
import yaml

from perses.app.setup_relative_calculation import run
from perses.utils.url_utils import retrieve_file_url
from perses.utils.url_utils import fetch_url_contents


class Executor:
    def __init__(self):
        self.
# STEPS
# 1. Get information from remote repository (no local clone)
    # 1.1. Target PDB file
        # 1.1.1. Get protein.db
        # 1.1.2. get cofactors
        # 1.1.3. concatenate protein.pdb with cofactors into target.pdb
    # 1.2. Ligands information
def create_ligands_file(ligands_yaml_path, output_path='ligands.sdf'):
    """
    Creates single ligands SDF-formatted file from ligands yaml file.

    Structure of files and dataset expected to match that one of
    https://github.com/openforcefield/protein-ligand-benchmark

    Parameters
    ----------
        ligands_yaml_path: Path or str
            Path to yaml file with ligands information.
        output_path: str, optional
            Path where to store the generated ligands SDF-formatted file (default is 'ligands.sdf')
    """
    base_repo_url = 'https://github.com/openforcefield/protein-ligand-benchmark'
    branch_or_tag = 'fix_data'  # release tag
    target_directory = "tyk2"

    ligands_url = f"{base_repo_url}/raw/{branch_or_tag}/data/{target_directory}/00_data/ligands.yml"
    with enter_temp_directory() as temp_dir:
        with fetch_url_contents(ligands_url) as response:
            ligands_dict = yaml.safe_load(response.read())
    print(ligands_dict)

    ligand_files = []
    for ligand in ligands_dict.keys():
        ligand_url = f"{base_repo_url}/raw/{branch_or_tag}/data/{target_directory}/02_ligands/{ligand}/crd/{ligand}.sdf"
        ligand_file = retrieve_file_url(ligand_url)
        ligand_files.append(ligand_file)

    # concatenate files
    concatenate_files(ligand_files, 'ligands.sdf', endline='\n')
    # 1.3. Edges information
# 2. Setup calculation
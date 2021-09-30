"""
Script to perform analysis of perses simulations executed using run_benchmarks.py script.

Intended to be used on systems from https://github.com/openforcefield/protein-ligand-benchmark
"""

import argparse
import urllib.request
import yaml
from perses.analysis.load_simulations import Simulation

# global variables
base_repo_url = "https://github.com/openforcefield/protein-ligand-benchmark"


# Defining command line arguments
# fetching targets from github repo
targets_url = f"{base_repo_url}/raw/master/data/targets.yml"
with urllib.request.urlopen(targets_url) as response:
    targets_dict = yaml.safe_load(response.read())
# get the possible choices from targets yaml file
target_choices = targets_dict.keys()

arg_parser = argparse.ArgumentParser(description='CLI tool for running perses protein-ligand benchmarks analysis.')
arg_parser.add_argument(
    "--target",
    type=str,
    help="Target biomolecule, use openff's plbenchmark names.",
    choices=target_choices,
    required=True
)
args = arg_parser.parse_args()
target = args.target

# TODO implement benchmark analysis

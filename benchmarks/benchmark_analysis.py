"""
Script to perform analysis of perses simulations executed using run_benchmarks.py script.

Intended to be used on systems from https://github.com/openforcefield/protein-ligand-benchmark
"""

import argparse
import glob
import numpy as np
import urllib.request
import yaml
from perses.analysis.load_simulations import Simulation

# global variables
base_repo_url = "https://github.com/openforcefield/protein-ligand-benchmark"


# Helper functions
# Copied straight from https://github.com/choderalab/qmlify
# TODO: Do all of this with openff.arsenic - make respective PRs if needed 
def absolute_from_relative(g, experimental, experimental_error):
    """
    Use DiffNet to compute absolute free energies, from computed relative free energies and assign absolute
    experimental values and uncertainties.

    Parameters
    ----------
    g : nx.DiGraph()
        A graph with relative results assigned to edges
    experimental : list
        list of experimental absolute affinities in kcal/mol
    experimental_error : list
        list of experimental absolute uncertainties in kcal/mol

    Returns
    -------

    """
    from openff.arsenic import stats
    # compute maximum likelihood estimates of free energies from calculated DDG attribute
    f_i_calc, c_calc = stats.mle(g, factor='calc_DDG')
    variance = np.diagonal(c_calc)  # variance of estimate is diagonal of the calculation matrix
    for n, f_i, df_i in zip(g.nodes(data=True), f_i_calc, variance**0.5):
        n[1]['calc_DG'] = f_i
        n[1]['calc_dDG'] = df_i
        n[1]['exp_DG'] = experimental[n[0]]
        n[1]['exp_dDG'] = experimental_error[n[0]]


def make_mm_graph(mm_results, expt, d_expt):
    """ Make a networkx graph from MM results

    Parameters
    ----------
    mm_results : list(perses.analysis.load_simulations.Simulation)
        List of perses simulation objects
    expt : list
        List of experimental values, in kcal/mol
    d_expt : list
        List of uncertainties in experimental values, in kcal/mol

    Returns
    -------
    nx.DiGraph
        Graph object with relative MM free energies

    """
    import networkx as nx
    mm_g = nx.DiGraph()
    for sim in mm_results:
        ligA = int(sim.directory[3:].split('_')[1])  # define edges
        ligB = int(sim.directory[3:].split('_')[2])
        exp_dDDG = (d_expt[ligA]**2 + d_expt[ligB]**2)**0.5  # define exp error
        mm_g.add_edge(ligA,
                      ligB,
                      calc_DDG=-sim.bindingdg/sim.bindingdg.unit,  # simulation reports A - B
                      calc_dDDG=sim.bindingddg/sim.bindingddg.unit,
                      exp_DDG=(expt[ligB] - expt[ligA]),
                      exp_dDDG=exp_dDDG)  # add edge to the digraph
    absolute_from_relative(mm_g, expt, d_expt)
    return mm_g


# Defining command line arguments
# fetching targets from github repo
# TODO: This part should be done using plbenchmarks API - once there is a conda pkg
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

# Download experimental data
# TODO: This part should be done using plbenchmarks API - once there is a conda pkg
target_dir = targets_dict[target]['dir']
ligands_url = f"{base_repo_url}/raw/master/data/{target_dir}/00_data/ligands.yml"
with urllib.request.urlopen(ligands_url) as response:
    ligands_dict = yaml.safe_load(response.read())
ligands_exp_values = []  # list where to store experimental values
for _, ligand_data in ligands_dict.items():
    ligands_exp_values.append(ligand_data['measurement']['value'])

# Load simulation from directories
out_dirs = [filepath.split('/')[0] for filepath in glob.glob('out*/*complex.nc')]
simulations = [Simulation(out_dir) for out_dir in out_dirs]

# create graph with results
results_graph = make_mm_graph(simulations, ligands_exp_values, [0]*len(ligands_exp_values))

print("DEBUG STOP")
"""
Script to perform analysis of perses simulations executed using run_benchmarks.py script.

Intended to be used on systems from https://github.com/openforcefield/protein-ligand-benchmark
"""

import argparse
import glob
import itertools
import re
import warnings

import numpy as np
import urllib.request
import yaml

from openmmtools.constants import kB
from perses.analysis.load_simulations import Simulation

from simtk import unit

# global variables
base_repo_url = "https://github.com/openforcefield/protein-ligand-benchmark"


# Helper functions

def get_simdir_list(reversed=False):
    """
    Get list of directories to extract simulation data.

    Attributes
    ----------
    reversed: bool, optional, default=False
        Whether to consider the reversed simulations or not.

    Returns
    -------
    dir_list: list
        List of directories paths for simulation results.
    """
    # Load all expected simulation from directories
    out_dirs = [filepath.split('/')[0] for filepath in glob.glob(f'out*/*complex.nc')]
    reg = re.compile(r'out_[0-9]+_[0-9]+_reversed')  # regular expression to deal with reversed directories
    if reversed:
        # Choose only reversed directories
        out_dirs = list(filter(reg.search, out_dirs))
    else:
        # Filter out reversed directories
        out_dirs = list(itertools.filterfalse(reg.search, out_dirs))
    return out_dirs


def get_simulations_data(simulation_dirs):
    """Generates a list of simulation data objects given the simulation directories paths."""
    simulations = []
    for out_dir in simulation_dirs:
        # Load complete or fully working simulations
        # TODO: Try getting better exceptions from openmmtools -- use non-generic exceptions
        try:
            simulation = Simulation(out_dir)
            simulations.append(simulation)
        except Exception:
            warnings.warn(f"Edge in {out_dir} could not be loaded. Check simulation output is complete.")
    return simulations


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
                      calc_DDG_dev=sim.bindingddg/sim.bindingddg.unit,
                      exp_DDG=(expt[ligB] - expt[ligA]),
                      exp_DDG_dev=exp_dDDG,
                      vac_DG=sim._vacdg/sim._vacdg.unit,
                      vac_DG_dev=sim._vacddg/sim._vacddg.unit,
                      sol_DG=sim._soldg/sim._soldg.unit,
                      sol_DG_dev=sim._solddg/sim._solddg.unit,
                      com_DG=sim._comdg / sim._comdg.unit,
                      com_DG_dev=sim._comddg / sim._comddg.unit
                      )  # add edge to the digraph
    # absolute_from_relative(mm_g, expt, d_expt)
    return mm_g


def map_from_edge_data(mm_graph, data_key='calc_DDG'):
    """Creates a map/dictionary from edge data using the given key"""
    data_map = {(edge[0], edge[1]): edge[2][data_key] for edge in mm_graph.edges(data=True)}
    return data_map


def compare_forward_reverse(data_key='calc_DDG'):
    """
    Compare reversed and forward simulations given the phase and the data keyword.
    """
    import matplotlib.pyplot as plt
    # get forward sim directories path
    forward_dirs = get_simdir_list(reversed=False)
    # load forward simulations
    forward_simulations = get_simulations_data(forward_dirs)
    # Make graph with just he calculated quantities, don't care about experimental here
    forward_graph = make_mm_graph(forward_simulations, [0]*len(forward_simulations), [0]*len(forward_simulations))

    # Do the same for reversed simulations
    reversed_dirs = get_simdir_list(reversed=True)
    reversed_simulations = get_simulations_data(reversed_dirs)
    reversed_graph = make_mm_graph(reversed_simulations, [0]*len(reversed_simulations), [0]*len(reversed_simulations))

    # get maps/dictionaries with edge and data
    forward_map = map_from_edge_data(forward_graph, data_key=data_key)
    reversed_map = map_from_edge_data(reversed_graph, data_key=data_key)
    # get maps/dictionaries with edge and data errors/deviations
    forward_map_dev = map_from_edge_data(forward_graph, data_key=f"{data_key}_dev")
    reversed_map_dev = map_from_edge_data(reversed_graph, data_key=f"{data_key}_dev")

    # Get common edges/keys
    common_edges = [key for key in forward_map.keys() if key in reversed_map.keys()]

    # extract data and errors/deviations from objects
    forward_data = [forward_map[edge] for edge in common_edges]
    reversed_data = [-reversed_map[edge] for edge in common_edges]  # flip the sign for reverse data
    forward_data_dev = [forward_map_dev[edge] for edge in common_edges]
    reversed_data_dev = [reversed_map_dev[edge] for edge in common_edges]
    # make the plot
    fig, ax = plt.subplots()
    plt.title(f"{data_key}")
    plt.xlabel(f"forward")
    plt.ylabel(f"reverse")
    ax.scatter(forward_data, reversed_data)
    ax.errorbar(forward_data,
                reversed_data,
                xerr=forward_data_dev,
                yerr=reversed_data_dev,
                linewidth=0.0,
                elinewidth=2.0,
                zorder=1
                )
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.show()


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
arg_parser.add_argument(
    "--reversed",
    action='store_true',
    help="Analyze reversed edge simulations. Helpful for consistency checks."
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
    # ligands_exp_values.append(ligand_data['measurement']['value']/1e6)
    ligands_exp_values.append(ligand_data['measurement']['value'])
# converting to kcal/mol
kBT = kB * 300 * unit.kelvin
ligands_exp_values = kBT.in_units_of(unit.kilocalorie_per_mole) * np.log(ligands_exp_values)

# Get paths for simulation output directories
# out_dirs = get_simdir_list(reversed=args.reversed)

# Generate list with simulation objects
# simulations = get_simulations_data(out_dirs)

# compare forward and backward simulations
# TODO: enable the comparison and data_key to be specified via cli
compare_forward_reverse(data_key='com_DG')

# create graph with results
# results_graph = make_mm_graph(simulations, ligands_exp_values, [0]*len(ligands_exp_values))

# Print FE comparison
# for edge in results_graph.edges(data=True):
#     print(edge[0], edge[1], edge[2]['calc_DDG'], edge[2]['exp_DDG'])


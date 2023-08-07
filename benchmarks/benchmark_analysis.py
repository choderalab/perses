"""
Script to perform analysis of perses simulations executed using run_benchmarks.py script.

Intended to be used on systems from https://github.com/openforcefield/protein-ligand-benchmark
"""

import argparse
import glob
import itertools
import os
import re
import traceback
import warnings

import numpy as np
import urllib.request
import yaml

from openmmtools.constants import kB
from perses.analysis.load_simulations import Simulation
from perses.utils.url_utils import fetch_url_contents

from simtk import unit

from cinnabar import plotting, wrangle

# global variables
base_repo_url = "https://github.com/openforcefield/protein-ligand-benchmark"


# Helper functions
def get_target_dir(target_name, branch="0.2.1"):
    """
    Retrieves the target subdirectory in upstream repo structure given the target name.
    """
    # Targets information on branch
    targets_url = f"{base_repo_url}/raw/{branch}/data/targets.yml"
    with fetch_url_contents(targets_url) as response:
        targets_dict = yaml.safe_load(response.read())
    target_dir = targets_dict[target]['dir']
    return target_dir


def get_simdir_list(base_dir='.', is_reversed=False):
    """
    Get list of directories to extract simulation data.

    Attributes
    ----------
    base_dir: str, optional, default='.'
        Base directory where to search for simulations results. Defaults to current directory.
    is_reversed: bool, optional, default=False
        Whether to consider the reversed simulations or not. Meant for testing purposes.

    Returns
    -------
    dir_list: list
        List of directories paths for simulation results.
    """
    # Load all expected simulation from directories
    out_dirs = ['/'.join(filepath.split('/')[:-1]) for filepath in glob.glob(f'{base_dir}/out*/*complex.nc')]
    reg = re.compile(r'out_[0-9]+_[0-9]+_reversed')  # regular expression to deal with reversed directories
    if is_reversed:
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
            # Print traceback anyway
            print(traceback.format_exc())
            warnings.warn(f"Edge in {out_dir} could not be loaded. Check simulation output is complete.")
    return simulations


def get_ligands_information(ligands_file="ligands.sdf"):
    """Extract ligands information, namely index and name, from sdf file.

    Parameters
    ----------
    ligands_file : str
        Path to sdf file where from extract the information.

    Returns
    -------
    ligands_dict : dict
        Dictionary with ligands information with (index, name) as key-value pairs.
    """
    from openff.toolkit.topology import Molecule
    molecules = Molecule.from_file(ligands_file)
    # Making sure the sdf file has a list of molecules
    if not isinstance(molecules, list):
        raise TypeError(f"Could not read a list of ligands from {ligands_file}.")
    ligands_dict = {}
    # Fill ligand dictionary information
    for index, molecule in enumerate(molecules):
        ligands_dict[index] = molecule.name

    return ligands_dict


def to_arsenic_csv(experimental_data: dict, simulation_data: list, out_csv: str = 'out_benchmark.csv',
                   ligands_file="ligands.sdf"):
    """
    Generates a csv file to be used with openff-arsenic. Energy units in kcal/mol.

    .. warning:: To be deprecated once arsenic object model is improved.

    Parameters
    ----------
        experimental_data: dict
            Python nested dictionary with experimental data in micromolar or nanomolar units.
            Example of entry:

                {'lig_ejm_31': {'measurement': {'comment': 'Table 4, entry 31',
                  'doi': '10.1016/j.ejmech.2013.03.070',
                  'error': -1,
                  'type': 'ki',
                  'unit': 'uM',
                  'value': 0.096},
                  'name': 'lig_ejm_31',
                  'smiles': '[H]c1c(c(c(c(c1[H])Cl)C(=O)N([H])c2c(c(nc(c2[H])N([H])C(=O)C([H])([H])[H])[H])[H])Cl)[H]'}

        simulation_data: list or iterable
            Python iterable object with perses Simulation objects as entries.
        out_csv: str
            Path to output csv file to be generated.
        ligands_file : str
            Path to sdf file where from extract the information.
    """
    # get ligand information from ligands file
    ligands_dict = get_ligands_information(ligands_file=ligands_file)
    kBT = kB * 300 * unit.kelvin  # useful when converting to kcal/mol
    # Write csv file
    with open(out_csv, 'w') as csv_file:
        # Experimental block
        # print header for block
        csv_file.write("# Experimental block\n")
        csv_file.write("# Ligand, expt_DG, expt_dDG\n")
        # Extract ligand name, expt_DG and expt_dDG from ligands dictionary
        for ligand_name, ligand_data in experimental_data.items():
            # TODO: Handle multiple measurement types
            unit_symbol = ligand_data['measurement']['unit']
            measurement_value = ligand_data['measurement']['value']
            measurement_error = ligand_data['measurement']['error']
            # Unit conversion
            # TODO: Let's persuade PLBenchmarks to use pint units
            unit_conversions = { 'M' : 1.0, 'mM' : 1e-3, 'uM' : 1e-6, 'nM' : 1e-9, 'pM' : 1e-12, 'fM' : 1e-15 }
            if unit_symbol not in unit_conversions:
                raise ValueError(f'Unknown units "{unit_symbol}"')
            value_to_molar= unit_conversions[unit_symbol]
            # Handle unknown errors
            # TODO: We should be able to ensure that all entries have more reasonable errors.
            if measurement_error == -1:
                # TODO: For now, we use a relative_error from the Tyk2 system 10.1016/j.ejmech.2013.03.070
                relative_error = 0.3
            else:
                relative_error = measurement_error / measurement_value
            # Convert to free energies
            expt_DG = kBT.value_in_unit(unit.kilocalorie_per_mole) * np.log(measurement_value * value_to_molar)
            expt_dDG = kBT.value_in_unit(unit.kilocalorie_per_mole) * relative_error
            csv_file.write(f"{ligand_name}, {expt_DG}, {expt_dDG}\n")

        # Calculated block
        # print header for block
        csv_file.write("# Calculated block\n")
        csv_file.write("# Ligand1,Ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)\n")
        # Loop through simulation, extract ligand1 and ligand2 indices, convert to names, create string with
        # ligand1, ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)
        # write string in csv file
        for simulation in simulation_data:
            out_dir = os.path.basename(simulation.directory)
            # getting integer indices
            ligand1_id, ligand2_id = int(out_dir.split('_')[-1]), int(out_dir.split('_')[-2])  # CHECK ORDER!
            # getting names of ligands
            ligand1, ligand2 = ligands_dict[ligand1_id], ligands_dict[ligand2_id]
            # getting calc_DDG in kcal/mol
            calc_DDG = simulation.bindingdg.value_in_unit(unit.kilocalorie_per_mole)
            # getting calc_dDDG in kcal/mol
            calc_dDDG = simulation.bindingddg.value_in_unit(unit.kilocalorie_per_mole)
            csv_file.write(
                f"{ligand1}, {ligand2}, {calc_DDG}, {calc_dDDG}, 0.0\n")  # hardcoding additional error as 0.0


# Defining command line arguments
# fetching targets from github repo
# TODO: This part should be done using plbenchmarks API - once there is a conda pkg
targets_url = f"{base_repo_url}/raw/main/data/targets.yml"  # Assuming main has the right targets info
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
    help="Analyze reversed edge simulations. Helpful for testing/consistency checks."
)
arg_parser.add_argument(
    "--revision",
    type=str,
    default="0.2.1",
    help="Specify revision, release or branch in upstream repo to use. Defaults to using the 0.2.1 release branch."
)
args = arg_parser.parse_args()
target = args.target
branch = args.revision

# Download experimental data
# TODO: This part should be done using plbenchmarks API - once there is a conda pkg
target_dir = get_target_dir(target, branch=branch)
ligands_url = f"{base_repo_url}/raw/{branch}/data/{target_dir}/00_data/ligands.yml"
with urllib.request.urlopen(ligands_url) as response:
    yaml_contents = response.read()
    print(yaml_contents)
    ligands_experimental_dict = yaml.safe_load(yaml_contents)

# DEBUG
# print('')
# print(yaml.dump(ligands_dict))

# Get paths for simulation output directories
out_dirs = get_simdir_list(is_reversed=args.reversed)

# Generate list with simulation objects
simulations = get_simulations_data(out_dirs)

# Generate csv file
csv_path = f'./{target}_arsenic.csv'
to_arsenic_csv(ligands_experimental_dict, simulations, out_csv=csv_path)


# TODO: Separate plotting in a different file
# Make plots and store
fe = wrangle.FEMap(csv_path)
# Relative plot
plotting.plot_DDGs(fe.graph,
                   target_name=f'{target}',
                   title=f'Relative binding energies - {target}',
                   figsize=5,
                   filename='./plot_relative.png'
                   )
# Absolute plot, with experimental data shifted to correct mean
experimental_mean_dg = np.asarray([node[1]["exp_DG"] for node in fe.graph.nodes(data=True)]).mean()
plotting.plot_DGs(fe.graph,
                  target_name=f'{target}',
                  title=f'Absolute binding energies - {target}',
                  figsize=5,
                  filename='./plot_absolute.png',
                  shift=experimental_mean_dg,
                  )

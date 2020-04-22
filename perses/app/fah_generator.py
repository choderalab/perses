#1/usr/bin/env python

__author__ = 'dominic rufa'

"""
Folding@Home perses executor

# prereqs
conda config --add channels omnia --add channels conda-forge
conda create -n perses python3.7 perses tqdm dicttoxml
pip uninstall --yes openmmtools
pip install git+https://github.com/choderalab/openmmtools.git

argv[1]: setup.yaml (argument for perses.app.setup_relative_calculation.getSetupOptions)
argv[2]: neq_setup.yaml (contains keywords for openmmtools.integrators.PeriodicNonequilibriumIntegrator arguments)
argv[3]: run_number (project run number; defined by f"setup_options['trajectory_directory']_phase/RUN_{run_number}")
"""
import yaml
import numpy as np
import pickle
import os
import sys
import simtk.unit as unit
from simtk import openmm
from perses.app.setup_relative_calculation import getSetupOptions, run_setup
from perses.annihilation.relative import HybridTopologyFactory
from perses.app.relative_setup import RelativeFEPSetup
from perses.annihilation.lambda_protocol import LambdaProtocol
from perses.dispersed.feptasks import minimize
from openmmtools import SamplerState, ThermodynamicState, CompoundThermodynamicState

from openmmtools import mcmc, utils
from perses.utils.smallmolecules import render_atom_mapping
from perses.tests.utils import validate_endstate_energies
from perses.dispersed.smc import SequentialMonteCarlo
from simtk.openmm import XmlSerializer
from perses.utils import data
from openmmtools.integrators import PeriodicNonequilibriumIntegrator
from copy import deepcopy
import tqdm
from openmmtools.integrators import LangevinIntegrator

import datetime
class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated
        delta = datetime.datetime.fromtimestamp(record.relativeCreated/1000.0) - datetime.datetime.fromtimestamp(last/1000.0)
        record.relative = '{0:.2f}'.format(delta.seconds + delta.microseconds/1000000.0)
        self.last = record.relativeCreated
        return True

fmt = logging.Formatter(fmt="%(asctime)s:(%(relative)ss):%(name)s:%(message)s")
#logging.basicConfig(level = logging.NOTSET)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
[hndl.addFilter(TimeFilter()) for hndl in _logger.handlers]
[hndl.setFormatter(fmt) for hndl in _logger.handlers]

setup_yaml = sys.argv[1]
neq_yaml = sys.argv[2]
ENERGY_THRESHOLD=1e-4
x = 'lambda'
DEFAULT_ALCHEMICAL_FUNCTIONS = {
                             'lambda_sterics_core': x,
                             'lambda_electrostatics_core': x,
                             'lambda_sterics_insert': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                             'lambda_sterics_delete': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
                             'lambda_electrostatics_insert': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
                             'lambda_electrostatics_delete': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                             'lambda_bonds': x,
                             'lambda_angles': x,
                             'lambda_torsions': x}

from openmmtools.constants import kB

def make_neq_integrator(setup_options_dict, neq_options_dict, alchemical_functions = DEFAULT_ALCHEMICAL_FUNCTIONS):
    integrator_kwargs = deepcopy(setup_options_dict)
    integrator_kwargs.update(neq_options_dict)
    integrator_kwargs['alchemical_functions'] = alchemical_functions
    integrator = PeriodicNonequilibriumIntegrator(**integrator_kwargs)
    return integrator

def relax_structure(temperature, system, positions, nminimize=100, nequil = 4, n_steps_per_iteration=250):
    integrator = LangevinIntegrator(temperature = temperature)
    context = openmm.Context(system, integrator)
    context.setPeriodicBoxVectors(*state.getDefaultPeriodicBoxVectors())
    context.setPositions(positions)
​
    # Minimize
    for iteration in tqdm(range(nminimize)):
        openmm.LocalEnergyMinimizer.minimize(context, 0.0, 1)
​
    # Equilibrate
    context.setVelocitiesToTemperature(temperature)
    for iteration in tqdm(range(nequil)):
        integrator.step(nsteps_per_iteration)
    context.setVelocitiesToTemperature(temperature)
​
    state = context.getState(getEnergy=True, getForces=True, getPositions=True, getVelocities=True, getParameters=True)

    del context, integrator
​
    return state


def run_neq_fah_setup(setup_yaml, neq_yaml, run_number, **kwargs):
    """
    given a run_number, read a perses setup yaml, a neq-specific yaml, and conduct the following:
        1.create HTF objects for all phases specified
        2.create a RUN_{run_number} directory of the specific transform
        3.serialize a hybrid_system, neq_integrator, state (relaxed), a core, and a reference pkl
    """
    _logger.info("Beginning Setup...")
    if setup_yaml is None:
       try:
          setup_yaml = sys.argv[1]
          _logger.info(f"Detected yaml file: {setup_yaml}")
       except IndexError as e:
           _logger.critical(f"You must specify the setup yaml file as an argument to the script.")

    _logger.info(f"Getting setup options from {setup_yaml}")
    setup_options = getSetupOptions(setup_yaml)
    neq_yaml_file = open(neq_yaml, 'r')
    neq_setup_options = yaml.load(neq_yaml_file, Loader=yaml.FullLoader)
    neq_yaml_file.close()

    _logger.info(f"Running setup...")
    setup_dict = run_setup(setup_options, serialize_systems=False, build_samplers=False)

    trajectory_directory = setup_options['trajectory_directory']

    n_equilibration_iterations = setup_options['n_equilibration_iterations'] #set this to 1 for neq_fep
    _logger.info(f"Equilibration iterations: {n_equilibration_iterations}.")

    topology_proposals = setup_dict['topology_proposals']
    htfs = setup_dict['hybrid_topology_factories']

    #create solvent and complex directories
    for phase in htfs.keys():
        dir = os.path.join(os.getcwd(), f"{trajectory_directory}_{phase}", f"RUN_{format(run_number, '03')}")
        os.mkdir(dir)

        #serialize the hybrid_system
        data.serialize(htfs[phase].hybrid_system, f"{dir}/system.xml")

        #make and serialize an integrator
        integrator = make_neq_integrator(setup_options, neq_setup_options, alchemical_functions = DEFAULT_ALCHEMICAL_FUNCTIONS)
        data.serialize(htfs[phase].hybrid_system, f"{dir}/integrator.xml")

        #create and serialize a state
        try:
            state = relax_structure(setup_dict['temperature'],
                            system = htfs[phase].hybrid_system,
                            positions = htfs[phase].hybrid_positions,
                            nminimize=100,
                            nequil = 4,
                            n_steps_per_iteration=250)

            data.serialize(state, f"{dir}/state.xml")
        except Exception as e:
            print(e)
            passed=False
        else:
            passed=True


        #lastly, make a core.xml
        #TODO: make core.xml

        #create a logger for reference
        references = {'start_ligand': setup_dict['old_ligand_index'],
                      'end_ligand': setup_dict['new_ligand_index'],
                      'protein_filename': setup_dict['protein_pdb'],
                      'passed_strucutre_relax': passed}
        with open(f"{dir}/reference.pkl", 'wb') as f:
            pickle.dump(references, f)


if __name__ == "__main__":
    setup_yaml, neq_setup_yaml, run_number = sys.argv[1], sys.argv[2], sys.argv[3] #define args

    #open the setup yaml to pull trajectory_directory
    yaml_file = open(setup_yaml, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()
    traj_dir = setup_options['trajectory_directory']

    gather all
    phases = setup_options['phases']
    for phase in phases:
        now_path = os.path.join(os.getcwd(), f"{traj_dir}_{phase}")
        if not os.path.exists(now_path):
            os.mkdir(now_path)
        else:
            raise Exception(f"{now_path} already exists.  Aborting.")
    run_neq_fah_setup(setup_yaml, neq_yaml, int(run_number), **kwargs)

###########################################
# IMPORTS
###########################################
from simtk.openmm import app
from simtk import unit, openmm
import numpy as np
import os
from nose.tools import nottest
from unittest import skipIf
from perses.app.setup_relative_calculation import *
from perses.annihilation.relative import HybridTopologyFactory
from perses.dispersed import parallel
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
from perses.app.relative_setup import RelativeFEPSetup
from perses.tests.utils import validate_endstate_energies
from perses.dispersed.smc import SequentialMonteCarlo
from simtk import openmm, unit
from openmmtools.constants import kB
#######################
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
ENERGY_THRESHOLD = 1e-6

istravis = os.environ.get('TRAVIS', None) == 'true'

fe_setup = RelativeFEPSetup(ligand_input = f"{os.getcwd()}/test.smi",
                            old_ligand_index = 0,
                            new_ligand_index = 1,
                            forcefield_files = ['gaff.xml'],
                            phases = ['vacuum'])

hybrid_factory = HybridTopologyFactory(topology_proposal = fe_setup._vacuum_topology_proposal,
                                       current_positions = fe_setup._vacuum_positions_old,
                                       new_positions = fe_setup._vacuum_positions_new,
                                       neglected_new_angle_terms = fe_setup._vacuum_forward_neglected_angles,
                                       neglected_old_angle_terms = fe_setup._vacuum_reverse_neglected_angles,
                                       softcore_LJ_v2 = True,
                                       interpolate_old_and_new_14s = False)

zero_state_error, one_state_error = validate_endstate_energies(fe_setup._vacuum_topology_proposal,
                                                               hybrid_factory,
                                                               added_energy = fe_setup._vacuum_added_valence_energy,
                                                               subtracted_energy = fe_setup._vacuum_subtracted_valence_energy,
                                                               beta = beta,
                                                               ENERGY_THRESHOLD = ENERGY_THRESHOLD)
#default arguments
lambda_protocol = 'default'
temperature = 300 * unit.kelvin
trajectory_directory = 'test_smc'
trajectory_prefix = 'out'
atom_selection = 'not water'
timestep = 1 * unit.femtoseconds
collision_rate = 1 / unit.picoseconds
eq_splitting_string = 'V R O R V'
neq_splitting_string = 'V R O R V'
ncmc_save_interval = None
measure_shadow_work = False
neq_integrator = 'langevin'
external_parallelism = None
internal_parallelism = {'library': ('dask', 'LSF'), 'num_processes': 2}
os.system(f"mkdir {trajectory_directory}")

def test_local_AIS():
    ne_fep = SequentialMonteCarlo(factory = hybrid_factory,
                                  lambda_protocol = lambda_protocol,
                                  temperature = temperature,
                                  trajectory_directory = trajectory_directory,
                                  trajectory_prefix = trajectory_prefix,
                                  atom_selection = atom_selection,
                                  timestep = timestep,
                                  eq_splitting_string = eq_splitting_string,
                                  neq_splitting_string = neq_splitting_string,
                                  collision_rate = collision_rate,
                                  ncmc_save_interval = ncmc_save_interval,
                                  external_parallelism = None,
                                  internal_parallelism = None)
    ne_fep.minimize_sampler_states()
    ne_fep.equilibrate(n_equilibration_iterations = 5,
                       n_steps_per_equilibration = 1,
                       endstates = [0,1],
                       decorrelate = True,
                       timer = True,
                       minimize = False)
    ne_fep.sMC_anneal(num_particles = 5,
                      protocols = {'forward': np.linspace(0,1,9), 'reverse': np.linspace(1,0,9)},
                      directions = ['forward', 'reverse'],
                      num_integration_steps = 1,
                      return_timer = True,
                      rethermalize = False,
                      trailblaze = None,
                      resample = None)
    try:
        os.system(f"rm -r {trajectory_directory}")
    except Exception as e:
        print(e)

import os
import pickle
import argparse
import logging
from pathlib import Path

from simtk import openmm
from simtk.openmm import unit

from openmmtools import mcmc
from openmmtools import cache, utils
from openmmtools.multistate import MultiStateReporter

from perses.dispersed.utils import configure_platform
from perses.samplers.multistate import HybridRepexSampler

from mdtraj.core.residue_names import _SOLVENT_TYPES

# Set up logger
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

# Configure platform
platform = configure_platform(utils.get_fastest_platform().getName())
#platform.setPropertyDefaultValue('UseBlockingSync', 'false')

# Load arguments
parser = argparse.ArgumentParser(description='run repex')
parser.add_argument('dir', type=str, help='path to input/output dir')
parser.add_argument('phase', type=str, help='phase of the simulation to use in storage filename')
parser.add_argument('n_states', type=int, help='number of states')
parser.add_argument('n_cycles', type=int, help='number of iterations to run')
parser.add_argument('t_max', type=int, help='maximum temperature to use for rest scaling')
parser.add_argument('--restraint', type=str, help="the atoms to restrain, if any: 'CA', 'heavy', 'heavy-solvent'")
parser.add_argument('--force_constant', type=float, help='the force constant to use for restraints in kcal/molA^2')
args = parser.parse_args()

# Load hybrid topology factory
directory_number = Path(args.dir).parts[-2]
htf = pickle.load(open(os.path.join(args.dir, f"{directory_number}_{args.phase}.pickle"), "rb" ))
hybrid_system = htf.hybrid_system
hybrid_positions = htf.hybrid_positions

# Make sure LRC is set correctly
force_dict = {force.getName(): index for index, force in enumerate(hybrid_system.getForces())}
htf_class_name = htf.__class__.__name__
custom_force_name = 'CustomNonbondedForce'
nonbonded_force_name = 'NonbondedForce'
if htf_class_name == 'RESTCapableHybridTopologyFactory':
    custom_force_name += '_sterics'
    nonbonded_force_name += '_sterics'
custom_force = hybrid_system.getForce(force_dict[custom_force_name])
nonbonded_force = hybrid_system.getForce(force_dict[nonbonded_force_name])
_logger.info(f"{custom_force_name} use LRC? {custom_force.getUseLongRangeCorrection()}")
_logger.info(f"{nonbonded_force_name} use LRC? {nonbonded_force.getUseDispersionCorrection()}")

# Add virtual bond for complex phase
if args.phase == 'complex':
    chain_A = 0
    chain_B = 2
    chains = list(htf.hybrid_topology.chains)
    atom_A = list(chains[chain_A].atoms)[0]
    atom_B = list(chains[chain_B].atoms)[0]
    force = openmm.CustomBondForce('0')
    force.addBond(atom_A.index, atom_B.index, [])
    hybrid_system.addForce(force)
    _logger.info(f"Added virtual bond between {atom_A} and {atom_B}")

# Add restraints
if args.restraint is not None:
    topology = htf.hybrid_topology
    solvent_types = list(_SOLVENT_TYPES)
    force_constant = args.force_constant*unit.kilocalories_per_mole/unit.angstrom**2 if args.force_constant is not None else 50*unit.kilocalories_per_mole/unit.angstrom**2
    _logger.info(f"Adding restraint to {args.restraint} atoms with force constant {force_constant}")

    if args.restraint == 'heavy':
        atom_indices = [atom.index for atom in topology.atoms if atom.residue.name not in solvent_types and atom.element.name != 'hydrogen']
    elif args.restraint == 'CA':
        atom_indices = [atom.index for atom in topology.atoms if atom.residue.name not in solvent_types and atom.name == 'CA']
    elif args.restraint == 'heavy-solvent':
        atom_indices = [atom.index for atom in topology.atoms if atom.element.name != 'hydrogen']
    else:
        raise Exception("Invalid restraint string specified")

    _logger.info(atom_indices)
    custom_cv_force = openmm.CustomCVForce('(K_RMSD/2)*(RMSD)^2')
    custom_cv_force.addGlobalParameter('K_RMSD', force_constant * 2)
    rmsd_force = openmm.RMSDForce(hybrid_positions, atom_indices)
    custom_cv_force.addCollectiveVariable('RMSD', rmsd_force)
    hybrid_system.addForce(custom_cv_force)

# Instantiate sampler 
_logger.setLevel(logging.DEBUG)
reporter_file = os.path.join(os.path.join(args.dir, f"{directory_number}_{args.phase}.nc"))
reporter = MultiStateReporter(reporter_file, checkpoint_interval=100)
move = mcmc.LangevinDynamicsMove(timestep= 4.0 * unit.femtoseconds,
                                          collision_rate=1.0 / unit.picosecond,
                                          n_steps=250,
                                          reassign_velocities=False,
                                          constraint_tolerance=1e-06)
sampler = HybridRepexSampler(mcmc_moves=move,
                             replica_mixing_scheme='swap-all',
                             hybrid_factory=htf, 
                             online_analysis_interval=None)
sampler.setup(n_states=args.n_states, temperature=300*unit.kelvin, t_max=args.t_max * unit.kelvin, storage_file=reporter, endstates=True)

# Create context caches
sampler.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
sampler.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)

# Run simulation
sampler.extend(args.n_cycles)


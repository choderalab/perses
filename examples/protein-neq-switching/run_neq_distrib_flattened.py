import logging
import numpy as np
from openmmtools.integrators import PeriodicNonequilibriumIntegrator
from simtk import unit
from simtk import openmm
import argparse
import os
import pathlib
import time
from perses.app.relative_point_mutation_setup import PointMutationExecutor

# Set up logger
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

# Read args
parser = argparse.ArgumentParser(description='run perses protein mutation on capped amino acid')
parser.add_argument('dir', type=str, help='path to input/output dir')
parser.add_argument('time_step', type=float, help='time step in femtoseconds')
parser.add_argument('eq_steps', type=int, help='Number of steps for equilibrium simulation')
parser.add_argument('neq_steps', type=int, help='Number of steps for non-equilibrium simulation')
parser.add_argument('--platform',
                    type=str,
                    help='compute platform: Reference, CPU, CUDA or OpenCL.',
                    default='OpenCL',
                    required=False)
parser.add_argument('--eq_save_period',
                    type=int,
                    help='Save period for equlibrium simulation, in steps.',
                    default=1000)
parser.add_argument('--neq_save_period',
                    type=int,
                    help='Save period for non-equlibrium simulation, in steps.',
                    default=1000)
args = parser.parse_args()

# Build HybridTopologyFactory
solvent_delivery = PointMutationExecutor("ala_vacuum.pdb",
                                         '1',
                                         '2',
                                         'ASP',
                                         ionic_strength=0.15 * unit.molar,
                                         flatten_torsions=True,
                                         flatten_exceptions=True,
                                         conduct_endstate_validation=False
                                         )
htf = solvent_delivery.get_apo_htf()

# Define lambda functions
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
    'lambda_torsions': x
}

# Define simulation parameters
nsteps_eq = args.eq_steps
nsteps_neq = args.neq_steps
neq_splitting = 'V R H O R V'
timestep = args.time_step * unit.femtosecond
platform_name = args.platform
temperature = 300 * unit.kelvin
save_freq_eq = args.eq_save_period
save_freq_neq = args.neq_save_period

system = htf.hybrid_system
positions = htf.hybrid_positions

# Set up integrator
integrator = PeriodicNonequilibriumIntegrator(DEFAULT_ALCHEMICAL_FUNCTIONS,
                                              nsteps_eq,
                                              nsteps_neq,
                                              neq_splitting,
                                              timestep=timestep,
                                              temperature=temperature)

# Set up context
platform = openmm.Platform.getPlatformByName(platform_name)
if platform_name in ['CUDA', 'OpenCL']:
    platform.setPropertyDefaultValue('Precision', 'mixed')
if platform_name in ['CUDA']:
    platform.setPropertyDefaultValue('DeterministicForces', 'true')
context = openmm.Context(system, integrator, platform)
context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

# Minimize
openmm.LocalEnergyMinimizer.minimize(context)

# Run neq
forward_works_master, reverse_works_master = list(), list()
forward_eq_old, forward_eq_new, forward_neq_old, forward_neq_new = list(), list(), list(), list()
reverse_eq_new, reverse_eq_old, reverse_neq_old, reverse_neq_new = list(), list(), list(), list()

# Equilibrium (lambda = 0)
for step in range(nsteps_eq):
    initial_time = time.time()
    integrator.step(1)
    elapsed_time = (time.time() - initial_time) * unit.seconds
    if step % save_freq_eq == 0:
        _logger.info(f'Step: {step}, equilibrating at lambda = 0, took: {elapsed_time / unit.seconds} seconds')
        pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
        old_pos = np.asarray(htf.old_positions(pos))
        new_pos = np.asarray(htf.new_positions(pos))
        forward_eq_old.append(old_pos)
        forward_eq_new.append(new_pos)

# Forward (0 -> 1)
forward_works = [integrator.get_protocol_work(dimensionless=True)]
for fwd_step in range(nsteps_neq):
    initial_time = time.time()
    integrator.step(1)
    elapsed_time = (time.time() - initial_time) * unit.seconds
    forward_works.append(integrator.get_protocol_work(dimensionless=True))
    if fwd_step % save_freq_neq == 0:
        _logger.info(f'forward NEQ step: {fwd_step}, took: {elapsed_time / unit.seconds} seconds')
        pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
        old_pos = np.asarray(htf.old_positions(pos))
        new_pos = np.asarray(htf.new_positions(pos))
        forward_neq_old.append(old_pos)
        forward_neq_new.append(new_pos)
forward_works_master.append(forward_works)

# Equilibrium (lambda = 1)
for step in range(nsteps_eq):
    initial_time = time.time()
    integrator.step(1)
    elapsed_time = (time.time() - initial_time) * unit.seconds
    if step % save_freq_eq == 0:
        _logger.info(f'Step: {step}, equilibrating at lambda = 1, took: {elapsed_time / unit.seconds} seconds')
        pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
        old_pos = np.asarray(htf.old_positions(pos))
        new_pos = np.asarray(htf.new_positions(pos))
        reverse_eq_new.append(new_pos)
        reverse_eq_old.append(old_pos)

# Reverse work (1 -> 0)
reverse_works = [integrator.get_protocol_work(dimensionless=True)]
for rev_step in range(nsteps_neq):
    initial_time = time.time()
    integrator.step(1)
    elapsed_time = (time.time() - initial_time) * unit.seconds
    reverse_works.append(integrator.get_protocol_work(dimensionless=True))
    if rev_step % save_freq_neq == 0:
        _logger.info(f'reverse NEQ step: {rev_step}, took: {elapsed_time / unit.seconds} seconds')
        pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
        old_pos = np.asarray(htf.old_positions(pos))
        new_pos = np.asarray(htf.new_positions(pos))
        reverse_neq_old.append(old_pos)
        reverse_neq_new.append(new_pos)
reverse_works_master.append(reverse_works)

# Save output
# create output directory if it does not exist
out_path = pathlib.Path(args.dir)
out_path.mkdir(parents=True, exist_ok=True)
# Save works
with open(os.path.join(out_path, f"forward.npy"), 'wb') as out_file:
    np.save(out_file, forward_works_master)
with open(os.path.join(out_path, f"reverse.npy"), 'wb') as out_file:
    np.save(out_file, reverse_works_master)

# Save trajs
with open(os.path.join(out_path, f"forward_eq_old.npy"), 'wb') as out_file:
    np.save(out_file, np.array(forward_eq_old))
with open(os.path.join(out_path, f"forward_eq_new.npy"), 'wb') as out_file:
    np.save(out_file, np.array(forward_eq_new))
with open(os.path.join(out_path, f"reverse_eq_new.npy"), 'wb') as out_file:
    np.save(out_file, np.array(reverse_eq_new))
with open(os.path.join(out_path, f"reverse_eq_old.npy"), 'wb') as out_file:
    np.save(out_file, np.array(reverse_eq_old))
with open(os.path.join(out_path, f"forward_neq_old.npy"), 'wb') as out_file:
    np.save(out_file, np.array(forward_neq_old))
with open(os.path.join(out_path, f"forward_neq_new.npy"), 'wb') as out_file:
    np.save(out_file, np.array(forward_neq_new))
with open(os.path.join(out_path, f"reverse_neq_old.npy"), 'wb') as out_file:
    np.save(out_file, np.array(reverse_neq_old))
with open(os.path.join(out_path, f"reverse_neq_new.npy"), 'wb') as out_file:
    np.save(out_file, np.array(reverse_neq_new))

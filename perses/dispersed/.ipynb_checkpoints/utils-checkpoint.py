import simtk.openmm as openmm
import openmmtools.cache as cache
from typing import List, Tuple, Union, NamedTuple
import os
import copy
import openmmtools.cache as cache

import openmmtools.mcmc as mcmc
import openmmtools.integrators as integrators
import openmmtools.states as states
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState
import numpy as np
import mdtraj as md
from perses.annihilation.relative import HybridTopologyFactory
import mdtraj.utils as mdtrajutils
import pickle
import simtk.unit as unit
import tqdm
from perses.tests.utils import compute_potential_components
from openmmtools.constants import kB
import pdb
import logging
import tqdm
from sys import getsizeof
import time
from collections import namedtuple
from perses.annihilation.lambda_protocol import LambdaProtocol
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol
import random
import pymbar
import dask.distributed as distributed
import tqdm
import time
from scipy.special import logsumexp

# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("sMC_utils")
_logger.setLevel(logging.DEBUG)

#cache.global_context_cache.platform = openmm.Platform.getPlatformByName('Reference') #this is just a local version
EquilibriumFEPTask = namedtuple('EquilibriumInput', ['sampler_state', 'inputs', 'outputs'])
              
#smc functions
def compute_survival_rate(sMC_particle_ancestries):
    """
    compute the time-series survival rate as a function of resamples

    Arguments
    ---------
    sMC_particle_ancestries : dict of {_direction : np.2darray(ints)}
        dict of the particle ancestor indices

    Returns
    -------
    survival_rate : dict of {_direction : np.array(float)}
        the particle survival rate as a function of step
    """
    survival_rate = {}
    for _direction in sMC_particle_ancestries.keys():
        rate = []
        num_starting_particles = np.multiply(*sMC_particle_ancestries[_direction][0].shape)
        for step in range(len(sMC_particle_ancestries[_direction])):
            rate.append(float(len(set(sMC_particle_ancestries[_direction][step].flatten()))) / num_starting_particles)
        survival_rate[_direction] = rate

    return survival_rate


def _pool_dict_results(_dict, _direction):
    """
    simple function to pool all of the actor results
    """
    _results_lst = list(_dict.values())
    flattened_result_lst = [item for sublist in _result_lst for item in sublist]
    return flattened_result_lst


def minimize(thermodynamic_state,
             sampler_state,
             max_iterations = 100):
    """
    Minimize the given system and state, up to a maximum number of steps.
    This does not return a copy of the samplerstate; it is simply an update-in-place.

    Arguments
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The state at which the system could be minimized
    sampler_state : openmmtools.states.SamplerState
        The starting state at which to minimize the system.
    max_iterations : int, optional, default 20
        The maximum number of minimization steps. Default is 100.

    Returns
    -------
    sampler_state : openmmtools.states.SamplerState
        The posititions and accompanying state following minimization
    """
    if type(cache.global_context_cache) == cache.DummyContextCache:
        integrator = openmm.VerletIntegrator(1.0) #we won't take any steps, so use a simple integrator
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state, integrator)
        _logger.debug(f"using dummy context cache")
    else:
        _logger.debug(f"using global context cache")
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
    sampler_state.apply_to_context(context, ignore_velocities = True)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations = max_iterations)
    sampler_state.update_from_context(context)

def multinomial_resample(total_works, num_resamples):
    """
    from a numpy array of total works and particle_labels, resample the particle indices N times with replacement
    from a multinomial distribution conditioned on the weights w_i \propto e^{-cumulative_works_i}
    Parameters
    ----------
    total_works : np.array of floats
        generalized accumulated works at time t for all particles
    num_resamples : int, default len(sampler_states)
        number of resamples to conduct; default doesn't change the number of particles

    Returns
    -------
    resampled_works : np.array([1.0/num_resamples]*num_resamples)
        resampled works (uniform)
    resampled_indices : np.array of ints
        resampled indices
    """
    normalized_weights = np.exp(-total_works - logsumexp(-total_works))
    resampled_indices = np.random.choice(len(normalized_weights), num_resamples, p=normalized_weights, replace = True)
    resampled_works = np.array([np.average(total_works)] * num_resamples)

    return resampled_works, resampled_indices

def ESS(works_prev, works_incremental):
    """
    compute the effective sample size (ESS) as given in Eq 3.15 in https://arxiv.org/abs/1303.3123.
    Parameters
    ----------
    works_prev: np.array
        np.array of floats representing the accumulated works at t-1 (unnormalized)
    works_incremental: np.array
        np.array of floats representing the incremental works at t (unnormalized)

    Returns
    -------
    ESS: float
        effective sample size
    """
    prev_weights_normalized = np.exp(-works_prev - logsumexp(-works_prev))
    incremental_weights_unnormalized = np.exp(-works_incremental)
    ESS = np.dot(prev_weights_normalized, incremental_weights_unnormalized)**2 / np.dot(np.power(prev_weights_normalized, 2), np.power(incremental_weights_unnormalized, 2))
    return ESS

def CESS(works_prev, works_incremental):
    """
    compute the conditional effective sample size (CESS) as given in Eq 3.16 in https://arxiv.org/abs/1303.3123.
    Parameters
    ----------
    works_prev: np.array
        np.array of floats representing the accumulated works at t-1 (unnormalized)
    works_incremental: np.array
        np.array of floats representing the incremental works at t (unnormalized)

    Returns
    -------
    CESS: float
        conditional effective sample size
    """
    prev_weights_normalization = np.exp(logsumexp(-works_prev))
    prev_weights_normalized = np.exp(-works_prev) / prev_weights_normalization
    incremental_weights_unnormalized = np.exp(-works_incremental)
    N = len(prev_weights_normalized)
    CESS = N * np.dot(prev_weights_normalized, incremental_weights_unnormalized)**2 / np.dot(prev_weights_normalized, np.power(incremental_weights_unnormalized, 2))
    return CESS

def compute_timeseries(reduced_potentials):
    """
    Use pymbar timeseries to compute the uncorrelated samples in an array of reduced potentials.  Returns the uncorrelated sample indices.
    """
    from pymbar import timeseries
    t0, g, Neff_max = timeseries.detectEquilibration(reduced_potentials) #computing indices of uncorrelated timeseries
    A_t_equil = reduced_potentials[t0:]
    uncorrelated_indices = timeseries.subsampleCorrelatedData(A_t_equil, g=g)
    A_t = A_t_equil[uncorrelated_indices]
    full_uncorrelated_indices = [i+t0 for i in uncorrelated_indices]

    return [t0, g, Neff_max, A_t, full_uncorrelated_indices]

def run_equilibrium(task):
    """
    Run n_iterations*nsteps_equil integration steps.  n_iterations mcmc moves are conducted in the initial equilibration, returning n_iterations
    reduced potentials.  This is the guess as to the burn-in time for a production.  After which, a single mcmc move of nsteps_equil
    will be conducted at a time, including a time-series (pymbar) analysis to determine whether the data are decorrelated.
    The loop will conclude when a single configuration yields an iid sample.  This will be saved.
    Parameters
    ----------
    task : FEPTask namedtuple
        The namedtuple should have an 'input' argument.  The 'input' argument is a dict characterized with at least the following keys and values:
        {
         thermodynamic_state: (<openmmtools.states.CompoundThermodynamicState>; compound thermodynamic state comprising state at lambda = 0 (1)),
         nsteps_equil: (<int>; The number of equilibrium steps that a move should make when apply is called),
         topology: (<mdtraj.Topology>; an MDTraj topology object used to construct the trajectory),
         n_iterations: (<int>; The number of times to apply the move. Note that this is not the number of steps of dynamics),
         splitting: (<str>; The splitting string for the dynamics),
         atom_indices_to_save: (<list of int, default None>; list of indices to save when excluding waters, for instance. If None, all indices are saved.),
         trajectory_filename: (<str, optional, default None>; Full filepath of trajectory files. If none, trajectory files are not written.),
         max_size: (<float>; maximum size of the trajectory numpy array allowable until it is written to disk),
         timer: (<bool, default False>; whether to time all parts of the equilibrium run),
         _minimize: (<bool, default False>; whether to minimize the sampler_state before conducting equilibration),
         file_iterator: (<int, default 0>; which index to begin writing files),
         timestep: (<unit.Quantity=float*unit.femtoseconds>; dynamical timestep)
         }
    """
    inputs = task.inputs

    timer = inputs['timer'] #bool
    timers = {}
    file_numsnapshots = []
    file_iterator = inputs['file_iterator']

    # creating copies in case computation is parallelized
    if timer: start = time.time()
    thermodynamic_state = copy.deepcopy(inputs['thermodynamic_state'])
    sampler_state = task.sampler_state
    if timer: timers['copy_state'] = time.time() - start

    if inputs['_minimize']:
        _logger.debug(f"conducting minimization")
        if timer: start = time.time()
        minimize(thermodynamic_state, sampler_state)
        if timer: timers['minimize'] = time.time() - start

    #get the atom indices we need to subset the topology and positions
    if timer: start = time.time()
    if not inputs['atom_indices_to_save']:
        atom_indices = list(range(inputs['topology'].n_atoms))
        subset_topology = inputs['topology']
    else:
        atom_indices = inputs['atom_indices_to_save']
        subset_topology = inputs['topology'].subset(atom_indices)
    if timer: timers['define_topology'] = time.time() - start

    n_atoms = subset_topology.n_atoms

    #construct the MCMove:
    mc_move = mcmc.LangevinSplittingDynamicsMove(n_steps=inputs['nsteps_equil'], splitting=inputs['splitting'], timestep = inputs['timestep'])
    mc_move.n_restart_attempts = 10

    #create a numpy array for the trajectory
    trajectory_positions, trajectory_box_lengths, trajectory_box_angles = list(), list(), list()
    reduced_potentials = list()

    #loop through iterations and apply MCMove, then collect positions into numpy array
    _logger.debug(f"conducting {inputs['n_iterations']} of production")
    if timer: eq_times = []

    init_file_iterator = inputs['file_iterator']
    for iteration in tqdm.trange(inputs['n_iterations']):
        if timer: start = time.time()
        _logger.debug(f"\tconducting iteration {iteration}")
        mc_move.apply(thermodynamic_state, sampler_state)

        #add reduced potential to reduced_potential_final_frame_list
        reduced_potentials.append(thermodynamic_state.reduced_potential(sampler_state))

        #trajectory_positions[iteration, :,:] = sampler_state.positions[atom_indices, :].value_in_unit_system(unit.md_unit_system)
        trajectory_positions.append(sampler_state.positions[atom_indices, :].value_in_unit_system(unit.md_unit_system))

        #get the box lengths and angles
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
        trajectory_box_lengths.append([a,b,c])
        trajectory_box_angles.append([alpha, beta, gamma])

        #if tajectory positions is too large, we have to write it to disk and start fresh
        if np.array(trajectory_positions).nbytes > inputs['max_size']:
            trajectory = md.Trajectory(np.array(trajectory_positions), subset_topology, unitcell_lengths=np.array(trajectory_box_lengths), unitcell_angles=np.array(trajectory_box_angles))
            if inputs['trajectory_filename'] is not None:
                new_filename = inputs['trajectory_filename'][:-2] + f'{file_iterator:04}' + '.h5'
                file_numsnapshots.append((new_filename, len(trajectory_positions)))
                file_iterator +=1
                write_equilibrium_trajectory(trajectory, new_filename)

                #re_initialize the trajectory positions, box_lengths, box_angles
                trajectory_positions, trajectory_box_lengths, trajectory_box_angles = list(), list(), list()

        if timer: eq_times.append(time.time() - start)

    if timer: timers['run_eq'] = eq_times
    _logger.debug(f"production done")

    #If there is a trajectory filename passed, write out the results here:
    if timer: start = time.time()
    if inputs['trajectory_filename'] is not None:
        #construct trajectory object:
        if trajectory_positions != list():
            #if it is an empty list, then the last iteration satistifed max_size and wrote the trajectory to disk;
            #in this case, we can just skip this
            trajectory = md.Trajectory(np.array(trajectory_positions), subset_topology, unitcell_lengths=np.array(trajectory_box_lengths), unitcell_angles=np.array(trajectory_box_angles))
            if file_iterator == init_file_iterator: #this means that no files have been written yet
                new_filename = inputs['trajectory_filename'][:-2] + f'{file_iterator:04}' + '.h5'
                file_numsnapshots.append((new_filename, len(trajectory_positions)))
            else:
                new_filename = inputs['trajectory_filename'][:-2] + f'{file_iterator+1:04}' + '.h5'
                file_numsnapshots.append((new_filename, len(trajectory_positions)))
            write_equilibrium_trajectory(trajectory, new_filename)

    if timer: timers['write_traj'] = time.time() - start

    if not timer:
        timers = {}

    return EquilibriumFEPTask(sampler_state = sampler_state, inputs = task.inputs, outputs = {'reduced_potentials': reduced_potentials, 'files': file_numsnapshots, 'timers': timers})

def write_equilibrium_trajectory(trajectory: md.Trajectory, trajectory_filename: str) -> float:
    """
    Write the results of an equilibrium simulation to disk. This task will append the results to the given filename.
    Parameters
    ----------
    trajectory : md.Trajectory
        the trajectory resulting from an equilibrium simulation
    trajectory_filename : str
        the name of the trajectory file to which we should append
    Returns
    -------
    True
    """
    if not os.path.exists(trajectory_filename):
        trajectory.save_hdf5(trajectory_filename)
        _logger.debug(f"{trajectory_filename} does not exist; instantiating and writing to.")
    else:
        _logger.debug(f"{trajectory_filename} exists; appending.")
        written_traj = md.load_hdf5(trajectory_filename)
        concatenated_traj = written_traj.join(trajectory)
        concatenated_traj.save_hdf5(trajectory_filename)

    return True

def write_nonequilibrium_trajectory(nonequilibrium_trajectory, trajectory_filename):
    """
    Write the results of a nonequilibrium switching trajectory to a file. The trajectory is written to an
    mdtraj hdf5 file.
    Parameters
    ----------
    nonequilibrium_trajectory : md.Trajectory
        The trajectory resulting from a nonequilibrium simulation
    trajectory_filename : str
        The full filepath for where to store the trajectory
    Returns
    -------
    True : bool
    """
    if nonequilibrium_trajectory is not None:
        nonequilibrium_trajectory.save_hdf5(trajectory_filename)

    return True

def compute_reduced_potential(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState) -> float:
    """
    Compute the reduced potential of the given SamplerState under the given ThermodynamicState.
    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state under which to compute the reduced potential
    sampler_state : openmmtools.states.SamplerState
        The sampler state for which to compute the reduced potential
    Returns
    -------
    reduced_potential : float
        unitless reduced potential (kT)
    """
    if type(cache.global_context_cache) == cache.DummyContextCache:
        integrator = openmm.VerletIntegrator(1.0) #we won't take any steps, so use a simple integrator
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state, integrator)
    else:
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
    sampler_state.apply_to_context(context, ignore_velocities=True)
    return thermodynamic_state.reduced_potential(context)


def compute_reduced_potential(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState) -> float:
    """
    Compute the reduced potential of the given SamplerState under the given ThermodynamicState.
    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state under which to compute the reduced potential
    sampler_state : openmmtools.states.SamplerState
        The sampler state for which to compute the reduced potential
    Returns
    -------
    reduced_potential : float
        unitless reduced potential (kT)
    """
    if type(cache.global_context_cache) == cache.DummyContextCache:
        integrator = openmm.VerletIntegrator(1.0) #we won't take any steps, so use a simple integrator
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state, integrator)
    else:
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
    sampler_state.apply_to_context(context, ignore_velocities=True)
    return thermodynamic_state.reduced_potential(context)
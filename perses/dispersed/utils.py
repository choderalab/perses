import simtk.openmm as openmm
import os
import copy

from openmmtools import cache
import openmmtools.mcmc as mcmc
import openmmtools.integrators as integrators
import openmmtools.states as states
from openmmtools.states import ThermodynamicState
import numpy as np
import mdtraj as md
import mdtraj.utils as mdtrajutils
import simtk.unit as unit
import tqdm
from openmmtools.constants import kB
import logging
import time
from collections import namedtuple
from perses.annihilation.lambda_protocol import LambdaProtocol
import dask.distributed as distributed
from scipy.special import logsumexp
import openmmtools.cache as cache
from openmmtools import utils

# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("sMC_utils")
_logger.setLevel(logging.INFO)
DISTRIBUTED_ERROR_TOLERANCE = 1e-6
EquilibriumFEPTask = namedtuple('EquilibriumInput', ['sampler_state', 'inputs', 'outputs'])

def check_platform(platform):
    """
    Check whether we can construct a simulation using this platform.
    From https://github.com/choderalab/integrator-benchmark/blob/bb307e6ebf476b652e62e41ae49730f530732da3/benchmark/testsystems/configuration.py#L17
    """
    from openmmtools.testsystems import HarmonicOscillator
    try:
        integrator = openmm.VerletIntegrator(1.0)
        testsystem = HarmonicOscillator()
        context = openmm.Context(testsystem.system, integrator, platform)
        del context, testsystem, integrator
    except Exception as e:
        print(f'Desired platform not supported. exception raised: {e}')
        raise Exception(e)


def configure_platform(platform_name='Reference', fallback_platform_name='CPU', precision='mixed'):
    """
    Retrieve the requested platform with platform-appropriate precision settings.
    platform_name : str, optional, default='Reference'
       The requested platform name
    fallback_platform_name : str, optional, default='CPU'
       If the requested platform cannot be provided, the fallback platform will be provided.
    Returns
    -------
    platform : simtk.openmm.Platform
       The requested platform with precision configured appropriately,
       or the fallback platform if this is not available.

    From https://github.com/choderalab/integrator-benchmark/blob/bb307e6ebf476b652e62e41ae49730f530732da3/benchmark/testsystems/configuration.py#L17
    """
    fallback_platform = openmm.Platform.getPlatformByName(fallback_platform_name)
    try:
        if platform_name.upper() == 'Reference'.upper():
            platform = openmm.Platform.getPlatformByName('Reference')
        elif platform_name.upper() == "CPU":
            platform = openmm.Platform.getPlatformByName("CPU")
        elif platform_name.upper() == 'OpenCL'.upper():
            platform = openmm.Platform.getPlatformByName('OpenCL')
            platform.setPropertyDefaultValue('OpenCLPrecision', precision)
        elif platform_name.upper() == 'CUDA'.upper():
            platform = openmm.Platform.getPlatformByName('CUDA')
            platform.setPropertyDefaultValue('CudaPrecision', precision)
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
        else:
            raise (ValueError("Invalid platform name"))

        check_platform(platform)

    except:
        print(
        "Warning: Returning {} platform instead of requested platform {}".format(fallback_platform_name, platform_name))
        platform = fallback_platform

    print(f"conducting subsequent work with the following platform: {platform.getName()}")
    return platform

#########
cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())
#########

#smc functions
def compute_survival_rate(sMC_particle_ancestries):
    """
    compute the time-series survival rate as a function of resamples

    Parameters
    ----------
    sMC_particle_ancestries : dict of {_direction : list(np.array(ints))}
        dict of the particle ancestor indices

    Returns
    -------
    survival_rate : dict of {_direction : np.array(float)}
        the particle survival rate as a function of step
    """
    survival_rate = {}
    for _direction, _lst in sMC_particle_ancestries.items():
        rate = []
        num_starting_particles = len(_lst[0])
        for step in range(len(sMC_particle_ancestries[_direction])):
            rate.append(float(len(set(sMC_particle_ancestries[_direction][step]))) / num_starting_particles)
        survival_rate[_direction] = rate

    return survival_rate

def minimize(thermodynamic_state,
             sampler_state,
             max_iterations = 100):
    """
    Minimize the given system and state, up to a maximum number of steps.
    This does not return a copy of the samplerstate; it is simply an update-in-place.

    Parameters
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
    r"""
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
    normalized_ESS: float
        effective sample size
    """
    prev_weights_normalized = np.exp(-works_prev - logsumexp(-works_prev))
    incremental_weights_unnormalized = np.exp(-works_incremental)
    ESS = np.dot(prev_weights_normalized, incremental_weights_unnormalized)**2 / np.dot(np.power(prev_weights_normalized, 2), np.power(incremental_weights_unnormalized, 2))
    normalized_ESS = ESS / len(prev_weights_normalized)
    assert normalized_ESS >= 0.0 - DISTRIBUTED_ERROR_TOLERANCE and normalized_ESS <= 1.0 + DISTRIBUTED_ERROR_TOLERANCE, f"the normalized ESS ({normalized_ESS} is not between 0 and 1)"
    return normalized_ESS

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
    CESS = np.dot(prev_weights_normalized, incremental_weights_unnormalized)**2 / np.dot(prev_weights_normalized, np.power(incremental_weights_unnormalized, 2))
    assert CESS >= 0.0 - DISTRIBUTED_ERROR_TOLERANCE and CESS <= 1.0 + DISTRIBUTED_ERROR_TOLERANCE, f"the CESS ({CESS} is not between 0 and 1)"
    return CESS

def compute_timeseries(reduced_potentials):
    """
    Use pymbar timeseries to compute the uncorrelated samples in an array of reduced potentials.  Returns the uncorrelated sample indices.

    Parameters
    ----------
    reduced_potentials : np.array of floats
        reduced potentials from which a timeseries is to be extracted

    Returns
    -------
    t0 : int
        production region index
    g : float
        statistical inefficiency
    Neff_max : int
        effective number of samples in production region
    full_uncorrelated_indices : list of ints
        uncorrelated indices

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
    task : EquilibriumFEPTask namedtuple
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

    Returns
    -------
    out_task : EquilibriumFEPTask namedtuple
        output EquilibriumFEPTask after equilibration
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
    mc_move = mcmc.LangevinSplittingDynamicsMove(n_steps=inputs['nsteps_equil'],
            splitting=inputs['splitting'], timestep = inputs['timestep'], context_cache = cache.ContextCache(capacity=None, time_to_live=None))
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

    out_task = EquilibriumFEPTask(sampler_state = sampler_state, inputs = task.inputs, outputs = {'reduced_potentials': reduced_potentials, 'files': file_numsnapshots, 'timers': timers})
    return out_task

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

def create_endstates(first_thermostate, last_thermostate):
    """
    utility function to generate unsampled endstates
    1. move all alchemical atom LJ parameters from CustomNonbondedForce to NonbondedForce
    2. delete the CustomNonbondedForce
    3. set PME tolerance to 1e-5
    4. enable LJPME to handle long range dispersion corrections in a physically reasonable manner

    Parameters
    ----------
    first_thermostate : openmmtools.states.CompoundThermodynamicState
        the first thermodynamic state for which an unsampled endstate will be created
    last_thermostate : openmmtools.states.CompoundThermodynamicState
        the last thermodynamic state for which an unsampled endstate will be created

    Returns
    -------
    unsampled_endstates : list of openmmtools.states.CompoundThermodynamicState
        the corrected unsampled endstates
    """
    unsampled_endstates = []
    for master_lambda, endstate in zip([0., 1.], [first_thermostate, last_thermostate]):
        dispersion_system = endstate.get_system()
        energy_unit = unit.kilocalories_per_mole
        # Find the NonbondedForce (there must be only one)
        forces = { force.__class__.__name__ : force for force in dispersion_system.getForces() }
        # Set NonbondedForce to use LJPME
        forces['NonbondedForce'].setNonbondedMethod(openmm.NonbondedForce.LJPME)
        # Set tight PME tolerance
        TIGHT_PME_TOLERANCE = 1.0e-5
        forces['NonbondedForce'].setEwaldErrorTolerance(TIGHT_PME_TOLERANCE)
        # Move alchemical LJ sites from CustomNonbondedForce back to NonbondedForce
        for particle_index in range(forces['NonbondedForce'].getNumParticles()):
            charge, sigma, epsilon = forces['NonbondedForce'].getParticleParameters(particle_index)
            sigmaA, epsilonA, sigmaB, epsilonB, unique_old, unique_new = forces['CustomNonbondedForce'].getParticleParameters(particle_index)
            if (epsilon/energy_unit == 0.0) and ((epsilonA > 0.0) or (epsilonB > 0.0)):
                sigma = (1-master_lambda)*sigmaA + master_lambda*sigmaB
                epsilon = (1-master_lambda)*epsilonA + master_lambda*epsilonB
                forces['NonbondedForce'].setParticleParameters(particle_index, charge, sigma, epsilon)
        # Delete the CustomNonbondedForce since we have moved all alchemical particles out of it
        for force_index, force in enumerate(list(dispersion_system.getForces())):
            if force.__class__.__name__ == 'CustomNonbondedForce':
                custom_nonbonded_force_index = force_index
                break
        dispersion_system.removeForce(custom_nonbonded_force_index)
        # Set all parameters to master lambda
        for force_index, force in enumerate(list(dispersion_system.getForces())):
            if hasattr(force, 'getNumGlobalParameters'):
                for parameter_index in range(force.getNumGlobalParameters()):
                    if force.getGlobalParameterName(parameter_index)[0:7] == 'lambda_':
                        force.setGlobalParameterDefaultValue(parameter_index, master_lambda)
        # Store the unsampled endstate
        unsampled_endstates.append(ThermodynamicState(dispersion_system, temperature = endstate.temperature))

    return unsampled_endstates

################################################################
##################Distributed Tasks#############################
################################################################
def activate_LocallyOptimalAnnealing(thermodynamic_state,
                                     remote_worker,
                                     lambda_protocol = 'default',
                                     timestep = 1 * unit.femtoseconds,
                                     collision_rate = 1 / unit.picoseconds,
                                     temperature = 300 * unit.kelvin,
                                     neq_splitting_string = 'V R O R V',
                                     ncmc_save_interval = None,
                                     topology = None,
                                     subset_atoms = None,
                                     measure_shadow_work = False,
                                     integrator = 'langevin',
                                     compute_endstate_correction = True):
    """
    Function to set worker attributes for annealing.
    """
    supported_integrators = ['langevin', 'hmc']

    if remote_worker == 'remote':
        _class = distributed.get_worker()
    else:
        _class = remote_worker

    _class.annealing_class = LocallyOptimalAnnealing()
    assert _class.annealing_class.initialize(thermodynamic_state = thermodynamic_state,
                                             lambda_protocol = lambda_protocol,
                                             timestep = timestep,
                                             collision_rate = collision_rate,
                                             temperature = temperature,
                                             neq_splitting_string = neq_splitting_string,
                                             ncmc_save_interval = ncmc_save_interval,
                                             topology = topology,
                                             subset_atoms = subset_atoms,
                                             measure_shadow_work = measure_shadow_work,
                                             integrator = integrator,
                                             compute_endstate_correction = compute_endstate_correction)

def deactivate_worker_attributes(remote_worker):
    """
    Function to remove worker attributes for annealing
    """
    if remote_worker == 'remote':
        _logger.debug(f"\t\tremote_worker is True, getting worker")
        _class = distributed.get_worker()
    else:
        _logger.debug(f"\t\tremote worker is not True; getting local worker as 'self'")
        _class = remote_worker

    delattr(_class, 'annealing_class')

    address = _class.address if remote_worker == 'remote' else 0
    return address

def call_anneal_method(remote_worker,
                       sampler_state,
                       lambdas,
                       noneq_trajectory_filename = None,
                       num_integration_steps = 1,
                       return_timer = False,
                       return_sampler_state = False,
                       rethermalize = False,
                       compute_incremental_work = True):
    """
    this function calls LocallyOptimalAnnealing.anneal;
    since we can only map functions with parallelisms (no actors), we need to submit a function that calls
    the LocallyOptimalAnnealing.anneal method.
    """
    if remote_worker == 'remote':
        _class = distributed.get_worker()
    else:
        _class = remote_worker

    incremental_work, new_sampler_state, timer, _pass, endstate_corrections = _class.annealing_class.anneal(sampler_state = sampler_state,
                                                                                      lambdas = lambdas,
                                                                                      noneq_trajectory_filename = noneq_trajectory_filename,
                                                                                      num_integration_steps = num_integration_steps,
                                                                                      return_timer = return_timer,
                                                                                      return_sampler_state = return_sampler_state,
                                                                                      rethermalize = rethermalize,
                                                                                      compute_incremental_work = compute_incremental_work)
    return incremental_work, new_sampler_state, timer, _pass, endstate_corrections



class LocallyOptimalAnnealing():
    """
    Actor for locally optimal annealed importance sampling.
    The initialize method will create an appropriate context and the appropriate storage objects,
    but must be called explicitly.
    """
    supported_integrators = ['langevin', 'hmc']

    def initialize(self,
                   thermodynamic_state,
                   lambda_protocol = 'default',
                   timestep = 1 * unit.femtoseconds,
                   collision_rate = 1 / unit.picoseconds,
                   temperature = 300 * unit.kelvin,
                   neq_splitting_string = 'V R O R V',
                   ncmc_save_interval = None,
                   topology = None,
                   subset_atoms = None,
                   measure_shadow_work = False,
                   integrator = 'langevin',
                   compute_endstate_correction = True):

        try:
            self.context_cache = cache.global_context_cache

            if measure_shadow_work:
                measure_heat = True
            else:
                measure_heat = False

            self.thermodynamic_state = thermodynamic_state
            if integrator == 'langevin':
                self.integrator = integrators.LangevinIntegrator(temperature = temperature,
                                                                 timestep = timestep,
                                                                 splitting = neq_splitting_string,
                                                                 measure_shadow_work = measure_shadow_work,
                                                                 measure_heat = measure_heat,
                                                                 constraint_tolerance = 1e-6,
                                                                 collision_rate = collision_rate)
            elif integrator == 'hmc':
                self.integrator = integrators.HMCIntegrator(temperature = temperature,
                                                            nsteps = 2,
                                                            timestep = timestep/2)
            else:
                raise Exception(f"integrator {integrator} is not supported. supported integrators include {self.supported_integrators}")

            self.lambda_protocol_class = LambdaProtocol(functions = lambda_protocol)

            #create temperatures
            self.beta = 1.0 / (kB*temperature)
            self.temperature = temperature

            self.save_interval = ncmc_save_interval

            self.topology = topology
            self.subset_atoms = subset_atoms

            #if we have a trajectory, set up some ancillary variables:
            if self.topology is not None:
                n_atoms = self.topology.n_atoms
                self._trajectory_positions = []
                self._trajectory_box_lengths = []
                self._trajectory_box_angles = []

            self.compute_endstate_correction = compute_endstate_correction
            if self.compute_endstate_correction:
                self.thermodynamic_state.set_alchemical_parameters(0.0, lambda_protocol = self.lambda_protocol_class)
                first_endstate = copy.deepcopy(self.thermodynamic_state)
                self.thermodynamic_state.set_alchemical_parameters(1.0, lambda_protocol = self.lambda_protocol_class)
                last_endstate = copy.deepcopy(self.thermodynamic_state)
                endstates = create_endstates(first_endstate, last_endstate)
                self.endstates = {0.0: endstates[0], 1.0: endstates[1]}
            else:
                self.endstates = None

            #set a bool variable for pass or failure
            self.succeed = True
            return True
        except Exception as e:
            _logger.error(e)
            self.succeed = False
            return False

    def anneal(self,
               sampler_state,
               lambdas,
               noneq_trajectory_filename = None,
               num_integration_steps = 1,
               return_timer = False,
               return_sampler_state = False,
               rethermalize = False,
               compute_incremental_work = True):
        """
        conduct annealing across lambdas.

        Parameters
        ----------
        sampler_state : openmmtools.states.SamplerState
            The starting state at which to minimize the system.
        noneq_trajectory_filename : str, default None
            Name of the nonequilibrium trajectory file to which we write
        lambdas : np.array
            numpy array of the lambdas to run
        num_integration_steps : np.array or int, default 1
            the number of integration steps to be conducted per proposal
        return_timer : bool, default False
            whether to time the annealing protocol
        return_sampler_state : bool, default False
            whether to return the last sampler state
        rethermalize : bool, default False,
            whether to re-initialize velocities after propagation step
        compute_incremental_work : bool, default True
            whether to compute the incremental work or simply anneal


        Returns
        -------
        incremental_work : np.array of shape (1, len(lambdas) - 1)
            cumulative works for every lambda
        sampler_state : openmmtools.states.SamplerState
            configuration at last lambda after proposal
        timer : np.array
            timers
        _pass : bool
            whether the annealing protocol passed
        compute_endstate_corrections : tuple of floats or None
            the endstate correction
            Convention:
        """
        #check if we can save the trajectory
        if noneq_trajectory_filename is not None:
            if self.save_interval is None:
                raise Exception(f"The save interval is None, but a nonequilibrium trajectory filename was given!")

        #check returnables for timers:
        if return_timer is not None:
            timer = np.zeros(len(lambdas) - 1)
        else:
            timer = None
        if compute_incremental_work:
            incremental_work = np.zeros(len(lambdas) - 1)
        #first set the thermodynamic state to the proper alchemical state and pull context, integrator
        self.sampler_state = sampler_state
        if self.compute_endstate_correction:
            endstate_rps = {_endstate: None for _endstate in self.endstates.keys()}
            self.thermodynamic_state.set_alchemical_parameters(lambdas[0], lambda_protocol = self.lambda_protocol_class)
            if lambdas[0] == 0.:
                try:
                    endstate_rps[0.] = self.endstates[0.].reduced_potential(self.sampler_state) - self.thermodynamic_state.reduced_potential(self.sampler_state)
                except:
                    pass
            elif lambdas[0] == 1.:
                try:
                    endstate_rps[1.] = self.endstates[1.].reduced_potential(self.sampler_state) - self.thermodynamic_state.reduced_potential(self.sampler_state)
                except:
                    pass
        else:
            endstate_rps = None

        if compute_incremental_work:
            self.dummy_sampler_state = copy.deepcopy(sampler_state) #use dummy to not update velocities and save bandwidth
        self.thermodynamic_state.set_alchemical_parameters(lambdas[0], lambda_protocol = self.lambda_protocol_class)
        self.context, integrator = self.context_cache.get_context(self.thermodynamic_state, self.integrator)
        self.sampler_state.apply_to_context(self.context, ignore_velocities=False)

        for idx, _lambda in enumerate(lambdas[1:]): #skip the first lambda
            try:
                if return_timer:
                    start_timer = time.time()
                if compute_incremental_work: #compute incremental work and update the context
                    _incremental_work = self.compute_incremental_work(_lambda)
                    assert np.isfinite(_incremental_work) #check to make sure that the incremental work doesn't blow up; not checking velocities
                    incremental_work[idx] = _incremental_work
                else: #simply update the context from the thermodynamic state
                    self.update_context(_lambda)

                integrator.step(num_integration_steps)

                if rethermalize:
                    self.context.setVelocitiesToTemperature(self.thermodynamic_state.temperature) #rethermalize
                if noneq_trajectory_filename is not None:
                    self.save_configuration(idx, sampler_state, context)
                if return_timer:
                    timer[idx] = time.time() - start_timer
            except Exception as e:
                print(f"failure: {e}")
                self.reset_dimensions()
                return None, None, None, False, None

        self.attempt_termination(noneq_trajectory_filename)

        #determine corrected endstates
        if self.compute_endstate_correction:
            self.thermodynamic_state.set_alchemical_parameters(lambdas[-1], lambda_protocol = self.lambda_protocol_class)
            if lambdas[-1] == 0.:
                try:
                    endstate_rps[0.] = self.endstates[0.].reduced_potential(self.sampler_state.update_from_context(self.context)) - self.thermodynamic_state.reduced_potential(self.sampler_state)
                except:
                    pass
            elif lambdas[-1] == 1.:
                try:
                    endstate_rps[1.] = self.endstates[1.].reduced_potential(self.sampler_state.update_from_context(self.context)) - self.thermodynamic_state.reduced_potential(self.sampler_state)
                except:
                    pass
            #now we can compute the corrections
            if all(type(i) == np.float64 for i in  np.array(list(endstate_rps.values()))) and all(not q for q in np.isinf(np.array(list(endstate_rps.values())))): #then we can perform a calculation
                return_endstate_corrections = endstate_rps
            else:
                return_endstate_corrections = None
        else:
            return_endstate_corrections = None


        #pull the last sampler state and return
        if return_sampler_state:
            if rethermalize:
                self.sampler_state.update_from_context(self.context, ignore_velocities=True)
            else:
                self.sampler_state.update_from_context(self.context, ignore_velocities=False)
                assert not self.sampler_state.has_nan()
            if not compute_incremental_work:
                incremental_work = None

            return (incremental_work, sampler_state, timer, True, return_endstate_corrections)
        else:
            return (incremental_work, None, timer, True, return_endstate_corrections)



    def attempt_termination(self, noneq_trajectory_filename):
        """
        Attempt to terminate the annealing protocol and return the Particle attributes.

        Parameters
        ----------
        noneq_trajectory_filename : str, default None
            Name of the nonequilibrium trajectory file to which we write
        """
        if noneq_trajectory_filename is not None:
            _logger.info(f"saving configuration")
            trajectory = md.Trajectory(np.array(self._trajectory_positions), self.topology, unitcell_lengths=np.array(self._trajectory_box_lengths), unitcell_angles=np.array(self._trajectory_box_angles))
            write_nonequilibrium_trajectory(trajectory, noneq_trajectory_filename)

        self.reset_dimensions()

    def reset_dimensions(self):
        """
        utility method to reset trajectory positions, box_lengths, and box_angles.
        """
        self._trajectory_positions = []
        self._trajectory_box_lengths = []
        self._trajectory_box_angles = []

    def compute_incremental_work(self, _lambda):
        """
        compute the incremental work of a lambda update on the thermodynamic state.
        function also updates the thermodynamic state and the context

        Parameters
        ----------
        _lambda : float
            the lambda value used to update the importance sample
        sampler_state : openmmtools.states.SamplerState
            sampler state with which to update

        Returns
        -------
        _incremental_work : float or None
            the incremental work returned from the lambda update; if None, then there is a numerical instability
        """
        self.dummy_sampler_state.update_from_context(self.context, ignore_velocities=True)
        assert not self.dummy_sampler_state.has_nan()
        old_rp = self.thermodynamic_state.reduced_potential(self.dummy_sampler_state)

        #update thermodynamic state and context
        self.update_context(_lambda)

        self.dummy_sampler_state.update_from_context(self.context, ignore_velocities=True)
        assert not self.dummy_sampler_state.has_nan()
        new_rp = self.thermodynamic_state.reduced_potential(self.dummy_sampler_state)
        _incremental_work = new_rp - old_rp

        return _incremental_work

    def update_context(self, _lambda):
        """
        utility function to update the class context

        Parameters
        ----------
        _lambda : float
            the lambda value that the self.context will be updated to
        """
        self.thermodynamic_state.set_alchemical_parameters(_lambda, lambda_protocol = self.lambda_protocol_class)
        self.thermodynamic_state.apply_to_context(self.context)


    def save_configuration(self, iteration, sampler_state, context):
        """
        pass a conditional save function

        Parameters
        ----------
        iteration : int
            the iteration index
        sampler_state : openmmtools.states.SamplerState
            sampler state to save
        context : simtk.openmm.app.Context
            context used to update the sampler state
        """
        if iteration % self.ncmc_save_interval == 0: #we save the protocol work if the remainder is zero
            _logger.debug(f"\t\tsaving protocol")
            #self._kinetic_energy.append(self._beta * context.getState(getEnergy=True).getKineticEnergy()) #maybe if we want kinetic energy in the future
            sampler_state.update_from_context(self.context, ignore_velocities=True) #save bandwidth by not updating the velocities

            if self.subset_atoms is None:
                self._trajectory_positions.append(sampler_state.positions[:, :].value_in_unit_system(unit.md_unit_system))
            else:
                self._trajectory_positions.append(sampler_state.positions[self.subset_atoms, :].value_in_unit_system(unit.md_unit_system))

                #get the box angles and lengths
                a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
                self._trajectory_box_lengths.append([a, b, c])
                self._trajectory_box_angles.append([alpha, beta, gamma])

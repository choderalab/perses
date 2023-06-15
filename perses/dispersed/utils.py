import simtk.openmm as openmm
import os
import copy

import openmmtools.mcmc as mcmc
import openmmtools.integrators as integrators
import openmmtools.states as states
from openmmtools import utils
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
from scipy.special import logsumexp
import openmmtools.cache as cache

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

# Instantiate logger
_logger = logging.getLogger("sMC_utils")
_logger.setLevel(logging.INFO)
DISTRIBUTED_ERROR_TOLERANCE = 1e-6
EquilibriumFEPTask = namedtuple('EquilibriumInput', ['sampler_state', 'inputs', 'outputs'])

# Default to fastest platform for compute heavy workflow
DEFAULT_PLATFORM = utils.get_fastest_platform()


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


def check_system(system):
    """
    Check OpenMM System object for pathologies, like duplicate atoms in torsions.

    Parameters
    ----------
    system : simtk.openmm.System

    """
    forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
    force = forces['PeriodicTorsionForce']
    for index in range(force.getNumTorsions()):
        [i, j, k, l, periodicity, phase, barrier] = force.getTorsionParameters(index)
        if len(set([i,j,k,l])) < 4:
            msg  = 'Torsion index %d of self._topology_proposal.new_system has duplicate atoms: %d %d %d %d\n' % (index,i,j,k,l)
            msg += 'Serialzed system to system.xml for inspection.\n'
            raise Exception(msg)
    from simtk.openmm import XmlSerializer
    serialized_system = XmlSerializer.serialize(system)
    outfile = open('system.xml', 'w')
    outfile.write(serialized_system)
    outfile.close()


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


# smc functions
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


def compute_potential_components(context, beta=beta, platform=DEFAULT_PLATFORM):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    context : simtk.openmm.Context
        The context from which to extract, System, parameters, and positions.

    """
    # Make a deep copy of the system.
    import copy

    from perses.dispersed.utils import configure_platform
    platform = configure_platform(platform.getName(), fallback_platform_name='Reference', precision='double')

    system = context.getSystem()
    system = copy.deepcopy(system)
    # Get positions.
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Get Parameters
    parameters = context.getParameters()
    # Segregate forces.
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
    # Create new Context.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for (parameter, value) in parameters.items():
        context.setParameter(parameter, value)
    energy_components = dict()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        forcename = force.getName()
        groups = 1 << index
        potential = beta * context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        energy_components[forcename] = potential
    del context, integrator
    return energy_components


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

    Works for `HybridTopologyFactory`
    Note that this function is kept for legacy purposes and that `create_endstates_from_real_systems' is a more
    general version of this function. For `HybridTopologyFactory`, both functions do the same thing, so either can be used
    when generating unsampled endstates for `HybridTopologyFactory`. By default, `create_endstates_from_real_systems` is used.

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


def generate_endpoint_thermodynamic_states(system, topology_proposal, repartitioned_endstate=None):
    """
    Generate endpoint thermodynamic states for the system

    Parameters
    ----------
    system : openmm.System
        System object corresponding to thermodynamic state
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        TopologyProposal representing transformation
    repartitioned_endstate : int, default None
        If the htf was generated using RepartitionedHybridTopologyFactory, use this argument to specify the endstate at
        which it was generated. Otherwise, leave as None.

    Returns
    -------
    nonalchemical_zero_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda zero endpoint
    nonalchemical_one_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda one endpoint
    lambda_zero_thermodynamic_state : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda zero
    lambda_one_thermodynamic_State : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda one
    """
    # Create the thermodynamic state
    from perses.annihilation.lambda_protocol import RelativeAlchemicalState

    check_system(system)

    # Create thermodynamic states for the nonalchemical endpoints
    nonalchemical_zero_thermodynamic_state = states.ThermodynamicState(topology_proposal.old_system, temperature=temperature)
    nonalchemical_one_thermodynamic_state = states.ThermodynamicState(topology_proposal.new_system, temperature=temperature)

    # Create the base thermodynamic state with the hybrid system
    thermodynamic_state = states.ThermodynamicState(system, temperature=temperature)

    if repartitioned_endstate == 0:
        lambda_zero_thermodynamic_state = thermodynamic_state
        lambda_one_thermodynamic_state = None
    elif repartitioned_endstate == 1:
        lambda_zero_thermodynamic_state = None
        lambda_one_thermodynamic_state = thermodynamic_state
    else:
        # Create relative alchemical states
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(system)
        lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

        # Ensure their states are set appropriately
        lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
        lambda_one_alchemical_state.set_alchemical_parameters(1.0)

        # Now create the compound states with different alchemical states
        lambda_zero_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_zero_alchemical_state])
        lambda_one_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_one_alchemical_state])

    return nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state


def create_endstates_from_real_systems(htf, for_testing=False):
    """
    Generates unsampled endstates using LJPME as the nonbonded method to more accurately account for
    long-range steric interactions at the lambda = 0 and lambda = 1 endstates.

    Works for `HybridTopologyFactory` and `RESTCapableHybridTopologyFactory`

    Tested in tests/test_relative.py::test_create_endstates()

    Parameters
    ----------
    htf : HybridTopologyFactory or RESTCapableHybridTopologyFactory
        hybrid factory from which to create the unsampled endstates
    for_testing : bool, default False
        whether to generate the unsampled endstates for testing
        For energy validation tests, we'll use PME as the nonbonded method, since the original hybrid system uses PME (not LJPME).

    Returns
    -------
    unsampled_endstates : list of ThermodynamicState
        the unsampled endstate at lambda = 0, then the unsampled endstate at lambda = 1
    """
    import openmm
    from perses.annihilation.relative import RESTCapableHybridTopologyFactory

    TIGHT_PME_TOLERANCE = 1.0e-5

    # Retrieve old and new NonbondedForces
    old_system_nbf = htf._old_system_forces['NonbondedForce']
    new_system_nbf = htf._new_system_forces['NonbondedForce']

    # Load original hybrid_system
    hybrid_system = htf.hybrid_system

    # Make a copy of the hybrid_system
    hybrid_system_0 = copy.deepcopy(hybrid_system)
    hybrid_system_1 = copy.deepcopy(hybrid_system)

    # Delete existing nonbonded-related forces, create a new NonbondedForce, copy particles/exceptions
    unsampled_endstates = []
    for lambda_val, hybrid_system in zip([0, 1], [hybrid_system_0, hybrid_system_1]):
        # Delete existing nonbonded-related forces
        forces = {hybrid_system.getForce(index).getName(): index for index in range(hybrid_system.getNumForces())}
        indices_to_remove = [index for name, index in forces.items() if 'Nonbonded' in name or 'exceptions' in name]
        for index in sorted(indices_to_remove, reverse=True):
            hybrid_system.removeForce(index)

            # Set defaults for global parameters depending on the factory
            htf_class = htf.__class__.__name__
            for force_index, force in enumerate(list(hybrid_system.getForces())):
                if hasattr(force, 'getNumGlobalParameters'): # Only custom forces will have global parameters to set
                    for parameter_index in range(force.getNumGlobalParameters()):
                        global_parameter_name = force.getGlobalParameterName(parameter_index)
                        if global_parameter_name[0:7] == 'lambda_':
                            if htf_class == 'HybridTopologyFactory':
                                force.setGlobalParameterDefaultValue(parameter_index, lambda_val)
                            elif htf_class == 'RESTCapableHybridTopologyFactory':
                                if 'old' in global_parameter_name:
                                    force.setGlobalParameterDefaultValue(parameter_index, 1 - lambda_val)
                                elif 'new' in global_parameter_name:
                                    force.setGlobalParameterDefaultValue(parameter_index, lambda_val)
                            else:
                                raise Exception(
                                    f"{htf_class} is not supported. Supported factories: HybridTopologyFactory, RESTCapableHybridTopologyFactory")

        # Create NonbondedForce
        nonbonded_force = openmm.NonbondedForce()
        hybrid_system.addForce(nonbonded_force)

        # Set nonbonded method and related attributes
        if not for_testing:
            nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.LJPME)
            nonbonded_force.setEwaldErrorTolerance(TIGHT_PME_TOLERANCE)
        else:
            nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
            [alpha_ewald, nx, ny, nz] = old_system_nbf.getPMEParameters()
            delta = old_system_nbf.getEwaldErrorTolerance()
            nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            nonbonded_force.setEwaldErrorTolerance(delta)
            nonbonded_force.setCutoffDistance(old_system_nbf.getCutoffDistance())

        # Retrieve name of atom class for which to zero the charges/epsilons
        atom_class_to_zero = 'unique_new_atoms' if lambda_val == 0 else 'unique_old_atoms'

        # Iterate over particles in the old NonbondedForce and add them to the NonbondedForce for the hybrid system
        done_indices = []
        for idx in range(old_system_nbf.getNumParticles()):
            # Given the atom index, get charge, sigma, epsilon, and hybrid index
            hybrid_idx = htf._old_to_hybrid_map[idx]
            alch_id, atom_class = RESTCapableHybridTopologyFactory.get_alch_identifier(htf, hybrid_idx)
            charge, sigma, epsilon = old_system_nbf.getParticleParameters(idx)
            if atom_class == atom_class_to_zero:
                charge, sigma, epsilon = charge * 0, sigma, epsilon * 0
            elif atom_class == 'core_atoms' and lambda_val == 1:  # Retrieve new system parameters for core atoms
                new_idx = htf._hybrid_to_new_map[hybrid_idx]
                charge, sigma, epsilon = new_system_nbf.getParticleParameters(new_idx)

            # Add particle to the NonbondedForce
            nonbonded_force.addParticle(charge, sigma, epsilon)

            # Keep track of indices that have already been added
            done_indices.append(hybrid_idx)

        # Iterate over particles in the new NonbondedForce and add them to the NonbondedForce for the hybrid system
        remaining_hybrid_indices = sorted(set(range(hybrid_system.getNumParticles())).difference(set(done_indices)))
        for hybrid_idx in remaining_hybrid_indices:
            # Given the atom index, get charge, sigma, and epsilon
            idx = htf._hybrid_to_new_map[hybrid_idx]
            alch_id, atom_class = RESTCapableHybridTopologyFactory.get_alch_identifier(htf, hybrid_idx)
            charge, sigma, epsilon = new_system_nbf.getParticleParameters(idx)
            if atom_class == atom_class_to_zero:
                charge, sigma, epsilon = charge * 0, sigma, epsilon * 0

            # Add particle to NonbondedForce, with zeroed charge and epsilon
            nonbonded_force.addParticle(charge, sigma, epsilon)

        # Now remove interactions between unique old/new
        unique_news = htf._atom_classes['unique_new_atoms']
        unique_olds = htf._atom_classes['unique_old_atoms']
        for new in unique_news:
            for old in unique_olds:
                nonbonded_force.addException(old,
                                             new,
                                             0.0 * unit.elementary_charge ** 2,
                                             1.0 * unit.nanometers,
                                             0.0 * unit.kilojoules_per_mole)

        # Now add add all nonzeroed exceptions to custom bond force
        old_term_collector = {}
        new_term_collector = {}

        # Gather the old system exceptions into a dict
        for term_idx in range(old_system_nbf.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = old_system_nbf.getExceptionParameters(term_idx)  # Grab the parameters
            hybrid_p1, hybrid_p2 = htf._old_to_hybrid_map[p1], htf._old_to_hybrid_map[p2]  # Make hybrid indices
            sorted_indices = tuple(sorted([hybrid_p1, hybrid_p2]))  # Sort the indices
            assert not sorted_indices in old_term_collector.keys(), f"this exception already exists"
            old_term_collector[sorted_indices] = [term_idx, chargeProd, sigma, epsilon]

        # Repeat for the new system exceptions
        for term_idx in range(new_system_nbf.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = new_system_nbf.getExceptionParameters(term_idx)
            hybrid_p1, hybrid_p2 = htf._new_to_hybrid_map[p1], htf._new_to_hybrid_map[p2]
            sorted_indices = tuple(sorted([hybrid_p1, hybrid_p2]))
            assert not sorted_indices in new_term_collector.keys(), f"this exception already exists"
            new_term_collector[sorted_indices] = [term_idx, chargeProd, sigma, epsilon]

        htf_class = htf.__class__.__name__

        # Iterate over the old_term_collector and add exceptions to the NonbondedForce for the hybrid system
        for hybrid_index_pair in old_term_collector.keys():
            # Get terms
            idx_set = set(list(hybrid_index_pair))
            alch_id, atom_class = RESTCapableHybridTopologyFactory.get_alch_identifier(htf, idx_set)
            idx, chargeProd, sigma, epsilon = old_term_collector[hybrid_index_pair]
            if atom_class == atom_class_to_zero:
                if htf_class == 'HybridTopologyFactory' and not htf._interpolate_14s:  # When _interpolate_14s is False, the exceptions always remain on
                    pass
                else:
                    chargeProd, sigma, epsilon = chargeProd * 0, sigma, epsilon * 0
            elif atom_class == 'core_atoms' and lambda_val == 1:  # Retrieve new system parameters for core atoms
                new_idx, chargeProd, sigma, epsilon = new_term_collector[hybrid_index_pair]

            # Add exception
            nonbonded_force.addException(hybrid_index_pair[0], hybrid_index_pair[1], chargeProd, sigma, epsilon)

        # Now iterate over the new_term_collector and add exceptions to the NonbondedForce for the hybrid system
        for hybrid_index_pair in new_term_collector.keys():
            if hybrid_index_pair not in old_term_collector.keys():
                # Get terms
                idx_set = set(list(hybrid_index_pair))
                alch_id, atom_class = RESTCapableHybridTopologyFactory.get_alch_identifier(htf, idx_set)
                idx, chargeProd, sigma, epsilon = new_term_collector[hybrid_index_pair]
                if atom_class == atom_class_to_zero:
                    if htf_class == 'HybridTopologyFactory' and not htf._interpolate_14s:  # When _interpolate_14s is False, the exceptions always remain on
                        pass
                    else:
                        chargeProd, sigma, epsilon = chargeProd * 0, sigma, epsilon * 0

                # Add exception
                nonbonded_force.addException(hybrid_index_pair[0], hybrid_index_pair[1], chargeProd, sigma, epsilon)

        unsampled_endstates.append(ThermodynamicState(hybrid_system, temperature=temperature))

    return unsampled_endstates


def validate_endstate_energies(topology_proposal,
                               htf,
                               added_energy,
                               subtracted_energy,
                               beta=1.0/kT,
                               ENERGY_THRESHOLD=1e-6,
                               platform=DEFAULT_PLATFORM,
                               trajectory_directory=None,
                               repartitioned_endstate=None):
    """
    ** Used for validating endstate energies for HybridTopologyFactory **

    Function to validate that the difference between the nonalchemical versus alchemical state at lambda = 0,1 is
    equal to the difference in valence energy (forward and reverse).

    Parameters
    ----------
    topology_proposal : perses.topology_proposal.TopologyProposal object
        top_proposal for relevant transformation
    htf : perses.new_relative.HybridTopologyFactory object
        hybrid top factory for setting alchemical hybrid states
    added_energy : float
        reduced added valence energy
    subtracted_energy : float
        reduced subtracted valence energy
    beta : float, default 1.0/kT
        unit-bearing inverse thermal energy
    ENERGY_THRESHOLD : float, default 1e-6
        threshold for ratio in energy difference at a particular endstate
    platform : str, default utils.get_fastest_platform()
        platform to conduct validation on (e.g. 'CUDA', 'Reference', 'OpenCL')
    trajectory_directory : str, default None
        path to save the save the serialized state to. If None, the state will not be saved
    repartitioned_endstate : int, default None
        if the htf was generated using RepartitionedHybridTopologyFactory, use this argument to specify the endstate at
        which it was generated. Otherwise, leave as None.

    Returns
    -------
    zero_state_energy_difference : float
        reduced potential difference of the nonalchemical and alchemical lambda = 0 state (corrected for valence energy).
    one_state_energy_difference : float
        reduced potential difference of the nonalchemical and alchemical lambda = 1 state (corrected for valence energy).
    """
    import copy
    from perses.dispersed.utils import configure_platform
    from perses.utils import data
    platform = configure_platform(platform.getName(), fallback_platform_name='Reference', precision='double')

    # Create copies of old/new systems and set the dispersion correction
    top_proposal = copy.deepcopy(topology_proposal)
    forces = { top_proposal._old_system.getForce(index).__class__.__name__ : top_proposal._old_system.getForce(index) for index in range(top_proposal._old_system.getNumForces()) }
    forces['NonbondedForce'].setUseDispersionCorrection(False)
    forces = { top_proposal._new_system.getForce(index).__class__.__name__ : top_proposal._new_system.getForce(index) for index in range(top_proposal._new_system.getNumForces()) }
    forces['NonbondedForce'].setUseDispersionCorrection(False)

    # Create copy of hybrid system, define old and new positions, and turn off dispersion correction
    hybrid_system = copy.deepcopy(htf.hybrid_system)
    hybrid_system_n_forces = hybrid_system.getNumForces()
    for force_index in range(hybrid_system_n_forces):
        forcename = hybrid_system.getForce(force_index).__class__.__name__
        if forcename == 'NonbondedForce':
            hybrid_system.getForce(force_index).setUseDispersionCorrection(False)

    old_positions, new_positions = htf._old_positions, htf._new_positions

    # Generate endpoint thermostates
    nonalch_zero, nonalch_one, alch_zero, alch_one = generate_endpoint_thermodynamic_states(hybrid_system, top_proposal, repartitioned_endstate)

    # Compute reduced energies for the nonalchemical systems...
    attrib_list = [('real-old', nonalch_zero, old_positions, top_proposal._old_system.getDefaultPeriodicBoxVectors()),
                    ('hybrid-old', alch_zero, htf._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    ('hybrid-new', alch_one, htf._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    ('real-new', nonalch_one, new_positions, top_proposal._new_system.getDefaultPeriodicBoxVectors())]

    rp_list = []
    for (state_name, state, pos, box_vectors) in attrib_list:
        if not state:
            rp_list.append(None)
        else:
            integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
            context = state.create_context(integrator, platform)
            samplerstate = states.SamplerState(positions=pos, box_vectors=box_vectors)
            samplerstate.apply_to_context(context)
            rp = state.reduced_potential(context)
            rp_list.append(rp)
            energy_comps = compute_potential_components(context)
            for name, force in energy_comps.items():
               print(f"\t\t\t{name}: {force}")
            _logger.debug(f'added forces:{sum(energy_comps.values())}')
            _logger.debug(f'rp: {rp}')
            if trajectory_directory is not None:
                _logger.info(f'Saving {state_name} state xml to {trajectory_directory}/{state_name}-state.gz')
                state = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True,
                                         getParameters=True)
                data.serialize(state, f'{trajectory_directory}-{state_name}-state.gz')
            del context, integrator

    nonalch_zero_rp, alch_zero_rp, alch_one_rp, nonalch_one_rp = rp_list[0], rp_list[1], rp_list[2], rp_list[3]

    if repartitioned_endstate == 0:
        zero_error = nonalch_zero_rp - alch_zero_rp + added_energy
        one_error = None
        ratio = abs((zero_error) / (nonalch_zero_rp + alch_zero_rp + added_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ZERO state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-zero: {nonalch_zero_rp} \n alc-zero: {alch_zero_rp} \nadded-valence: {added_energy}"
    elif repartitioned_endstate == 1:
        zero_error = None
        one_error = nonalch_one_rp - alch_one_rp + subtracted_energy
        ratio = abs((one_error) / (nonalch_one_rp + alch_one_rp + subtracted_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ONE state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-one: {nonalch_one_rp} \n alc-one: {alch_one_rp} \nsubtracted-valence: {subtracted_energy}"
    else:
        zero_error = nonalch_zero_rp - alch_zero_rp + added_energy
        one_error = nonalch_one_rp - alch_one_rp + subtracted_energy
        ratio = abs((zero_error) / (nonalch_zero_rp + alch_zero_rp + added_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ZERO state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-zero: {nonalch_zero_rp} \n alc-zero: {alch_zero_rp} \nadded-valence: {added_energy}"
        ratio = abs((one_error) / (nonalch_one_rp + alch_one_rp + subtracted_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ONE state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-one: {nonalch_one_rp} \n alc-one: {alch_one_rp} \nsubtracted-valence: {subtracted_energy}"

    return zero_error, one_error


def validate_endstate_energies_point(input_htf, endstate=0, minimize=False):
    """
    ** Used for validating endstate energies for RESTCapableHybridTopologyFactory **

    Check that the hybrid system's energy (without unique old/new valence energy) matches the original system's energy for the positions in the htf.

    E.g. at endstate=0, the hybrid system's energy (with unique new valence terms zeroed) should match the old system's energy.

    .. note ::
    Note that this function assumes that the RESTCapableHybridTopologyFactory hybrid system contains the following
    forces

    ['MonteCarloBarostat', 'CustomBondForce', 'CustomAngleForce', 'CustomTorsionForce',
     'CustomNonbondedForce_electrostatics', 'CustomNonbondedForce_sterics', 'CustomBondForce_exceptions',
     'NonbondedForce_reciprocal', 'NonbondedForce_sterics'].

    It may fail if there have been changes to the forces or force names, so proceed with caution"

    Parameters
    ----------
    input_htf : RESTCapableHybridTopologyFactory
        the RESTCapableHybridTopologyFactory to test
    endstate : int, default=0
        the endstate to test (0 or 1)
    minimize : bool, default=False
        whether to minimize the positions before testing that the energies match
    """
    from perses.dispersed import feptasks

    # Check that endstate is 0 or 1
    assert endstate in [0, 1], "Endstate must be 0 or 1"

    # Make deep copy to ensure original object remains unaltered
    htf = copy.deepcopy(input_htf)

    # Get original system
    system = htf._topology_proposal.old_system if endstate == 0 else htf._topology_proposal.new_system

    # Get hybrid system, positions, and forces
    hybrid_system = htf.hybrid_system
    hybrid_positions = htf.hybrid_positions

    force_dict = {force.getName(): force for force in hybrid_system.getForces()}
    bond_force = force_dict['CustomBondForce']
    angle_force = force_dict['CustomAngleForce']
    torsion_force = force_dict['CustomTorsionForce']
    electrostatics_force = force_dict['CustomNonbondedForce_electrostatics']
    scaled_sterics_force = force_dict['CustomNonbondedForce_sterics']
    exceptions_force = force_dict['CustomBondForce_exceptions']
    reciprocal_force = force_dict['NonbondedForce_reciprocal']
    nonscaled_sterics_force = force_dict['NonbondedForce_sterics']

    forces = [bond_force, angle_force, torsion_force, electrostatics_force, scaled_sterics_force]
    force_names = ['bonds', 'angles', 'torsions', 'electrostatics', 'sterics']

    # For this test, we need to turn the LRC on for the CustomNonbondedForce scaled steric interactions,
    # since there is no way to turn the LRC on for the non-scaled interactions only in the real systems
    scaled_sterics_force.setUseLongRangeCorrection(True)

    # Set global parameters for valence + electrostatics/scaled_sterics forces
    lambda_old = 1 if endstate == 0 else 0
    lambda_new = 0 if endstate == 0 else 1
    for force, name in zip(forces, force_names):
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == f'lambda_alchemical_{name}_old':
                force.setGlobalParameterDefaultValue(i, lambda_old)
            if force.getGlobalParameterName(i) == f'lambda_alchemical_{name}_new':
                force.setGlobalParameterDefaultValue(i, lambda_new)

    # Set global parameters for exceptions force
    old_parameter_names = ['lambda_alchemical_electrostatics_exceptions_old',
                           'lambda_alchemical_sterics_exceptions_old']
    new_parameter_names = ['lambda_alchemical_electrostatics_exceptions_new',
                           'lambda_alchemical_sterics_exceptions_new']
    for i in range(exceptions_force.getNumGlobalParameters()):
        if exceptions_force.getGlobalParameterName(i) in old_parameter_names:
            exceptions_force.setGlobalParameterDefaultValue(i, lambda_old)
        elif exceptions_force.getGlobalParameterName(i) in new_parameter_names:
            exceptions_force.setGlobalParameterDefaultValue(i, lambda_new)

    # Set global parameters for reciprocal force
    for i in range(reciprocal_force.getNumGlobalParameters()):
        if reciprocal_force.getGlobalParameterName(i) == 'lambda_alchemical_electrostatics_reciprocal':
            reciprocal_force.setGlobalParameterDefaultValue(i, lambda_new)

    # Zero the unique old/new valence terms at lambda = 1/0
    hybrid_to_bond_indices = htf._hybrid_to_new_bond_indices if endstate == 0 else htf._hybrid_to_old_bond_indices
    hybrid_to_angle_indices = htf._hybrid_to_new_angle_indices if endstate == 0 else htf._hybrid_to_old_angle_indices
    hybrid_to_torsion_indices = htf._hybrid_to_new_torsion_indices if endstate == 0 else htf._hybrid_to_old_torsion_indices
    for hybrid_idx, idx in hybrid_to_bond_indices.items():
        p1, p2, hybrid_params = bond_force.getBondParameters(hybrid_idx)
        hybrid_params = list(hybrid_params)
        hybrid_params[-2] *= 0  # zero K_old
        hybrid_params[-1] *= 0  # zero K_new
        bond_force.setBondParameters(hybrid_idx, p1, p2, hybrid_params)
    for hybrid_idx, idx in hybrid_to_angle_indices.items():
        p1, p2, p3, hybrid_params = angle_force.getAngleParameters(hybrid_idx)
        hybrid_params = list(hybrid_params)
        hybrid_params[-2] *= 0
        hybrid_params[-1] *= 0
        angle_force.setAngleParameters(hybrid_idx, p1, p2, p3, hybrid_params)
    for hybrid_idx, idx in hybrid_to_torsion_indices.items():
        p1, p2, p3, p4, hybrid_params = torsion_force.getTorsionParameters(hybrid_idx)
        hybrid_params = list(hybrid_params)
        hybrid_params[-2] *= 0
        hybrid_params[-1] *= 0
        torsion_force.setTorsionParameters(hybrid_idx, p1, p2, p3, p4, hybrid_params)

    # Get energy components of hybrid system
    thermostate_hybrid = states.ThermodynamicState(system=hybrid_system, temperature=temperature)
    integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
    if minimize:
        sampler_state = states.SamplerState(hybrid_positions)
        feptasks.minimize(thermostate_hybrid, sampler_state)
        hybrid_positions = sampler_state.positions
    context_hybrid.setPositions(hybrid_positions)
    components_hybrid = compute_potential_components(context_hybrid, beta=beta)

    # Get energy components of original system
    thermostate_other = states.ThermodynamicState(system=system, temperature=temperature)
    integrator_other = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context_other = thermostate_other.create_context(integrator_other)
    positions = htf.old_positions(hybrid_positions) if endstate == 0 else htf.new_positions(hybrid_positions)
    context_other.setPositions(positions)
    components_other = compute_potential_components(context_other, beta=beta)

    # Check that each of the valence force energies are concordant
    # TODO: Instead of checking with np.isclose(), check whether the ratio of differences is less than a specified energy threshold (like in validate_endstate_energies())
    # Build map between other and rest force keys - keys are for other, values are for rest
    bonded_keys_other_to_hybrid = {'HarmonicBondForce': 'CustomBondForce', 'HarmonicAngleForce': 'CustomAngleForce',
                                   'PeriodicTorsionForce': 'CustomTorsionForce'}
    for other_key, hybrid_key in bonded_keys_other_to_hybrid.items():
        other_value = components_other[other_key]
        hybrid_value = components_hybrid[hybrid_key]
        print(f"{other_key} -- og: {other_value}, hybrid: {hybrid_value}")
        assert np.isclose(other_value, hybrid_value)

    # Check that the nonbonded (rest of the components) force energies are concordant
    nonbonded_hybrid_values = [components_hybrid[key] for key in components_hybrid.keys()
                               if key not in bonded_keys_other_to_hybrid.values()]
    print(
        f"Nonbondeds -- og: {components_other['NonbondedForce']}, hybrid: {np.sum(nonbonded_hybrid_values)}"
    )
    assert np.isclose([components_other['NonbondedForce']], np.sum(nonbonded_hybrid_values))

    print(f"Success! Energies are equal at lambda {endstate}!")


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
    import dask.distributed as distributed
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
    import dask.distributed as distributed
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
    import dask.distributed as distributed
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
                    self.save_configuration(idx, sampler_state)
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


    def save_configuration(self, iteration, sampler_state):
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

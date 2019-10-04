import simtk.openmm as openmm
import openmmtools.cache as cache
from typing import List, Tuple, Union, NamedTuple
import os
import copy

import openmmtools.mcmc as mcmc
import openmmtools.integrators as integrators
import openmmtools.states as states
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

# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("feptasks")
_logger.setLevel(logging.DEBUG)

#cache.global_context_cache.platform = openmm.Platform.getPlatformByName('Reference') #this is just a local version
EquilibriumResult = namedtuple('EquilibriumResult', ['sampler_state', 'reduced_potentials', 'files', 'timers', 'nonalchemical_perturbations'])
NonequilibriumResult = namedtuple('NonequilibriumResult', ['succeed', 'sampler_state', 'protocol_work', 'cumulative_work', 'shadow_work', 'kinetic_energies','timers'])
iter_tuple = namedtuple('iter_tuple', ['prep_time', 'step_time', 'save_config_time'])

class NonequilibriumSwitchingMove():
    """
    This class represents an MCMove that runs a nonequilibrium switching protocol using the AlchemicalNonequilibriumLangevinIntegrator.
    It is simply a wrapper around the aforementioned integrator, which must be provided in the constructor.
    WARNING: take care in writing trajectory file as saving positions to memory is costly.  Either do not write the configuration or save sparse positions.
    Parameters
    ----------
    nsteps : int
        number of annealing steps in the protocol
    direction : str
        whether the protocol runs 'forward' or 'reverse'
    splitting : str, default 'V R O R V'
        Splitting string for integrator
    temperature : unit.Quantity(float, units = unit.kelvin)
        temperature at which to run the simulation
    timestep : float
        size of timestep (units of time)
    work_save_interval : int, default None
        The frequency with which to record the cumulative total work. If None, only save the total work at the end
    top: md.Topology, default None
        The topology to use to write the positions along the protocol. If None, don't write anything.
    subset_atoms : np.array, default None
        The indices of the subset of atoms to write. If None, write all atoms (if writing is enabled)
    save_configuration : bool, default False
        whether to save the ncmc trajectory
    measure_shadow_work : bool, default False
        whether to measure the shadow work from the integrator
    Attributes
    ----------
    context_cache : openmmtools.cache.global_context_cache
        Global context cache to deal with context instantiation
    _timers : dict
        dictionary of timers corresponding to various processes in the protocol
    _direction : str
        whether the protocol runs 'forward' or 'reverse'
    _integrator : openmmtools.integrators.LangevinIntegrator
        integrator for propagation kernel
    _n_lambda_windows : int
        number of lambda windows (including 0,1); equal to nsteps + 1
    _nsteps : int
        number of annealing steps in protocol
    _beta : unit.kcal_per_mol**-1
        inverse temperature
    _work_save_interval : int
        how often to save protocol work and trajectory
    _save_configuration : bool
        whether to save trajectory
    _measure_shadow_work : bool
        whether to measure the shadow work from the LangevinIntegrator
    _cumulative_work : float
        total -log(weight) of annealing
    _shadow_work : float
        total shadow work accumulated by kernel; if measure_shadow_work == False: _shadow_work = 0.0
    _protocol_work : numpy.array
        protocol accumulated works at save snapshots
    _heat : float
        total heat gathered by kernel
    _kinetic_energy : list[float]
        reduced kinetic energies after propagation step which save every _work_save_interval (except last interval since there is no propagation step)
    _topology : md.Topology
        topology or subset defined by the mask to save
    _subset_atoms : numpy.array
        atom indices to save
    _trajectory : md.Trajectory
        trajectory to save
    _trajectory_positions : list of simtk.unit.Quantity()
        particle positions
    _trajectory_box_lengths : list of triplet float
        length of box in nm
    _trajectory_box_angles : list of triplet float
        angles of box in rads
    _
    """

    def __init__(self, nsteps: int, direction: str, splitting: str= 'V R O R V', temperature: unit.Quantity=300*unit.kelvin, timestep: unit.Quantity=1.0*unit.femtosecond,
        work_save_interval: int=None, top: md.Topology=None, subset_atoms: np.array=None, save_configuration: bool=False, measure_shadow_work: bool=False, **kwargs):

        start = time.time()
        self._timers = {} #instantiate timer

        self.context_cache = cache.global_context_cache

        if measure_shadow_work:
            measure_heat = True
        else:
            measure_heat = False

        assert direction == 'forward' or direction == 'reverse', f"The direction of the annealing protocol ({direction}) is invalid; must be specified as 'forward' or 'reverse'"

        self._direction = direction

        self._integrator = integrators.LangevinIntegrator(temperature = temperature, timestep = timestep, splitting = splitting, measure_shadow_work = measure_shadow_work, measure_heat = measure_heat)

        self._nsteps = nsteps
        self._beta = 1.0 / (kB*temperature)
        self._temperature = temperature

        if not work_save_interval:
            self._work_save_interval = self._nsteps
        else:
            self._work_save_interval = work_save_interval

        self._save_configuration = save_configuration
        self._measure_shadow_work = measure_shadow_work

        #check that the work write interval is a factor of the number of steps, so we don't accidentally record the
        #work before the end of the protocol as the end
        if self._nsteps % self._work_save_interval != 0:
            raise ValueError("The work writing interval must be a factor of the total number of steps")

        #use the number of step moves plus one, since the first is always zero
        self._cumulative_work = 0.0
        self._shadow_work = 0.0
        self._protocol_work = [0.0]
        self._heat = 0.0
        self._kinetic_energy = []

        self._topology = top
        self._subset_atoms = subset_atoms
        self._trajectory = None

        #if we have a trajectory, set up some ancillary variables:
        if self._topology is not None:
            n_atoms = self._topology.n_atoms
            self._trajectory_positions = []
            self._trajectory_box_lengths = []
            self._trajectory_box_angles = []
        else:
            self._save_configuration = False

        self._timers['instantiate'] = time.time() - start

        #set a bool variable for pass or failure
        self.succeed = True

    def apply(self, thermodynamic_state, sampler_state, lambda_protocol = 'default'):
        """Propagate the state through the integrator.
        This updates the SamplerState after the integration. It will apply the full NCMC protocol.
        Parameters
        ----------
        thermodynamic_state : openmmtools.states.CompoundThermodynamicState
           The compound thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        lambda_protocol : str (or dict), default 'default'
           Which protocol to use for the lambda update
        """
        """Propagate the state through the integrator.
        This updates the SamplerState after the integration. It also logs
        benchmarking information through the utils.Timer class.
        See Also
        --------
        openmmtools.utils.Timer
        """
        start = time.time()
        thermodynamic_state = copy.deepcopy(thermodynamic_state)
        sampler_state = copy.deepcopy(sampler_state)

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        #define the lambda schedule (linear)
        if self._direction == 'forward':
            start_lambda = 0.0
            end_lambda = 1.0
        elif self._direction == 'reverse':
            start_lambda = 1.0
            end_lambda = 0.0
        else:
            raise Error(f"direction must be 'forward' or 'reverse'")

        # define the lambda_schedule
        lambda_schedule = np.linspace(start_lambda, end_lambda, self._nsteps + 1)[1:]

        # # set the thermodynamic state
        # thermodynamic_state.set_alchemical_parameters(start_lambda)

        context, integrator = context_cache.get_context(thermodynamic_state, self._integrator)
        sampler_state.apply_to_context(context, ignore_velocities=True)
        context.setVelocitiesToTemperature(thermodynamic_state.temperature) #randomize velocities @ temp
        init_state = context.getState(getEnergy=True)
        initial_energy = self._beta * (init_state.getPotentialEnergy() + init_state.getKineticEnergy())
        sampler_state.update_from_context(context, ignore_velocities=True)

        if self._direction == 'forward':
            assert context.getParameter('lambda_sterics_core') == 0.0, f"Direction was specified as {self._direction} but initial master lambda is defined as {context.getParameter('lambda_sterics_core')}" #lambda_sterics_core maps master_lambda (see lambda_protocol.py)
        elif self._direction == 'reverse':
            assert context.getParameter('lambda_sterics_core') == 1.0, f"Direction was specified as {self._direction} but initial master lambda is defined as {context.getParameter('lambda_sterics_core')}" #lambda_sterics_core maps master_lambda (see lambda_protocol.py)


        # reset the integrator after it is bound to a context and before annealing
        integrator.reset()

        #save the initial configuration
        if self._save_configuration:
            if self._subset_atoms is None:
                self._trajectory_positions.append(sampler_state.positions[:, :].value_in_unit_system(unit.md_unit_system))
            else:
                self._trajectory_positions.append(sampler_state.positions[self._subset_atoms, :].value_in_unit_system(unit.md_unit_system))

            #get the box angles and lengths
            a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
            self._trajectory_box_lengths.append([a, b, c])
            self._trajectory_box_angles.append([alpha, beta, gamma])

        self._timers['prepare_neq_switching'] = time.time() - start

        timer_list = []
        lambda_protocol_class = LambdaProtocol(functions = lambda_protocol)

        #loop through the number of times we have to apply in order to collect the requested work and trajectory statistics.
        for iteration, master_lambda in zip(range(self._nsteps), lambda_schedule):
            _logger.debug(f"\tconducting iteration {iteration} at master lambda {master_lambda}")
            try:
                prep_start = time.time()
                old_rp = self._beta * context.getState(getEnergy=True).getPotentialEnergy()
                thermodynamic_state.set_alchemical_parameters(master_lambda, lambda_protocol = lambda_protocol_class)
                thermodynamic_state.apply_to_context(context)
                #context.setVelocitiesToTemperature(thermodynamic_state.temperature) #resample velocities
                new_rp = self._beta * context.getState(getEnergy=True).getPotentialEnergy()
                work = new_rp - old_rp
                self._cumulative_work += work
                _logger.debug(f"\twork: {work}")
                prep_time = time.time() - prep_start


                if not master_lambda == end_lambda: #we don't have to propagate dynamics if it is the last iteration
                    step_start = time.time()
                    integrator.step(1)
                    step_time = time.time() - step_start

                if (iteration+1) % self._work_save_interval == 0: #we save the protocol work if the remainder is zero
                    _logger.debug(f"\tconducting work save")
                    #_logger.debug(f"\tsampler state: {vars(sampler_state)}")
                    self._protocol_work.append(self._cumulative_work)
                    timer_list.append((prep_time, step_time))
                    self._kinetic_energy.append(self._beta * context.getState(getEnergy=True).getKineticEnergy())
                    sampler_state.update_from_context(context, ignore_velocities=True) #save bandwidth by not updating the velocities

                    #if we have a trajectory, we'll also write to it
                    save_start = time.time()
                    if self._save_configuration:

                        #record positions for writing to trajectory
                        #we need to check whether the user has requested to subset atoms (excluding water, for instance)

                        if self._subset_atoms is None:
                            self._trajectory_positions.append(sampler_state.positions[:, :].value_in_unit_system(unit.md_unit_system))
                        else:
                            self._trajectory_positions.append(sampler_state.positions[self._subset_atoms, :].value_in_unit_system(unit.md_unit_system))

                        #get the box angles and lengths
                        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
                        self._trajectory_box_lengths.append([a, b, c])
                        self._trajectory_box_angles.append([alpha, beta, gamma])

                    save_time = time.time() - save_start

                    timer_list.append(iter_tuple(prep_time = prep_time, step_time = step_time, save_config_time = save_time))

                else:
                    _logger.debug(f"\tomitting work save (work save interval is {self._work_save_interval} but iteration is {iteration+1})")
                    _logger.debug(f"\t result of calc is {(iteration+1) % self._work_save_interval}")



            except Exception as e:
                _logger.debug(f"the simulation failed")
                if self._save_configuration:
                    self._trajectory = md.Trajectory(np.array(self._trajectory_positions), self._topology, unitcell_lengths=np.array(self._trajectory_box_lengths), unitcell_angles=np.array(self._trajectory_box_angles))
                self._timers['neq_switching'] = timer_list
                self._shadow_work = 0.0
                self.succeed = False
                return



        self._timers['neq_switching'] = timer_list

        # assertion for full lambda protocol
        if self._direction == 'forward':
            assert master_lambda == 1.0, f"Direction is forward but the end of the protocol returned a master lambda of {master_lambda}"
            assert context.getParameter('lambda_sterics_core') == 1.0, f"Direction was specified as {self._direction} but final master lambda is defined as {context.getParameter('lambda_sterics_core')}"
        elif self._direction == 'reverse':
            assert master_lambda == 0.0, f"Direction is reverse but the end of the protocol returned a master lambda of {master_lambda}"
            assert context.getParameter('lambda_sterics_core') == 0.0, f"Direction was specified as {self._direction} but initial master lambda is defined as {context.getParameter('lambda_sterics_core')}"


        if self._save_configuration:
            self._trajectory = md.Trajectory(np.array(self._trajectory_positions), self._topology, unitcell_lengths=np.array(self._trajectory_box_lengths), unitcell_angles=np.array(self._trajectory_box_angles))

        if self._measure_shadow_work:
            total_heat = integrator.get_heat(dimensionless=True)
            final_state = context.getState(getEnergy=True)
            final_energy = self._beta * (final_state.getPotentialEnergy() + final_state.getKineticEnergy())
            total_energy_change = final_energy - initial_energy
            self._shadow_work = total_energy_change - (total_heat + self._cumulative_work)
        else:
            self._shadow_work = 0.0


def run_protocol(thermodynamic_state: states.CompoundThermodynamicState, equilibrium_result: EquilibriumResult,
                 direction: str, topology: md.Topology, nsteps_neq: int = 1000, forward_functions: str = None, work_save_interval: int = 1, splitting: str='V R O R V',
                 atom_indices_to_save: List[int] = None, trajectory_filename: str = None, write_configuration: bool = False, timestep: unit.Quantity=1.0*unit.femtoseconds, measure_shadow_work: bool=False, timer: bool=True) -> dict:
    """
    Perform a nonequilibrium switching protocol and return the nonequilibrium protocol work. Note that it is expected
    that this will perform an entire protocol, that is, switching lambda completely from 0 to 1, in increments specified
    by the ne_mc_move. The trajectory that results, along with the work values, will contain n_iterations elements.
    Parameters
    ----------
    thermodynamic_state : openmmtools.states.CompoundThermodynamicState
        The thermodynamic state at which to run the protocol
    equilibrium_result : EquilibriumResult
        the equilibrium result from which the sampler state will be extracted
    direction : str
        whether the protocol runs 'forward' or 'reverse'
    topology : mdtraj.Topology
        An MDtraj topology for the system to generate trajectories
    nsteps_neq : int
        The number of nonequilibrium steps in the protocol
    forward_functions : str, default None
        which option to call as the forward function for the lambda protocol
    work_save_interval : int
        How often to write the work and, if requested, configurations
    splitting : str, default 'V R O R V'
        The splitting string to use for the Langevin integration
    atom_indices_to_save : list of int, default None
        list of indices to save (when excluding waters, for instance). If None, all indices are saved.
    trajectory_filename : str, default None
        Full filepath of output trajectory, if desired. If None, no trajectory file is written.
    write_configuration : bool, default False
        Whether to also write configurations of the trajectory at the requested interval.
    timestep : unit.Quantity, default 1 fs
        The timestep to use in the integrator
    measure_shadow_work : bool, default False
        Whether to compute the shadow work; there is additional overhead in the integrator cost
    Returns
    -------
    cumulative_work : np.array
        the cumulative work as a function of the integration step; len(cumulative_work) = work_save_interval
    protocol_work : np.array
        the work per iteration of annealing; len(protocol_work) = work_save_interval
    shadow_work : float
        the shadow work accumulated by the discrete integrator
    """
    timers = {}

    # creating copies in case computation is parallelized
    thermodynamic_state = copy.deepcopy(thermodynamic_state)
    sampler_state = copy.deepcopy(equilibrium_result.sampler_state)

    #forward Functions
    if not forward_functions:
        forward_functions = 'default'


    #get the temperature needed for the simulation
    temperature = thermodynamic_state.temperature

    #get the atom indices we need to subset the topology and positions
    if atom_indices_to_save is None:
        atom_indices = list(range(topology.n_atoms))
        subset_topology = topology
    else:
        subset_topology = topology.subset(atom_indices_to_save)
        atom_indices = atom_indices_to_save

    _logger.debug(f"Instantiating NonequilibriumSwitchingMove class")
    ne_mc_move = NonequilibriumSwitchingMove(nsteps = nsteps_neq,
                                             direction = direction,
                                             splitting = splitting,
                                             temperature = temperature,
                                             timestep = timestep,
                                             work_save_interval = work_save_interval,
                                             top = subset_topology,
                                             subset_atoms = atom_indices,
                                             save_configuration = write_configuration,
                                             measure_shadow_work=measure_shadow_work)

    #apply the nonequilibrium move; sampler state gets updated
    _logger.debug(f"applying thermodynamic state and sampler state to the ne_mc_move")
    ne_mc_move.apply(thermodynamic_state, sampler_state, lambda_protocol = forward_functions)

    #get the cumulative work
    cumulative_work = ne_mc_move._cumulative_work

    #get the protocol work
    protocol_work = ne_mc_move._protocol_work

    #get the kinetic energies
    kinetic_energies = ne_mc_move._kinetic_energy

    #if we're measuring shadow work, get that. Otherwise just fill in zeros:
    if measure_shadow_work:
        shadow_work = ne_mc_move._shadow_work
    else:
        shadow_work = np.zeros_like(protocol_work)


    #if desired, write nonequilibrium trajectories:
    if trajectory_filename is not None:
        #if writing configurations was requested, get the trajectory
        if write_configuration:
                trajectory = ne_mc_move._trajectory
                write_nonequilibrium_trajectory(trajectory, trajectory_filename)
                _logger.debug(f"successfully wrote nonequilibrium trajectory to {trajectory_filename}")

    if not timer:
        timers = {}
    else:
        timers = ne_mc_move._timers

    if ne_mc_move.succeed:
        neq_result = NonequilibriumResult(succeed = True,
                                          sampler_state = sampler_state,
                                          protocol_work = protocol_work,
                                          cumulative_work = cumulative_work,
                                          shadow_work = shadow_work,
                                          kinetic_energies = kinetic_energies,
                                          timers = timers)
    else:
        neq_result = NonequilibriumResult(succeed = False,
                                          sampler_state = sampler_state,
                                          protocol_work = protocol_work,
                                          cumulative_work = cumulative_work,
                                          shadow_work = shadow_work,
                                          kinetic_energies = kinetic_energies,
                                          timers = timers)

    return neq_result

def run_equilibrium(thermodynamic_state: states.CompoundThermodynamicState, eq_result: EquilibriumResult,
                    nsteps_equil: int, topology: md.Topology, n_iterations : int,
                    atom_indices_to_save: List[int] = None, trajectory_filename: str = None,
                    splitting: str="V R O R V", timestep: unit.Quantity=1.0*unit.femtoseconds,
                    max_size: float=1024*1e3, timer: bool=False, _minimize: bool = False, file_iterator: int = 0,
                    nonalchemical_perturbation_args: dict = None) -> EquilibriumResult:
    """
    Run n_iterations*nsteps_equil integration steps (likely at the lambda 0 state).  n_iterations mcmc moves are conducted in the initial equilibration, returning n_iterations
    reduced potentials.  This is the guess as to the burn-in time for a production.  After which, a single mcmc move of nsteps_equil
    will be conducted at a time, including a time-series (pymbar) analysis to determine whether the data are decorrelated.
    The loop will conclude when a single configuration yields an iid sample.  This will be saved.
    Parameters
    ----------
    thermodynamic_state : openmmtools.states.CompoundThermodynamicState
        The thermodynamic state (including context parameters) that should be used
    eq_result : EquilibriumResult
        The EquilibriumResult from which the sampler_state will be extracted
    nsteps_equil : int
        The number of equilibrium steps that a move should make when apply is called
    topology : mdtraj.Topology
        an MDTraj topology object used to construct the trajectory
    n_iterations : int
        The minimum number of times to apply the move. Note that this is not the number of steps of dynamics; it is
        n_iterations*n_steps (which is set in the MCMove).
    splitting: str, default "V R O R V"
        The splitting string for the dynamics
    atom_indices_to_save : list of int, default None
        list of indices to save (when excluding waters, for instance). If None, all indices are saved.
    trajectory_filename : str, optional, default None
        Full filepath of trajectory files. If none, trajectory files are not written.
    max_size: float
        maximum size of the trajectory numpy array allowable until it is written to disk
    timer: bool, default False
        whether to time all parts of the equilibrium run
    _minimize : bool, default False
        whether to minimize the sampler_state before conducting equilibration
    file_iterator : int, default 0
        which index to begin writing files
    """
    timers = {}
    file_numsnapshots = []
    _logger.debug(f"running equilibrium")

    # creating copies in case computation is parallelized
    if timer: start = time.time()
    thermodynamic_state = copy.deepcopy(thermodynamic_state)
    sampler_state = copy.deepcopy(eq_result.sampler_state)
    if timer: timers['copy_state'] = time.time() - start

    if _minimize:
        _logger.debug(f"conducting minimization; (minimize = {_minimize})")
        if timer: start = time.time()
        minimize(thermodynamic_state, sampler_state)
        if timer: timers['minimize'] = time.time() - start

    #get the atom indices we need to subset the topology and positions
    if timer: start = time.time()
    if atom_indices_to_save is None:
        atom_indices = list(range(topology.n_atoms))
        subset_topology = topology
    else:
        subset_topology = topology.subset(atom_indices_to_save)
        atom_indices = atom_indices_to_save
    if timer: timers['define_topology'] = time.time() - start

    n_atoms = subset_topology.n_atoms

    #construct the MCMove:
    mc_move = mcmc.LangevinSplittingDynamicsMove(n_steps=nsteps_equil, splitting=splitting, timestep = timestep)
    mc_move.n_restart_attempts = 10

    #create a numpy array for the trajectory
    trajectory_positions, trajectory_box_lengths, trajectory_box_angles = list(), list(), list()
    reduced_potentials = list()

    #loop through iterations and apply MCMove, then collect positions into numpy array
    _logger.debug(f"conducting {n_iterations} of production")
    if timer: eq_times = []

    init_file_iterator = file_iterator
    for iteration in tqdm.trange(n_iterations):
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
        if np.array(trajectory_positions).nbytes > max_size:
            trajectory = md.Trajectory(np.array(trajectory_positions), subset_topology, unitcell_lengths=np.array(trajectory_box_lengths), unitcell_angles=np.array(trajectory_box_angles))
            if trajectory_filename is not None:
                new_filename = trajectory_filename[:-2] + f'{file_iterator:04}' + '.h5'
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
    if trajectory_filename is not None:
        #construct trajectory object:
        if trajectory_positions != list():
            #if it is an empty list, then the last iteration satistifed max_size and wrote the trajectory to disk;
            #in this case, we can just skip this
            trajectory = md.Trajectory(np.array(trajectory_positions), subset_topology, unitcell_lengths=np.array(trajectory_box_lengths), unitcell_angles=np.array(trajectory_box_angles))
            if file_iterator == init_file_iterator: #this means that no files have been written yet
                new_filename = trajectory_filename[:-2] + f'{file_iterator:04}' + '.h5'
                file_numsnapshots.append((new_filename, len(trajectory_positions)))
            else:
                new_filename = trajectory_filename[:-2] + f'{file_iterator+1:04}' + '.h5'
                file_numsnapshots.append((new_filename, len(trajectory_positions)))
            write_equilibrium_trajectory(trajectory, new_filename)

    if timer: timers['write_traj'] = time.time() - start

    if timer: start = time.time()
    if nonalchemical_perturbation_args != None:
        #then we will conduct a perturbation on the given sampler state with the appropriate arguments
        valence_energy, nonalchemical_reduced_potential, hybrid_reduced_potential = compute_nonalchemical_perturbation(nonalchemical_perturbation_args['hybrid_thermodynamic_states'][0],
                                                                                                                       nonalchemical_perturbation_args['_endpoint_growth_thermostates'][0],
                                                                                                                       sampler_state,
                                                                                                                       nonalchemical_perturbation_args['factory'],
                                                                                                                       nonalchemical_perturbation_args['nonalchemical_thermostates'][0],
                                                                                                                       nonalchemical_perturbation_args['lambdas'][0])
        alt_valence_energy, alt_nonalchemical_reduced_potential, alt_hybrid_reduced_potential = compute_nonalchemical_perturbation(nonalchemical_perturbation_args['hybrid_thermodynamic_states'][1],
                                                                                                                       nonalchemical_perturbation_args['_endpoint_growth_thermostates'][1],
                                                                                                                       sampler_state,
                                                                                                                       nonalchemical_perturbation_args['factory'],
                                                                                                                       nonalchemical_perturbation_args['nonalchemical_thermostates'][1],
                                                                                                                       nonalchemical_perturbation_args['lambdas'][1])
        nonalch_perturbations = {'valence_energies': (valence_energy, alt_valence_energy),
                                 'nonalchemical_reduced_potentials': (nonalchemical_reduced_potential, alt_nonalchemical_reduced_potential),
                                 'hybrid_reduced_potentials': (hybrid_reduced_potential, alt_hybrid_reduced_potential)}
    else:
        nonalch_perturbations = {}

    if timer: timers['perturbation'] = time.time() - start

    if not timer:
        timers = {}

    equilibrium_result = EquilibriumResult(sampler_state = sampler_state,
                                           reduced_potentials = reduced_potentials,
                                           files = file_numsnapshots,
                                           timers = timers,
                                           nonalchemical_perturbations = nonalch_perturbations)

    return equilibrium_result



def minimize(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState,
             max_iterations: int=100) -> states.SamplerState:
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
        The maximum number of minimization steps. Default is 20.
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

def write_nonequilibrium_trajectory(nonequilibrium_trajectory: md.Trajectory, trajectory_filename: str) -> float:
    """
    Write the results of a nonequilibrium switching trajectory to a file. The trajectory is written to an
    mdtraj hdf5 file, whereas the cumulative work is written to a numpy file.
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

def compute_nonalchemical_perturbation(alchemical_thermodynamic_state: states.ThermodynamicState,  growth_thermodynamic_state: states.ThermodynamicState, hybrid_sampler_state: states.SamplerState, hybrid_factory: HybridTopologyFactory, nonalchemical_thermodynamic_state: states.ThermodynamicState, lambda_state: int) -> tuple:
    """
    Compute the perturbation of transforming the given hybrid equilibrium result into the system for the given nonalchemical_thermodynamic_state
    Parameters
    ----------
    alchemical_thermodynamic_state: states.ThermodynamicState
        alchemical thermostate
    growth_thermodynamic_state : states.ThermodynamicState
    hybrid_sampler_state: states.SamplerState
        sampler state for the alchemical thermodynamic_state
    hybrid_factory : HybridTopologyFactory
        Hybrid factory necessary for getting the positions of the nonalchemical system
    nonalchemical_thermodynamic_state : states.ThermodynamicState
        ThermodynamicState of the nonalchemical system
    lambda_state : int
        Whether this is lambda 0 or 1
    Returns
    -------
    work : float
        perturbation in kT from the hybrid system to the nonalchemical one
    """
    #get the objects we need to begin
    hybrid_reduced_potential = compute_reduced_potential(alchemical_thermodynamic_state, hybrid_sampler_state)
    hybrid_positions = hybrid_sampler_state.positions

    #get the positions for the nonalchemical system
    if lambda_state==0:
        nonalchemical_positions = hybrid_factory.old_positions(hybrid_positions)
        nonalchemical_alternate_positions = hybrid_factory.new_positions(hybrid_positions)
    elif lambda_state==1:
        nonalchemical_positions = hybrid_factory.new_positions(hybrid_positions)
        nonalchemical_alternate_positions = hybrid_factory.old_positions(hybrid_positions)
    else:
        raise ValueError("lambda_state must be 0 or 1")

    nonalchemical_sampler_state = states.SamplerState(nonalchemical_positions, box_vectors=hybrid_sampler_state.box_vectors)
    nonalchemical_alternate_sampler_state = states.SamplerState(nonalchemical_alternate_positions, box_vectors=hybrid_sampler_state.box_vectors)

    nonalchemical_reduced_potential = compute_reduced_potential(nonalchemical_thermodynamic_state, nonalchemical_sampler_state)

    #now for the growth system (set at lambda 0 or 1) so we can get the valence energy
    if growth_thermodynamic_state:
        valence_energy = compute_reduced_potential(growth_thermodynamic_state, nonalchemical_alternate_sampler_state)
    else:
        valence_energy = 0.0

    #now, the corrected energy of the system (for dispersion correction) is the nonalchemical_reduced_potential + valence_energy
    return (valence_energy, nonalchemical_reduced_potential, hybrid_reduced_potential)

def compute_timeseries(reduced_potentials: np.array) -> list:
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

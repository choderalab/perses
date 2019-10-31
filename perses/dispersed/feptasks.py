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
_logger.setLevel(logging.WARNING)

#cache.global_context_cache.platform = openmm.Platform.getPlatformByName('Reference') #this is just a local version
EquilibriumFEPTask = namedtuple('EquilibriumInput', ['sampler_state', 'inputs', 'outputs'])
NonequilibriumFEPTask = namedtuple('NonequilibriumFEPTask', ['particle', 'inputs'])

class Particle():
    """
    This class represents an sMC particle that runs a nonequilibrium switching protocol.
    It is simply a wrapper around the aforementioned integrator, which must be provided in the constructor.
    WARNING: take care in writing trajectory file as saving positions to memory is costly.  Either do not write the configuration or save sparse positions.
    Parameters
    ----------
    thermodynamic_state: openmmtools.states.CompoundThermodynamicState
        compound thermodynamic state comprising state at some lambda
    sampler_state: openmmtools.states.SamplerState
        sampler state from which to conduct annealing
    nsteps : int
        number of annealing steps in the protocol
    direction : str
        whether the protocol runs 'forward' or 'reverse'
    splitting : str, default 'V R O R V'
        Splitting string for integrator
    temperature : unit.Quantity(float, units = unit.kelvin)
        temperature at which to run the simulation
    collision_rate : unit.Quantity(float, units = 1/unit.picoseconds)
        integrator collision rate
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
    lambda_protocol: str, default 'default'
        which lambda protocol from which to anneal
    measure_shadow_work : bool, default False
        whether to measure the shadow work from the integrator
    label : int, default None
        particle label
    trajectory_filename : str
        name of the trajectory filename prefix to which we write

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

    def __init__(self,
                 thermodynamic_state,
                 sampler_state,
                 nsteps,
                 direction,
                 splitting = 'V R O R V',
                 temperature = 300*unit.kelvin,
                 collision_rate = np.inf/unit.picoseconds,
                 timestep = 1.0*unit.femtosecond,
                 work_save_interval = None,
                 top = None,
                 subset_atoms = None,
                 save_configuration = False,
                 lambda_protocol = 'default',
                 measure_shadow_work = False,
                 label = None,
                 trajectory_filename  = None):

        _logger.debug(f"Initializing Particle...")
        start = time.time()
        self._timers = {} #instantiate timer
        self.label = [label]

        self.context_cache = cache.global_context_cache

        if measure_shadow_work:
            measure_heat = True
        else:
            measure_heat = False

        assert direction == 'forward' or direction == 'reverse', f"The direction of the annealing protocol ({direction}) is invalid; must be specified as 'forward' or 'reverse'"

        self._direction = direction

        #define the lambda schedule (linear)
        if self._direction == 'forward':
            self.start_lambda = 0.0
            self.end_lambda = 1.0
        elif self._direction == 'reverse':
            self.start_lambda = 1.0
            self.end_lambda = 0.0
        else:
            raise Error(f"direction must be 'forward' or 'reverse'")

        #create lambda protocol
        self._nsteps = nsteps
        if self._nsteps is None:
            self.trailblaze = True
            self._work_save_interval = work_save_interval # this is allowed to be None, in which case, the save_config method will never be called
            #likewise, if work work save interval is longer than the trailblazed lambda protocol, the save_configuration method will never be called
            self.current_lambda = self.start_lambda
            self.importance_samples = 1 #including the zero state
            #if we opt to trailblaze, we don't define a self.lambdas linsapce
            #instead, we define a self.current lambda and set it to the value of the start lambda
        else:
            self.trailblaze = False
            self.lambdas = np.linspace(self.start_lambda, self.end_lambda, self._nsteps)
            self.current_index = int(0)
            self.current_lambda = self.lambdas[self.current_index]
            if work_save_interval is None:
                self._work_save_interval = self._nsteps
            else:
                self._work_save_interval = work_save_interval
            #check that the work write interval is a factor of the number of steps, so we don't accidentally record the
            #work before the end of the protocol as the end
            if self._nsteps % self._work_save_interval != 0:
                raise ValueError("The work writing interval must be a factor of the total number of steps")

        #create sampling objects
        self.sampler_state = sampler_state
        self.thermodynamic_state = thermodynamic_state
        _logger.debug(f"thermodynamic state: {self.thermodynamic_state}")
        self.integrator = integrators.LangevinIntegrator(temperature = temperature, timestep = timestep, splitting = splitting, measure_shadow_work = measure_shadow_work, measure_heat = measure_heat, constraint_tolerance = 1e-6, collision_rate = collision_rate)
        #platform = openmm.Platform.getPlatformByName(platform_name)
        self.context = self.context_cache.get_context(self.thermodynamic_state, self.integrator)
        #self.context, self.integrator = self.context_cache.get_context(self.thermodynamic_state, integrator)
        _logger.debug(f"context: {self.context}")
        self.lambda_protocol_class = LambdaProtocol(functions = lambda_protocol)
        self.thermodynamic_state.set_alchemical_parameters(self.start_lambda, lambda_protocol = self.lambda_protocol_class)
        self.thermodynamic_state.apply_to_context(self.context)
        self.sampler_state.apply_to_context(self.context, ignore_velocities=True)
        self.context.setVelocitiesToTemperature(self.thermodynamic_state.temperature) #randomize velocities @ temp
        self.integrator.step(1)
        self.sampler_state.update_from_context(self.context)


        #create temperatures
        self._beta = 1.0 / (kB*temperature)
        self._temperature = temperature

        init_state = self.context.getState(getEnergy=True)
        self.initial_energy = self._beta * (init_state.getPotentialEnergy() + init_state.getKineticEnergy())

        self._save_configuration = save_configuration
        self._measure_shadow_work = measure_shadow_work
        if self._save_configuration:
            if trajectory_filename is None:
                raise Exception(f"cannot save configuration when trajectory_filename is None")
            else:
                self._trajectory_filename = trajectory_filename

        #use the number of step moves plus one, since the first is always zero
        self._cumulative_work = [0.0]
        self._shadow_work = 0.0
        self._heat = 0.0

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
        self._timers['protocol'] = []
        self._timers['save'] = []

        #set a bool variable for pass or failure
        self.succeed = True
        self.failures = []
        _logger.debug(f"Initialization complete!")

    def anneal(self, new, num_integration_steps = 1):
        """Propagate the state through the integrator.
        This updates the SamplerState after the integration. It will an mcmc protocol specified by the next lambda

        Parameters
        ----------
        new: int or float
            if float: is presumed to be the new lambda for the next importance weight
            if int: is presumed to be the index of the corresponding lambda for annealing
        num_integration_steps: int, default 1
            the number of integration steps to propagate the particle at a given lambda
        """
        self.context, self.integrator = self.context_cache.get_context(self.thermodynamic_state, self.integrator)
        self.sampler_state.apply_to_context(self.context, ignore_velocities=True)
        self.context.setVelocitiesToTemperature(self.thermodynamic_state.temperature)
        #first, we must check the trailblaze
        if type(new) == float or isinstance(new, np.float64):
            if not self.trailblaze:
                raise Exception(f"Annealing to lambda = {new}, but Particle is not trailblazing.  Aborting")
            _logger.info(f"Computing importance weight and propagating\
                with respect to lambda = {new} from {self.current_lambda} and propagating {num_integration_steps} steps...")
            self.current_lambda = new
            try:
                prep_start = time.time()
                self.compute_incremental_work(new)
                self.integrator.step(num_integration_steps)
                self._timers['protocol'].append(time.time() - prep_start)
                self.current_lambda = new
                self.importance_samples += 1
                iteration = self.importance_samples
                self.save_configuration(iteration)
                #we can resample outside of this if necessary
            except Exception as e:
                _logger.info(f"\tdetected failure: {e}")
                self.succeed = False
                self.failures.append(e)

        elif type(new) == int:
            if self.trailblaze:
                raise Exception(f"Annealing to index = {new}, but Particle is trailblazing.  Aborting")
            num_steps = new - self.current_index
            if self.current_index + num_steps >= len(self.lambdas):
                raise Exception(f"you specified {num_steps} steps, but there are only {len(self.lambdas)} lambda values, or {len(self.lambdas) - 1} steps to be conducted...aborting!")
            _logger.info(f"Computing importance weight and propagating to index {new}. Annealing from lambda = {self.lambdas[self.current_index]}\
                to {self.lambdas[new]}, corresponding to {num_steps} steps.")
            for _ in range(num_steps): #anneal for step steps
                try:
                    prep_start = time.time()
                    current_lambda = self.lambdas[self.current_index]
                    new_lambda = self.lambdas[self.current_index + 1]
                    self.compute_incremental_work(new_lambda)
                    self.current_index += increment
                    self.integrator.step(num_integration_steps)
                    self._timers['protocol'].append(time.time() - prep_start)
                    iteration = self.current_index
                    self.save_configuration(iteration)
                    self.current_lambda = new_lambda
                    #we can resample outside of this if necessary
                except Exception as e:
                    _logger.info(f"\tdetected failure: {e}")
                    self.succeed = False
                    self.failures.append(e)

        else:
            raise Exception(f"new ({new} is neither a float nor an integer  it is a{type(new)}.  Aborting.)")

        #attempt termination
        self.attempt_termination()

    def attempt_termination(self):
        """
        Attempt to terminate the annealing protocol and return the Particle attributes.
        """
        if self.current_lambda == self.end_lambda: #then we terminate
            _logger.info(f"detected final lambda")
            if self._save_configuration:
                _logger.info(f"saving configuration")
                self._trajectory = md.Trajectory(np.array(self._trajectory_positions), self._topology, unitcell_lengths=np.array(self._trajectory_box_lengths), unitcell_angles=np.array(self._trajectory_box_angles))
                Particle.write_nonequilibrium_trajectory(self._trajectory, self._trajectory_filename)

            if self._measure_shadow_work:
                _logger.info(f"measuring shadow work")
                total_heat = self.integrator.get_heat(dimensionless=True)
                final_state = self.context.getState(getEnergy=True)
                self.final_energy = self._beta * (final_state.getPotentialEnergy() + final_state.getKineticEnergy())
                total_energy_change = final_energy - self.initial_energy
                self._shadow_work = total_energy_change - (total_heat + self._cumulative_work)
            else:
                self._shadow_work = 0.0



    def compute_incremental_work(self, _lambda):
        """
        compute the incremental work of a lambda update
        """
        old_rp = self._beta * self.context.getState(getEnergy=True).getPotentialEnergy()

        #update thermodynamic state and context
        self.thermodynamic_state.set_alchemical_parameters(_lambda, lambda_protocol = self.lambda_protocol_class)
        self.thermodynamic_state.apply_to_context(self.context)
        new_rp = self._beta * self.context.getState(getEnergy=True).getPotentialEnergy()
        self._cumulative_work.append(self._cumulative_work[-1] + (new_rp - old_rp))

    def save_configuration(self, iteration):
        """
        pass a conditional save function
        """
        #now to save the protocol work if necessary
        if self._work_save_interval is None:
            return
        if iteration % self._work_save_interval == 0: #we save the protocol work if the remainder is zero
            _logger.debug(f"\t\tsaving protocol")
            save_start = time.time()
            #self._kinetic_energy.append(self._beta * context.getState(getEnergy=True).getKineticEnergy()) #maybe if we want kinetic energy in the future
            self.sampler_state.update_from_context(self.context, ignore_velocities=True) #save bandwidth by not updating the velocities

            #if we have a trajectory, we'll also write to it
            if self._save_configuration:
                _logger.debug(f"\t\tsaving configuration")

                #record positions for writing to trajectory
                #we need to check whether the user has requested to subset atoms (excluding water, for instance)

                if self._subset_atoms is None:
                    self._trajectory_positions.append(self.sampler_state.positions[:, :].value_in_unit_system(unit.md_unit_system))
                else:
                    self._trajectory_positions.append(self.sampler_state.positions[self._subset_atoms, :].value_in_unit_system(unit.md_unit_system))

                #get the box angles and lengths
                a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*self.sampler_state.box_vectors)
                self._trajectory_box_lengths.append([a, b, c])
                self._trajectory_box_angles.append([alpha, beta, gamma])
            self._timers['save'].append(time.time() - save_start)


    @staticmethod
    def launch_particle(task):

        """
        Instantiate the Particle class and place into distributed memory.

        Parameters
        ----------
        task : NonequilibriumFEPTask namedtuple
            The namedtuple should have an 'input' argument.  The 'input' argument is a dict characterized as follows:
            {
             thermodynamic_state: (<openmmtools.states.CompoundThermodynamicState>; compound thermodynamic state comprising state at some lambda),
             sampler_state: (<openmmtools.states.SamplerState>; sampler state from which to anneal)
             direction: (<str>; 'forward' or 'reverse')
             topology: (<mdtraj.Topology>; an MDTraj topology object used to construct the trajectory),
             n_lambdas: (<int, default None; number of nonequilibrium steps in the protocol>),
             work_save_interval: (<int>; how often to write the work and, if requested, configurations),
             splitting: (<str>; The splitting string for the dynamics),
             atom_indices_to_save: (<list of int, default None>; list of indices to save when excluding waters, for instance. If None, all indices are saved.),
             trajectory_filename: (<str, optional, default None>; Full filepath of trajectory files. If none, trajectory files are not written.),
             write_configuration: (<boolean, default False>; Whether to also write configurations of the trajectory at the requested interval.),
             timestep: (<unit.Quantity=float*unit.femtoseconds>; dynamical timestep),
             collision_rate : (unit.Quantity = float/unit.picoseconds; collision rate)
             measure_shadow_work: (<bool, default False>; Whether to compute the shadow work; there is additional overhead in the integrator cost),
             lambda_protocol: (<str, default 'default'; lambda protocol with which to conduct annealing >)
            }
        """

        inputs_dict = task.inputs
        Particle.check_NonequilibriumFEPTask(task)

        thermodynamic_state = inputs_dict['thermodynamic_state']
        sampler_state = inputs_dict['sampler_state']

        #forward Functions
        if not inputs_dict['lambda_protocol']:
            inputs_dict['lambda_protocol'] = 'default'


        #get the atom indices we need to subset the topology and positions
        if inputs_dict['atom_indices_to_save'] is None:
            atom_indices = list(range(inputs_dict['topology'].n_atoms))
            subset_topology = inputs_dict['topology']
        else:
            subset_topology = inputs_dict['topology'].subset(inputs_dict['atom_indices_to_save'])
            atom_indices = inputs_dict['atom_indices_to_save']

        _logger.debug(f"Instantiating NonequilibriumSwitchingMove class")
        particle = Particle(thermodynamic_state = thermodynamic_state,
                            sampler_state = sampler_state,
                            nsteps = inputs_dict['n_lambdas'],
                            direction = inputs_dict['direction'],
                            splitting = inputs_dict['splitting'],
                            temperature = thermodynamic_state.temperature,
                            timestep = inputs_dict['timestep'],
                            work_save_interval = inputs_dict['work_save_interval'],
                            top = subset_topology,
                            subset_atoms = atom_indices,
                            save_configuration = inputs_dict['write_configuration'],
                            lambda_protocol = inputs_dict['lambda_protocol'],
                            measure_shadow_work=inputs_dict['measure_shadow_work'],
                            label = inputs_dict['label'],
                            trajectory_filename = inputs_dict['trajectory_filename'])

        return NonequilibriumFEPTask(particle = particle, inputs = inputs_dict)

    @staticmethod
    def distribute_anneal(task, new, num_integration_steps = 1):
        """
        client-callable function to call Particle.anneal method

        Parameters
        ----------
        task: NonequilibriumFEPTask
            the particle-containing task on which to call Particle.anneal()
        new: int or float
            if float: is presumed to be the new lambda for the next importance weight
            if int: is presumed to be the index of the corresponding lambda for annealing
        num_integration_steps: int, default 1
            the number of integration steps to propagate the particle at a given lambda

        Returns
        -------
        task: NonequilibriumFEPTask
            the particle-containing task on which anneal was called

        """
        task.particle.anneal(new, num_integration_steps)
        return task

    @staticmethod
    def update_incremental_work(task, index):
        """
        client-callable function to compute importance weight given a new index
        (w.r.t. current index), and set the new self.current_index
        """
        task.particle.compute_incremental_work(index)
        task.particle._cumulative_work -= task.particle.incremental_work[-1]
        task.current_index = index
        return task


    @staticmethod
    def pull_sampler_state(task):
        """
        client-callable function to pull the sampler state from the task.particle class
        """
        task.particle.sampler_state.update_from_context(task.particle.context, ignore_velocities=True)
        return task.particle.sampler_state

    @staticmethod
    def pull_cumulative_work(task):
        """
        client-callable function to pull the cumulative_work from the task.particle class
        """
        return task.particle._cumulative_work[-1]

    @staticmethod
    def pull_cumulative_work_profile(task):
        """
        client-callable function to pull the cumulative_work profile from the task.particle class
        """
        return task.particle._cumulative_work


    @staticmethod
    def pull_work_increments(task):
        """
        client-callable function to pull the last two cumulative works from the task.particle class

        Returns
        -------
        work_increments: tuple(float, float)
            last two accumulated work values of the Particle._cumulative_work list
        """

        return task.particle._cumulative_work[-2:]

    @staticmethod
    def pull_current_lambda(task):
        """
        client-callable function to pull the current index from the task.particle class

        Returns
        -------
        current_index: int
            the index of the particle iterator
        """
        return task.particle.current_lambda

    @staticmethod
    def pull_shadow_work(task):
        """
        client-callable function to pull the shadow_work from the task.particle class
        """
        return task.particle._shadow_work

    @staticmethod
    def pull_timers(task):
        """
        client-callable function to pull the _timers dict from the task.particle class
        """
        return task.particle._timers

    @staticmethod
    def pull_success(task):
        """
        client-callable function to pull the success bool from the task.particle_class
        """
        return task.particle.succeed

    @staticmethod
    def pull_labels(task):
        """
        client-callable function to pull the label from the task.particle_class
        """
        return task.particle.label

    @staticmethod
    def pull_last_label(task):
        """
        client-callable function to pull the last label from the task.particle_class
        """
        return task.particle.label[-1]


    @staticmethod
    def push_resamples(task, sampler_state, label, work):
        """
        client-callable function to push a local sampler state, label, and cumulative_work to the task.particle class.
        It simply overwrites the particle.sampler_state, appends to self.label and corrects the work
        """
        task.particle.sampler_state = sampler_state
        task.particle.sampler_state.apply_to_context(task.particle.context, ignore_velocities=True)
        task.particle.context.setVelocitiesToTemperature(task.particle.thermodynamic_state.temperature) #randomize velocities @ temp
        task.particle.label.append(label)
        task.particle._cumulative_work[-1] = work
        return task

    @staticmethod
    def push_cumulative_work(task, work):
        """
        client-callable function to push a local float (cumulative work) to the task.particle class.
        """
        task.particle._cumulative_work[-1] = work
        return task

    @staticmethod
    def check_NonequilibriumFEPTask(task):
        """
        checks the NonequilibriumFEPTask for proper parameters as specified by Particle class

        Parameters
        ----------
        task : NonequilibriumFEPTask
        """
        input_dict = task.inputs
        assert type(input_dict) == dict, f"the NonquilibriumFEPTask.input entry is not a dictionary; type = {type(input_dict)}"

        valid_keys = ['thermodynamic_state',
                      'sampler_state',
                      'n_lambdas',
                      'direction',
                      'splitting',
                      'timestep',
                      'work_save_interval',
                      'topology',
                      'atom_indices_to_save',
                      'write_configuration',
                      'lambda_protocol',
                      'collision_rate',
                      'measure_shadow_work',
                      'label',
                      'trajectory_filename']

        #assert all valid keys are in the inputs keys
        for _key in input_dict.keys():
            if _key not in valid_keys:
                raise Exception(f"{_key} is not in NonequilibriumFEPTask.inputs")

    @staticmethod
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

def check_EquilibriumFEPTask(task):
    """
    checks the EquilibriumFEPTask for run_equilibrium parameters

    Parameters
    ----------
    task : EquilibriumFEPTask
    """
    input_dict = task.inputs
    #assert input_dict is a dict
    assert type(input_dict) == dict, f"the EquilibriumFEPTask.input entry is not a dictionary; type = {type(input_dict)}"
    valid_keys = ['thermodynamic_state', 'nsteps_equil', 'topology', 'n_iterations', 'splitting', 'atom_indices_to_save', 'trajectory_filename',
                  'max_size', 'timer', '_minimize', 'file_iterator', 'timestep']

    #assert all valid keys are in the inputs keys
    for _key in input_dict.keys():
        if _key not in valid_keys:
            raise Exception(f"{_key} is not in EquilibriumFEPTask.inputs")


def minimize(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState,
             max_iterations: int=100) -> states.SamplerState:
    """
    Minimize the given system and state, up to a maximum number of steps.
    This does not return a copy of the samplerstate; it is an update-in-place.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The state at which the system could be minimized
    sampler_state : openmmtools.states.SamplerState
        The starting state at which to minimize the system.
    max_iterations : int, optional, default 100
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

def write_nonequilibrium_trajectory(nonequilibrium_trajectory, trajectory_filename) -> float:
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
    valence_energy: float
        reduced potential energy of the valence contribution of the alternate endstate
    nonalchemical_reduced_potential : float
        reduced potential energy of the nonalchemical endstate
    hybrid_reduced_potential: float
        reduced potential energy of the alchemical endstate
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
    #first, we must check the input variable of the EquilibriumFEPTask and define the dictionary
    check_EquilibriumFEPTask(task)
    inputs = task.inputs

    timer = inputs['timer'] #bool
    timers = {}
    file_numsnapshots = []
    file_iterator = inputs['file_iterator']
    _logger.debug(f"running equilibrium")

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

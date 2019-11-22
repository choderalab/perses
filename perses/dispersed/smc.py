import simtk.openmm as openmm
import openmmtools.cache as cache
from typing import List, Tuple, Union, NamedTuple
import os
import copy
import openmmtools.cache as cache
from perses.dispersed.utils import *

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

# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("sMC")
_logger.setLevel(logging.DEBUG)

#cache.global_context_cache.platform = openmm.Platform.getPlatformByName('Reference') #this is just a local version
EquilibriumFEPTask = namedtuple('EquilibriumInput', ['sampler_state', 'inputs', 'outputs'])

class DaskClient(object):
    """
    This class manages the dask scheduler.
    Parameters
    ----------
    LSF: bool, default False
        whether we are using the LSF dask Client
    num_processes: int, default 2
        number of processes to run.  If not LSF, this argument does nothing
    adapt: bool, default False
        whether to use an adaptive scheduler.  If not LSF, this argument does nothing
    """
    def __init__(self):
        _logger.info(f"Initializing DaskClient")

    def activate_client(self,
                        LSF = True,
                        num_processes = 2,
                        adapt = False,
                        timeout = 1800):

        if LSF:
            from dask_jobqueue import LSFCluster
            cluster = LSFCluster()
            self._adapt = adapt
            self.num_processes = num_processes

            if self._adapt:
                _logger.debug(f"adapting cluster from 1 to {self.num_processes} processes")
                cluster.adapt(minimum = 2, maximum = self.num_processes, interval = "1s")
            else:
                _logger.debug(f"scaling cluster to {self.num_processes} processes")
                cluster.scale(self.num_processes)

            _logger.debug(f"scheduling cluster with client")
            self.client = distributed.Client(cluster, timeout = timeout)
            while len(self.client.nthreads()) != self.num_processes:
                _logger.debug(f"workers: {self.client.nthreads()}.  waiting for {self.num_processes} workers...")
                time.sleep(3)
            worker_threads = self.client.nthreads()
            self.workers = {i: _worker for i, _worker in zip(range(len(worker_threads)), worker_threads.keys())}
            self.worker_counter = 0

            #now we wait for all of our workers.
        else:
            self.client = None
            self._adapt = False
            self.num_processes = 0

    def deactivate_client(self):
        """
        NonequilibriumSwitchingFEP is not pickleable with the self.client or self.cluster activated.
        This must be called before pickling
        """
        if self.client is not None:
            self.client.close()
            self.client = None
            self.workers = None
            self.worker_counter = 0

    def scatter(self, df):
        """
        wrapper to scatter the local data df
        """
        if self.client is None:
            #don't actually scatter
            return df
        else:

            return self.client.scatter(df)

    def deploy(self, func, arguments):
        """
        wrapper to map a function and its arguments to the client for scheduling
        Arguments
        ---------
        func : function to map
            arguments: tuple of the arguments that the function will take
        argument : tuple of argument lists
        Returns
        ---------
        futures
        """
        if self.client is None:
            if len(arguments) == 1:
                futures = [func(plug) for plug in arguments[0]]
            else:
                futures = [func(*plug) for plug in zip(*arguments)]
        else:
            futures = self.client.map(func, *arguments)
        return futures

    def gather_results(self, futures):
        """
        wrapper to gather a function given its arguments
        Arguments
        ---------
        futures : future pointers

        Returns
        ---------
        results
        """
        if self.client is None:
            return futures
        else:
            results = self.client.gather(futures)
            return results

    def gather_actor_result(self, future):
        """
        wrapper to pull the .result() of a method called to an actor
        """
        if self.client is None:
            return future
        else:
            distributed.progress(future)
            result = future.result()
            return result

    def launch_actor(self, _class):
        """
        wrapper to launch an actor

        Arguments
        ---------
        _class : class object
            class to put on a worker

        Returns
        ---------
        actor : dask.distributed.Actor pointer (future)
        """
        if self.client is not None:
            future = self.client.submit(_class, workers = [self.workers[self.worker_counter]], actor=True)  # Create a _class on a worker
            self.worker_counter += 1
            distributed.progress(future)
            actor = future.result()                    # Get back a pointer to that object
            return actor
        else:
            actor = _class()
            return actor

    def progress(self, futures):
        """
        wrapper to log the progress of futures
        """
        if self.client is None:
            pass
        else:
            distributed.progress(futures)

    def wait(self, futures):
        """
        wrapper to wait until futures are complete.
        """
        if self.client is None:
            pass
        else:
            distributed.wait(futures)

class SequentialMonteCarlo(DaskClient):
    """
    This class represents an sMC particle that runs a nonequilibrium switching protocol.
    It is a batteries-included engine for conducting sequential Monte Carlo sampling.

    WARNING: take care in writing trajectory file as saving positions to memory is costly.  Either do not write the configuration or save sparse positions.
    """
    
    supported_resampling_methods = {'multinomial': multinomial_resample}
    supported_observables = {'ESS': ESS, 'CESS': CESS}

    def __init__(self,
                 factory,
                 lambda_protocol = 'default',
                 temperature = 300 * unit.kelvin,
                 trajectory_directory = 'test',
                 trajectory_prefix = 'out',
                 atom_selection = 'not water',
                 timestep = 1 * unit.femtoseconds,
                 collision_rate = 1 / unit.picoseconds,
                 eq_splitting_string = 'V R O R V',
                 neq_splitting_string = 'V R O R V',
                 ncmc_save_interval = None,
                 measure_shadow_work = False,
                 neq_integrator = 'langevin',
                 LSF = False,
                 num_processes = 2,
                 adapt = False):
        """
        Parameters
        ----------
        factory : perses.annihilation.relative.HybridTopologyFactory - compatible object
        lambda_protocol : str, default 'default'
            the flavor of scalar lambda protocol used to control electrostatic, steric, and valence lambdas
        temperature : float unit.Quantity
            Temperature at which to perform the simulation, default 300K
        trajectory_directory : str, default 'test'
            Where to write out trajectories resulting from the calculation. If None, no writing is done.
        trajectory_prefix : str, default None
            What prefix to use for this calculation's trajectory files. If none, no writing is done.
        atom_selection : str, default not water
            MDTraj selection syntax for which atomic coordinates to save in the trajectories. Default strips
            all water.
        timestep : float unit.Quantity, default 1 * units.femtoseconds
            the timestep for running MD
        collision_rate : float unit.Quantity, default 1 / unit.picoseconds
            the collision rate for running MD
        eq_splitting_string : str, default 'V R O R V'
            The integrator splitting to use for equilibrium simulation
        neq_splitting_string : str, default 'V R O R V'
            The integrator splitting to use for nonequilibrium switching simulation
        ncmc_save_interval : int, default None
            interval with which to write ncmc trajectory.  If None, trajectory will not be saved.
            We will assert that the n_lambdas % ncmc_save_interval = 0; otherwise, the protocol will not be complete
        measure_shadow_work : bool, default False
            whether to measure the shadow work of the integrator.
            WARNING : this is not currently supported
        neq_integrator : str, default 'langevin'
            which integrator to use
        LSF: bool, default False
            whether we are using the LSF dask Client
        num_processes: int, default 2
            number of processes to run.  If not LSF, this argument does nothing
        adapt: bool, default False
            whether to use an adaptive scheduler.  If not LSF, this argument does nothing
        """

        _logger.info(f"Initializing SequentialMonteCarlo")

        #pull necessary attributes from factory
        self.factory = factory

        #context cache
        self.context_cache = cache.global_context_cache

        #use default protocol
        self.lambda_protocol = lambda_protocol

        #handle both eq and neq parameters
        self.temperature = temperature
        self.timestep = timestep
        self.collision_rate = collision_rate

        self.measure_shadow_work = measure_shadow_work
        self.neq_integrator = neq_integrator
        if measure_shadow_work:
            raise Exception(f"measure_shadow_work is not currently supported.  Aborting!")


        #handle equilibrium parameters
        self.eq_splitting_string = eq_splitting_string

        #handle storage and names
        self.trajectory_directory = trajectory_directory
        self.trajectory_prefix = trajectory_prefix
        self.atom_selection = atom_selection

        #handle neq parameters
        self.neq_splitting_string = neq_splitting_string
        self.ncmc_save_interval = ncmc_save_interval

        #lambda states:
        self.lambda_endstates = {'forward': [0.0,1.0], 'reverse': [1.0, 0.0]}

        #instantiate trajectory filenames
        if self.trajectory_directory and self.trajectory_prefix:
            self.write_traj = True
            self.eq_trajectory_filename = {lambda_state: os.path.join(os.getcwd(), self.trajectory_directory, f"{self.trajectory_prefix}.eq.lambda_{lambda_state}.h5") for lambda_state in self.lambda_endstates['forward']}
            self.neq_traj_filename = {direct: os.path.join(os.getcwd(), self.trajectory_directory, f"{self.trajectory_prefix}.neq.lambda_{direct}") for direct in self.lambda_endstates.keys()}
            self.topology = self.factory.hybrid_topology
        else:
            self.write_traj = False
            self.eq_trajectory_filename = {0: None, 1: None}
            self.neq_traj_filename = {'forward': None, 'reverse': None}
            self.topology = None

        # subset the topology appropriately:
        self.atom_selection_string = atom_selection
        # subset the topology appropriately:
        if self.atom_selection_string is not None:
            atom_selection_indices = self.factory.hybrid_topology.select(self.atom_selection_string)
            self.atom_selection_indices = atom_selection_indices
        else:
            self.atom_selection_indices = None

        # instantiating equilibrium file/rp collection dicts
        self._eq_dict = {0: [], 1: [], '0_decorrelated': None, '1_decorrelated': None, '0_reduced_potentials': [], '1_reduced_potentials': []}
        self._eq_files_dict = {0: [], 1: []}
        self._eq_timers = {0: [], 1: []}
        self._neq_timers = {'forward': [], 'reverse': []}

        #instantiate nonequilibrium work dicts: the keys indicate from which equilibrium thermodynamic state the neq_switching is conducted FROM (as opposed to TO)
        self.cumulative_work = {'forward': [], 'reverse': []}
        self.incremental_work = copy.deepcopy(self.cumulative_work)
        self.shadow_work = copy.deepcopy(self.cumulative_work)
        self.nonequilibrium_timers = copy.deepcopy(self.cumulative_work)
        self.total_jobs = 0
        #self.failures = copy.deepcopy(self.cumulative_work)
        self.dg_EXP = copy.deepcopy(self.cumulative_work)
        self.dg_BAR = None


        # create an empty dict of starting and ending sampler_states
        self.start_sampler_states = {_direction: [] for _direction in ['forward', 'reverse']}
        self.end_sampler_states = {_direction: [] for _direction in ['forward', 'reverse']}

        #instantiate thermodynamic state
        lambda_alchemical_state = RelativeAlchemicalState.from_system(self.factory.hybrid_system)
        lambda_alchemical_state.set_alchemical_parameters(0.0, LambdaProtocol(functions = self.lambda_protocol))
        self.thermodynamic_state = CompoundThermodynamicState(ThermodynamicState(self.factory.hybrid_system, temperature = self.temperature),composable_states = [lambda_alchemical_state])

        # set the SamplerState for the lambda 0 and 1 equilibrium simulations
        sampler_state = SamplerState(self.factory.hybrid_positions,
                                          box_vectors=self.factory.hybrid_system.getDefaultPeriodicBoxVectors())
        self.sampler_states = {0: copy.deepcopy(sampler_state), 1: copy.deepcopy(sampler_state)}

        #Dask implementables
        self.LSF = LSF
        self.num_processes = num_processes
        self.adapt = adapt

    def launch_LocallyOptimalAnnealing(self):
        """
        Call LocallyOptimalAnnealing with the number of particles and a protocol.

        Returns
        -------
        LOA_actor : dask.distributed.actor (or class object) of smc.LocallyOptimalAnnealing
            actor pointer if self.LSF, otherwise class object
        """
        start_timer = time.time()
        if self.LSF:
            LOA_actor = self.launch_actor(LocallyOptimalAnnealing)
        else:
            LOA_actor = self.launch_actor(LocallyOptimalAnnealing)

        actor_bool = LOA_actor.initialize(thermodynamic_state = self.thermodynamic_state,
                                          lambda_protocol = self.lambda_protocol,
                                          timestep = self.timestep,
                                          collision_rate = self.collision_rate,
                                          temperature = self.temperature,
                                          neq_splitting_string = self.neq_splitting_string,
                                          ncmc_save_interval = self.ncmc_save_interval,
                                          topology = self.topology,
                                          subset_atoms = self.topology.select(self.atom_selection_string),
                                          measure_shadow_work = self.measure_shadow_work,
                                          integrator = self.neq_integrator)
        if self.LSF:
            assert self.gather_actor_result(actor_bool), f"Dask initialization failed"
        else:
            assert actor_bool, f"local initialization failed"

        end_timer = time.time() - start_timer
        _logger.info(f"\t\t launch_LocallyOptimalAnnealing took {end_timer} seconds")

        return LOA_actor

    def _actor_distribution(self, directions, num_particles):
        """
        wrapper to decide the distribution of actors

        Arguments
        ----------
        directions : list of str
            the directions to run
        num_particles : int
            number of particles per direction

        Returns
        -------
        num_actors : int
            number of actors that will be launched
        num_actors_per_direction : int
            number of actors per direction
        num_particles_per_actor : int
            number of particles to be launched per actor
        sMC actors : dict {_direction: {actor: [actor_future]}}
            the actor dict that manages bookkeeping
        """
        if self.LSF: #we have to figure out how many actors to make
            if not self.adapt:
                num_actors = self.num_processes
                num_actors_per_direction = num_actors // len(directions)
                if num_actors % len(directions) != 0:
                    raise Exception(f"the number of actors per direction does not divide evenly (num_actors = {num_actors} while num_directions = {len(directions)}).")
                num_particles_per_actor = num_particles // num_actors_per_direction
                if num_particles % num_actors_per_direction != 0:
                    raise Exception(f"the number of particles per actor does not divide evenly (num_particles = {num_particles} while num_actors_per_direction = {num_actors_per_direction}).")
            else:
                raise Exception(f"the client is adaptable, but AIS does not currently support an adaptive client")
        else:
            #we only create one local actor and put all of the particles on it
            num_actors = len(directions)
            num_actors_per_direction = 1
            num_particles_per_actor = num_particles

        _logger.info(f"num_actors: {num_actors}")
        _logger.info(f"particles per actor: {num_particles_per_actor}")

        #now we have to launch the LocallyOptimalAnnealing actors
        _logger.info(f"Instantiating sMC actors...")
        sMC_actors = {_direction: [] for _direction in directions}

        for _direction in directions:
            _logger.info(f"launching {_direction} actors...")
            for actor_idx in range(num_actors_per_direction):
                _logger.info(f"\tlaunching actor {actor_idx + 1} of {num_actors_per_direction}.")
                _actor = self.launch_LocallyOptimalAnnealing()
                sMC_actors[_direction].append(_actor)

        return num_actors, num_actors_per_direction, num_particles_per_actor, sMC_actors

<<<<<<< HEAD
=======
    def _actor_collection(self, _dict):
        """
        wrapper for actor future collections

        Arguments
        ----------
        _dict : dict (dict of directions of 2d list of dask.distributed.ActorFutures)

        Returns
        -------
        results_dict : _dict of 2d list of SequentialMonteCarlo.anneal results
        """

        #now we collect the jobs
        for _direction in directions:
            actor_dict = AIS_actors[_direction]
            for _actor in actor_dict.keys():
                _future = AIS_actors[_direction][_actor][-1]
                result = self.gather_actor_result(_future)
                AIS_actors[_direction][_actor][-1] = result

>>>>>>> 51995494c40d6fa3f62816b5977c79ffbd196017

    def AIS(self,
            num_particles,
            protocol_length,
            directions = ['forward','reverse'],
            num_integration_steps = 1,
            return_timer = False,
            rethermalize = False):
        """
        Conduct vanilla AIS. with a linearly interpolated lambda protocol

        Arguments
        ----------
        num_particles : int
            number of particles to run in each direction
        protocol_length : int
            number of lambdas
        directions : list of str, default ['forward', 'reverse']
            the directions to run.
        num_integration_steps : int
            number of integration steps per proposal
        return_timer : bool, default False
            whether to time the annealing protocol
        rethermalize : bool, default False
            whether to rethermalize velocities after proposal
        """
        self.activate_client(LSF = self.LSF,
                            num_processes = self.num_processes,
                            adapt = self.adapt)

        for _direction in directions:
            assert _direction in ['forward', 'reverse'], f"direction {_direction} is not an appropriate direction"
        protocols = {}
        for _direction in directions:
            if _direction == 'forward':
                protocol = np.linspace(0, 1, protocol_length)
            elif _direction == 'reverse':
                protocol = np.linspace(1, 0, protocol_length)
            protocols.update({_direction: protocol})

        num_actors, num_actors_per_direction, num_particles_per_actor, AIS_actors = self._actor_distribution(directions = directions, 
                                                                                                             num_particles = num_particles)
        AIS_actor_futures = {_direction: [[[None] * num_particles_per_actor] for _ in range(num_actors_per_direction)]}
        AIS_actor_results = copy.deepcopy(AIS_actor_futures)

        for job in tqdm.trange(num_particles_per_actor):
            start_timer = time.time()
            for _direction in directions:
                _logger.info(f"entering {_direction} direction to launch annealing jobs.")
                for _actor in range(num_actors_per_direction):
                    _logger.info(f"\tretrieving actor {_actor}.")
                    sampler_state = self.pull_trajectory_snapshot(0) if _direction == 'forward' else self.pull_trajectory_snapshot(1)
                    if self.ncmc_save_interval is not None: #check if we should make 'trajectory_filename' not None
                        noneq_trajectory_filename = self.neq_traj_filename[_direction] + f".iteration_{self.total_jobs:04}.h5"
                        self.total_jobs += 1
                    else:
                        noneq_trajectory_filename = None

                    actor_future = AIS_actors[_direction][_actor].anneal(sampler_state = sampler_state,
                                                                         lambdas = protocols[_direction],
                                                                         label = self.total_jobs,
                                                                         noneq_trajectory_filename = noneq_trajectory_filename,
                                                                         num_integration_steps = num_integration_steps,
                                                                         return_timer = return_timer,
                                                                         return_sampler_state = False,
                                                                         rethermalize = rethermalize,
                                                                         )
                    AIS_actor_futures[_direction][_actor][job] = actor_future


            #now we collect the jobs
            for _direction in directions:
                for _actor in range(num_actors_per_direction):
                    _future = AIS_actors_futures[_direction][_actor][job]
                    result = self.gather_actor_result(_future)
                    AIS_actor_results[_direction][_actor][job] = result

            end_timer = time.time() - start_timer
            _logger.info(f"iteration took {end_timer} seconds.")


        #now that the actors are gathered, we can collect the results and put them into class attributes
        _logger.info(f"organizing all results...")
        for _direction in AIS_actor_results.keys():
            _result_lst = AIS_actor_results[_direction]
            _logger.info(f"collecting {_direction} actor results...")
            flattened_result_lst = [item for sublist in _result_lst for item in sublist]
            [self.incremental_work[_direction].append(item[0]) for item in flattened_result_lst]
            [self.nonequilibrium_timers[_direction].append(item[2]) for item in flattened_result_lst]

        #compute the free energy
        self.compute_free_energy()

        #deactivate_client
        self.deactivate_client()

    def sMC_anneal(self,
                   num_particles,
                   protocols = {'forward': np.linspace(0,1, 1000), 'reverse': np.linspace(1,0,1000)},
                   directions = ['forward','reverse'],
                   num_integration_steps = 1,
                   return_timer = False,
                   rethermalize = False,
                   trailblaze = None,
                   resample = None):
        """
        Conduct generalized sMC annealing with trailblazing functionality.

        Arguments
        ----------
        num_particles : int
            number of particles to run in each direction
        protocols : dict of {direction : np.array}, default {'forward': np.linspace(0,1, 1000), 'reverse': np.linspace(1,0,1000)},
            the dictionary of forward and reverse protocols.  if None, the protocols will be trailblazed.
        directions : list of str, default ['forward', 'reverse']
            the directions to run.
        num_integration_steps : int
            number of integration steps per proposal
        return_timer : bool, default False
            whether to time the annealing protocol
        rethermalize : bool, default False
            whether to rethermalize velocities after proposal
        trailblaze : dict, default None
            which observable/criterion to use for trailblazing and the threshold
            if None, trailblazing is not conducted;
            else: the dict must have the following format:
                {'criterion': str, 'threshold': float}
        resample : dict, default None
            the resample dict specifies the resampling criterion and threshold, as well as the resampling method used.  if None, no resampling is conduced;
            otherwise, the resample dict must take the following form:
            {'criterion': str, 'method': str, 'threshold': float}
        """
        #first some bookkeeping/validation
        for _direction in directions:
            assert _direction in ['forward', 'reverse'], f"direction {_direction} is not an appropriate direction"

        if protocols is not None:
            assert set(protocols.keys()) == set(directions), f"protocols are specified, but do not match the directions in the specified directions"
            if trailblaze is not None:
                raise Exception(f"the protocols were specified, as was the trailblaze criterion.  Only one can be defined")
            _trailblaze = False
            starting_lines = {_direction: protocols[_direction][0] for _direction in directions}
            finish_lines = {_direction: protocols[_direction][-1] for _direction in directions}
            self.protocols = protocols
        else:
            assert trailblaze is not None, f"the protocols weren't specified, and neither was the trailblaze criterion; one must be specified"
            assert set(list(trailblaze.keys())).issubset(set(['criterion', 'threshold'])), f"trailblaze does not contain 'criterion' and 'threshold'"
            _trailblaze = True
            starting_lines, finish_lines = {}
            if 'forward' in directions:
                finish_lines['forward'] = 1.0
                starting_lines['forward'] = 0.0
            if 'reverse' in directions:
                finish_lines['reverse'] = 0.0
                starting_lines['reverse'] = 1.0
            self.protocols = {_direction : [starting_lines[_direction]] for _direction in directions}


        if resample is not None:
            assert resample['criterion'] in list(supported_observables.keys()), f"the specified resampling criterion is not supported."
            assert resample_method['method'] in list(supported_resampling_methods), f"the specified resampling method is not supported."
            _resample = True
        else:
            _resample = False

        for _direction in directions:
            assert _direction in ['forward', 'reverse'], f"direction {_direction} is not an appropriate direction"

        #initialize recording lists
        _logger.info(f"initializing organizing dictionaries...")
        
        num_actors, num_actors_per_direction, num_particles_per_actor, sMC_actors = self._actor_distribution(directions, num_particles)
        _logger.debug(f"\tsMC_actors: {sMC_actors}")
        
        sMC_actor_futures = {_direction: [[None for _ in range(num_particles_per_actor)] for __ in range(num_actors_per_direction)] for _direction in directions}
        _logger.debug(f"\tsMC_actor_futures: {sMC_actor_futures}")
        
        sMC_sampler_states = {_direction: np.array([[self.pull_trajectory_snapshot(int(starting_lines[_direction])) for _ in range(num_particles_per_actor)] for __ in range(num_actors_per_direction)]) for _direction in directions}
        _logger.debug(f"\tsMC_sampler_states: {sMC_sampler_states}")
        
        sMC_particle_ancestries = {_direction : np.arange(num_actors_per_direction * num_particles_per_actor).reshape(num_actors_per_direction, num_particles_per_actor) for _direction in directions}
        _logger.debug(f"\tsMC_particle_ancestries: {sMC_particle_ancestries}")
        
        sMC_cumulative_works = {_direction : [np.zeros((num_actors_per_direction, num_particles_per_actor))] for _direction in directions}
        _logger.debug(f"\tsMC_cumulative_works: {sMC_cumulative_works}")
        
        sMC_observables = {_direction : [1.] for _direction in directions}
        _logger.debug(f"\tsMC_observables: {sMC_observables}")

        #now we can launch annealing jobs and manage them on-the-fly
        current_lambdas = starting_lines
        iteration_number = 0
        _logger.info(f"current protocols : {self.protocols}")

        while current_lambdas != finish_lines: # respect the while loop; it is a dangerous creature
            _logger.info(f"entering iteration {iteration_number}; current lambdas are: {current_lambdas}")
            start_timer = time.time()
            if (not _trailblaze) and (not _resample):
                local_incremental_work_collector = {_direction : np.zeros((num_actors_per_direction, num_particles_per_actor, self.protocols[_direction].shape[0])) for _direction in directions}
            else:
                local_incremental_work_collector = {_direction : np.zeros((num_actors_per_direction, num_particles_per_actor)) for _direction in directions}
            start_timer = time.time()
            #if trailblaze is true, we have to choose the next lambda from the previous sampler states and weights
            if _trailblaze:
                for _direction in directions:
                    if current_lambdas[_direction] == finish_lines[_direction]: #if this direction is done...
                        pass #we pass the direction
                    else: #we have to choose the next lambda value
                        #gather sampler states and cumulative works in a concurrent manner (i.e. flatten them)
                        sampler_states = sMC_sampler_states[_direction].flatten()
                        cumulative_works = sMC_cumulative_works[_direction].flatten()
                        if iteration_number == 0:
                            initial_guess = None
                        else:
                            initial_guess = min([2 * protocols[_direction][-1] - protocol[_direction][-2], 1.0]) if _direction == 'forward' else max([2 * protocol[_direction][-1] - protocol[_direction][-2], 0.0])

                        _new_lambda, observable = self.binary_search(sampler_states = sampler_states,
                                                                     cumulative_works = cumulative_works,
                                                                     start_val = current_lambdas[_direction],
                                                                     end_val = finish_lines[_direction],
                                                                     observable = supported_observables[trailblaze['criterion']],
                                                                     observable_threshold = trailblaze['threshold'],
                                                                     initial_guess = initial_guess)
                        protocols[_direction].append(_new_lambda)
                        sMC_observables[_direction].append(observable)


            for job in tqdm.trange(num_particles_per_actor):
                for _direction in directions:
                    if current_lambdas[_direction] == finish_lines[_direction]:
                        pass
                    actor_list = sMC_actors[_direction]
                    _logger.info(f"\tentering {_direction} direction to launch annealing jobs.")
                    for _actor in range(num_actors_per_direction):
                        _logger.info(f"\tretrieving actor {_actor}.")

                        if self.ncmc_save_interval is not None: #check if we should make 'trajectory_filename' not None
                            noneq_trajectory_filename = self.neq_traj_filename[_direction] + f".iteration_{(_actor * num_particles_per_actor + job):04}.h5"
                        else:
                            noneq_trajectory_filename = None

                        if (not _trailblaze) and (not _resample):
                            #then we are just doing vanilla AIS, in which case, it is not necessary to perform a single incremental lambda perturbation
                            #instead, we can run the entire defined protocol
                            _lambdas = self.protocols[_direction]
                        else:
                            _lambdas = np.array([self.protocols[_direction][iteration_number], self.protocols[_direction][iteration_number + 1]])

                        sampler_state = sMC_sampler_states[_direction][_actor, job]
                        actor_future = actor_list[_actor].anneal(sampler_state = sampler_state,
                                                                 lambdas = _lambdas,
                                                                 noneq_trajectory_filename = noneq_trajectory_filename,
                                                                 num_integration_steps = num_integration_steps,
                                                                 return_timer = return_timer,
                                                                 return_sampler_state = True,
                                                                 rethermalize = rethermalize)

                        sMC_actor_futures[_direction][_actor][job] = actor_future
                        current_lambdas[_direction] = _lambdas[-1]

                #now we collect the jobs
                for _direction in directions:
                    _logger.info(f"\tentering {_direction} direction to collect annealing jobs.")
                    actor_list = sMC_actors[_direction]
                    for _actor in range(num_actors_per_direction):
                        _logger.info(f"\t\tretrieving actor {_actor}.")
                        
                        _future = sMC_actor_futures[_direction][_actor][job]
                        _incremental_work, _sampler_state, _timer = self.gather_actor_result(_future)
                        _logger.debug(f"\t\tincremental work: {_incremental_work}")
                        if _incremental_work.shape[0] == 1:
                            local_incremental_work_collector[_direction][_actor, job] += _incremental_work[0]
                        else: #vanilla AIS
                            local_incremental_work_collector[_direction][_actor, job, 1:] = _incremental_work
                        sMC_sampler_states[_direction][_actor, job] = _sampler_state

            #report the updated logger dicts
            _logger.debug(f"\tsMC_sampler_states: {sMC_sampler_states}")
            _logger.debug(f"\tsMC_local_incremental_work_collector: {local_incremental_work_collector}")
            
            
            
            
            
            #resample if necessary
            if _resample:
                for _direction in directions:
                    if current_lambdas[_direction] == finish_lines[_direction]:
                        pass
                    normalized_observable_value, resampled_works, resampled_indices = self._resample(incremental_works = local_incremental_work_collector[_direction].flatten(),
                                                                                                     cumulative_works = sMC_cumulative_works[_direction][-1],
                                                                                                     observable = resample['criterion'],
                                                                                                     resampling_method = resample['method'],
                                                                                                     resample_observable_threshold = resample['threshold'])
                    sMC_observables[_direction].append(normalized_observable_value)

                    sMC_cumulative_works[_direction].append(resampled_works.reshape(num_actors_per_direction, num_particles_per_actor))

                    flattened_sampler_states = sMC_sampler_states.flatten()
                    new_sampler_states = np.array([flattened_sampler_states[i] for i in resampled_indices]).reshape(num_actors_per_direction, num_particles_per_actor)
                    sMC_sampler_states[_direction] = new_sampler_states

                    flattened_particle_ancestries = sMC_particle_ancestries[_direction][-1].flatten()
                    new_particle_ancestries = np.array([flattened_particle_ancestries[i] for i in resampled_indices]).reshape(num_actors_per_direction, num_particles_per_actor)
                    sMC_particle_ancestries[_direction].append(new_particle_ancestries)
            else:
                if (not _trailblaze) and (not _resample): #we have to make an exception to bookkeeping if we are doing vanilla AIS
                    sMC_cumulative_works = {}
                    cumulative_works = {}
                    for _direction in directions:
                        cumulative_works[_direction] = local_incremental_work_collector[_direction]
                        for i in range(cumulative_works[_direction].shape[0]):
                            for j in range(cumulative_works[_direction].shape[1]):
                                cumulative_works[_direction][i,j,:] = np.cumsum(local_incremental_work_collector[_direction][i,j,:]) 
                        sMC_cumulative_works[_direction] = [cumulative_works[_direction][:,:,i] for i in range(cumulative_works[_direction].shape[2])]
                else:           
                    for _direction in directions:
                        sMC_cumulative_works[_direction].append(np.add(sMC_cumulative_works[_direction][-1], local_incremental_work_collector[_direction]))

            end_timer = time.time() - start_timer
            iteration_number += 1
            _logger.info(f"iteration took {end_timer} seconds.")

        self.compute_sMC_free_energy(sMC_cumulative_works)
        self.sMC_observables = sMC_observables
        if _resample:
            self.survival_rate = compute_survival_rate(sMC_particle_ancestries)
            self.particle_ancestries = {_direction : np.array([q.flatten() for q in sMC_particle_ancestries[_direction]]) for _direction in sMC_particle_ancestries.keys()}

    def compute_sMC_free_energy(self, cumulative_work_dict):
        _logger.debug(f"computing free energies...")
        self.cumulative_work = {_direction: None for _direction in cumulative_work_dict.keys()}
        self.dg_EXP = {}
        for _direction, _lst in cumulative_work_dict.items():
            flat_cumulative_work = np.array([arr.flatten() for arr in _lst])
            self.cumulative_work[_direction] = flat_cumulative_work
            self.dg_EXP[_direction] = np.array([pymbar.EXP(flat_cumulative_work[i, :]) for i in range(flat_cumulative_work.shape[0])])
        _logger.info(f"cumulative_work: {self.cumulative_work}")
        if (len(list(self.cumulative_work.keys())) == 2):
            self.dg_BAR = pymbar.BAR(self.cumulative_work['forward'][-1, :], self.cumulative_work['reverse'][-1, :])

    def minimize_sampler_states(self):
        # initialize by minimizing
        for state in self.lambda_endstates['forward']: # 0.0, 1.0
            self.thermodynamic_state.set_alchemical_parameters(state, LambdaProtocol(functions = self.lambda_protocol))
            minimize(self.thermodynamic_state, self.sampler_states[int(state)])

    def pull_trajectory_snapshot(self, endstate):
        """
        Draw randomly a single snapshot from self._eq_files_dict

        Parameters
        ----------
        endstate: int
            lambda endstate from which to extract an equilibrated snapshot, either 0 or 1
        Returns
        -------
        sampler_state: openmmtools.SamplerState
            sampler state with positions and box vectors if applicable
        """
        #pull a random index
        index = random.choice(self._eq_dict[f"{endstate}_decorrelated"])
        files = [key for key in self._eq_files_dict[endstate].keys() if index in self._eq_files_dict[endstate][key]]
        assert len(files) == 1, f"files: {files} doesn't have one entry; index: {index}, eq_files_dict: {self._eq_files_dict[endstate]}"
        file = files[0]
        file_index = self._eq_files_dict[endstate][file].index(index)

        #now we load file as a traj and create a sampler state with it
        traj = md.load_frame(file, file_index)
        positions = traj.openmm_positions(0)
        box_vectors = traj.openmm_boxes(0)
        sampler_state = SamplerState(positions, box_vectors = box_vectors)

        return sampler_state

    def equilibrate(self,
                    n_equilibration_iterations = 1,
                    n_steps_per_equilibration = 5000,
                    endstates = [0,1],
                    max_size = 1024*1e3,
                    decorrelate=False,
                    timer = False,
                    minimize = False):
        """
        Run the equilibrium simulations a specified number of times at the lambda 0, 1 states. This can be used to equilibrate
        the simulation before beginning the free energy calculation.

        Parameters
        ----------
        n_equilibration_iterations : int; default 1
            number of equilibrium simulations to run, each for lambda = 0, 1.
        n_steps_per_equilibration : int, default 5000
            number of integration steps to take in an equilibration iteration
        endstates : list, default [0,1]
            at which endstate(s) to conduct n_equilibration_iterations (either [0] ,[1], or [0,1])
        max_size : float, default 1.024e6 (bytes)
            number of bytes allotted to the current writing-to file before it is finished and a new equilibrium file is initiated.
        decorrelate : bool, default False
            whether to parse all written files serially and remove correlated snapshots; this returns an ensemble of iid samples in theory.
        timer : bool, default False
            whether to trigger the timing in the equilibration; this adds an item to the EquilibriumResult, which is a list of times for various
            processes in the feptask equilibration scheme.
        minimize : bool, default False
            Whether to minimize the sampler state before conducting equilibration. This is passed directly to feptasks.run_equilibration
        Returns
        -------
        equilibrium_result : perses.dispersed.feptasks.EquilibriumResult
            equilibrium result namedtuple
        """

        _logger.debug(f"conducting equilibration")
        for endstate in endstates:
            assert endstate in [0, 1], f"the endstates contains {endstate}, which is not in [0, 1]"

        # run a round of equilibrium
        _logger.debug(f"iterating through endstates to submit equilibrium jobs")
        EquilibriumFEPTask_list = []
        for state in endstates: #iterate through the specified endstates (0 or 1) to create appropriate EquilibriumFEPTask inputs
            _logger.debug(f"\tcreating lambda state {state} EquilibriumFEPTask")
            self.thermodynamic_state.set_alchemical_parameters(float(state), lambda_protocol = LambdaProtocol(functions = self.lambda_protocol))
            input_dict = {'thermodynamic_state': copy.deepcopy(self.thermodynamic_state),
                          'nsteps_equil': n_steps_per_equilibration,
                          'topology': self.factory.hybrid_topology,
                          'n_iterations': n_equilibration_iterations,
                          'splitting': self.eq_splitting_string,
                          'atom_indices_to_save': None,
                          'trajectory_filename': None,
                          'max_size': max_size,
                          'timer': timer,
                          '_minimize': minimize,
                          'file_iterator': 0,
                          'timestep': self.timestep}


            if self.write_traj:
                _logger.debug(f"\twriting traj to {self.eq_trajectory_filename[state]}")
                equilibrium_trajectory_filename = self.eq_trajectory_filename[state]
                input_dict['trajectory_filename'] = equilibrium_trajectory_filename
            else:
                _logger.debug(f"\tnot writing traj")

            if self._eq_dict[state] == []:
                _logger.debug(f"\tself._eq_dict[{state}] is empty; initializing file_iterator at 0 ")
            else:
                last_file_num = int(self._eq_dict[state][-1][0][-7:-3])
                _logger.debug(f"\tlast file number: {last_file_num}; initiating file iterator as {last_file_num + 1}")
                file_iterator = last_file_num + 1
                input_dict['file_iterator'] = file_iterator
            task = EquilibriumFEPTask(sampler_state = self.sampler_states[state], inputs = input_dict, outputs = None)
            EquilibriumFEPTask_list.append(task)

        _logger.debug(f"scattering and mapping run_equilibrium task")
        self.activate_client(LSF = self.LSF,
                            num_processes = 2,
                            adapt = self.adapt)

        scatter_futures = self.scatter(EquilibriumFEPTask_list)
        futures = self.deploy(run_equilibrium, (scatter_futures,))
        self.progress(futures)
        eq_results = self.gather_results(futures)
        self.deactivate_client()

        for state, eq_result in zip(endstates, eq_results):
            _logger.debug(f"\tcomputing equilibrium task future for state = {state}")
            self._eq_dict[state].extend(eq_result.outputs['files'])
            self._eq_dict[f"{state}_reduced_potentials"].extend(eq_result.outputs['reduced_potentials'])
            self.sampler_states.update({state: eq_result.sampler_state})
            self._eq_timers[state].append(eq_result.outputs['timers'])

        _logger.debug(f"collections complete.")
        if decorrelate: # if we want to decorrelate all sample
            _logger.debug(f"decorrelating data")
            for state in endstates:
                _logger.debug(f"\tdecorrelating lambda = {state} data.")
                traj_filename = self.eq_trajectory_filename[state]
                if os.path.exists(traj_filename[:-2] + f'0000' + '.h5'):
                    _logger.debug(f"\tfound traj filename: {traj_filename[:-2] + f'0000' + '.h5'}; proceeding...")
                    [t0, g, Neff_max, A_t, uncorrelated_indices] = compute_timeseries(np.array(self._eq_dict[f"{state}_reduced_potentials"]))
                    _logger.debug(f"\tt0: {t0}; Neff_max: {Neff_max}; uncorrelated_indices: {uncorrelated_indices}")
                    self._eq_dict[f"{state}_decorrelated"] = uncorrelated_indices

                    #now we just have to turn the file tuples into an array
                    _logger.debug(f"\treorganizing decorrelated data; files w/ num_snapshots are: {self._eq_dict[state]}")
                    iterator, corrected_dict = 0, {}
                    for tupl in self._eq_dict[state]:
                        new_list = [i + iterator for i in range(tupl[1])]
                        iterator += len(new_list)
                        decorrelated_list = [i for i in new_list if i in uncorrelated_indices]
                        corrected_dict[tupl[0]] = decorrelated_list
                    self._eq_files_dict[state] = corrected_dict
                    _logger.debug(f"\t corrected_dict for state {state}: {corrected_dict}")

        def _resample(self,
                            incremental_works,
                            cumulative_works,
                            observable = 'ESS',
                            resampling_method = 'multinomial',
                            resample_observable_threshold = 0.5):
            """
            Attempt to resample particles given an observable diagnostic and a resampling method.

            Arguments
            ----------
            incremental_works : np.array() of floats
                the incremental work accumulated from importance sampling at time t
            cumulative_works : np.array() of floats
                the cumulative work accumulated from importance sampling from time t = 1 : t-1
            observable : str, default 'ESS'
                the observable used as a resampling diagnostic; this calls a key in supported_observables
            resampling_method: str, default 'multinomial'
                method used to resample, this calls a key in supported_resampling_methods
            resample_observable_threshold : float, default 0.5
                the threshold to diagnose a resampling event.
                If None, will automatically return without observables

            Returns
            -------
            normalized_observable_value: float
                the value of the observable
            resampled_works : np.array() floats
                the resampled total works at iteration t
            resampled_indices : np.array() int
                resampled particle indices


            """
            _logger.debug(f"\tAttempting to resample...")
            num_particles = labels.shape[0]

            normalized_observable_value = supported_observables[observable](cumulative_works, incremental_works) / num_particles
            total_works = np.add(cumulative_works, incremental_works)

            #decide whether to resample
            _logger.debug(f"\tnormalized observable value: {normalized_observable_value}")
            if normalized_observable_value <= resample_observable_threshold: #then we resample
                _logger.debug(f"\tnormalized observable value ({normalized_observable_value}) <= {resample_observable_threshold}.  Resampling")

                #resample
                resampled_works, resampled_indices = self.allowable_resampling_methods[resampling_method](total_works = total_works,
                                                                                                          num_resamples = num_particles)

                normalized_observable_value = 1.0
            else:
                _logger.debug(f"\tnormalized observable value ({normalized_observable_value}) > {resample_observable_threshold}.  Skipping resampling.")
                resampled_works = total_works
                resampled_indices = np.arange(resampled_works)
                normalized_observable_value = normalized_observable_value


        return normalized_observable_value, resampled_works, resampled_indices


        def binary_search(self,
                      sampler_states,
                      cumulative_works,
                      start_val,
                      end_val,
                      observable,
                      observable_threshold,
                      max_iterations=20,
                      initial_guess = None,
                      precision_threshold = None):
            """
            Given corresponding start_val and end_val of observables, conduct a binary search to find min value for which the observable threshold
            is exceeded.
            Arguments
            ----------
            sampler_states : np.array(openmmtools.states.SamplerState)
                numpy array of sampler states
            cumulative_works : np.array(float)
                cumulative works of corresponding sampler states
            start_val: float
                start value of binary search
            end_val: float
                end value of binary search
            observable : function
                function to compute an observable
            observable_threshold : float
                the threshold of the observable used to satisfy the binary search criterion
            max_iterations: int, default 20
                maximum number of interations to conduct
            initial_guess: float, default None
                guess where the threshold is achieved
            precision_threshold: float, default None
                precision threshold below which, the max iteration will break

            Returns
            -------
            midpoint: float
                maximum value that doesn't exceed threshold
            _observable : float
                observed value of observable
            """
            _base_end_val = end_val
            _logger.debug(f"\t\t\tmin, max values: {start_val}, {end_val}. ")
            self.thermodynamic_state.set_alchemical_parameters(start_val, LambdaProtocol(functions = self.lambda_protocol))
            current_rps = np.array([compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])

            if initial_guess is not None:
                midpoint = initial_guess
            else:
                midpoint = (start_val + end_val) * 0.5
            _logger.debug(f"\t\t\tinitial midpoint is: {midpoint}")

            for _ in range(max_iterations):
                _logger.debug(f"\t\t\titeration {_}: current lambda: {midpoint}")
                self.thermodynamic_state.set_alchemical_parameters(midpoint, LambdaProtocol(functions = self.lambda_protocol))
                new_rps = np.array([compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])
                _observable = observable(cumulative_works, new_rps - current_rps) / len(current_rps)
                _logger.debug(f"\t\t\tobservable: {_observable}")
                if _observable <= observable_threshold:
                    _logger.debug(f"\t\t\tobservable {_observable} <= observable_threshold {observable_threshold}")
                    end_val = midpoint
                else:
                    _logger.debug(f"\t\t\tobservable {_observable} > observable_threshold {observable_threshold}")
                    start_val = midpoint
                midpoint = (start_val + end_val) * 0.5
                if precision_threshold is not None:
                    if abs(_base_end_val - midpoint) <= precision_threshold:
                        _logger.debug(f"\t\t\tthe difference between the original max val ({_base_end_val}) and the midpoint is less than the precision_threshold ({precision_threshold}).  Breaking with original max val.")
                        midpoint = _base_end_val
                        thermodynamic_state = self.modify_thermodynamic_state(thermodynamic_state, current_lambda = midpoint)
                        new_rps = np.array([feptasks.compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])
                        _observable = observable(cumulative_works, new_rps - current_rps) / len(current_rps)
                        break
                    elif abs(end_val - start_val) <= precision_threshold:
                        _logger.debug(f"\t\t\tprecision_threshold: {precision_threshold} is exceeded.  Breaking")
                        midpoint = end_val
                        thermodynamic_state = self.modify_thermodynamic_state(thermodynamic_state, current_lambda = midpoint)
                        new_rps = np.array([feptasks.compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])
                        _observable = observable(cumulative_works, new_rps - current_rps) / len(current_rps)
                        break

            return midpoint, _observable


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
                   integrator = 'langevin'):

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
            raise Exception(f"integrator {integrator} is not supported. supported integrators include {supported_integrators}")

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

        #set a bool variable for pass or failure
        self.succeed = True
        return True

    def anneal(self,
               sampler_state,
               lambdas,
               noneq_trajectory_filename = None,
               num_integration_steps = 1,
               return_timer = False,
               return_sampler_state = False,
               rethermalize = False,
               initial_propagation = True):
        """
        conduct annealing across lambdas.

        Arguments
        ---------
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
        initial_propagation : bool, default True
            whether to take an initial propagation step before a proposal/weight

        Returns
        -------
        incremental_work : np.array of shape (1, len(lambdas) - 1)
            cumulative works for every lambda
        sampler_state : openmmtools.states.SamplerState
            configuration at last lambda after proposal
        timer : np.array
            timers
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

        incremental_work = np.zeros(len(lambdas) - 1)
        #first set the thermodynamic state to the proper alchemical state and pull context, integrator
        self.thermodynamic_state.set_alchemical_parameters(lambdas[0], lambda_protocol = self.lambda_protocol_class)
        self.context, integrator = self.context_cache.get_context(self.thermodynamic_state, self.integrator)
        if initial_propagation:
            sampler_state.apply_to_context(self.context, ignore_velocities=True)
            self.context.setVelocitiesToTemperature(self.thermodynamic_state.temperature)
            integrator.step(num_integration_steps) #we have to propagate the start state
        else:
            sampler_state.apply_to_context(self.context, ignore_velocities=False)

        for idx, _lambda in enumerate(lambdas[1:]): #skip the first lambda
            try:
                if return_timer:
                    start_timer = time.time()
                incremental_work[idx] = self.compute_incremental_work(_lambda)
                integrator.step(num_integration_steps)
                if rethermalize:
                    self.context.setVelocitiesToTemperature(self.thermodynamic_state.temperature) #rethermalize
                if noneq_trajectory_filename is not None:
                    self.save_configuration(idx, sampler_state, context)
                if return_timer:
                    timer[idx] = time.time() - start_timer
            except Exception as e:
                print(f"failure: {e}")
                return e

        self.attempt_termination(noneq_trajectory_filename)

#         try:
#             _logger.debug(f"\t\t\t\tintegrator acceptance rate: {integrator.acceptance_rate}")
#         except:
#             pass

        #pull the last sampler state and return
        if return_sampler_state:
            if rethermalize:
                sampler_state.update_from_context(self.context, ignore_velocities=True)
            else:
                sampler_state.update_from_context(self.context, ignore_velocities=True)

            return (incremental_work, sampler_state, timer)
        else:
            return (incremental_work, None, timer)



    def attempt_termination(self, noneq_trajectory_filename):
        """
        Attempt to terminate the annealing protocol and return the Particle attributes.
        """
        if noneq_trajectory_filename is not None:
            _logger.info(f"saving configuration")
            trajectory = md.Trajectory(np.array(self._trajectory_positions), self.topology, unitcell_lengths=np.array(self._trajectory_box_lengths), unitcell_angles=np.array(self._trajectory_box_angles))
            write_nonequilibrium_trajectory(trajectory, noneq_trajectory_filename)

        self._trajectory_positions = []
        self._trajectory_box_lengths = []
        self._trajectory_box_angles = []


    def compute_incremental_work(self, _lambda):
        """
        compute the incremental work of a lambda update on the thermodynamic state.
        function also updates the thermodynamic state and the context
        """
        old_rp = self.beta * self.context.getState(getEnergy=True).getPotentialEnergy()

        #update thermodynamic state and context
        self.thermodynamic_state.set_alchemical_parameters(_lambda, lambda_protocol = self.lambda_protocol_class)
        self.thermodynamic_state.apply_to_context(self.context)
        new_rp = self.beta * self.context.getState(getEnergy=True).getPotentialEnergy()
        _incremental_work = new_rp - old_rp

        return _incremental_work

    def save_configuration(self, iteration, sampler_state, context):
        """
        pass a conditional save function
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


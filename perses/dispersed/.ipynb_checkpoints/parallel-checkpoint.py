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
_logger = logging.getLogger("parallelism")
_logger.setLevel(logging.DEBUG)

class Parallelism(object):
    supported_libraries = {'dask': ['LSF']}

    def activate_client(self,
                        library = ('dask', 'LSF'),
                        num_processes = 2,
                        timeout = 1800):
        """
        Parameters
        ----------
        library : tuple(str, str), default ('dask', 'LSF')
            parallelism and scheduler tuple
        num_processes : int or None
            number of workers to run with the new client
            if None, num_processes will be adaptive
        timeout : int
            number of seconds to wait to fulfill the workers order
        """
        assert library[0] in list(self.supported_libraries.keys()), f"{library[0]} is not a supported parallelism. (supported parallelisms are {self.supported_libraries.keys()})"
        assert library[1] in list(self.supported_libraries[library[0]]), f"{library[1]} is not a supported . (supported parallelisms are {self.supported_libraries[library[0]]})"
        self.library = library
        if not library:
            self.client = None
            self._adapt = False
            self.num_processes = 0
            self.workers = {}

        elif library[0] == 'dask':
            if library[1] == 'LSF':
                from dask_jobqueue import LSFCluster
                cluster = LSFCluster()
                if num_processes is None:
                    self._adapt = True
                    cluster.adapt(minimum = 1, interval = '1s')
                else:
                    self._adapt = False
                    self.num_processes = num_processes
                    cluster.scale(self.num_processes)

                self.client = distributed.Client(cluster, timeout = timeout)
                if not self._adapt:
                    while len(self.client.nthreads()) != self.num_processes:
                        time.sleep(5)
                worker_threads = self.client.nthreads()
                self.workers = {i: _worker for i, _worker in zip(range(len(worker_threads)), worker_threads.keys())}
            else:
                raise Exception(f"{library[1]} is supported, but without client-activation functionality!")

        # if LSF:
        #     from dask_jobqueue import LSFCluster
        #     cluster = LSFCluster()
        #     self._adapt = adapt
        #     self.num_processes = num_processes
        #
        #     if self._adapt:
        #         _logger.debug(f"adapting cluster from 1 to {self.num_processes} processes")
        #         cluster.adapt(minimum = 2, maximum = self.num_processes, interval = "1s")
        #     else:
        #         _logger.debug(f"scaling cluster to {self.num_processes} processes")
        #         cluster.scale(self.num_processes)
        #
        #     _logger.debug(f"scheduling cluster with client")
        #     self.client = distributed.Client(cluster, timeout = timeout)
        #     while len(self.client.nthreads()) != self.num_processes:
        #         _logger.debug(f"workers: {self.client.nthreads()}.  waiting for {self.num_processes} workers...")
        #         time.sleep(3)
        #     worker_threads = self.client.nthreads()
        #     self.workers = {i: _worker for i, _worker in zip(range(len(worker_threads)), worker_threads.keys())}
        #     self.worker_counter = 0
        #
        #     #now we wait for all of our workers.
        # else:
        #     self.client = None
        #     self._adapt = False
        #     self.num_processes = 0

    def deactivate_client(self):
        """
        NonequilibriumSwitchingFEP is not pickleable with the self.client or self.cluster activated.
        This must be called before pickling
        """
        if self.library is not None:
            if self.library[0] == 'dask':
                if self.client is not None:
                    self.client.close()
                else:
                    _logger.warning(f"the client is NoneType.")
        else:
            _logger.warning(f"the library is NoneType.")
        
        _attrs_to_delete = ['library', 'client', '_adapt', 'num_processes', 'workers']
        for _attr in _attrs_to_delete:
            delattr(self, _attr)

    def scatter(self, df, workers = None):
        """
        wrapper to scatter the local data df
        """
        if self.client is None:
            #don't actually scatter
            return df
        else:
            if self.library[0] == 'dask':
                if workers is None:
                    return self.client.scatter(df)
                else:
                    return self.client.scatter(df, workers)
            else:
                raise Exception(f"the client is not NoneType but the library is not supported")

    def deploy(self, func, arguments, workers = None):
        """
        wrapper to map a function and its arguments to the client for scheduling

        Arguments
        ---------
        func : function to map
            arguments: tuple of the arguments that the function will take
        argument : tuple of argument lists, default None
            if None, then the default workers are all workers

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
            if workers is None:
                _workers = list(self.workers.values())
            if self.library[0] == 'dask':
                futures = self.client.map(func, *arguments, workers = _workers)
            else:
                raise Exception(f"{self.library} is supported, but without deployment functionality!")

        return futures

    def run_all(self, func, arguments, workers):
        """
        wrapper to run a function and its arguments to the client for scheduling.
        the same function and arguments is run on all workers

        Arguments
        ---------
        func : function to map
            arguments: tuple of the arguments that the function will take
        argument : tuple of arguments

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
            if self.library[0] == 'dask':
                futures = self.client.run(func, *arguments, workers = workers)
            else:
                raise Exception(f"{self.library} is supported, but without deployment functionality!")

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
            if self.library[0] == 'dask':
                results = self.client.gather(futures)
                return results
            else:
                raise Exception(f"{self.library} is supported, but without gather-results functionality!")

    def gather_actor_result(self, future):
        """
        wrapper to pull the .result() of a method called to an actor
        """
        if self.client is None:
            return future
        else:
            if self.library[0] == 'dask':
                distributed.progress(future)
                result = future.result()
                return result
            else:
                raise Exception(f"{self.library} is supported, but without actor-gather functionality!")

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
            if self.library[0] == 'dask':
                future = self.client.submit(_class, workers = [self.workers[self.worker_counter]], actor=True)  # Create a _class on a worker
                distributed.progress(future)
                actor = future.result()                    # Get back a pointer to that object
                return actor
            else:
                raise Exception(f"{self.library} is supported, but without actor launch functionality!")
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
            if self.library[0] == 'dask':
                distributed.progress(futures)
            else:
                raise Exception(f"{self.library} is supported, but without actor launch functionality!")


    def wait(self, futures):
        """
        wrapper to wait until futures are complete.
        """
        if self.client is None:
            pass
        else:
            distributed.wait(futures)

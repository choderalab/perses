from typing import List, Tuple, Union, NamedTuple
import os
import copy
import numpy as np
import pickle
import tqdm
import logging
import tqdm
from sys import getsizeof
import time
from collections import namedtuple
import random
import dask.distributed as distributed
import tqdm
import time
from scipy.special import logsumexp
from dask_jobqueue import LSFCluster
from dask_jobqueue.lsf import lsf_detect_units, lsf_format_bytes_ceil
from simtk import unit

# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("parallelism")
_logger.setLevel(logging.INFO)

class Parallelism(object):
    """
    This class maintains a running parallelism (with support for the parallelism libraries and schedulers
    specified in `supported libraries`).
    The class can currently support the following generalized parallel functions:
        - cluster and client activation/deactivation and maintenance
        - scatter local data to distributed memory
        - deploy/gather a function and list arguments to distributed workers
        - deploy/gather a function with a single set of appropriate arguments to all workers
        - launch and perform operations on actors
        - block until computation is complete with 'wait' or monitor progress
    """
    supported_libraries = {'dask': ['LSF']}

    def activate_client(self,
                        library = ('dask', 'LSF'),
                        num_processes = 2,
                        timeout = 1800,
                        processor = 'gpu'):
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
        processor : str
            current support is 'gpu' or 'cpu'
        """
        self.library = library
        if library is not None:
            _logger.debug(f"library is not None")
            assert library[0] in list(self.supported_libraries.keys()), f"{library[0]} is not a supported parallelism. (supported parallelisms are {self.supported_libraries.keys()})"
            assert library[1] in list(self.supported_libraries[library[0]]), f"{library[1]} is not a supported . (supported parallelisms are {self.supported_libraries[library[0]]})"
        elif library is None:
            _logger.debug(f"library is None")
            self.client = None
            self._adapt = False
            self.num_processes = 0
            self.workers = {}
            return

        if library[0] == 'dask':
            _logger.debug(f"detected dask parallelism...")
            if library[1] == 'LSF':
                _logger.debug(f"detected LSF scheduler")
                backend = LSFDaskBackend()
                _logger.debug(f"creating cluster...")
                cluster = backend.create_cluster(processor = processor)
                if num_processes is None:
                    _logger.debug(f"adaptive cluster")
                    self._adapt = True
                    cluster.adapt(minimum = 1, interval = '1s')
                else:
                    _logger.debug(f"nonadaptive cluster")
                    self._adapt = False
                    self.num_processes = num_processes
                    cluster.scale(self.num_processes)

                _logger.debug(f"creating client with cluster")
                self.client = distributed.Client(cluster, timeout = timeout)
                if not self._adapt:
                    while len(self.client.nthreads()) != self.num_processes:
                        _logger.debug(f"waiting for worker request fulfillment...")
                        time.sleep(5)
                worker_threads = self.client.nthreads()
                self.workers = {i: _worker for i, _worker in zip(range(len(worker_threads)), worker_threads.keys())}
                _logger.debug(f"workers initialized: {self.workers}")
            else:
                raise Exception(f"{library[1]} is supported, but without client-activation functionality!")

    def deactivate_client(self):
        """
        deactivate a cluster that is maintained by a local client.  attributes associated with
        client instantiation are deleted.
        """
        _logger.debug(f"attempting to deactivate client...")
        if self.library is not None:
            _logger.debug(f"library ({self.library}) is not None; attempting to close client")
            if self.library[0] == 'dask':
                _logger.debug(f"detected dask parallelism...")
                if self.client is not None:
                    _logger.debug(f"closing client...")
                    self.client.cluster.close()
                    self.client = None
                    _logger.debug(f"client closed successfully")
                else:
                    _logger.warning(f"the client is NoneType.")
        else:
            _logger.warning(f"the library is NoneType.")

        _attrs_to_delete = ['library', 'client', '_adapt', 'num_processes', 'workers']
        assert self.client is None, f"the client is not None!"
        for _attr in _attrs_to_delete:
            _logger.debug(f"deleting parallelism attribute {_attr}")
            delattr(self, _attr)

    def scatter(self, df, workers = None):
        """
        wrapper to scatter the local data to distributed memory

        Arguments
        ---------
        df : object
            any python object to be distributed to workers
        workers : list of str
            worker addresses

        Return
        ------
        scatter_future : <generalized> future
            scattered future
        """
        if self.client is None:
            #don't actually scatter
            return df
        else:
            if self.library[0] == 'dask':
                if workers is None:
                    scatter_future = self.client.scatter(df)
                    return scatter_future
                else:
                    scatter_future = self.client.scatter(df, workers)
                    return scatter_future
            else:
                raise Exception(f"the client is not NoneType but the library is not supported")

    def deploy(self, func, arguments, workers = None):
        """
        wrapper to map a function and its arguments to the client for scheduling

        Arguments
        ---------
        func : function
            python function to distribute
        arguments : tuple of lists, default None
            if None, then the default workers are all workers
        workers : list of str, default None
            worker address list

        Returns
        ---------
        futures: <generalized> future object
            futures of the map
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
        distribute single function with single set of arguments to all workers

        Arguments
        ---------
        func : function
            python function to distribute
        arguments : tuple of args, default None
            if None, then the default workers are all workers
        workers : list of str, default None
            worker address list

        Returns
        ---------
        futures: <generalized> future object
            futures of the map
        """
        if self.client is None:
            futures = func(*arguments)
        else:
            if self.library[0] == 'dask':
                futures = self.client.run(func, *arguments, workers = workers)
            else:
                raise Exception(f"{self.library} is supported, but without deployment functionality!")

        return futures

    def gather_results(self, futures, omit_errors = False):
        """
        wrapper to gather a function given its arguments

        Arguments
        ---------
        futures : list of <generalized> futures
            futures that are to be gathered
        omit_errors : bool, default False
            whether to skip or raise errors from the workers.
            WARNING: this fails if self.client is None (i.e. there is no distributed scheduler).
                     If this is the case, then the function/method pointer to `futures` must be safe

        Returns
        ---------
        results: <generalized> function output
            the results of the futures
        """
        if self.client is None:
            return futures
        else:
            if self.library[0] == 'dask':
                _errors = 'raise' if not omit_errors else 'skip'
                results = self.client.gather(futures, errors = _errors)
                return results
            else:
                raise Exception(f"{self.library} is supported, but without gather-results functionality!")

    def gather_actor_result(self, future):
        """
        wrapper to pull the .result() of a method called to an actor

        Arguments
        ---------
        future : <generalized> future
            the future object to be collected from an actor
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

        Arguments
        ---------
        futures : list of <generalized> futures
            futures that are to be gathered
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

        Arguments
        ---------
        futures : list of <generalized> futures
            futures that are to be gathered
        """
        if self.client is None:
            pass
        else:
            distributed.wait(futures)

    @staticmethod
    def get_remote_worker(remote_worker, library = ('dask', 'LSF')):
        """
        Staticmethod to attempt to gather the worker from a remote given a library on the remote.

        Arguments
        ---------
        remote_worker : bool
            whether we are attempting to pull a worker from a remote or otherwise
        library : tup('str', 'str'), default ('dask', 'LSF')
            specification of the parallelism and scheduler, respectively

        Returns
        -------
        _worker : generalized worker or None
            if remote_worker is True, then the parallelism/scheduler will attempt to pull the worker class;
            otherwise, None is returned
        """
        parallelism, scheduler = library[0], library[1]
        if remote_worker == True:
            try:
                if parallelism == 'dask':
                    if scheduler == 'LSF':
                        _worker = distributed.get_worker()
                    else:
                        raise Exception(f"the scheduler {scheduler} is not supported")
                else:
                    raise Exception(f"the parallelism {parallelism} is not supported")
            except Exception as e:
                _logger.warning(e)
                _worker = None
        else:
            _worker = None

        return _worker


class LSFDaskBackend(object):
    """
    dask-specific LSF backend for building a Parallelism
    """
    from dask_jobqueue.lsf import lsf_detect_units, lsf_format_bytes_ceil
    supported_processors = {
                            'gpu':
                                   {'queue_name': 'gpuqueue',
                                    'cores': 1,
                                    'walltime': '04:00',
                                    'memory': f"{lsf_format_bytes_ceil(6e9, lsf_units=lsf_detect_units())}{lsf_detect_units().upper()}",
                                    'mem': 6e9,
                                    'job_extra': ['-gpu num=1:j_exclusive=yes:mode=shared:mps=no:', '-m "ls-gpu lt-gpu"'],
                                    'env_extra': ['module load cuda/9.2'],
                                    'extra': ['--no-nanny'],
                                    'local_directory': 'dask-worker-space'},
                            'cpu':
                                   {'queue_name': 'cpuqueue',
                                    'cores': 2,
                                    'walltime': '04:00',
                                    'memory': f"{lsf_format_bytes_ceil(3e9, lsf_units=lsf_detect_units())}{lsf_detect_units().upper()}",
                                    'mem': 6e9,
                                    'job_extra': None,
                                    'env_extra': ['module load cuda/9.2'],
                                    'extra': ['--no-nanny'],
                                    'local_directory': 'dask-worker-space'}
                                    }
    def modify_lsf_cluster(processor, attribute):
        """
        give a processor and an attribute to modify

        Arguments
        ---------
        processor : str
            one of the supported processors
        attrbute : dict
            dict of the attribute to change in self.supported_processors[processor].keys()
        """
        assert processor in list(self.supported_processors.keys()), f"processor {processor} is not supported"
        assert set(attribute.keys()).issubset(set(self.supported_processors[processor].keys())), f"the attributes {attribute.keys()} is not supported."
        self.supported_processors[processor].update(attribute)

    def create_cluster(processor):
        """
        return an LSF cluster given the self.supported_processors and the number of workers

        Arguments
        ---------
        processor : str
            one of the supported processors

        Returns
        -------
        cluster : LSFCluster
            a cluster given a processor
        """
        attributes = self.supported_processors[processor]
        cluster = LSFCluster(queue = attributes['queue_name'],
                             cores = attributes['cores'],
                             walltime = attributes['walltime'],
                             memory = attributes['memory'],
                             mem = attributes['mem'],
                             job_extra = attributes['job_extra'],
                             env_extra = attributes['env_extra'],
                             extra = attributes['extra'],
                             local_directory = attributes['local_directory'])
        return cluster

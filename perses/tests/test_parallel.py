###########################################
# IMPORTS
###########################################
from simtk.openmm import app
from simtk import unit, openmm
import numpy as np
import os
from nose.tools import nottest
from unittest import skipIf

import copy
import pymbar

###
from perses.dispersed import parallel
###

istravis = os.environ.get('TRAVIS', None) == 'true'



def test_Parallelism_local():
    """
    following function will create a local Parallelism instance and run all of the used methods.
    """
    _parallel = parallel.Parallelism()

    #test client activation
    _parallel.activate_client(library = None, num_processes = None)

    data = np.arange(10)

    run_parallelism(_parallel, data)

@skipIf(istravis, "Skip helper function on travis")
def test_Parallelism_distributed():
    """
    following function will create a distributed Parallelism instance and run all of the used methods.
    Note : this can only be run on a nosetest since travis cannot access dask_jobqueue or any python distributed libraries.
    """
    _parallel = parallel.Parallelism()

    #test client activation
    _parallel.activate_client(library = ('dask', 'LSF'), num_processes = 2)
    data = np.arange(10)
    run_parallelism(_parallel, data)









@nottest
@skipIf(istravis, "Skip helper function on travis")
def run_parallelism(_parallel, data):
    """
    helper function to run through the parallelism tests

    Arguments
    ---------
    _parallel : perses.dispersed.parallelism.Parallelism
        parallelism object to run tests on
    data : np.array
        test python object to distribute
    """
    #test scatter
    df = _parallel.scatter(data)

    #test deploy
    futures = _parallel.deploy(dummy_function,
                              (df,),
                              workers = list(_parallel.workers.values()))
    #progress
    _parallel.progress(futures)

    #wait
    _parallel.wait(futures)

    #collect deployment
    results = _parallel.gather_results(futures)

    #attempt a run all
    run_all_futures = _parallel.run_all(dummy_function,
                                        (data,),
                                        workers = list(_parallel.workers.values()))


@nottest
@skipIf(istravis, "Skip helper function on travis")
def dummy_function(_arg):
    """
    dummy function to distribute;
    """
    return _arg

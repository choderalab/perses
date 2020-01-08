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
    assert hasattr(_parallel, 'library')
    assert _parallel.client == None
    assert _parallel._adapt == False
    assert _parallel.num_processes == 0
    assert _parallel.workers == {}

    data = np.arange(10)

    run_parallelism(_parallel, data)


# Run this test on a cluster to test parallelism
# skipping for GH and travis
#@skipIf(istravis, "Skip helper function on travis")
#def test_Parallelism_distributed():
#    """
#    following function will create a distributed Parallelism instance and run all of the used methods.
#    Note : this can only be run on a nosetest since travis cannot access dask_jobqueue or any python distributed libraries.
#    """
#    _parallel = parallel.Parallelism()
#
#    #test client activation
#    _parallel.activate_client(library = ('dask', 'LSF'), num_processes = 2)
#    data = np.arange(10)
#    run_parallelism(_parallel, data)


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
    _remote = False if _parallel.client == None else True
    #test scatter
    df = _parallel.scatter(data)
    if not _remote:
        assert all(i == j for i,j in zip(df, data)), f"local worker but scattered data is not identical to data"
    else:
        pass


    #test deploy
    futures = _parallel.deploy(dummy_function,
                              (df,),
                              workers = list(_parallel.workers.values()))
    if not _remote:
        locals = [dummy_function(i) for i in data]
        assert all(i == j for i, j in zip(locals, futures)), f"futures is {futures}, but should be {[dummy_function(i) for i in data]}"
    else:
        pass

    #progress
    _parallel.progress(futures) #this is a dummy function for local parallelism
    if not _remote:
        pass
    else:
        pass

    #wait
    _parallel.wait(futures) #this is a dummy function for local parallelism
    if not _remote:
        pass
    else:
        pass

    #collect deployment
    results = _parallel.gather_results(futures)
    if not _remote:
        assert results == [dummy_function(i) for i in data], f"futures is {futures}, but should be {[dummy_function(i) for i in data]}"
    else:
        pass


    #attempt a run all
    run_all_futures = _parallel.run_all(dummy_function,
                                        (data,),
                                        workers = list(_parallel.workers.values()))
    if not _remote:
        locals = [dummy_function(i) for i in data]
        assert all(i == j for i, j in zip(locals, run_all_futures)), f"run_all is returning {run_all_futures} instead of {dummy_function(data)}"


@nottest
@skipIf(istravis, "Skip helper function on travis")
def dummy_function(_arg):
    """
    dummy function to distribute;
    """
    return _arg

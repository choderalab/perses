"""
Test storage layer.

TODO:
* Write tests

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
import logging
import tempfile
from functools import partial

from perses.storage import NetCDFStorage
import perses.tests.testsystems

import perses.rjmc.topology_proposal as topology_proposal
import perses.bias.bias_engine as bias_engine
import perses.rjmc.geometry as geometry
import perses.annihilation.ncmc_switching as ncmc_switching

################################################################################
# TEST STORAGE
################################################################################

def test_storage_create():
    """Test storage layer creating a new file.
    """
    tmpfile = tempfile.NamedTemporaryFile()
    storage = NetCDFStorage(tmpfile.name, mode='w')
    storage.close()

def test_storage_append():
    """Test storage layer appending to a file.
    """
    tmpfile = tempfile.NamedTemporaryFile()
    storage = NetCDFStorage(tmpfile.name, mode='w')
    storage.close()
    storage = NetCDFStorage(tmpfile.name, mode='a')
    storage.close()

def test_write_quantity():
    """Test writing of a quantity.
    """
    tmpfile = tempfile.NamedTemporaryFile()
    storage = NetCDFStorage(tmpfile.name, mode='w')

    storage.write_quantity('envname', 'modname', 'singleton', 1.0)

    for iteration in range(10):
        storage.write_quantity('envname', 'modname', 'varname', float(iteration), iteration=iteration)

    for iteration in range(10):
        assert (storage._ncfile['/envname/modname/varname'][iteration] == float(iteration))

def test_sync():
    """Test writing of a quantity.
    """
    tmpfile = tempfile.NamedTemporaryFile()
    storage = NetCDFStorage(tmpfile.name, mode='w')
    storage.write_quantity('envname', 'modname', 'singleton', 1.0)
    storage.sync()

def test_storage_with_samplers():
    """Test storage layer inside all samplers.
    """
    testsystem_names = ['ValenceSmallMoleculeLibraryTestSystem']
    niterations = 5 # number of iterations to run

    for testsystem_name in testsystem_names:
        # Create storage.
        tmpfile = tempfile.NamedTemporaryFile()
        storage = NetCDFStorage(tmpfile.name, mode='w')

        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        # Instantiate test system.
        testsystem = testsystem_class()
        # Test MCMCSampler samplers.
        for environment in testsystem.environments:
            mcmc_sampler = testsystem.mcmc_samplers[environment]
            mcmc_sampler.storage = storage
            f = partial(mcmc_sampler.run, niterations)
            f.description = "Testing MCMC sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test ExpandedEnsembleSampler samplers.
        for environment in testsystem.environments:
            exen_sampler = testsystem.exen_samplers[environment]
            exen_sampler.storage = storage
            f = partial(exen_sampler.run, niterations)
            f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            sams_sampler.storage = storage
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler, if present.
        if hasattr(testsystem, 'designer') and (testsystem.designer is not None):
            f = partial(testsystem.designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f

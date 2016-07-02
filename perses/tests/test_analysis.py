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
import cPickle as pickle

from perses.storage import NetCDFStorage, NetCDFStorageView
import perses.tests.testsystems
from perses.analysis import Analysis

################################################################################
# TEST ANALYSIS
################################################################################

def test_analysis():
    """Test analysis tools.
    """
    testsystem_names = ['T4LysozymeInhibitorsTestSystem']
    niterations = 5 # number of iterations to run

    for testsystem_name in testsystem_names:
        # Create storage.
        tmpfile = tempfile.NamedTemporaryFile()
        storage_filename = tmpfile.name

        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)

        # Instantiate test system.
        testsystem = testsystem_class(storage_filename=storage_filename)

        # Alter settings
        for environment in testsystem.environments:
            testsystem.mcmc_samplers[environment].verbose = False
            testsystem.mcmc_samplers[environment].nsteps = 5 # use fewer MD steps to speed things up
            testsystem.exen_samplers[environment].verbose = False
            testsystem.exen_samplers[environment].ncmc_engine.nsteps = 5 # NCMC switching
            testsystem.sams_samplers[environment].verbose = False
        testsystem.designer.verbose = False
        testsystem.designer.run(niterations=5)

        # Analyze file.
        # TODO: Use temporary filenames
        analysis = Analysis(storage_filename)
        analysis.plot_ncmc_work('ncmc.pdf')

if __name__ == '__main__':
    analysis = Analysis('output.nc')
    analysis.plot_ncmc_work('ncmc.pdf')

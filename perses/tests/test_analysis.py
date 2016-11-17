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
import json

from perses.storage import NetCDFStorage, NetCDFStorageView
import perses.tests.testsystems
from perses.analysis import Analysis

################################################################################
# TEST ANALYSIS
################################################################################

def test_analysis():
    """Test analysis tools.
    """
    testsystem_names = ['ValenceSmallMoleculeLibraryTestSystem']

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

        # Run test simulations.
        niterations = 5 # just a few iterations
        if testsystem.designer is not None:
            # Run the designer
            testsystem.designer.verbose = False
            testsystem.designer.run(niterations=niterations)
        else:
            # Run the SAMS samplers.
            for environment in testsystem.environments:
                testsystem.sams_samplers[environment].run(niterations=niterations)

        # Analyze file.
        # TODO: Use temporary filenames
        analysis = Analysis(storage_filename)
        analysis.plot_ncmc_work('ncmc.pdf')

if __name__ == '__main__':
    #analysis = Analysis('output-10000.nc')
    #analysis.plot_ncmc_work('ncmc-10000.pdf')
    test_analysis()

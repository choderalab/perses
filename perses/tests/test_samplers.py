"""
Samplers for perses automated molecular design.

TODO:
* Refactor tests into a test class so that AlanineDipeptideSAMS test system only needs to be constructed once for a battery of tests.
* Generalize tests of samplers to iterate over all PersesTestSystem subclasses

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
from functools import partial

import perses.tests.testsystems

import perses.rjmc.topology_proposal as topology_proposal
import perses.bias.bias_engine as bias_engine
import perses.rjmc.geometry as geometry
import perses.annihilation.ncmc_switching as ncmc_switching

################################################################################
# TEST MCMCSAMPLER
################################################################################

def test_valence():
    """
    Test valence-only test system.
    """
    # TODO: Test that proper statistics (equal sampling, zero free energy differences) are obtained.

    testsystem_names = ['ValenceSmallMoleculeLibraryTestSystem']
    niterations = 2 # number of iterations to run
    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        # Instantiate test system.
        testsystem = testsystem_class()
        # Test ExpandedEnsembleSampler samplers.
        #for environment in testsystem.environments:
        #    exen_sampler = testsystem.exen_samplers[environment]
        #    f = partial(exen_sampler.run, niterations)
        #    f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
        #    yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            testsystem.exen_samplers[environment].pdbfile = open('sams.pdb', 'w') # DEBUG
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler for implicit hydration free energy
        from perses.samplers.samplers import MultiTargetDesign
        # Construct a target function for identifying mutants that maximize the peptide implicit solvent hydration free energy
        for environment in testsystem.environments:
            target_samplers = { testsystem.sams_samplers[environment] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
            designer = MultiTargetDesign(target_samplers)
            f = partial(designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f

def test_samplers():
    """
    Test samplers on multiple test systems.
    """
    testsystem_names = ['T4LysozymeMutationTestSystem', 'ValenceSmallMoleculeLibraryTestSystem', 'T4LysozymeInhibitorsTestSystem', 'KinaseInhibitorsTestSystem', 'AlkanesTestSystem', 'AlanineDipeptideTestSystem', 'AblImatinibResistanceTestSystem', 'AblAffinityTestSystem']
    niterations = 5 # number of iterations to run

    testsystem_names = ['T4LysozymeInhibitorsTestSystem']
    # If TESTSYSTEMS environment variable is specified, test those systems.
    if 'TESTSYSTEMS' in os.environ:
        testsystem_names = os.environ['TESTSYSTEMS'].split(' ')

    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        # Instantiate test system.
        testsystem = testsystem_class()
        # Test MCMCSampler samplers.
        for environment in testsystem.environments:
            mcmc_sampler = testsystem.mcmc_samplers[environment]
            f = partial(mcmc_sampler.run, niterations)
            f.description = "Testing MCMC sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test ExpandedEnsembleSampler samplers.
        for environment in testsystem.environments:
            exen_sampler = testsystem.exen_samplers[environment]
            f = partial(exen_sampler.run, niterations)
            f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler for implicit hydration free energy
        from perses.samplers.samplers import MultiTargetDesign
        # Construct a target function for identifying mutants that maximize the peptide implicit solvent hydration free energy
        for environment in testsystem.environments:
            target_samplers = { testsystem.sams_samplers[environment] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
            designer = MultiTargetDesign(target_samplers)
            f = partial(designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f


if __name__=="__main__":
    for t in test_samplers():
        print(t.description)
        if(t.description) == "Testing MultiTargetDesign sampler with T4LysozymeInhibitorsTestSystem transfer free energy from vacuum -> vacuum":
            t()

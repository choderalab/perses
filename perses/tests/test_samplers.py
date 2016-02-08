"""
Samplers for perses automated molecular design.

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

import perses.tests.testsystems

import perses.rjmc.topology_proposal as topology_proposal
import perses.bias.bias_engine as bias_engine
import perses.rjmc.geometry as geometry
import perses.annihilation.ncmc_switching as ncmc_switching

################################################################################
# TEST SAMPLERS
################################################################################

def test_peptide_mutations_implicit():
    """
    Test mutation of alanine dipeptide in implcit solvent hydration free energy calculation.
    """
    # Retrieve the test system.
    from perses.tests.testsystems import AlanineDipeptideSAMS
    testsystem = AlanineDipeptideSAMS()
    # Create the design sampler
    from perses.samplers.samplers import MultiTargetDesign
    # Construct a target function for identifying mutants that maximize the peptide implicit solvent hydration free energy
    target_samplers = { testsystem.sams_samplers['implicit'] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
    # Set up the design sampler.
    designer = MultiTargetDesign(target_samplers)
    # Run the designer engine
    designer.run()

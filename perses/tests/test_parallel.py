###########################################
# IMPORTS
###########################################
from simtk.openmm import app
from simtk import unit, openmm
import numpy as np
import os
from nose.tools import nottest

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from perses.annihilation.relative import HybridTopologyFactory
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, SystemGenerator, TopologyProposal
from perses.tests import utils
import openeye.oechem as oechem
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
import openmmtools.mcmc as mcmc
import openmmtools.cache as cache
from unittest import skipIf

import pymbar.timeseries as timeseries

import copy
import pymbar

###
from perses.dispersed import parallel
###

istravis = os.environ.get('TRAVIS', None) == 'true'

try:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")
except Exception:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")

#############################################
# CONSTANTS
#############################################
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01
ENERGY_THRESHOLD = 1e-1
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CPU")






@skipIf(istravis, "Skip helper function on travis")
def test_Parallelism_local():
    """
    following function will create a local Parallelism instance and run all of the used methods.
    """
    

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
    tmpfile = tempfile.NamedTemporaryFile()
    storage = NetCDFStorage(tmpfile.name, mode='w')
    storage.close()

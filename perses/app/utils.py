import yaml
import numpy as np
import pickle
import os
import sys
import simtk.unit as unit
import logging
from perses.utils.data import load_smi

from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
from perses.annihilation.relative import HybridTopologyFactory
from perses.app.relative_setup import NonequilibriumSwitchingFEP, RelativeFEPSetup
from perses.annihilation.lambda_protocol import LambdaProtocol

from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter, sams, replicaexchange
from perses.utils.smallmolecules import render_atom_mapping
from perses.tests.utils import validate_endstate_energies
from openmoltools import forcefield_generators
from perses.utils.openeye import *

logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("utils")
_logger.setLevel(logging.INFO)

def generate_fully_connected_adjacency_matrix(num_nodes):
    """
    create a fully-connected adjacency_matrix

    Arguments
    ---------
    num_nodes : int
        number of nodes

    Returns
    -------
    adjacency_matrix = np.ndarray(2)
        adjacency_matrix of connection with equal weights
    """
    zeros = np.zeros((num_nodes, num_nodes))
    np.fill_diagonal(zeros, -np.inf)
    return np.zeros((num_nodes, num_nodes))

class Simulation(object):
    """
    Simulation object: maintains API for sampling strategy.
    This class wraps the setup and execution functionality of all flavors of free energy sampling (e.g. nonequilibrium, staged equilibrium)
    simulation methods.

    In the alchemical network, at least one Simulation object will be placed on each phase of each edge of each alchemical transformation.
    The Simulation object will be subsequently called by the internal/external parallelism to sample the free energy along a protocol
    specified by lambda
    """
    common_parameters = {}

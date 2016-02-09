"""
Test systems for perses automated design.

Examples
--------

Alanine dipeptide in various environments (vacuum, implicit, explicit):

>>> from perses.tests.testsystems import AlaninDipeptideSAMS
>>> testsystem = AlanineDipeptideSAMS()
>>> system_generator = testsystem.system_generator['explicit']
>>> sams_sampler = testsystem.sams_sampler['explicit']

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
from functools import partial
from pkg_resources import resource_filename
from openeye import oechem
from openmmtools import testsystems

################################################################################
# CONSTANTS
################################################################################

from perses.samplers.thermodynamics import kB

################################################################################
# TEST SYSTEMS
################################################################################

class SAMSTestSystem(object):
    def __init__(self):
        self.environments = list()
        self.topologies = dict()
        self.positions = dict()
        self.system_generators = dict()
        self.proposal_engines = dict()
        self.thermodynamic_states = dict()
        self.sams_samplers = dict()

class AlanineDipeptideSAMS(SAMSTestSystem):
    """
    Create a consistent set of SAMS samplers useful for testing PointMutationEngine on alanine dipeptide in various solvents.
    This is useful for testing a variety of components.

    Properties
    ----------
    environments : list of str
        Available environments: ['vacuum', 'explicit', 'implicit']
    topologies : dict of simtk.openmm.app.Topolog
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments


    Examples
    --------

    >>> from perses.tests.testsystems import AlanineDipeptideSAMS
    >>> testsystem = AlanineDipeptideSAMS()
    # Build a system
    >>> system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['implicit']

    """
    def __init__(self):
        super(AlanineDipeptideSAMS, self).__init__()
        environments = ['explicit', 'implicit', 'vacuum']

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator(['amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : app.HBonds })
        system_generators['implicit'] = SystemGenerator(['amber99sbildn.xml', 'amber99_obc.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : app.OBC2, 'constraints' : app.HBonds })
        system_generators['vacuum'] = SystemGenerator(['amber99sbildn.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : app.HBonds })

        # Create peptide in solvent.
        from openmmtools.testsystems import AlanineDipeptideExplicit, AlanineDipeptideImplicit, AlanineDipeptideVacuum
        testsystems = dict()
        testsystems['explicit'] = AlanineDipeptideExplicit()
        testsystems['implicit'] = AlanineDipeptideImplicit()
        testsystems['vacuum']    = AlanineDipeptideVacuum()

        # Store topologies and positions.
        topologies = dict()
        positions = dict()
        for environment in environments:
            topologies[environment] = testsystems[environment].topology
            positions[environment] = testsystems[environment].positions

        # Set up the proposal engines.
        from perses.rjmc.topology_proposal import PointMutationEngine
        proposal_metadata = { 'ffxmls' : ['amber99sbildn.xml'] }
        proposal_engines = dict()
        for environment in environments:
            proposal_engines[environment] = PointMutationEngine(system_generators[environment], max_point_mutants=1, proposal_metadata=proposal_metadata)

        # Define thermodynamic state of interest.
        from perses.samplers.thermodynamics import ThermodynamicState
        thermodynamic_states = dict()
        thermodynamic_states['explicit'] = ThermodynamicState(system=testsystems['explicit'].system, temperature=300*unit.kelvin, pressure=1*unit.atmospheres)
        thermodynamic_states['implicit'] = ThermodynamicState(system=testsystems['implicit'].system, temperature=300*unit.kelvin)
        thermodynamic_states['vacuum']   = ThermodynamicState(system=testsystems['vacuum'].system, temperature=300*unit.kelvin)

        # Create SAMS samplers
        from perses.samplers.samplers import SAMSSampler
        sams_samplers = dict()
        for environment in environments:
            sams_samplers[environment] = SAMSSampler(thermodynamic_states[environment],
                system_generator=system_generators[environment], proposal_engine=proposal_engines[environment],
                topology=topologies[environment], positions=positions[environment])

        # Store things.
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.system_generators = system_generators
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.sams_samplers = sams_samplers

def test_AlanineDipeptideSAMS():
    """
    Testing AlanineDipeptideSAMS...
    """
    from perses.tests.testsystems import AlanineDipeptideSAMS
    testsystem = AlanineDipeptideSAMS()
    # Build a system
    system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
    # Retrieve a SAMSSampler
    sams_sampler = testsystem.sams_samplers['implicit']

"""
This is a template for code that creates alchemical eliminations
for NCMC switching between compounds or residues.
"""
from simtk import openmm, unit
import alchemy
import logging
class AlchemicalEliminationEngine(object):
    """
    This class is the template for generating systems with the appropriate atoms alchemically modified

    Arguments
    ---------
    metadata : dict
        metadata for alchemical modification of systems

    Examples
    --------

    Create a null transformation for an alanine dipeptide test system

    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from perses.rjmc.topology_proposal import TopologyProposal
    >>> new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    >>> topology_proposal = TopologyProposal(old_system=testsystem.system, old_topology=testsystem.topology, old_positions=testsystem.positions, new_system=testsystem.system, new_topology=testsystem.topology, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())

    Create some alchemical systems

    >>> engine = AlchemicalEliminationEngine()
    >>> alchemical_system_create = engine.make_alchemical_system(testsystem.system, topology_proposal, direction='insert')
    >>> alchemical_system_delete = engine.make_alchemical_system(testsystem.system, topology_proposal, direction='delete')

    """

    def __init__(self, metadata=None):
        pass

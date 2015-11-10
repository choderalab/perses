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
    >>> from perses.rjmc import topology_proposal
    >>> new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    >>> topology_proposal = topology_proposal.TopologyProposal(old_topology=testsystem.topology, new_topology=testsystem.topology, logp=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())

    Create some alchemical systems

    >>> alchemical_system_insert = make_alchemical_system(testsystem.system, topology_proposal, direction='create')
    >>> alchemical_system_delete = make_alchemical_system(testsystem.system, topology_proposal, direction='delete')

    """

    def __init__(self, metadata):
        pass


    def make_alchemical_system(self, unmodified_system, topology_proposal, direction='create'):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal

        Arguments
        ---------
        unmodified_system : openmm.System object
            The unmodified system to get alchemical modifications
        topology_proposal : TopologyProposal namedtuple
            Contains old topology, proposed new topology, and atom mapping
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms.

        Returns
        -------
        alchemical_system : openmm.System object
            The system with appropriate atoms alchemically modified

        """
        atom_map = topology_proposal.new_to_old_atom_map
        n_atoms = unmodified_system.getNumParticles()

        #take the unique atoms as those not in the {new_atom : old_atom} atom map
        if direction == 'delete':
            alchemical_atoms = [atom for atom in range(n_atoms) if atom not in atom_map.values()]
        elif direction == 'create':
            alchemical_atoms = [atom for atom in range(n_atoms) if atom not in atom_map.keys()]
        else:
            raise Exception("direction must be one of ['delete', 'create']; found '%s' instead" % direction)

        logging.debug(alchemical_atoms)
        # Create an alchemical factory.
        from alchemy import AbsoluteAlchemicalFactory
        alchemical_factory = AbsoluteAlchemicalFactory(unmodified_system, ligand_atoms=alchemical_atoms)

        # Return the alchemically-modified system.
        alchemical_system = alchemical_factory.createPerturbedSystem()
        return alchemical_system

"""
This is a template for code that creates alchemical eliminations
for NCMC switching between compounds or residues.
"""
import simtk.openmm as openmm
class  AlchemicalEliminationEngine(object):
    """
    This class is the template for generating systems with the appropriate atoms alchemically modified

    Arguments
    ---------
    metadata : dict
        metadata for alchemical modification of systems
    """

    def __init__(self, metadata):
        pass


    def make_alchemical_system(self, unmodified_system, topology_proposal):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal

        Arguments
        ---------
        unmodified_system : openmm.System object
            The unmodified system to get alchemical modifications
        topology_proposal : TopologyProposal namedtuple
            Contains old topology, proposed new topology, and atom mapping

        Returns
        -------
        alchemical_system : openmm.System object
            The system with appropriate atoms alchemically modified
        """
        return

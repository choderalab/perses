"""
Contains base utility class to generate an openmm System
object from the topology proposal
"""
import simtk.openmm as openmm


class SystemGenerator(object):
    """
    This is the base class for utility functions that generate a System
    object from TopologyProposal namedtuple

    Arguments
    ---------
    metadata : dict
        contains metadata (such as forcefield) for system creation
    """

    def __init__(self, metadata):
        pass

    def new_system(self, top_proposal):
        """
	Create a system with ligand and protein

	Arguments
	---------
	top_proposal : TopologyProposal namedtuple
	    Contains the topology of the new system to be made
	
	Returns
	-------
	system : simtk.openmm.System object
	    openmm System containing the protein and ligand(s)
	"""
        return openmm.System()

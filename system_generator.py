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
    proposal : TopologyProposal namedtuple
        Contains the proposed new topology and metadata

    Properties
    ----------
    complex_system : simtk.openmm.System object, read only
        The system object for the complex
    ligand_system : simtk.openmm.System object, read-only
        The system object for the ligand
    """

    def __init__(self, proposal):
        pass

    @property
    def complex_system(self):
        return openmm.System()

    @property
    def ligand_system(self):
        return openmm.System()

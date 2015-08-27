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
    system : simtk.openmm.System object
        The system that should be alchemically modified
    topology_proposal : Proposal namedtuple
        namedtuple of topology proposal
    
    Properties
    ----------
    alchemical_system : simtk.openmm.System object, read-only
        The alchemically-modified system of interest.
    """
    
    def __init__(self, system, topology_proposal):
        self._alchemically_modified_system = openmm.System()
        
    @property
    def alchemical_system(self):
         return self._alchemically_modified_system

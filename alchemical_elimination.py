"""
This is a template for code that creates alchemical eliminations
for NCMC switching between compounds or residues.
"""
import simtk.openmm as openmm
def AlchemicalEliminationEngine(object):
    """
    This class is the template for generating systems with the appropriate atoms alchemically modified
    
    Arguments
    ---------
    system : simtk.openmm.System object
        The system that should be alchemically modified
    proposal : Proposal namedtuple
        namedtuple of topology proposal
    """
    
    def __init__(self, system, proposal):
        self._alchemically_modified_system = openmm.System()
        
    @property
    def alchemical_system(self):
         """
         Returns a system with the unique atoms alchemically modified.
         
         Returns
         -------
         alchemical_system : simtk.openmm.System object
             An alchemically modified system.
         """
         return self._alchemically_modified_system

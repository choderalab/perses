"""
This is a template for code that creates alchemical eliminations
for NCMC switching between compounds or residues.
"""

def AlchemicalEliminationEngine(object):
    """
    This class is the template for generating systems with the appropriate atoms alchemically modified
    """
    
    def __init__(self, system, proposal):
        pass

    @property
    def alchemical_system(self):
         """
         Returns a system with the unique atoms alchemically modified.
         
         Returns
         -------
         alchemical_system : simtk.openmm.System object
             An alchemically modified system.
         """
         return self._alchemically_moodified_system

import numpy as np

class NCMCEngine(object):
    """
    This is the base class for NCMC switching between two different systems.
    
    Arguments
    ---------
    alchemical_system : simtk.openmm.System object
        alchemically-modified system with atoms to be eliminated
    alchemical_protocol : dict?
        The protocol to use for alchemical introduction or elimination
    initial_positions : [n, 3] numpy.ndarray
        positions of the atoms in the old system
    
    Properties
    ---------
    log_ncmc : float, read-only
         The contribution of the NCMC move to the acceptance probability
    final_positions : [n,3] numpy.ndarray, read-only
        positions of the system after NCMC switching
    """

    def __init__(self, alchemical_system, alchemical_protocol, initial_positions):
         pass

    def integrate(self):
         """
         Performs NCMC switching according to the provided
         alchemical_protocol
         """
         pass
    
    @property
    def log_ncmc(self):
        return 0

    @property
    def final_positions(self):
        return np.array([0.0,0.0,0.0])

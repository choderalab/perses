"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple
import numpy as np

GeometryProposal = namedtuple('GeometryProposal',['new_positions','logp_ratio'])

class GeometryEngine(object):
    """
    This is the base class for the geometry engine.
    
    Arguments
    ---------
    topology_proposal : TopologyProposal namedtuple
        The result of the topology proposal, containing the atom mapping and topologies.
    current_positions : numpy.ndarray of floats
        The positions of the old system, used for the new geometry proposal
    """
    
    def __init__(self, topology_proposal, current_positions):
         pass


    def propose(self):
        """
        Make a geometry proposal for the appropriate atoms.
        
        Returns
        -------
        proposal : GeometryProposal namedtuple
             Contains the new positions as well as the logp ratio
             of the proposal.
        """
        return GeometryProposal(np.array([0.0,0.0,0.0]), 0)


                  

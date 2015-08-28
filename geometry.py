"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple
import numpy as np

GeometryProposal = namedtuple('GeometryProposal',['new_positions','logp'])

class GeometryEngine(object):
    """
    This is the base class for the geometry engine.
    
    Arguments
    ---------
    metadata : dict
        GeometryEngine-related metadata as a dict
    """
    
    def __init__(self, metadata):
         pass


    def propose(self, topology_proposal, sampler_state):
        """
        Make a geometry proposal for the appropriate atoms.
        
	Arguments
        ----------
        topology_proposal : TopologyProposal namedtuple
            The result of the topology proposal, containing the atom mapping and topologies.
        sampler_state : SamplerState namedtuple
	    The current state of the sampler
	
        Returns
        -------
        proposal : GeometryProposal namedtuple
             Contains the new positions as well as the logp ratio
             of the proposal.
        """
        return GeometryProposal(np.array([0.0,0.0,0.0]), 0)


                  

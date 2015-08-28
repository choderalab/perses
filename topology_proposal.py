"""
This file contains the base classes for topology proposals
"""

import simtk.openmm.app as app
from collections import namedtuple
TopologyProposal = namedtuple('Proposal',['old_topology','new_topology','logp', 'new_to_old_atom_map'])
SamplerState = namedtuple('SamplerState',['topology','system','positions', 'metadata'])

class Transformation(object):
    """
    This defines a type which, given the requisite metadata, can produce Proposals (namedtuple)
    of new topologies.
    
    Arguments
    --------
    proposal_metadata : dict
        Contains information necessary to initialize proposal engine
    """
    
    def __init__(self, proposal_metadata):
        pass
    
    def propose(self, sampler_state):
        """
        Base interface for proposal method.
        
        Arguments
	---------
	sampler_state : SamplerState namedtuple
            namedtuple containing the current state of the sampler

        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        return TopologyProposal(app.Topology(), app.Topology(), 0.0, {0 : 0})

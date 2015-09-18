"""
This file contains the base classes for topology proposals
"""

import simtk.openmm.app as app
from collections import namedtuple
TopologyProposal = namedtuple('TopologyProposal',['old_topology','new_topology','logp', 'new_to_old_atom_map', 'metadata'])
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
    
    def propose(self, current_system, current_topology, current_positions, current_metadata):
        """
        Base interface for proposal method.
        
        Arguments
	---------
        current_system : simtk.openmm.System object
            The current system object
        current_topology : simtk.openmm.app.Topology object
            The current topology
        current_positions : [n,3] ndarray of floats
            The current positions of the system
        current_metadata : dict
            Additional metadata about the state
        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
	return TopologyProposal(app.Topology(), app.Topology(), 0.0, {0 : 0}, {'molecule_smiles' : 'CC'})

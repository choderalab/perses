"""
This file contains the base classes for topology proposals
"""


from collections import namedtuple
Proposal = namedtuple('Proposal',['old_topology','new_topology','ln_selection', 'new_to_old_atom_map'])

class Transformation(object):
    """
    This defines a type which, given the requisite metadata, can produce Proposals (namedtuple)
    of new topologies.
    
    Arguments
    --------
    current_topology : simtk.openmm.app.Topology object
        The topology which currently defines the system
    current_system : simtk.openmm.System object
        The system object with all relevant parameters
    metadata : object
        Data necessary for the generation of proposals
    """
    
    def __init__(self, current_topology, current_system, metadata):
        pass
    
    def propose(self):
        """
        Base interface for proposal method. Raises NotImplementedError
        
        Returns
        -------
        proposal : Proposal
            NamedTuple of type Proposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        raise NotImplementedError


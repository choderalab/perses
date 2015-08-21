"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple

GeometryProposal = namedtuple('GeometryProposal',['new_coordinates','logp_ratio'])

class GeometryEngine(object):
    """
    This is the base class for the geometry engine.
    
    Arguments
    ---------
    topology_proposal : TopologyProposal namedtuple
        The result of the topology proposal, containing the atom mapping and topologies.
    current_positions : numpy.ndarray of floats
        The positions of the old system, used for the new geometry proposal

    Properties
    ----------
    proposal : GeometryProposal namedtuple
        The result of the geometry proposal, containing the new positions
        and the logp_ratio of the proposal

    """
    
    def __init__(self, topology_proposal, current_positions):
         pass

    def propose(self):
        """
        Proposes a new geometry for each atom in the list of atoms requiring proposal.
        Additionally calculates probability of reverse proposal.

        Returns
        -------
        new_positions : namedtuple of type NewPositionsProposal
             The new positions, as well as the jacobian-corrected log-probability ratio (logp difference) 
        """
        pass

    @property
    def proposal(self):
        """
        The current geometry proposal.
        
        Returns
        -------
        proposal : GeometryProposal namedtuple
             Contains the new positions as well as the logp ratio
             of the proposal.
        """
        return GeometryProposal(np.array([0.0,0.0,0.0]), 0)


                  

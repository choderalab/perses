"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple
AtomicPositionProposal = namedtuple("PositionProposal",['atom_index','logp','cartesian_coord'])

GeometryProposal = namedtuple('NewPositionsProposal',['new_coordinates','logp_ratio'])

class GeometryEngine(object):
    """
    This is the base class for the geometry engine.
    
    Arguments
    ---------
    topology_proposal : TopologyProposal namedtuple
        The result of the topology proposal, containing the atom mapping and topologies.
    """
    
    def __init__(self, topology_proposal, current_positions):
        self.topology_proposal = topology_proposal   
        self.atoms_for_proposal = []
        self.atoms_for_reverse_proposal = []
    def _transfer_positions(self):
        for a in self.topology_proposal.new_topology.atoms:
              if a.index in self.topology_proposal.new_to_old_atom_mapping.values():
                  self.new_positions[a.index] = self.current_positions[self.topology_proposal.atom_mapping[a.index]]
              else:
                  self.atoms_for_proposal.append(a.index)

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

                  

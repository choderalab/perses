"""
This file contains the base class to calculate weights for each
state
"""

class StateWeight(object):
    """
    Provides facilities for calculating the weight of a given state
    (such as a molecule, or mutant).
    
    Arguments
    ---------
    proposal : namedtuple of type TopologyProposal
        Contains the newly-proposed topology and
        associated metadata.
    """
    
    def __init__(self, proposal):
        pass

    @property
    def weight(self):
        """
        Return the calculated weight of this state
        """
        pass


"""
This file contains the base class to calculate weights for each
state
"""

class StateWeight(object):
    """
    Provides facilities for calculating the weight of a given state
    (such as a molecule, or mutant). 
    """
    
    def __init__(self):
        pass

    def log_weight(self, sampler_state):
        """
	Calculate the log-weight for a given state
	
	Arguments
	---------
	top_proposal : TopologyProposal namedtuple
            Contains the old and new topology and metadata
	    for the system relevant to weight calculation
	"""
        return 0

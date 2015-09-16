"""
This is the base class for generating a biasing potential
for expanded ensemble simulation
"""

class BiasEngine(object):
    """
    Generates the bias for expanded ensemble simulations
    
    Arguments
    ---------
    metadata : dict
        Dictionary containing metadata relevant to the implementation
    """
    
    def __init__(self, metadata):
        pass

    def generate_bias(self, topology_proposal):
        """
        Generate a biasing weight g_k for the state indicated.
        
        Arguments
	--------
	topology_proposal : namedtuple of type TopologyProposal
	    Contains information about the state for which g_k should be generated

        Returns
        -------
	g_k : float
            Bias for the given state
        """
	return 0

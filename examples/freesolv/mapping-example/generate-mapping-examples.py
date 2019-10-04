from perses.tests import test_topology_proposal 
import itertools

pairs_of_smiles = [('Cc1ccccc1','c1ccc(cc1)N'),('Cc1ccccc1','c1ccc(cc1)C=O'),('c1ccc(cc1)N','c1ccc(cc1)C=O')]
test_topology_proposal.test_mapping_strength_levels(pairs_of_smiles,test=False)

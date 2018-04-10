import numpy as np
from simtk import unit
from perses.dispersed.relative_setup import HybridTopologyFactory, HybridSAMSSampler
import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    mcl1tp = np.load('/home/ballen/mcl1_solvent_021218/mcl1topology_proposals.npy').item()
    factory = HybridTopologyFactory(mcl1tp['complex_topology_proposal'], mcl1tp['complex_old_positions'],
                                    mcl1tp['complex_new_positions'], False)
    chss = HybridSAMSSampler(hybrid_factory=factory)
    chss.setup(n_states=100, temperature=300.0 * unit.kelvin, storage_file='complex_test.nc', checkpoint_interval=1)
    chss.extend(1000)
    print("DONE FINALLY!!!")
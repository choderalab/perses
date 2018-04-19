import numpy as np
from simtk import unit
from openmmtools import mcmc
from perses.dispersed.relative_setup import HybridTopologyFactory, HybridSAMSSampler
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    mcl1tp = np.load('/home/ballen/mcl1_solvent_021218/mcl1topology_proposals.npy').item()
    factory = HybridTopologyFactory(mcl1tp['complex_topology_proposal'], mcl1tp['complex_old_positions'],
                                    mcl1tp['complex_new_positions'])
    chss = HybridSAMSSampler(mcmc_moves=mcmc.LangevinDynamicsMove(timestep=2.0 * unit.femtosecond,
                                                                  collision_rate=5.0 / unit.picosecond,
                                                                  n_steps=5000,
                                                                  reassign_velocities=True,
                                                                  n_restart_attempts=6),
                             hybrid_factory=factory)
    chss.setup(n_states=50, temperature=300.0 * unit.kelvin, storage_file='complex_test_50.nc', checkpoint_interval=1)
    chss.minimize()
    chss.equilibrate(10)
    chss.extend(1000)
    print("DONE FINALLY!!!")

import numpy as np
from simtk import unit
from openmmtools import mcmc
from perses.samplers.multistate import HybridSAMSSampler
from perses.dispersed.relative_setup import HybridTopologyFactory
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    mcl1tp = np.load('/home/ballen/PycharmProjects/perses/examples/cdk2-example/cdk2_sams_hbonds/cdk2topology_proposals.npy').item()
    factory = HybridTopologyFactory(mcl1tp['complex_topology_proposal'], mcl1tp['complex_old_positions'],
                                    mcl1tp['complex_new_positions'])
    chss = HybridSAMSSampler(mcmc_moves=mcmc.LangevinDynamicsMove(timestep=2.0 * unit.femtosecond,
                                                                  collision_rate=5.0 / unit.picosecond,
                                                                  n_steps=1000,
                                                                  reassign_velocities=False,
                                                                  n_restart_attempts=6),
                             hybrid_factory=factory)
    chss.setup(n_states=100, temperature=300.0 * unit.kelvin,
               storage_file='/media/ballen/overflow/perses/complex_test_100.nc', checkpoint_interval=1)
    chss.minimize()
    chss.equilibrate(10)
    chss.extend(10000)
    print("DONE FINALLY!!!")

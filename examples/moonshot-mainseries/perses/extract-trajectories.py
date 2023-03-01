"""
Extract replica or state trajectories from a perses replica exchange simulation.

This requires the same version of perses used to generate the simulations to be installed.

"""

import rich
import click
import openmm
from openeye import oechem
from simtk.openmm import unit, app

import os
import pickle
import mdtraj as md
import numpy as np

# Configure logging
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
from rich.console import Console
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
log = logging.getLogger("rich")

def new_positions(htf, hybrid_positions):
    n_atoms_new = htf._topology_proposal.n_atoms_new
    hybrid_indices = [htf._new_to_hybrid_map[idx] for idx in range(n_atoms_new)]
    return hybrid_positions[hybrid_indices, :]

def old_positions(htf, hybrid_positions):
    n_atoms_old = htf._topology_proposal.n_atoms_old
    hybrid_indices = [htf._old_to_hybrid_map[idx] for idx in range(n_atoms_old)]
    return hybrid_positions[hybrid_indices, :]

def extract_single_trajectory(phase, htf, state):
    """
    Retrieve pdbs/dcds of the old and new positions for a given thermodynamic state index. 
    
    Adapted from Hannah: https://github.com/hannahbrucemacdonald/endstate_pdbs/blob/master/scripts/input_for_pol_calc.py

    Parameters
    ----------
    phase
    htf


    """
    
    
    # Load nc files
    from perses.analysis.utils import open_netcdf
    nc = open_netcdf(os.path.join(out_dir, f"{i}_{phase}.nc"))
    nc_checkpoint = open_netcdf(os.path.join(out_dir, f"{i}_{phase}_checkpoint.nc"))
    checkpoint_interval = nc_checkpoint.CheckpointInterval
    all_positions = nc_checkpoint.variables['positions']
    n_iter, n_replicas, n_atoms, _ = np.shape(all_positions)
    box_vectors = np.array(nc_checkpoint['box_vectors'])
    
    # Retrieve positions 
    all_pos_new = np.zeros(shape=(n_iter, new_top.n_atoms, 3))
    all_pos_old = np.zeros(shape=(n_iter, old_top.n_atoms, 3))
    all_box_vectors = np.zeros(shape=(n_iter, 3, 3))
    from rich.progress import track
    for iteration in track(range(n_iter)):
        replica_id = np.where(nc.variables['states'][iteration*checkpoint_interval] == state)[0]
        pos = all_positions[iteration,replica_id,:,:][0] *unit.nanometers
        all_pos_new[iteration] = new_positions(htf, pos).value_in_unit_system(unit.md_unit_system)
        all_pos_old[iteration] = old_positions(htf, pos).value_in_unit_system(unit.md_unit_system)
        all_box_vectors[iteration] = box_vectors[iteration,replica_id,:,:]
    
    # Create trajectories
    trajs = dict()
    trajs['old'] = md.Trajectory(all_pos_old, old_top)
    trajs['new'] = md.Trajectory(all_pos_new, new_top)
    
    # Set unit cell vectors in traj 
    trajs['old'].unitcell_vectors = all_box_vectors
    trajs['new'].unitcell_vectors = all_box_vectors
    
    return trajs

def read_htfs(htf_filename):
    """Read a HybridTopologyFactory 

    Parameters
    ----------
    htf_filename : str
        The filename of the .npy.npz serialized hybrid topology factories
    
    Returns
    -------
    htfs : dict of str : HybridTopologyFactory
       htfs[phase] is the HybridTopologyFactory corresponding to phase
       phase is one of ['complex', 'solvent']
    """
    htfs = np.load(htf_filename, allow_pickle=True)['arr_0'].item(0) 
    return htfs

@click.command()
@click.option('--path', required=True, help='path to perses trajectories')
@click.option('--prefix', default='out', help='prefix for perses output filenames')
@click.option('--extract', default='replicas', type=click.Choice(['replicas', 'states'], case_sensitive=False))
@click.option('--outpath', required=True, help='output path for replica trajectories, written as PDB + XTC pairs')
@click.option('--phase', default=None, help='If specified, write only this phase')
@click.option('--index', default=None, help='If specified, write only these replica or state index')
def extract_trajectories(path, prefix, extract, outpath, phase, index):
    """Extract replica trajectories from a perses simulation into PDB/XTC pairs.
    """
    # Read hybrid topology file (HTF) containing information about the transformation
    import numpy as np
    from openeye import oechem
    import os
    htf_filename = os.path.join(path, f'{prefix}-hybrid_factory.npy.npz')
    log.info(f':clock1: Extracting {htf_filename}...')
    # htfs[phase] is HybridTopologyFactory for phase; phase is one of ['complex', 'solvent']
    htfs = read_htfs(htf_filename)
    
    # Create directory for storing output trajectories
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    if phase is None:        
        PHASES = ['solvent', 'complex']
    else:
        PHASES = [phase]

    for phase in PHASES:
        log.info(f':clock: Extracting {phase} phase...')
        htf = htfs[phase]

        # Extract MDTraj topology
        mdtop = htf._hybrid_topology

        # Extract old and new Topology objects
        #new_top = md.Topology.from_openmm(htf._topology_proposal.new_topology)
        #old_top = md.Topology.from_openmm(htf._topology_proposal.old_topology)

        # Extract trajectories
        # Load nc files
        from perses.analysis.utils import open_netcdf
        nc = open_netcdf(os.path.join(path, f"{prefix}-{phase}.nc"))
        positions = nc.variables['positions']
        n_iterations, n_replicas, n_atoms, n_dim = np.shape(positions)
        hybrid_atom_indices = np.array(nc.variables['analysis_particle_indices'])
        mdtop = htf._hybrid_topology
        solute_mdtop = mdtop.subset(hybrid_atom_indices)

        if index is None:
            indices = np.arange(n_replicas)
        else:
            indices = [int(index)]

        #nc_checkpoint = open_netcdf(os.path.join(out_dir, f"{i}_{phase}_checkpoint.nc"))
        #checkpoint_interval = nc_checkpoint.CheckpointInterval
        #all_positions = nc_checkpoint.variables['positions']
        #box_vectors = np.array(nc_checkpoint['box_vectors'])    

        log.info(f':clock: Writing {extract}...')
        if extract == 'replicas':
            for replica_index in indices:
                log.info(f':clock: Writing {extract} : {replica_index} / {n_replicas}...')
                # Create MDTraj trajectory
                trajectory = md.Trajectory(positions[:,replica_index,:,:], solute_mdtop)
                output_prefix = os.path.join(outpath, f'{phase}-{extract}-{replica_index:03d}')

                # Write first frame in PDB, subsequent frames in XTC
                trajectory[0].save(output_prefix + '.pdb')
                trajectory[1:].save(output_prefix + '.xtc')
        elif extract == 'states':
            states = np.array(nc.variables['states'])
            state_positions = np.zeros([n_iterations, n_atoms, n_dim], np.float32)
            for state_index in indices:
                log.info(f':clock: Writing {extract} : {state_index} / {n_replicas}...')
                # Extract positions
                for iteration in range(n_iterations):
                    replica_index = np.where(states[iteration,:] == state_index)[0][0]
                    state_positions[iteration,:,:] = positions[iteration,replica_index,:,:]

                trajectory = md.Trajectory(state_positions, solute_mdtop)
                output_prefix = os.path.join(outpath, f'{phase}-{extract}-{state_index:03d}')

                # Write first frame in PDB, subsequent frames in XTC
                trajectory[0].save(output_prefix + '.pdb')
                trajectory[1:].save(output_prefix + '.xtc')
        else:
            raise ParameterError(f'extract must be one of [replica, state]')


    


if __name__ == '__main__':
    extract_trajectories()
    stop

    # DEBUG
    import numpy as np
    from openeye import oechem
    import os
    path = "step1-His41(0)-Cys145(0)-His163(0)-0-2"
    prefix = "out"
    htf_filename = os.path.join(path, f'{prefix}-hybrid_factory.npy.npz')
    log.info(f':clock1: Reading {htf_filename}...')
    htfs = np.load(htf_filename, allow_pickle=True)['arr_0'].item(0)
    phase = 'complex'
    htf = htfs[phase]
    from perses.analysis.utils import open_netcdf
    nc = open_netcdf(os.path.join(path, f"{prefix}-{phase}.nc"))
    positions = nc.variables['positions']
    n_iter, n_replicas, n_atoms, n_dim = np.shape(positions)
    hybrid_atom_indices = np.array(nc.variables['analysis_particle_indices'])
    mdtop = htf._hybrid_topology

    solute_mdtop = mdtop.subset(hybrid_atom_indices)
    traj = md.Trajectory(positions[:,0,:,:], solute_mdtop)
    traj[0].save('test.pdb')
    traj[1:].save('test.xtc')

    # Project onto old and new
    #for projection in ['old', 'new']:
    #    # Determine which atoms from the real system appear in the solute-only trajectory file
    #    real_to_hybrid_map = getattr(htf, f'_{projection}_to_hybrid_map') # hybrid_from_real_indices[real_index] is the hybrid index corresponding to real_index
    #    stored_atom_indices = np.array([hybrid_atom_indices.index(hybrid_atom_index) for old_atom_index, hybrid_atom_index in real_to_hybrid_map if hybrid_atom_index in hybrid_atom_indices ])
    #    traj = md.Trajectory(np.array(nc.variables['positions'][:,replica_index,real_atom_indices,:]), mdtop.subset(atom_indices))
    #output_prefix = os.path.join(outpath, f"{phase}-state{index:03d}-{molecule}")

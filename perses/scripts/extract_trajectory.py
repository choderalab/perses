"""

Functions to extract trajectory from a perses relative calculation

"""


import numpy as np
import mdtraj as md
from perses.scripts.utils import open_netcdf

def get_hybrid_topology(file):
    hybrid_factory = np.load(file)
    hybrid_factory = hybrid_factory.flatten()[0]

    phases = []
    topologies = []
    for phase in hybrid_factory.keys():
        topologies.append(hybrid_factory[phase].hybrid_topology)

    return phases, topologies

def get_positions(file):
    ncfile = open_netcdf(file)

    all_positions = ncfile.variables['positions']
    results = []
    for i,pos in enumerate(all_positions):
        coords = []
        pos = pos.tolist()
        results.append(pos[0])
    return results


def write_trajectory(positions, topology, outputfile='trajectory.pdb',center=True,offline=None):
    if offline != None:
        traj = md.Trajectory(positions[0::offline],topology)
    else:
        traj = md.Trajectory(positions, topology)
    if center == True:
        traj.center_coordinates()
    traj.save_pdb(outputfile)

    return

if __name__ == '__main__':
    import sys

    ncfilename = sys.argv[1]
    factoryfilename = sys.argv[2]

    positions = get_positions(ncfilename)
    _, topology = get_hybrid_topology(factoryfilename)

    write_trajectory(positions,topology[0])

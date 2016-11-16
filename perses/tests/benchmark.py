from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
from unittest import skipIf
import numpy as np
from functools import partial
from pkg_resources import resource_filename
from openeye import oechem
if sys.version_info >= (3, 0):
    from io import StringIO
    from subprocess import getstatusoutput
else:
    from cStringIO import StringIO
    from commands import getstatusoutput
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt

################################################################################
# NUMBER OF ATTEMPTS
################################################################################
niterations = 50
################################################################################
# CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

def simulate(system, positions, nsteps=500, timestep=1.0*unit.femtoseconds, temperature=temperature, collision_rate=5.0/unit.picoseconds, platform=None):
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    if platform == None:
        context = openmm.Context(system, integrator)
    else:
        context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(nsteps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)
    return [positions, velocities]

def check_hybrid_null_elimination(NullProposal, ncmc_nsteps=50, NSIGMA_MAX=6.0, geometry=False):
    """
    Test alchemical elimination engine on null transformations, where some atoms are deleted and then reinserted in a cycle.

    Parameters
    ----------
    NullProposal : class
        subclass of testsystems.NullTestSystem
    ncmc_nsteps : int, optional, default=50
        Number of NCMC switching steps, or 0 for instantaneous switching.
    NSIGMA_MAX : float, optional, default=6.0
        Number of standard errors away from analytical solution tolerated before Exception is thrown
    geometry : bool, optional, default=None
        If True, will also use geometry engine in the middle of the null transformation.
    """
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : '1.0',  
        'lambda_angles' : '0.1*lambda+0.9', 
        'lambda_torsions' : '0.7*lambda+0.3'
    }

    testsystem = NullProposal(storage_filename=None, scheme='geometry-ncmc-geometry',options={'functions': functions, 'nsteps': ncmc_nsteps})
    for key in testsystem.environments: # only one key: vacuum
        # run a single iteration to generate item in number_of_state_visits dict
        ncmc_engine = testsystem.exen_samplers[key].ncmc_engine

        topology = testsystem.topologies[key]
        system = testsystem.system_generators[key].build_system(topology)
        topology_proposal = testsystem.proposal_engines[key].propose(system, topology)
        positions = testsystem.positions[key]

    nequil = 5 # number of equilibration iterations
    logP_n = np.zeros([niterations], np.float64)
    for iteration in range(nequil):
        [positions, velocities] = simulate(topology_proposal.old_system, positions)
    for iteration in range(niterations):
        # Equilibrate
        [positions, velocities] = simulate(topology_proposal.old_system, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Hybrid NCMC from old to new
        [positions, new_old_positions, logP] = ncmc_engine.integrate(topology_proposal, positions, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on Hybrid NCMC switch")

        # Compute total probability
        logP_n[iteration] = logP

    work_n = - logP_n
    print(work_n.mean(), work_n.std())
    return work_n.mean(), work_n.std()

def check_alchemical_null_elimination(NullProposal, ncmc_nsteps=50, NSIGMA_MAX=6.0, geometry=False):
    """
    Test alchemical elimination engine on null transformations, where some atoms are deleted and then reinserted in a cycle.

    Parameters
    ----------
    NullProposal : class
        subclass of testsystems.NullTestSystem
    ncmc_nsteps : int, optional, default=50
        Number of NCMC switching steps, or 0 for instantaneous switching.
    NSIGMA_MAX : float, optional, default=6.0
        Number of standard errors away from analytical solution tolerated before Exception is thrown
    geometry : bool, optional, default=None
        If True, will also use geometry engine in the middle of the null transformation.
    """
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : '1.0', # don't soften bonds
        'lambda_angles' : '0.1*lambda+0.9', # don't soften angles
        'lambda_torsions' : '0.7*lambda+0.3'
    }
    testsystem = NullProposal(storage_filename=None, scheme='ncmc-geometry-ncmc',options={'functions': functions, 'nsteps': ncmc_nsteps})

    for key in testsystem.environments: # only one key: vacuum
        # run a single iteration to generate item in number_of_state_visits dict
        ncmc_engine = testsystem.exen_samplers[key].ncmc_engine

        topology = testsystem.topologies[key]
        system = testsystem.system_generators[key].build_system(topology)
        topology_proposal = testsystem.proposal_engines[key].propose(system, topology)
        positions = testsystem.positions[key]

    nequil = 5 # number of equilibration iterations
    logP_insert_n = np.zeros([niterations], np.float64)
    logP_delete_n = np.zeros([niterations], np.float64)
    logP_switch_n = np.zeros([niterations], np.float64)
    for iteration in range(nequil):
        [positions, velocities] = simulate(topology_proposal.old_system, positions)
    for iteration in range(niterations):
        # Equilibrate
        [positions, velocities] = simulate(topology_proposal.old_system, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Delete atoms
        [positions, logP_delete, potential_delete] = ncmc_engine.integrate(topology_proposal, positions, direction='delete')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC deletion")

        # Insert atoms
        [positions, logP_insert, potential_insert] = ncmc_engine.integrate(topology_proposal, positions, direction='insert')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC insertion")

        # Compute probability of switching geometries.
        logP_switch = - (potential_insert - potential_delete)

        # Compute total probability
        logP_delete_n[iteration] = logP_delete
        logP_insert_n[iteration] = logP_insert
        logP_switch_n[iteration] = logP_switch

    logP_n = logP_delete_n + logP_insert_n + logP_switch_n
    work_n = - logP_n
    print(work_n.mean(), work_n.std())
    return work_n.mean(), work_n.std()

def plot_ncmc_logP(mol_name, ncmc_type, mean, sigma):
    x = mean.keys()
    x.sort()
    y = [mean[steps] for steps in x]
    dy = [sigma[steps] for steps in x]
    plt.fill_between(x, [mean - dev for mean, dev in zip(y, dy)], [mean + dev for mean, dev in zip(y, dy)])
    plt.plot(x, y, 'k')
    plt.xscale('log')

    plt.title("Log acceptance probability of {0} NCMC for {1}".format(ncmc_type, mol_name))
    plt.ylabel('logP NCMC')
    plt.xlabel('ncmc steps')
    plt.savefig('{0}_{1}NCMC_logP'.format(mol_name, ncmc_type))
    print('Saved plot to {0}_{1}NCMC_logP.png'.format(mol_name, ncmc_type))
    plt.clf()

def benchmark_ncmc_null_protocols():
    """
    Check convergence of logP for hybrid and delete/insert NCMC schemes for 3
    small molecule null proposals

    Does not run geometry engine

    Plot mean and standard deviation of logP for 0, 1, 10, 100, 1000, 10000
    ncmc steps.
    """
    from perses.tests.testsystems import NaphthaleneTestSystem, ButaneTestSystem, PropaneTestSystem
    molecule_names = {
        'naphthalene' : NaphthaleneTestSystem,
        'butane' : ButaneTestSystem,
        'propane' : PropaneTestSystem,
    }
    for molecule_name, NullProposal in molecule_names.items():
        print('\nNow testing {0} null transformations'.format(molecule_name))
        mean = dict()
        sigma = dict()
        for ncmc_nsteps in [0, 1, 10, 100, 1000, 10000]:
            print('Running {0} hybrid NCMC steps for {1} iterations'.format(ncmc_nsteps, niterations))
            mean[ncmc_nsteps], sigma[ncmc_nsteps] = check_hybrid_null_elimination(NullProposal, ncmc_nsteps=ncmc_nsteps)
        plot_ncmc_logP(molecule_name, 'hybrid', mean, sigma)

        mean = dict()
        sigma = dict()
        for ncmc_nsteps in [0, 1, 10, 100, 1000, 10000]:
            print('Running {0} delete-insert NCMC steps for {1} iterations'.format(ncmc_nsteps, niterations))
            mean[ncmc_nsteps], sigma[ncmc_nsteps] = check_alchemical_null_elimination(NullProposal, ncmc_nsteps=ncmc_nsteps)
        plot_ncmc_logP(molecule_name, 'two-stage', mean, sigma)

if __name__ == "__main__":
    benchmark_ncmc_null_protocols()


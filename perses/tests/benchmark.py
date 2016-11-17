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
ENV = 'vacuum'
################################################################################
# CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

functions_hybrid = {
    'lambda_sterics' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_bonds' : '1.0',
    'lambda_angles' : '0.1*lambda+0.9',
    'lambda_torsions' : '0.7*lambda+0.3'
}
functions_twostage = {
    'lambda_sterics' : '(2*lambda)^4 * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
    'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
    'lambda_bonds' : '1.0', # don't soften bonds
    'lambda_angles' : '0.1*lambda+0.9',
    'lambda_torsions' : '0.7*lambda+0.3'
}

def plot_logPs(logps, molecule_name, scheme, component):
    """
    Create line plot of mean and standard deviation of given logPs.

    Arguments:
    ----------
        logps: dict { int : np.ndarray }
            key : number of total NCMC steps
            value : array of `niterations` logP values
        molecule_name : str
            The molecule featured in the NullTestSystem being analyzed
            in ['naphthalene','butane','propane']
        scheme : str
            Which NCMC scheme is being used
            in ['hybrid','two-stage']
        component : str
            Which logP is being plotted
            in ['NCMC','EXEN']
    """
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : '1.0',  
        'lambda_angles' : '0.1*lambda+0.9', 
        'lambda_torsions' : '0.7*lambda+0.3'
    }

    testsystem = NullProposal(storage_filename=None, scheme='geometry-ncmc-geometry',options={'functions': functions, 'nsteps': ncmc_nsteps})
    for key in [ENV]: #testsystem.environments: # only one key: vacuum
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

    for key in [ENV]: #testsystem.environments: # only one key: vacuum
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
    y = [logps[steps].mean() for steps in x]
    dy = [logps[steps].std() for steps in x]
    plt.fill_between(x, [mean - dev for mean, dev in zip(y, dy)], [mean + dev for mean, dev in zip(y, dy)])
    plt.plot(x, y, 'k')
    plt.xscale('log')

    plt.title("Log acceptance probability of {0} ExpandedEnsemble for {1}".format(scheme, molecule_name))
    plt.ylabel('logP')
    plt.xlabel('ncmc steps')
    plt.savefig('{0}_{1}_{2}NCMC_logP'.format(ENV, mol_name, ncmc_type))
    print('Saved plot to {0}_{1}_{2}NCMC_logP.png'.format(ENV, mol_name, ncmc_type))
    plt.clf()

def plot_exen_logP(mol_name, ncmc_type, mean, sigma):
    x = mean.keys()
    x.sort()
    y = [mean[steps] for steps in x]
    dy = [sigma[steps] for steps in x]
    plt.fill_between(x, [mean - dev for mean, dev in zip(y, dy)], [mean + dev for mean, dev in zip(y, dy)])
    plt.plot(x, y, 'k')
    plt.xscale('log')

    plt.title("Log acceptance probability of {0} ExpandedEnsemble for {1}".format(ncmc_type, mol_name))
    plt.ylabel('logP')
    plt.xlabel('ncmc steps')
    plt.savefig('{0}_{1}_{2}EXEN_logP'.format(ENV, mol_name, ncmc_type))
    print('Saved plot to {0}_{1}_{2}EXEN_logP.png'.format(ENV, mol_name, ncmc_type))
    plt.clf()

def benchmark_exen_ncmc_protocol(analyses, molecule_name, scheme):
    """
    For each combination of system and scheme, results are analyzed for
    the following:
    * Over the whole range of total steps:
        * Plot mean and standard deviation of NCMC logP as a function of
          total steps
        * Plot mean and standard deviation of EXEN logP as a function of
          total steps

    Arguments:
    ----------
        analyses : dict { int : perses.Analysis }
            key : number of total NCMC steps
            value : analysis object contained stored information
        molecule_name : str
            The molecule featured in the NullTestSystem being analyzed
            in ['naphthalene','butane','propane']
        scheme : str
            Which NCMC scheme is being used
            in ['hybrid','two-stage']

    Creates 2 plots every time it is called
    """
    components = {
        'logp_accept' : 'EXEN',
        'logp_ncmc' : 'NCMC',
    }

    for component in components.keys():
        print('Finding {0} over nsteps for {1} with {2} NCMC'.format(component, molecule_name, scheme))
        logps = dict()
        for nsteps, analysis in analyses.items():
            ee_sam = analysis._ncfile.groups['ExpandedEnsembleSampler']
            niterations = ee_sam.variables[component].shape[0]
            logps[nsteps] = np.zeros(niterations, np.float64)
            for n in range(niterations):
                logps[nsteps][n] = ee_sam.variables[component][n]
        plot_logPs(logps, molecule_name, scheme, components[component])

def benchmark_ncmc_work_during_protocol():
    """
    Run 50 iterations of ExpandedEnsembleSampler for NullTestSystems
    over a range of total NCMC steps [0, 1, 10, 100, 1000, 10000].

    Benchmark is repeated for Naphthalene, Butane, and Propane test
    systems, using two-stage and hybrid NCMC.

    For each combination of system and scheme, results are analyzed for
    the following:
    * For a given total number of steps:
        * For NCMC steps 100 and above, plot work done by ncmc integrator
          over the course of the protocol
        * Plot histograms of the contributions of each component to the
          overall log acceptance probability
    * Over the whole range of total steps:
        * Plot mean and standard deviation of NCMC logP as a function of
          total steps
        * Plot mean and standard deviation of EXEN logP as a function of
          total steps
    """
    from perses.tests.testsystems import NaphthaleneTestSystem, ButaneTestSystem, PropaneTestSystem
    from perses.analysis import Analysis
    import netCDF4 as netcdf
    import pickle
    import codecs
    molecule_names = {
        'naphthalene' : NaphthaleneTestSystem,
        'butane' : ButaneTestSystem,
        'propane' : PropaneTestSystem,
    }
    methods = {
        'hybrid' : check_hybrid_null_elimination,
        'two-stage' : check_alchemical_null_elimination,
    }
    for molecule_name, NullProposal in molecule_names.items():
        print('\nNow testing {0} null transformations'.format(molecule_name))
        for scheme, method in methods.items():
            mean = dict()
            sigma = dict()
            for ncmc_nsteps in [0, 1, 10, 100, 1000, 10000]:
                print('Running {0} delete-insert NCMC steps for {1} iterations'.format(ncmc_nsteps, niterations))
                mean[ncmc_nsteps], sigma[ncmc_nsteps] = method(NullProposal, ncmc_nsteps=ncmc_nsteps)
            plot_ncmc_logP(molecule_name, scheme, mean, sigma)

def check_alchemical_exen(NullProposal, scheme, ncmc_nsteps, functions):
    testsystem = NullProposal(storage_filename=None, scheme=scheme,options={'functions': functions, 'nsteps': ncmc_nsteps})
    for key in [ENV]: #testsystem.environments: # only one key: vacuum
        # run a single iteration to generate item in number_of_state_visits dict
        exen_sampler = testsystem.exen_samplers[key]
        exen_sampler.verbose = False
        topology = testsystem.topologies[key]
        system = testsystem.system_generators[key].build_system(topology)
        topology_proposal = testsystem.proposal_engines[key].propose(system, topology)
        positions = testsystem.positions[key]

    nequil = 5 # number of equilibration iterations
    for iteration in range(nequil):
        [positions, velocities] = simulate(topology_proposal.old_system, positions)
    testsystem.positions[ENV] = positions
    return testsystem, exen_sampler

def check_hybrid_exen(NullProposal, ncmc_nsteps=50):
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : '1.0',
        'lambda_angles' : '0.1*lambda+0.9',
        'lambda_torsions' : '0.7*lambda+0.3'
    }
    testsystem, exen_sampler = check_alchemical_exen(NullProposal, 'geometry-ncmc-geometry', ncmc_nsteps, functions)
    logP_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        exen_sampler.update_positions()
        logP, _ = exen_sampler._geometry_ncmc_geometry()
        logP_n[iteration] = logP

    print(logP_n.mean(), logP_n.std())
    return logP_n.mean(), logP_n.std()

def check_twostage_exen(NullProposal, ncmc_nsteps=50):
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : '1.0', # don't soften bonds
        'lambda_angles' : '0.1*lambda+0.9', # don't soften angles
        'lambda_torsions' : '0.7*lambda+0.3'
    }
    testsystem, exen_sampler = check_alchemical_exen(NullProposal, 'ncmc-geometry-ncmc', ncmc_nsteps, functions)
    logP_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        exen_sampler.update_positions()
        logP, _ = exen_sampler._ncmc_geometry_ncmc()
        logP_n[iteration] = logP

    print(logP_n.mean(), logP_n.std())
    return logP_n.mean(), logP_n.std()

def benchmark_exen_ncmc_null_protocols():
    """
    Relationship of overall ExpandedEnsemble logp_accept to n_ncmc_steps

    Check convergence of logP for hybrid and delete/insert NCMC schemes for 3
    small molecule null proposals

    Plot mean and standard deviation of logP for 0, 1, 10, 100, 1000, 10000
    ncmc steps.
    """
    from perses.tests.testsystems import NaphthaleneTestSystem, ButaneTestSystem, PropaneTestSystem
    molecule_names = {
        'naphthalene' : NaphthaleneTestSystem,
        'butane' : ButaneTestSystem,
        'propane' : PropaneTestSystem,
    }
    methods = {
        'hybrid' : check_hybrid_exen,
        'two-stage' : check_twostage_exen,
    }
    for molecule_name, NullProposal in molecule_names.items():
        print('\nNow testing {0} null transformations'.format(molecule_name))
        for scheme, method in methods.items():
            mean = dict()
            sigma = dict()
            for ncmc_nsteps in [0, 1, 10, 100, 1000, 10000]:
                print('Running {0} hybrid NCMC steps for {1} iterations'.format(ncmc_nsteps, niterations))
                mean[ncmc_nsteps], sigma[ncmc_nsteps] = method(NullProposal, ncmc_nsteps=ncmc_nsteps)
            plot_exen_logP(molecule_name, scheme, mean, sigma)

if __name__ == "__main__":
    benchmark_exen_ncmc_null_protocols()

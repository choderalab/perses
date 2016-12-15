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
from perses.samplers import ExpandedEnsembleSampler

import matplotlib.pyplot as plt

################################################################################
# NUMBER OF ATTEMPTS
################################################################################
niterations = 1000
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
    'lambda_bonds' : 'lambda',#'1.0',
    'lambda_angles' : 'lambda',#'0.1*lambda+0.9',
    'lambda_torsions' : 'lambda',#'0.7*lambda+0.3'
}
functions_twostage = {
    'lambda_sterics' : '(2*lambda)^4 * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
    'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
    'lambda_bonds' : '1.0', # don't soften bonds
    'lambda_angles' : '0.1*lambda+0.9',
    'lambda_torsions' : '0.7*lambda+0.3'
}

class BenchmarkExEnSampler(ExpandedEnsembleSampler):
    """
    Subclass of ExpandedEnsembleSampler

    For use in comparing hybrid and two-stage NCMC schemes resultant logPs
    when 0 NCMC steps are used (therefore logP should be exactly the same).

    Will run the geometry engine once, and give the proposed positions to
    both a hybrid NCMC engine and two-stage NCMC engine and print the
    resulting logP components
    Only prints geometry_logp and ncmc_logp because logp_proposal and
    log_weights are not being used (all 0)
    Reports two-stage ncmc_logp as the sum of
    (switch_logp + ncmc_elimination_logp + ncmc_introduction_logp)
    for direct comparison with hybrid ncmc_logp

    sampler.run() will behave the same as the super class for chosen scheme
    """
    def __init__(self, sampler, topology, state_key, proposal_engine, geometry_engine, log_weights=None, scheme='ncmc-geometry-ncmc', options=None, platform=None, envname=None, storage=None):
        """
        Difference from super class:
            self.ncmc_engine is not defined
            Instead, two ncmc_engines (self.hybrid_ncmc_engine
            and self.two_ncmc_engine) are instantiated and
            self.ncmc_engine is used as a pointer to each as necessary
        """
        super(BenchmarkExEnSampler, self).__init__(sampler, topology, state_key, proposal_engine, geometry_engine, log_weights=log_weights, scheme=scheme, options=options, platform=platform, envname=envname, storage=storage)
        del(self.ncmc_engine)
        from perses.annihilation.ncmc_switching import NCMCHybridEngine
        self.hybrid_ncmc_engine = NCMCHybridEngine(temperature=self.sampler.thermodynamic_state.temperature, timestep=options['timestep'], nsteps=options['nsteps'], functions=options['functions'], platform=platform, storage=self.storage)
        from perses.annihilation.ncmc_switching import NCMCEngine
        self.two_ncmc_engine = NCMCEngine(temperature=self.sampler.thermodynamic_state.temperature, timestep=options['timestep'], nsteps=options['nsteps'], functions=options['functions'], platform=platform, storage=self.storage)

    def _geometry_once_ncmc_twice(self, topology_proposal, positions, old_log_weight, new_log_weight):
        """
        For direct comparison of two ncmc schemes, uses a single geometry
        proposal as input to both ncmc schemes

        First runs the geometry engine for the forward logp and proposed
        positions

        Then calculates expanded ensemble logp based on hybrid scheme:
        New positions and ncmc logp are calculated by running hybrid
        ncmc_engine
        Reverse geometry logp is calculated from new positions

        Finally calculates expanded ensemble logp based on two-stage scheme:
        New positions for old topology are calculated by running two-stage
        ncmc_engine with direction='delete'
        Normally, geometry proposal would be based off of these new positions
        In this case, the geometry proposal was done at the beginning
        The new positions should be equivalent to the old positions because
        no steps of NCMC were actually run
        Reverse geometry logp is calculated from proposed positions to end
        of NCMC positions
        New positions for new topology are calculated by running two-stage
        ncmc_engine with direction='insert'
        """
        geometry_old_positions = positions
        geometry_new_positions, geometry_logp_propose = self._geometry_forward(topology_proposal, geometry_old_positions)

        self.ncmc_engine = self.hybrid_ncmc_engine
        hybrid_ncmc_new_positions, hybrid_ncmc_old_positions, hybrid_ncmc_logp = self._ncmc_hybrid(topology_proposal, positions, geometry_new_positions)
        hybrid_geometry_logp_reverse = self._geometry_reverse(topology_proposal, hybrid_ncmc_new_positions, hybrid_ncmc_old_positions)
        hybrid_geometry_logp = hybrid_geometry_logp_reverse - geometry_logp_propose
        hybrid_logp_accept = topology_proposal.logp_proposal + hybrid_geometry_logp + hybrid_ncmc_logp + new_log_weight - old_log_weight
        if True:
            print("hybrid    logp_accept = %+10.4e [%+10.4e = geometry_logp, %+10.4e = ncmc_logp]"
                % (hybrid_logp_accept, hybrid_geometry_logp, hybrid_ncmc_logp))
        del(self.ncmc_engine)

        self.ncmc_engine = self.two_ncmc_engine
        two_ncmc_old_positions, two_ncmc_elimination_logp, potential_delete = self._ncmc_delete(topology_proposal, positions)
        geometry_old_positions = two_ncmc_old_positions
        two_geometry_logp_reverse = self._geometry_reverse(topology_proposal, geometry_new_positions, geometry_old_positions)
        two_geometry_logp = two_geometry_logp_reverse - geometry_logp_propose
        two_ncmc_new_positions, two_ncmc_introduction_logp, potential_insert = self._ncmc_insert(topology_proposal, geometry_new_positions)
        switch_logp = - (potential_insert - potential_delete)
        two_logp_accept = topology_proposal.logp_proposal + two_geometry_logp + switch_logp + two_ncmc_elimination_logp + two_ncmc_introduction_logp + new_log_weight - old_log_weight
        if True:
            print("two-stage logp_accept = %+10.4e [%+10.4e = geometry_logp, %+10.4e = switch_logp + ncmc_elimination_logp + ncmc_introduction_logp]\n" 
                % (two_logp_accept, two_geometry_logp, (switch_logp + two_ncmc_elimination_logp + two_ncmc_introduction_logp)))
        del(self.ncmc_engine)

        return hybrid_logp_accept, hybrid_ncmc_new_positions, two_logp_accept, two_ncmc_new_positions

    def _geometry_ncmc_geometry(self, topology_proposal, positions, old_log_weight, new_log_weight):
        """
        Actually runs both hybrid and two-stage ncmc engines but returns
        logp_accept and ncmc_new_positions based on hybrid
        """
        logp_accept, ncmc_new_positions, _ , _ = self._geometry_once_ncmc_twice(topology_proposal, positions, old_log_weight, new_log_weight)

        return logp_accept, ncmc_new_positions

    def _ncmc_geometry_ncmc(self, topology_proposal, positions, old_log_weight, new_log_weight):
        """
        Actually runs both hybrid and two-stage ncmc engines but returns
        logp_accept and ncmc_new_positions based on two-stage
        """
        _ , _ , logp_accept, ncmc_new_positions = self._geometry_once_ncmc_twice(topology_proposal, positions, old_log_weight, new_log_weight)

        return logp_accept, ncmc_new_positions


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
    x = logps.keys()
    x.sort()
    y = [logps[steps].mean() for steps in x]
    dy = [logps[steps].std() for steps in x]
    plt.fill_between(x, [mean - dev for mean, dev in zip(y, dy)], [mean + dev for mean, dev in zip(y, dy)])
    plt.plot(x, y, 'k')
    plt.xscale('log')

    plt.title("Log acceptance probability of {0} ExpandedEnsemble for {1}".format(scheme, molecule_name))
    plt.ylabel('logP')
    plt.xlabel('ncmc steps')
    plt.savefig('{0}_{1}_{2}{3}_logP'.format(ENV, molecule_name, scheme, component))
    print('Saved plot to {0}_{1}_{2}{3}_logP.png'.format(ENV, molecule_name, scheme, component))
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
    from perses.tests.testsystems import NaphthaleneTestSystem, ButaneTestSystem, PropaneTestSystem, check_null_deltaG
    from perses.analysis import Analysis
    molecule_names = {
        'naphthalene' : NaphthaleneTestSystem,
        'butane' : ButaneTestSystem,
        'propane' : PropaneTestSystem,
    #    't4' : T4LysozymeInhibitorsTestSystem,
    }
    methods = {
        'hybrid' : ['geometry-ncmc-geometry', functions_hybrid],
        'two-stage' : ['ncmc-geometry-ncmc', functions_twostage],
    }

    for molecule_name, NullProposal in molecule_names.items():
        print('\nNow testing {0} null transformations'.format(molecule_name))
        for name, [scheme, functions] in methods.items():
            analyses = dict()
            for ncmc_nsteps in [0, 1, 10, 100, 1000, 10000]:
                print('Running {0} {2} ExpandedEnsemble steps for {1} iterations'.format(ncmc_nsteps, niterations, name))
                testsystem = NullProposal(storage_filename='{0}_{1}-{2}steps.nc'.format(molecule_name, name, ncmc_nsteps), scheme=scheme, options={'functions' : functions, 'nsteps' : ncmc_nsteps})

                testsystem.exen_samplers[ENV].verbose = False
                testsystem.exen_samplers[ENV].sampler.verbose = False
                if name == 'hybrid':
                    testsystem.exen_samplers[ENV].ncmc_engine.softening = 1.0
                testsystem.exen_samplers[ENV].run(niterations=niterations)
                try:
                    check_null_deltaG(testsystem)
                except Exception as e:
                    print(e)

                analysis = Analysis(testsystem.storage_filename)
                print(analysis.get_environments())
                if ncmc_nsteps > 99:
                    analysis.plot_ncmc_work('{0}_{1}-ncmc_work_over_{2}_steps.pdf'.format(molecule_name, name, ncmc_nsteps))
                analysis.plot_exen_logp_components()
                analyses[ncmc_nsteps] = analysis
            benchmark_exen_ncmc_protocol(analyses, molecule_name, name)

def benchmark_logp_0ncmc_schemes():
    from perses.tests.testsystems import NaphthaleneTestSystem, ButaneTestSystem, PropaneTestSystem, check_null_deltaG
    molecule_names = {
        'naphthalene' : NaphthaleneTestSystem,
        'butane' : ButaneTestSystem,
        'propane' : PropaneTestSystem,
    }
    name = 'hybrid'
    scheme = 'geometry-ncmc-geometry'
    functions = functions_hybrid
    ncmc_nsteps = 0
    trials = 10

    for molecule_name, NullProposal in molecule_names.items():
        print('\nNow testing {0} null transformations'.format(molecule_name))
        print('Running {0} {2} ExpandedEnsemble steps for {1} iterations'.format(ncmc_nsteps, trials, name))
        options={'functions' : functions, 'nsteps' : ncmc_nsteps}
        testsystem = NullProposal(storage_filename='{0}_{1}-{2}steps.nc'.format(molecule_name, name, ncmc_nsteps), scheme=scheme, options=options)

        topology = testsystem.topologies[ENV]
        proposal_engine = testsystem.proposal_engines[ENV]
        chemical_state_key = proposal_engine.compute_state_key(topology)
        exen_sampler = BenchmarkExEnSampler(testsystem.mcmc_samplers[ENV], topology, chemical_state_key, proposal_engine, testsystem.geometry_engine, scheme=scheme, options=options, storage=testsystem.storage)

        exen_sampler.hybrid_ncmc_engine.softening = 1.0
        testsystem.exen_samplers[ENV] = exen_sampler
        testsystem.exen_samplers[ENV].verbose = False
        testsystem.exen_samplers[ENV].sampler.verbose = False
        testsystem.exen_samplers[ENV].run(niterations=trials)

if __name__ == "__main__":
    benchmark_logp_0ncmc_schemes()
#    benchmark_ncmc_work_during_protocol()

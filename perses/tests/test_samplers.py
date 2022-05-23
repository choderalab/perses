"""
Samplers for perses automated molecular design.

TODO:
* Refactor tests into a test class so that AlanineDipeptideSAMS test system only needs to be constructed once for a battery of tests.
* Generalize tests of samplers to iterate over all PersesTestSystem subclasses

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from nose.plugins.attrib import attr

import os
import os.path
from functools import partial
from unittest import skipIf


running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

################################################################################
# TEST MCMCSAMPLER
################################################################################

@skipIf(running_on_github_actions, "Skip analysis test on GH Actions.  Currently broken")
def test_valence():
    """
    Test valence-only test system.
    """
    # TODO: Test that proper statistics (equal sampling, zero free energy differences) are obtained.

    testsystem_names = ['ValenceSmallMoleculeLibraryTestSystem']
    niterations = 2 # number of iterations to run
    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        # Instantiate test system.
        testsystem = testsystem_class()
        # Test ExpandedEnsembleSampler samplers.
        #for environment in testsystem.environments:
        #    exen_sampler = testsystem.exen_samplers[environment]
        #    f = partial(exen_sampler.run, niterations)
        #    f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
        #    yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            testsystem.exen_samplers[environment].pdbfile = open('sams.pdb', 'w') # DEBUG
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler for implicit hydration free energy
        from perses.samplers.samplers import MultiTargetDesign
        # Construct a target function for identifying mutants that maximize the peptide implicit solvent hydration free energy
        for environment in testsystem.environments:
            target_samplers = { testsystem.sams_samplers[environment] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
            designer = MultiTargetDesign(target_samplers)
            f = partial(designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f

def test_testsystems_gh_actions():
    """
    Test samplers on basic test systems for GH Actions.
    """
    # These tests have to work for the first paper.
    testsystem_names = ['ValenceSmallMoleculeLibraryTestSystem', 'AlkanesTestSystem', 'FusedRingsTestSystem', 'T4LysozymeInhibitorsTestSystem']

    # If TESTSYSTEMS environment variable is specified, test those systems.
    if 'TESTSYSTEMS' in os.environ:
        testsystem_names = os.environ['TESTSYSTEMS'].split(' ')

    run_samplers(testsystem_names)

@attr('advanced')
def test_testsystems_advanced():
    """
    Test samplers on advanced test systems.
    """
    testsystem_names = ['ImidazoleProtonationStateTestSystem', 'AblImatinibResistanceTestSystem', 'KinaseInhibitorsTestSystem', 'AlanineDipeptideTestSystem', 'AblAffinityTestSystem', 'T4LysozymeMutationTestSystem']
    run_samplers(testsystem_names)

def run_samplers(testsystem_names, niterations=5):
    """
    Run sampler stack on named test systems.

    Parameters
    ----------
    testsystem_names : list of str
        Names of test systems to run
    niterations : int, optional, default=5
        Number of iterations to run

    """
    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        # Instantiate test system.
        testsystem = testsystem_class()
        # Test MCMCSampler samplers.
        for environment in testsystem.environments:
            mcmc_sampler = testsystem.mcmc_samplers[environment]
            f = partial(mcmc_sampler.run, niterations)
            f.description = "Testing MCMC sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test ExpandedEnsembleSampler samplers.
        for environment in testsystem.environments:
            exen_sampler = testsystem.exen_samplers[environment]
            f = partial(exen_sampler.run, niterations)
            f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler, if present.
        if hasattr(testsystem, 'designer') and (testsystem.designer is not None):
            f = partial(testsystem.designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f

# HBM - this seems quite broken so removing for now. Think it will be addressed in protein-mutations PR
#def test_hybrid_scheme():
#    """
#    Test ncmc hybrid switching
#    """
#    from perses.tests.testsystems import AlanineDipeptideTestSystem
#    niterations = 5 # number of iterations to run
#
#    if 'TESTSYSTEMS' in os.environ:
#        testsystem_names = os.environ['TESTSYSTEMS'].split(' ')
#        if 'AlanineDipeptideTestSystem' not in testsystem_names:
#            return
#
#    # Instantiate test system.
#    testsystem = AlanineDipeptideTestSystem()
#    # Test MCMCSampler samplers.
#    testsystem.environments = ['vacuum']
#    # Test ExpandedEnsembleSampler samplers.
#    from perses.samplers.samplers import ExpandedEnsembleSampler
#    for environment in testsystem.environments:
#        chemical_state_key = testsystem.proposal_engines[environment].compute_state_key(testsystem.topologies[environment])
#        testsystem.exen_samplers[environment] = ExpandedEnsembleSampler(testsystem.mcmc_samplers[environment], testsystem.topologies[environment], chemical_state_key, testsystem.proposal_engines[environment], geometry.FFAllAngleGeometryEngine(metadata={}), options={'nsteps':1})
#        exen_sampler = testsystem.exen_samplers[environment]
#        exen_sampler.verbose = True
#        f = partial(exen_sampler.run, niterations)
#        f.description = "Testing expanded ensemble sampler with AlanineDipeptideTestSystem '%s'" % environment
#        yield f

def test_RESTCapableHybridTopologyFactory_repex_neutral_mutation():
    """
    Run ALA->THR and THR->ALA repex with the RESTCapableHybridTopologyFactory and make sure that the free energies are equal and opposite.

    """

    import tempfile
    from pkg_resources import resource_filename
    import numpy as np

    from openmm import unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from perses.dispersed.utils import configure_platform
    from perses.samplers.multistate import HybridRepexSampler

    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer
    from openmmtools import cache, utils, mcmc
    platform = configure_platform(utils.get_fastest_platform().getName())

    data = {}
    n_iterations = 2
    mutations = [('ala', 'thr'), ('thr', 'ala')]
    with tempfile.TemporaryDirectory() as temp_dir:
        for wt_name, mutant_name in mutations:
            # Generate htf
            pdb_filename = resource_filename("perses", f"data/{wt_name}_vacuum.pdb")
            solvent_delivery = PointMutationExecutor(
                pdb_filename,
                "1",
                "2",
                mutant_name.upper(),
                phase="solvent",
                w_scale=0.3,
                generate_unmodified_hybrid_topology_factory=False,
                generate_rest_capable_hybrid_topology_factory=True,
                conduct_endstate_validation=False
            )
            htf = solvent_delivery.get_apo_rest_htf()

            # Make sure LRC is set correctly
            hybrid_system = htf.hybrid_system
            force_dict = {force.getName(): index for index, force in enumerate(hybrid_system.getForces())}
            custom_force = hybrid_system.getForce(force_dict['CustomNonbondedForce_sterics'])
            nonbonded_force = hybrid_system.getForce(force_dict['NonbondedForce_sterics'])
            custom_force.setUseLongRangeCorrection(False)
            nonbonded_force.setUseDispersionCorrection(True)

            # Set up repex simulation
            reporter_file = os.path.join(temp_dir, f"{wt_name}-{mutant_name}")
            reporter = MultiStateReporter(reporter_file, checkpoint_interval=10)
            hss = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=4.0 * unit.femtoseconds,
                                                                                   collision_rate=1.0 / unit.picosecond,
                                                                                   n_steps=250,
                                                                                   reassign_velocities=True,
                                                                                   n_restart_attempts=20,
                                                                                   splitting="V R R R O R R R V",
                                                                                   constraint_tolerance=1e-06),
                                                                                   replica_mixing_scheme='swap-all',
                                    hybrid_factory=htf,
                                    online_analysis_interval=None)
            hss.setup(n_states=12, temperature=300 * unit.kelvin, t_max=300 * unit.kelvin,
                      storage_file=reporter, endstates=False)
            hss.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
            hss.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)

            # Run simulation
            hss.extend(n_iterations)

            # Retrieve free energy
            reporter.close()
            reporter = MultiStateReporter(reporter_file)
            analyzer = MultiStateSamplerAnalyzer(reporter, max_n_iterations=n_iterations)
            f_ij, df_ij = analyzer.get_free_energy()
            data[f"{wt_name}-{mutant_name}"] = {'free_energy': f_ij[0, -1], 'error': df_ij[0, -1]}

        assert np.isclose(np.array([data['ala-thr']['free_energy']]), np.array([data['thr-ala']['free_energy'] * -1 ]))

def test_RESTCapableHybridTopologyFactory_repex_charge_mutation():
    """
    Run ARG->ALA->ARG and LYS->ALA->LYS repex with the RESTCapableHybridTopologyFactory and make sure that the free energies are the same.

    We do this because the free energy of ARG->ALA will not be equal and opposite of that of ALA->ARG. There will be an energy mismatch
    because we introduce counterions by turning a water into an ion, so the ARG system in ARG->ALA is not the same as in ALA->ARG.

    Therefore, check whether two different + -> 0 -> + transformations are equal to make sure that introducing a counterion is self consistent.

    Note: Another major difference between this test and test_RESTCapableHybridTopologyFactory_repex_neutral_mutation is
    the use of solvated PDBs as input (vs the neutral mutation test uses vacuum PDBs and requires the PointMutationExecutor to solvate).
    This difference is because the ARG and LYS dipeptide PDBs were generated using the geometry engine and were therefore clashy,
    so we needed to run equilibration before using them as inputs. The ALA and THR PDBs were not clashy.
    """

    import tempfile
    from pkg_resources import resource_filename
    import numpy as np

    from openmm import unit, app

    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from perses.dispersed.utils import configure_platform
    from perses.samplers.multistate import HybridRepexSampler

    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer
    from openmmtools import cache, utils, mcmc
    platform = configure_platform(utils.get_fastest_platform().getName())

    data = {}
    n_iterations = 2
    d_mutations = {'forward': [('arg', 'ala'), ('lys', 'ala')], 'reverse': [('ala', 'arg'), ('ala', 'lys')]}

    with tempfile.TemporaryDirectory() as temp_dir:
        for mutation_type, mutations in d_mutations.items():
            for wt_name, mutant_name in mutations:
                # Generate htf
                pdb_filename = resource_filename("perses", f"data/{wt_name}_solvated.cif") if mutation_type == 'forward' else os.path.join(temp_dir, f"{wt_name}.cif")
                solvent_delivery = PointMutationExecutor(
                    pdb_filename,
                    "1",
                    "2",
                    mutant_name.upper(),
                    solvate=False,
                    w_scale=0.3,
                    generate_unmodified_hybrid_topology_factory=False,
                    generate_rest_capable_hybrid_topology_factory=True,
                    conduct_endstate_validation=False
                )
                htf = solvent_delivery.get_apo_rest_htf()

                # Save the new
                if mutation_type == 'forward':
                    app.PDBxFile.writeFile(htf._topology_proposal.new_topology,
                                          htf.new_positions(htf.hybrid_positions),
                                          open(os.path.join(temp_dir, f"{mutant_name}.cif"), "w"),
                                          keepIds=True)

                # Make sure LRC is set correctly
                hybrid_system = htf.hybrid_system
                force_dict = {force.getName(): index for index, force in enumerate(hybrid_system.getForces())}
                custom_force = hybrid_system.getForce(force_dict['CustomNonbondedForce_sterics'])
                nonbonded_force = hybrid_system.getForce(force_dict['NonbondedForce_sterics'])
                custom_force.setUseLongRangeCorrection(False)
                nonbonded_force.setUseDispersionCorrection(True)

                # Set up repex simulation
                reporter_file = os.path.join(temp_dir, f"{wt_name}-{mutant_name}.nc")
                reporter = MultiStateReporter(reporter_file, checkpoint_interval=10)
                hss = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=4.0 * unit.femtoseconds,
                                                                                       collision_rate=1.0 / unit.picosecond,
                                                                                       n_steps=250,
                                                                                       reassign_velocities=True,
                                                                                       n_restart_attempts=20,
                                                                                       splitting="V R R R O R R R V",
                                                                                       constraint_tolerance=1e-06),
                                                                                       replica_mixing_scheme='swap-all',
                                        hybrid_factory=htf,
                                        online_analysis_interval=None)
                hss.setup(n_states=12, temperature=300 * unit.kelvin, t_max=300 * unit.kelvin,
                          storage_file=reporter, endstates=False)
                hss.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
                hss.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)

                # Run simulation
                hss.extend(n_iterations)

                # Retrieve free energy
                reporter.close()
                reporter = MultiStateReporter(reporter_file)
                analyzer = MultiStateSamplerAnalyzer(reporter, max_n_iterations=n_iterations)
                f_ij, df_ij = analyzer.get_free_energy()
                data[f"{wt_name}-{mutant_name}"] = {'free_energy': f_ij[0, -1], 'error': df_ij[0, -1]}

        assert np.isclose(np.array([data['arg-ala']['free_energy'] + data['ala-arg']['free_energy']]), np.array([data['lys-ala']['free_energy'] + data['ala-lys']['free_energy']]))



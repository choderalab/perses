import os
import pytest
from perses.tests.utils import enter_temp_directory

# Tolerance for energy differences tests
ENERGY_DIFF_TOLERANCE_KT = 0.5


@pytest.mark.gpu_needed
def test_RESTCapableHybridTopologyFactory_repex_neutral_mutation():
    """
    Run ALA->THR and THR->ALA repex with the RESTCapableHybridTopologyFactory and make sure that the free energies are
    equal and opposite.

    """

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
    n_iterations = 250
    mutations = [('ala', 'thr'), ('thr', 'ala')]
    with enter_temp_directory() as temp_dir:
        for wt_name, mutant_name in mutations:
            # Generate htf
            pdb_filename = resource_filename("perses", f"data/{wt_name}_vacuum.pdb")
            solvent_delivery = PointMutationExecutor(
                pdb_filename,
                "1",
                "2",
                mutant_name.upper(),
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
                      storage_file=reporter, endstates=True)
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

        forward_data = data['ala-thr']['free_energy']
        reverse_data = data['thr-ala']['free_energy'] * -1
        assert np.isclose([forward_data], [reverse_data], atol=ENERGY_DIFF_TOLERANCE_KT), \
            f"ALA-THR is {forward_data}. THR-ALA is {reverse_data}."


@pytest.mark.gpu_needed
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
    n_iterations = 2000
    d_mutations = {'forward': [('arg', 'ala'), ('lys', 'ala')], 'reverse': [('ala', 'arg'), ('ala', 'lys')]}

    with enter_temp_directory() as temp_dir:
        for mutation_type, mutations in d_mutations.items():
            for wt_name, mutant_name in mutations:
                # Generate htf
                pdb_filename = resource_filename("perses", f"data/{wt_name}_solvated.cif") if mutation_type == 'forward' else os.path.join(temp_dir, f"{wt_name}.cif")
                solvent_delivery = PointMutationExecutor(
                    pdb_filename,
                    "1",
                    "2",
                    mutant_name.upper(),
                    is_solvated=True,
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
                          storage_file=reporter, endstates=True)
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

        arg_ala_forward = data['arg-ala']['free_energy']
        arg_ala_reverse = data['ala-arg']['free_energy']
        lys_ala_forward = data['lys-ala']['free_energy']
        lys_ala_reverse = data['ala-lys']['free_energy']
        arg_ala_arg = arg_ala_forward + arg_ala_reverse
        lys_ala_lys = lys_ala_forward + lys_ala_reverse

        assert np.isclose([arg_ala_arg], [lys_ala_lys], atol=ENERGY_DIFF_TOLERANCE_KT), \
            f"ARG-ALA-ARG is {arg_ala_arg}. LYS-ALA-LYS is {lys_ala_lys}."

def test_RESTCapableHybridTopologyFactory_repex_neutral_transformation():
    """
    Run CCC->CCCC and CCCC->CCC repex with the RESTCapableHybridTopologyFactory and make sure that the free energies are
    equal and opposite.

    """

    from pkg_resources import resource_filename
    import numpy as np

    from openmm import unit

    from perses.dispersed.utils import configure_platform
    from perses.app.relative_setup import RelativeFEPSetup
    from perses.samplers.multistate import HybridRepexSampler
    from perses.annihilation.lambda_protocol import LambdaProtocol
    from perses.annihilation.relative import RESTCapableHybridTopologyFactory

    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer
    from openmmtools import cache, utils, mcmc
    platform = configure_platform(utils.get_fastest_platform().getName())

    data = {}
    n_iterations = 3000
    n_states = 12
    transformations = [(0, 1), (1, 0)]  # Ligand indices in perses/data/test.smi

    with enter_temp_directory() as temp_dir:
        for ligand_A_index, ligand_B_index in transformations:
            # Generate topology proposal and positions
            smiles_filename = resource_filename("perses", os.path.join("data", "test.smi"))
            fe_setup = RelativeFEPSetup(
                ligand_input=smiles_filename,
                old_ligand_index=ligand_A_index,
                new_ligand_index=ligand_B_index,
                forcefield_files=["amber14/tip3p.xml"],
                small_molecule_forcefield="gaff-2.11",
                phases=["solvent"],
                solvent_padding=1.7 * unit.nanometers)

            # Generate htf
            htf = RESTCapableHybridTopologyFactory(
                topology_proposal=fe_setup.solvent_topology_proposal,
                current_positions=fe_setup.solvent_old_positions,
                new_positions=fe_setup.solvent_new_positions
            )

            # Create a reporter
            selection = "not water"
            reporter_file = os.path.join(temp_dir, f"{ligand_A_index}_{ligand_B_index}.nc")
            reporter = MultiStateReporter(
                reporter_file,
                analysis_particle_indices=htf.hybrid_topology.select(selection),
                checkpoint_interval=10)

            # Build the hybrid repex sampler
            sampler = HybridRepexSampler(
                mcmc_moves=mcmc.LangevinSplittingDynamicsMove(
                    timestep=4.0 * unit.femtoseconds,
                    collision_rate=1.0 / unit.picosecond,
                    n_steps=50,
                    reassign_velocities=False,
                    n_restart_attempts=20,
                    splitting="V R R R O R R R V",
                    constraint_tolerance=1e-06),
                replica_mixing_scheme='swap-all',
                hybrid_factory=htf,
                online_analysis_interval=10)

            sampler.setup(
                n_states=n_states,
                temperature=300 * unit.kelvin,
                storage_file=reporter,
                minimisation_steps=0,
                endstates=True)

            sampler.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
            sampler.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)

            # Run repex
            sampler.extend(n_iterations)

            # Retrieve free energy
            reporter.close()
            reporter = MultiStateReporter(reporter_file)
            analyzer = MultiStateSamplerAnalyzer(reporter, max_n_iterations=n_iterations)
            f_ij, df_ij = analyzer.get_free_energy()
            data[f"{ligand_A_index}-{ligand_B_index}"] = {'free_energy': f_ij[0, -1], 'error': df_ij[0, -1]}

        DDG = abs(data['0-1']['free_energy'] - data['1-0']['free_energy'] * -1)
        dDDG = np.sqrt(data['0-1']['error'] ** 2 + data['1-0']['error'] ** 2)
        assert DDG < 6 * dDDG, f"DDG ({DDG}) is greater than 6 * dDDG ({6 * dDDG})"



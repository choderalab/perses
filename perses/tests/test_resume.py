def test_cli_resume():
    import os
    import subprocess
    import tempfile

    import yaml
    from pkg_resources import resource_filename

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        # Need to get path to examples dir
        protein_pdb = resource_filename(
            "perses", os.path.join("examples", "cdk2-example", "CDK2_protein.pdb")
        ).replace("/perses", "", 1)
        ligand_file = resource_filename(
            "perses",
            os.path.join("examples", "cdk2-example", "CDK2_ligands_shifted.sdf"),
        ).replace("/perses", "", 1)

        document = """
        atom_selection: not water
        checkpoint_interval: 5
        fe_type: repex
        forcefield_files:
        - amber/ff14SB.xml
        - amber/tip3p_standard.xml
        - amber/tip3p_HFE_multivalent.xml
        - amber/phosaa10.xml
        n_cycles: 10
        n_equilibration_iterations: 10
        n_states: 3
        n_steps_per_move_application: 50
        new_ligand_index: 15
        old_ligand_index: 14
        phases:
        - vacuum
        pressure: 1.0
        save_setup_pickle_as: fesetup_hbonds.pkl
        small_molecule_forcefield: openff-1.0.0
        solvent_padding: 9.0
        temperature: 300.0
        timestep: 4.0
        trajectory_directory: cdk2_repex_hbonds
        trajectory_prefix: cdk2
        """

        y_doc = yaml.load(document, Loader=yaml.UnsafeLoader)
        y_doc["protein_pdb"] = protein_pdb
        y_doc["ligand_file"] = ligand_file
        with open("test.yml", "w") as outfile:
            yaml.dump(y_doc, outfile)
        subprocess.run(["perses-relative", "test.yml"])


def test_resume_small_molecule(tmp_path):
    import logging
    import os

    import simtk.unit as unit
    from openmmtools import mcmc
    from openmmtools.multistate import MultiStateReporter
    from pkg_resources import resource_filename

    from perses.annihilation.lambda_protocol import LambdaProtocol
    from perses.annihilation.relative import HybridTopologyFactory
    from perses.app.relative_setup import RelativeFEPSetup
    from perses.samplers.multistate import HybridRepexSampler

    os.chdir(tmp_path)

    smiles_filename = resource_filename("perses", os.path.join("data", "test.smi"))
    fe_setup = RelativeFEPSetup(
        ligand_input=smiles_filename,
        old_ligand_index=0,
        new_ligand_index=1,
        forcefield_files=[],
        small_molecule_forcefield="gaff-2.11",
        phases=["vacuum"],
    )

    htf = HybridTopologyFactory(
        topology_proposal=fe_setup._vacuum_topology_proposal,
        current_positions=fe_setup._vacuum_positions_old,
        new_positions=fe_setup._vacuum_positions_new,
        neglected_new_angle_terms=fe_setup._vacuum_forward_neglected_angles,
        neglected_old_angle_terms=fe_setup._vacuum_reverse_neglected_angles,
        softcore_LJ_v2=True,
        interpolate_old_and_new_14s=False,
    )

    # Build the hybrid repex samplers
    _logger = logging.getLogger()
    _logger.setLevel(logging.DEBUG)
    selection = "not water"
    checkpoint_interval = 5
    n_states = 3
    n_cycles = 10
    lambda_protocol = LambdaProtocol(functions="default")
    reporter_file = tmp_path / "cdk2_solvent.nc"
    reporter = MultiStateReporter(
        reporter_file,
        analysis_particle_indices=htf.hybrid_topology.select(selection),
        checkpoint_interval=checkpoint_interval,
    )
    hss = HybridRepexSampler(
        mcmc_moves=mcmc.LangevinSplittingDynamicsMove(
            timestep=4.0 * unit.femtoseconds,
            collision_rate=5.0 / unit.picosecond,
            n_steps=250,
            reassign_velocities=False,
            n_restart_attempts=20,
            splitting="V R R R O R R R V",
            constraint_tolerance=1e-06,
        ),
        hybrid_factory=htf,
        online_analysis_interval=10,
    )
    hss.setup(
        n_states=n_states,
        temperature=300 * unit.kelvin,
        storage_file=reporter,
        lambda_protocol=lambda_protocol,
    )
    hss.extend(n_cycles)
    del hss

    # Load repex simulation
    reporter = MultiStateReporter(reporter_file, checkpoint_interval=10)
    simulation = HybridRepexSampler.from_storage(reporter)

    # Resume simulation
    simulation.extend(5)

    assert simulation.iteration == 15


def test_resume_protien_mutation_with_checkpoint(tmp_path):
    import logging

    import simtk.openmm.app as app
    from openeye import oechem
    from openmmtools import cache, mcmc
    from openmmtools.multistate import MultiStateReporter
    from pkg_resources import resource_filename
    from simtk import unit

    from perses.annihilation.lambda_protocol import LambdaProtocol
    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from perses.samplers.multistate import HybridRepexSampler

    pdb_filename = resource_filename("perses", "data/ala_vacuum.pdb")
    solvent_delivery = PointMutationExecutor(
        pdb_filename,
        "1",
        "2",
        "ASP",
        flatten_torsions=True,
        flatten_exceptions=True,
        conduct_endstate_validation=False,
        barostat=None,
        phase="vaccum",
        periodic_forcefield_kwargs=None,
        nonperiodic_forcefield_kwargs={"nonbondedMethod": app.NoCutoff},
    )
    htf = solvent_delivery.get_apo_htf()

    # Build the hybrid repex samplers
    _logger = logging.getLogger()
    _logger.setLevel(logging.DEBUG)
    selection = "not water"
    checkpoint_interval = 5
    n_states = 3
    n_cycles = 10
    lambda_protocol = LambdaProtocol(functions="default")
    reporter_file = tmp_path / "cdk2_solvent.nc"
    reporter = MultiStateReporter(
        reporter_file,
        analysis_particle_indices=htf.hybrid_topology.select(selection),
        checkpoint_interval=checkpoint_interval,
    )
    hss = HybridRepexSampler(
        mcmc_moves=mcmc.LangevinSplittingDynamicsMove(
            timestep=4.0 * unit.femtoseconds,
            collision_rate=5.0 / unit.picosecond,
            n_steps=250,
            reassign_velocities=False,
            n_restart_attempts=20,
            splitting="V R R R O R R R V",
            constraint_tolerance=1e-06,
        ),
        hybrid_factory=htf,
        online_analysis_interval=10,
    )
    hss.setup(
        n_states=n_states,
        temperature=300 * unit.kelvin,
        storage_file=reporter,
        lambda_protocol=lambda_protocol,
    )
    hss.extend(n_cycles)
    del hss

    # Load repex simulation
    reporter = MultiStateReporter(reporter_file, checkpoint_interval=10)
    simulation = HybridRepexSampler.from_storage(reporter)

    # Resume simulation
    simulation.extend(5)

    assert simulation.iteration == 15


def test_resume_protein_mutation_no_checkpoint(tmp_path):
    import logging

    import simtk.openmm.app as app
    import simtk.unit as unit
    from openmmtools import mcmc
    from openmmtools.multistate import MultiStateReporter
    from pkg_resources import resource_filename

    from perses.annihilation.lambda_protocol import LambdaProtocol
    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from perses.samplers.multistate import HybridRepexSampler

    pdb_filename = resource_filename("perses", "data/ala_vacuum.pdb")

    solvent_delivery = PointMutationExecutor(
        pdb_filename,
        "1",
        "2",
        "ASP",
        flatten_torsions=True,
        flatten_exceptions=True,
        conduct_endstate_validation=False,
        barostat=None,
        phase="vaccum",
        periodic_forcefield_kwargs=None,
        nonperiodic_forcefield_kwargs={"nonbondedMethod": app.NoCutoff},
    )
    htf = solvent_delivery.get_apo_htf()

    # Build the hybrid repex samplers
    _logger = logging.getLogger()
    _logger.setLevel(logging.DEBUG)
    selection = "not water"
    checkpoint_interval = 5
    n_states = 3
    n_cycles = 10
    lambda_protocol = LambdaProtocol(functions="default")
    reporter_file = tmp_path / "cdk2_solvent.nc"
    reporter = MultiStateReporter(
        reporter_file,
        analysis_particle_indices=htf.hybrid_topology.select(selection),
        checkpoint_interval=checkpoint_interval,
    )
    hss = HybridRepexSampler(
        mcmc_moves=mcmc.LangevinSplittingDynamicsMove(
            timestep=4.0 * unit.femtoseconds,
            collision_rate=5.0 / unit.picosecond,
            n_steps=250,
            reassign_velocities=False,
            n_restart_attempts=20,
            splitting="V R R R O R R R V",
            constraint_tolerance=1e-06,
        ),
        hybrid_factory=htf,
        online_analysis_interval=10,
    )
    hss.setup(
        n_states=n_states,
        temperature=300 * unit.kelvin,
        storage_file=reporter,
        lambda_protocol=lambda_protocol,
    )

    hss.extend(n_cycles)
    hss.extend(5)

    assert hss.iteration == 15

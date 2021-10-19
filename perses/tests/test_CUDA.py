import pytest


@pytest.mark.gpu_ci
def test_example_barnase_barstar_neq_switching(tmp_path):
    import logging
    import os
    import pathlib
    import time

    import numpy as np
    from openmmtools.integrators import PeriodicNonequilibriumIntegrator
    from pkg_resources import resource_filename
    from simtk import openmm, unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor

    # change to temp dir
    os.chdir(tmp_path)

    # Set up logger
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    # Simulation parameters -- modify as needed
    nsteps_eq = 2
    nsteps_neq = 32
    neq_splitting = "V R H O R V"
    timestep = 4.0 * unit.femtosecond
    platform_name = "CUDA"
    temperature = 300 * unit.kelvin
    save_freq_eq = 1
    save_freq_neq = 2
    outdir_path = "output/"

    # Build HybridTopologyFactory
    solvent_delivery = PointMutationExecutor(
        resource_filename(
            "perses",
            os.path.join("data", "barstar-mutation", "1brs_barstar_renumbered.pdb"),
        ),
        "1",
        "2",
        "ALA",
        ligand_input=resource_filename(
            "perses",
            os.path.join("data", "barstar-mutation", "1brs_barnase_renumbered.pdb"),
        ),
        ionic_strength=0.05 * unit.molar,
        flatten_torsions=True,
        flatten_exceptions=True,
        conduct_endstate_validation=False,
    )
    htf = solvent_delivery.get_apo_htf()

    # Define lambda functions
    x = "lambda"
    DEFAULT_ALCHEMICAL_FUNCTIONS = {
        "lambda_sterics_core": x,
        "lambda_electrostatics_core": x,
        "lambda_sterics_insert": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
        "lambda_sterics_delete": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
        "lambda_electrostatics_insert": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
        "lambda_electrostatics_delete": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
        "lambda_bonds": x,
        "lambda_angles": x,
        "lambda_torsions": x,
    }

    system = htf.hybrid_system
    positions = htf.hybrid_positions

    # Set up integrator
    integrator = PeriodicNonequilibriumIntegrator(
        DEFAULT_ALCHEMICAL_FUNCTIONS,
        nsteps_eq,
        nsteps_neq,
        neq_splitting,
        timestep=timestep,
        temperature=temperature,
    )

    # Set up context
    platform = openmm.Platform.getPlatformByName(platform_name)
    if platform_name in ["CUDA", "OpenCL"]:
        platform.setPropertyDefaultValue("Precision", "mixed")
    if platform_name in ["CUDA"]:
        platform.setPropertyDefaultValue("DeterministicForces", "true")
    context = openmm.Context(system, integrator, platform)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    # Minimize
    openmm.LocalEnergyMinimizer.minimize(context)

    # Run neq
    forward_works_master, reverse_works_master = list(), list()
    forward_eq_old, forward_eq_new, forward_neq_old, forward_neq_new = (
        list(),
        list(),
        list(),
        list(),
    )
    reverse_eq_new, reverse_eq_old, reverse_neq_old, reverse_neq_new = (
        list(),
        list(),
        list(),
        list(),
    )
    # Equilibrium (lambda = 0)
    for step in range(nsteps_eq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        if step % save_freq_eq == 0:
            _logger.info(
                f"Step: {step}, equilibrating at lambda = 0, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            old_pos = np.asarray(htf.old_positions(pos))
            new_pos = np.asarray(htf.new_positions(pos))
            forward_eq_old.append(old_pos)
            forward_eq_new.append(new_pos)

    # Forward (0 -> 1)
    forward_works = [integrator.get_protocol_work(dimensionless=True)]
    for fwd_step in range(nsteps_neq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        forward_works.append(integrator.get_protocol_work(dimensionless=True))
        if fwd_step % save_freq_neq == 0:
            _logger.info(
                f"forward NEQ step: {fwd_step}, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            old_pos = np.asarray(htf.old_positions(pos))
            new_pos = np.asarray(htf.new_positions(pos))
            forward_neq_old.append(old_pos)
            forward_neq_new.append(new_pos)
    forward_works_master.append(forward_works)

    # Equilibrium (lambda = 1)
    for step in range(nsteps_eq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        if step % save_freq_eq == 0:
            _logger.info(
                f"Step: {step}, equilibrating at lambda = 1, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            old_pos = np.asarray(htf.old_positions(pos))
            new_pos = np.asarray(htf.new_positions(pos))
            reverse_eq_new.append(new_pos)
            reverse_eq_old.append(old_pos)

    # Reverse work (1 -> 0)
    reverse_works = [integrator.get_protocol_work(dimensionless=True)]
    for rev_step in range(nsteps_neq):
        initial_time = time.time()
        integrator.step(1)
        elapsed_time = (time.time() - initial_time) * unit.seconds
        reverse_works.append(integrator.get_protocol_work(dimensionless=True))
        if rev_step % save_freq_neq == 0:
            _logger.info(
                f"reverse NEQ step: {rev_step}, took: {elapsed_time / unit.seconds} seconds"
            )
            pos = context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(asNumpy=True)
            old_pos = np.asarray(htf.old_positions(pos))
            new_pos = np.asarray(htf.new_positions(pos))
            reverse_neq_old.append(old_pos)
            reverse_neq_new.append(new_pos)
    reverse_works_master.append(reverse_works)

    # Save output
    # create output directory if it does not exist
    out_path = pathlib.Path(outdir_path)
    out_path.mkdir(parents=True, exist_ok=True)
    # Save works
    with open(os.path.join(out_path, f"forward.npy"), "wb") as out_file:
        np.save(out_file, forward_works_master)
    with open(os.path.join(out_path, f"reverse.npy"), "wb") as out_file:
        np.save(out_file, reverse_works_master)

    # Save trajs
    with open(os.path.join(out_path, f"forward_eq_old.npy"), "wb") as out_file:
        np.save(out_file, np.array(forward_eq_old))
    with open(os.path.join(out_path, f"forward_eq_new.npy"), "wb") as out_file:
        np.save(out_file, np.array(forward_eq_new))
    with open(os.path.join(out_path, f"reverse_eq_new.npy"), "wb") as out_file:
        np.save(out_file, np.array(reverse_eq_new))
    with open(os.path.join(out_path, f"reverse_eq_old.npy"), "wb") as out_file:
        np.save(out_file, np.array(reverse_eq_old))
    with open(os.path.join(out_path, f"forward_neq_old.npy"), "wb") as out_file:
        np.save(out_file, np.array(forward_neq_old))
    with open(os.path.join(out_path, f"forward_neq_new.npy"), "wb") as out_file:
        np.save(out_file, np.array(forward_neq_new))
    with open(os.path.join(out_path, f"reverse_neq_old.npy"), "wb") as out_file:
        np.save(out_file, np.array(reverse_neq_old))
    with open(os.path.join(out_path, f"reverse_neq_new.npy"), "wb") as out_file:
        np.save(out_file, np.array(reverse_neq_new))

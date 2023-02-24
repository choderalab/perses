from typing import Optional, Iterable, List, Dict, Any

import datetime
import logging
import time

from gufe.settings.models import ProtocolSettings
from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import ComponentMapping
from gufe.protocols import (
    Protocol,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
)

from perses.app.relative_setup import RelativeFEPSetup
from perses.app.setup_relative_calculation import get_openmm_platform
from perses.annihilation.relative import HybridTopologyFactory

from openff.units.openmm import to_openmm

# Specific instance of logger for this module
logger = logging.getLogger(__name__)


class SimulationUnit(ProtocolUnit):
    """
    Monolithic unit for simulation. It runs NEQ switching simulation from chemical systems and stores the
    work computed in numpy-formatted files, to be analyzed by another unit.
    """
    @staticmethod
    def _check_states_compatibility(state_a, state_b):
        """
        Checks that both states have the same solvent parameters and receptor.

        Parameters
        ----------
        state_a : gufe.state.State
            Origin state for the alchemical transformation.
        state_b :
            Destination state for the alchemical transformation.
        """
        # If any of them has a solvent, check the parameters are the same
        if any(["solvent" in state.components for state in (state_a, state_b)]):
            assert state_a.get("solvent") == state_b.get("solvent"), "Solvent parameters differ between solvent components."
        # check protein component is the same in both states if protein component is found
        if any(["protein" in state.components for state in (state_a, state_b)]):
            assert state_a.get("protein") == state_b.get("protein"), "Receptors in states are not compatible."

    @staticmethod
    def _detect_phase(state_a, state_b):
        """
        Detect phase according to the components in the input chemical state.

        Complex state is assumed if both states have ligands and protein components.

        Solvent state is assumed

        Vacuum state is assumed if only either a ligand or a protein is present
        in each of the states.

        Parameters
        ----------
        state_a : gufe.state.State
            Source state for the alchemical transformation.
        state_b : gufe.state.State
            Destination state for the alchemical transformation.

        Returns
        -------
        phase : str
            Phase name. "vacuum", "solvent" or "complex".
        component_keys : list[str]
            List of component keys to extract from states.
        """
        states = (state_a, state_b)
        # where to store the data to be returned

        # Order of phases is important! We have to check complex first and solvent second.
        key_options = {
            "complex": ["ligand", "protein", "solvent"],
            "solvent": ["ligand", "solvent"],
            "vacuum": ["ligand"]
        }
        for phase, keys in key_options.items():
            if all([key in state for state in states for key in keys]):
                detected_phase = phase
                break
        else:
            raise ValueError(
                "Could not detect phase from system states. Make sure the component in both systems match.")

        return detected_phase

    def _execute(self, ctx, *, state_a, state_b, mapping, settings, **inputs):
        """
        Execute the simulation part of the Nonequilibrium switching protocol using GUFE objects.

        Parameters
        ----------
        ctx: gufe.protocols.protocolunit.Context
            The gufe context for the unit.

        state_a : gufe.ChemicalSystem
            The initial chemical system.

        state_b : gufe.ChemicalSystem
            The objective chemical system.

        mapping : gufe.mapping.ComponentMapping
            The mapping between the two chemical systems.

        settings : gufe.settings.model.ProtocolSettings
            The settings for the protocol.

        Returns
        -------
        dict : dict[str, str]
            Dictionary with paths to work arrays, both forward and reverse, and trajectory coordinates for systems
            A and B.
        """
        # needed imports
        import numpy as np
        import openmm
        import openmm.unit as openmm_unit
        from openmmtools.integrators import PeriodicNonequilibriumIntegrator
        from perses.utils.openeye import generate_unique_atom_names
        from perses.utils.logging_utils import YAMLFormatter

        # Setting up logging to file in shared filesystem
        output_log_path = ctx.shared / "perses_neq_cycling_log.yaml"
        file_handler = logging.FileHandler(output_log_path)
        file_handler.setLevel(logging.DEBUG)  # TODO: Set to INFO in production
        log_formatter = YAMLFormatter(datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

        # Check compatibility between states (same receptor and solvent)
        self._check_states_compatibility(state_a, state_b)
        phase = self._detect_phase(state_a, state_b)

        # Get components from systems if found (None otherwise) -- NOTE: Uses hardcoded keys!
        receptor_a = state_a.components.get("protein")
        # receptor_b = state_b.components.get("protein")  # Should not be needed
        ligand_a = state_a.components.get("ligand")
        ligand_b = state_b.components.get("ligand")
        solvent_a = state_a.components.get("solvent")
        # solvent_b = state_b.components.get("solvent")  # Should not be needed


        # Check first state for receptor if not get receptor from second one
        if receptor_a:
            receptor_top = receptor_a.to_openmm_topology()
            receptor_pos = receptor_a.to_openmm_positions()
        else:
            receptor_top, receptor_pos = None, None

        # Get ligands cheminformatics molecules
        ligand_a = ligand_a.to_openeye()
        ligand_b = ligand_b.to_openeye()
        # Generating unique atom names for ligands -- openmmforcefields needs them
        ligand_a = generate_unique_atom_names(ligand_a)
        ligand_b = generate_unique_atom_names(ligand_b)

        # Get solvent parameters from component
        if solvent_a:
            ion_concentration = solvent_a.ion_concentration.to_openmm()
            positive_ion = solvent_a.positive_ion
            negative_ion = solvent_a.negative_ion
        else:
            ion_concentration, positive_ion, negative_ion = None, None, None

        # Get settings
        protocol_settings = settings.protocol_settings
        thermodynamic_settings = settings.thermo_settings
        phase = self._detect_phase(state_a, state_b)
        save_frequency = protocol_settings.save_frequency

        # Get the ligand mapping from ComponentMapping object
        ligand_mapping = mapping.componentA_to_componentB

        # Setup relative FE calculation
        fe_setup = RelativeFEPSetup(
            old_ligand=ligand_a,
            new_ligand=ligand_b,
            receptor=receptor_top,
            receptor_positions=receptor_pos,
            forcefield_files=settings.forcefield_settings.forcefields,
            small_molecule_forcefield=settings.forcefield_settings.small_molecule_forcefield,
            phases=[phase],
            transformation_atom_map=ligand_mapping,  # Handle atom mapping between systems
            ionic_strength=ion_concentration,
            positive_ion=positive_ion,
            negative_ion=negative_ion,
        )

        topology_proposals = fe_setup.topology_proposals
        old_positions = fe_setup.old_positions
        new_positions = fe_setup.new_positions

        # Generate Hybrid Topology Factory - Generic HTF
        htf = HybridTopologyFactory(
            topology_proposal=topology_proposals[phase],
            current_positions=old_positions[phase],
            new_positions=new_positions[phase],
            softcore_LJ_v2=protocol_settings.softcore_LJ_v2,
            interpolate_old_and_new_14s=protocol_settings.interpolate_old_and_new_14s,
        )

        system = htf.hybrid_system
        positions = htf.hybrid_positions

        # Set up integrator
        temperature = to_openmm(thermodynamic_settings.temperature)
        neq_steps = protocol_settings.eq_steps
        eq_steps = protocol_settings.neq_steps
        timestep = to_openmm(protocol_settings.timestep)
        splitting = protocol_settings.neq_splitting
        integrator = PeriodicNonequilibriumIntegrator(alchemical_functions=protocol_settings.lambda_functions,
                                                      nsteps_neq=neq_steps,
                                                      nsteps_eq=eq_steps,
                                                      splitting=splitting,
                                                      timestep=timestep,
                                                      temperature=temperature, )

        # Set up context
        platform = get_openmm_platform(protocol_settings.platform)
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        context.setPositions(positions)

        # Minimize
        openmm.LocalEnergyMinimizer.minimize(context)

        # Equilibrate
        context.setVelocitiesToTemperature(temperature)

        # Prepare objects to store data -- empty lists so far
        forward_eq_old, forward_eq_new, forward_neq_old, forward_neq_new = list(), list(), list(), list()
        reverse_eq_new, reverse_eq_old, reverse_neq_old, reverse_neq_new = list(), list(), list(), list()

        #  Also get the GPU information (plain try-except with nvidia-smi)

        # Equilibrium (lambda = 0)
        # start timer
        start_time = time.perf_counter()
        for step in range(neq_steps):
            integrator.step(1)  # TODO: Avoid overhead to have to check everytime of save_frequency. Use greatest common denominator also handle save_freq=0
            # Store positions and works
            if step % save_frequency == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                forward_eq_old.append(old_pos)
                forward_eq_new.append(new_pos)
        eq_forward_time = time.perf_counter()
        eq_forward_walltime = datetime.timedelta(seconds=eq_forward_time - start_time)
        logger.info(f"replicate_{self.name} Forward (lamba = 0) equilibration time : {eq_forward_walltime}")

        # Run neq
        # Forward (0 -> 1)
        forward_works = [integrator.get_protocol_work(dimensionless=True)]
        for fwd_step in range(eq_steps):
            integrator.step(1)  # TODO: step for a certain number and then retrieve work
            forward_works.append(integrator.get_protocol_work(dimensionless=True))  # TODO: we want to call this less frequently
            if fwd_step % save_frequency == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                forward_neq_old.append(old_pos)
                forward_neq_new.append(new_pos)
        neq_forward_time = time.perf_counter()
        neq_forward_walltime = datetime.timedelta(seconds=neq_forward_time - eq_forward_time)
        logger.info(f"replicate_{self.name} Forward nonequilibrium time (lambda 0 -> 1): {neq_forward_walltime}")

        # Equilibrium (lambda = 1)
        # TODO: Same thing as before
        for step in range(neq_steps):
            integrator.step(1)
            if step % save_frequency == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                reverse_eq_new.append(new_pos)  # TODO: Maybe better naming not old/new initial/final
                reverse_eq_old.append(old_pos)
        eq_reverse_time = time.perf_counter()
        eq_reverse_walltime = datetime.timedelta(seconds=eq_reverse_time - neq_forward_time)
        logger.info(f"replicate_{self.name} Reverse (lambda 1) time: {eq_reverse_walltime}")

        # Reverse work (1 -> 0)
        # TODO: Same thing as before
        reverse_works = [integrator.get_protocol_work(dimensionless=True)]
        for rev_step in range(eq_steps):
            integrator.step(1)
            reverse_works.append(integrator.get_protocol_work(dimensionless=True))
            if rev_step % save_frequency == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                reverse_neq_old.append(old_pos)
                reverse_neq_new.append(new_pos)
        neq_reverse_time = time.perf_counter()
        neq_reverse_walltime = datetime.timedelta(seconds=neq_reverse_time - eq_reverse_time)
        logger.info(f"replicate_{self.name} Reverse nonequilibrium time (lambda 1 -> 0): {neq_reverse_walltime}")
        # Elapsed time for the whole cycle
        cycle_walltime = datetime.timedelta(seconds=neq_reverse_time - start_time)
        logger.info(f"replicate_{self.name} Nonequilibrium cycle total walltime: {cycle_walltime}")

        # Computing performance in ns/day
        simulation_time = 2*(eq_steps + neq_steps)*timestep
        walltime_in_seconds = cycle_walltime.total_seconds() * openmm_unit.seconds
        estimated_performance = simulation_time.in_units_of(openmm_unit.nanosecond) / walltime_in_seconds.in_units_of(
            openmm_unit.days)  # Note: ends up unit-less
        logger.info(f"replicate_{self.name} Estimated performance: {estimated_performance} ns/day")

        # Save output
        # TODO: Assume initially we want the trajectories to understand when something wrong/weird happens.
        # Save works
        forward_work_path = ctx.shared / f"forward_{phase}_{self.name}.npy"
        reverse_work_path = ctx.shared / f"reverse_{phase}_{self.name}.npy"
        with open(forward_work_path, 'wb') as out_file:
            np.save(out_file, forward_works)
        with open(reverse_work_path, 'wb') as out_file:
            np.save(out_file, reverse_works)

        # TODO: Do we need to save the trajectories?
        # Save trajs
        # trajectory paths
        forward_eq_old_path = ctx.shared / f"forward_eq_old_{phase}_{self.name}.npy"
        forward_eq_new_path = ctx.shared / f"forward_eq_new_{phase}_{self.name}.npy"
        forward_neq_old_path = ctx.shared / f"forward_neq_old_{phase}_{self.name}.npy"
        forward_neq_new_path = ctx.shared / f"forward_neq_new_{phase}_{self.name}.npy"
        reverse_eq_new_path = ctx.shared / f"reverse_eq_new_{phase}_{self.name}.npy"
        reverse_eq_old_path = ctx.shared / f"reverse_eq_old_{phase}_{self.name}.npy"
        reverse_neq_old_path = ctx.shared / f"reverse_neq_old_{phase}_{self.name}.npy"
        reverse_neq_new_path = ctx.shared / f"reverse_neq_new_{phase}_{self.name}.npy"

        with open(forward_eq_old_path, 'wb') as out_file:
            np.save(out_file, np.array(forward_eq_old))
        with open(forward_eq_new_path, 'wb') as out_file:
            np.save(out_file, np.array(forward_eq_new))
        with open(reverse_eq_old_path, 'wb') as out_file:
            np.save(out_file, np.array(reverse_eq_old))
        with open(reverse_eq_new_path, 'wb') as out_file:
            np.save(out_file, np.array(reverse_eq_new))
        with open(forward_neq_old_path, 'wb') as out_file:
            np.save(out_file, np.array(forward_neq_old))
        with open(forward_neq_new_path, 'wb') as out_file:
            np.save(out_file, np.array(forward_neq_new))
        with open(reverse_neq_old_path, 'wb') as out_file:
            np.save(out_file, np.array(reverse_neq_old))
        with open(reverse_neq_new_path, 'wb') as out_file:
            np.save(out_file, np.array(reverse_neq_new))

        # Explicitly cleanup for GPU resources
        del context, integrator

        return {
            'forward_work': forward_work_path,
            'reverse_work': reverse_work_path,
            'forward_eq_old': forward_eq_old_path,
            'forward_eq_new': forward_eq_new_path,
            'forward_neq_old': forward_neq_old_path,
            'forward_neq_new': forward_neq_new_path,
            'reverse_eq_old': reverse_eq_old_path,
            'reverse_eq_new': reverse_eq_new_path,
            'reverse_neq_old': reverse_neq_old_path,
            'reverse_neq_new': reverse_neq_new_path,
            'log': output_log_path,
            'eq_forward_time': str(eq_forward_walltime),
            'neq_forward_time': str(neq_forward_walltime),
            'eq_reverse_time': str(eq_reverse_walltime),
            'neq_reverse_time': str(neq_reverse_walltime),
        }


class ResultUnit(ProtocolUnit):
    """
    Returns cumulated work and paths for output files.
    """

    @staticmethod
    def _execute(ctx, *, simulations, **inputs):
        import numpy as np
        # TODO: This can take the settings and process a debug flag, and populate all the paths for trajectories as needed
        # Load the works from shared serialized objects
        # import pdb
        # pdb.set_trace()
        # TODO: We need to make sure the array is the CUMULATIVE work and that we just want the last value
        cumulated_forward_works = []
        cumulated_reverse_works = []
        for simulation in simulations:
            forward_work = np.load(simulation.outputs['forward_work'])
            cumulated_forward = forward_work[-1] - forward_work[0]
            reverse_work = np.load(simulation.outputs['reverse_work'])
            cumulated_reverse = reverse_work[-1] - reverse_work[0]
            cumulated_forward_works.append(cumulated_forward)
            cumulated_reverse_works.append(cumulated_reverse)

        # Path dictionary
        paths_dict = {"forward_work": [simulation.outputs["forward_work"] for simulation in simulations],
                      "reverse_work": [simulation.outputs["reverse_work"] for simulation in simulations]}

        return {"forward_work": cumulated_forward_works,
                "reverse_work": cumulated_reverse_works,
                "paths": paths_dict,
                }


class NonEquilibriumCyclingProtocolResult(ProtocolResult):
    """
    Gathers results from different runs and computes the free energy estimates using BAR and its errors using
    bootstrapping.
    """

    def get_estimate(self):
        """
        Get a free energy estimate using bootstrap and BAR.

        Parameters
        ----------
        n_bootstraps: int
            Number of bootstrapped samples to use.

        Returns
        -------
        free_energy: float
            Free energy estimate in units of kT.

        """
        import numpy as np
        import numpy.typing as npt
        import pymbar

        forward_work: npt.NDArray[float] = np.array(self.data["forward_work"])
        reverse_work: npt.NDArray[float] = np.array(self.data["reverse_work"])
        free_energy, error = pymbar.bar.BAR(forward_work, reverse_work)

        return free_energy

    def get_uncertainty(self, n_bootstraps=1000):
        """
        Estimate uncertainty using standard deviation of the distribution of bootstrapped
        free energy (dg) samples.

        Parameters
        ----------
        n_bootstraps

        Returns
        -------
        free_energy_uncertainty: float
            Uncertainty on the free energy estimate in units of kT.

        """
        import numpy as np
        import numpy.typing as npt

        forward: npt.NDArray[float] = np.array(self.data["forward_work"])
        reverse: npt.NDArray[float] = np.array(self.data["reverse_work"])

        all_dgs = self._do_bootstrap(forward, reverse, n_bootstraps=n_bootstraps)

        # TODO: Check if standard deviation is a good uncertainty estimator
        return np.std(all_dgs)

    def get_rate_of_convergence(self):
        ...

    # @lru_cache()
    def _do_bootstrap(self, forward, reverse, n_bootstraps=1000):
        """
        Performs bootstrapping from forward and reverse cumulated works.

        Parameters
        ----------
        forward: np.ndarray[float]
            Array of cumulated works for the forward transformation
        reverse: np.ndarray[float]
            Array of cumulated works for the reverse transformation


        Returns
        -------
        free_energies: np.ndarray[Float]
            List of bootstrapped free energies in units of kT.
        """
        import pymbar
        import numpy as np
        import numpy.typing as npt

        # Check to make sure forward and reverse work values match in length
        assert len(forward) == len(reverse), "Forward and reverse work values are not paired"

        all_dgs: npt.NDArray[float] = np.zeros(n_bootstraps)  # initialize dgs array

        traj_size = len(forward)
        for i in range(n_bootstraps):
            # Sample trajectory indices with replacement
            indices = np.random.choice(np.arange(traj_size), size=[traj_size], replace=True)
            dg, ddg = pymbar.bar.BAR(forward[indices], reverse[indices])
            all_dgs[i] = dg

        return all_dgs


class NonEquilibriumCyclingProtocol(Protocol):
    """
    Run RBFE calculations between two chemical states using alchemical NEQ cycling and `gufe` objects.

    Chemical states are assumed to have the same component keys, as in, stateA should be composed
    of the same type of components as components in stateB.
    """

    result_cls = NonEquilibriumCyclingProtocolResult

    def __init__(self, settings: ProtocolSettings):
        super().__init__(settings)

    @classmethod
    def _default_settings(cls) -> ProtocolSettings:
        from perses.protocols import settings
        non_eq_settings = settings.NonEqCyclingSettings()
        return non_eq_settings

    # NOTE: create method should be really fast, since it would be running in the work units not the clients!!
    def _create(
            self,
            stateA: ChemicalSystem,
            stateB: ChemicalSystem,
            mapping: Optional[ComponentMapping] = None,
            extends: Optional[ProtocolDAGResult] = None,
    ) -> List[ProtocolUnit]:

        # Handle parameters
        # if mapping is None:
        #     raise ValueError("`mapping` is required for this Protocol")
        # if 'ligand' not in mapping:
        #     raise ValueError("'ligand' must be specified in `mapping` dict")
        # if extends:
        #     raise NotImplementedError("Can't extend simulations yet")

        # inputs to `ProtocolUnit.__init__` should either be `Gufe` objects
        # or JSON-serializable objects
        num_replicates = self.settings.protocol_settings.num_replicates

        simulations = [
            SimulationUnit(state_a=stateA, state_b=stateB, mapping=mapping, settings=self.settings, name=f"{replicate}")
            for replicate in range(num_replicates)]

        end = ResultUnit(name="result", simulations=simulations)

        return [*simulations, end]

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> Dict[str, Any]:

        from collections import defaultdict
        import numpy as np
        outputs = defaultdict(list)
        for pdr in protocol_dag_results:
            for pur in pdr.protocol_unit_results:
                if pur.name == "result":
                    outputs["forward_work"].extend(pur.outputs["forward_work"])
                    outputs["reverse_work"].extend(pur.outputs["reverse_work"])
                    outputs["work_file_paths"].extend(pur.outputs["paths"])

        # This can be populated however we want
        return outputs

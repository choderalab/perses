from typing import Optional, Iterable, List, Dict, Any

import os

from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import ComponentMapping
from gufe.protocols import (
    Protocol,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
    execute
)

from perses.app.relative_setup import RelativeFEPSetup
from perses.app.setup_relative_calculation import get_openmm_platform
from perses.annihilation.relative import HybridTopologyFactory

from openff.units.openmm import to_openmm


class SimulationUnit(ProtocolUnit):
    """
    Monolithic unit for simulation. It runs NEQ switching simulation from chemical systems and stores the
    work computed in numpy-formatted files, to be analyzed by another unit.
    """

    # TODO: This is not the best way to check for this, maybe we just want a couple of ifs instead of catching exceptions
    def _check_state_receptor(self, state):
        """
        Check receptor is found in the state, returning its topology and positions if found.

        Assumes the receptor is under the 'protein' key for the state components.

        Parameters
        ----------
        state : gufe.state.State
            The state to check for the receptor.

        Returns
        -------
        receptor_topology : openmm.app.Topology
            The topology of the receptor. None if not found.
        receptor_positions : openmm.unit.Quantity of dimension (natoms, 3) with units compatible with angstroms
            The positions of the receptor. None if not found.
        """
        try:
            receptor_component = state.components['protein']
        except KeyError:
            print("Receptor not found in chemical system. Assuming non-complex phase for system.")
            receptor_top = None
            receptor_pos = None
        else:
            receptor_top = receptor_component.to_openmm_topology()
            receptor_pos = receptor_component.to_openmm_positions()
        return receptor_top, receptor_pos

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
        """
        # needed imports
        import numpy as np
        import openmm
        from openmmtools.integrators import PeriodicNonequilibriumIntegrator
        from perses.utils.openeye import generate_unique_atom_names

        # Get receptor and ligands from states
        # NOTE: This assumes both states have/want same receptor
        # Check first state for receptor if not get receptor from second one
        receptor_top, receptor_pos = self._check_state_receptor(state_a)
        if not receptor_top:
            receptor_top, receptor_pos = self._check_state_receptor(state_b)

        # Get ligands -- using hardcoded keys for now
        ligand_a_component = state_a.components['ligand']
        ligand_b_component = state_b.components['ligand']
        ligand_a = ligand_a_component.to_openeye()
        ligand_b = ligand_b_component.to_openeye()

        # Generating unique atom names for ligands -- openmmforcefields needs them
        ligand_a = generate_unique_atom_names(ligand_a)
        ligand_b = generate_unique_atom_names(ligand_b)


        # Get settings
        forcefield_settings = settings.forcefield_settings
        alchemical_settings = settings.alchemical_settings
        integrator_settings = settings.integrator_settings
        thermodynamic_settings = settings.thermodynamic_settings
        miscellaneous_settings = settings.miscellaneous_settings
        phase = miscellaneous_settings.phase
        save_frequency = miscellaneous_settings.save_frequency

        # Get the ligand mapping from ComponentMapping object
        ligand_mapping = mapping.componentA_to_componentB

        # Interactive debugging
        # import pdb
        # pdb.set_trace()
        # Setup relative FE calculation
        fe_setup = RelativeFEPSetup(
            old_ligand=ligand_a,
            new_ligand=ligand_b,
            receptor=receptor_top,
            receptor_positions=receptor_pos,
            forcefield_files=forcefield_settings.forcefield_files,
            small_molecule_forcefield=forcefield_settings.small_molecule_forcefield,
            phases=[phase],
            transformation_atom_map=ligand_mapping,  # Handle atom mapping between systems
        )

        topology_proposals = fe_setup.topology_proposals
        old_positions = fe_setup.old_positions
        new_positions = fe_setup.new_positions

        # Generate Hybrid Topology Factory - Generic HTF
        htf = HybridTopologyFactory(
            topology_proposal=topology_proposals[phase],
            current_positions=old_positions[phase],
            new_positions=new_positions[phase],
            softcore_LJ_v2=alchemical_settings.softcore_LJ_v2,
            interpolate_old_and_new_14s=alchemical_settings.interpolate_old_and_new_14s,
        )

        system = htf.hybrid_system
        positions = htf.hybrid_positions

        # Set up integrator
        temperature = to_openmm(thermodynamic_settings.temperature)
        neq_steps = integrator_settings.eq_steps
        eq_steps = integrator_settings.neq_steps
        timestep = to_openmm(integrator_settings.timestep)
        splitting = integrator_settings.neq_splitting
        integrator = PeriodicNonequilibriumIntegrator(alchemical_functions=alchemical_settings.lambda_functions,
                                                      nsteps_neq=neq_steps,
                                                      nsteps_eq=eq_steps,
                                                      splitting=splitting,
                                                      timestep=timestep,
                                                      temperature=temperature, )

        # Set up context
        platform = get_openmm_platform(miscellaneous_settings.platform)
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        context.setPositions(positions)

        # Minimize
        openmm.LocalEnergyMinimizer.minimize(context)

        # Equilibrate
        context.setVelocitiesToTemperature(temperature)

        # Prepare objects to store data -- empty lists so far
        forward_works_main, reverse_works_main = list(), list()
        forward_eq_old, forward_eq_new, forward_neq_old, forward_neq_new = list(), list(), list(), list()
        reverse_eq_new, reverse_eq_old, reverse_neq_old, reverse_neq_new = list(), list(), list(), list()

        # Equilibrium (lambda = 0)
        for step in range(neq_steps):
            integrator.step(1)
            # Store positions and works
            if step % save_frequency == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                forward_eq_old.append(old_pos)
                forward_eq_new.append(new_pos)

        # Run neq
        # Forward (0 -> 1)
        forward_works = [integrator.get_protocol_work(dimensionless=True)]
        for fwd_step in range(eq_steps):
            integrator.step(1)
            forward_works.append(integrator.get_protocol_work(dimensionless=True))
            if fwd_step % save_frequency == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                forward_neq_old.append(old_pos)
                forward_neq_new.append(new_pos)
        forward_works_main.append(forward_works)

        # Equilibrium (lambda = 1)
        for step in range(neq_steps):
            integrator.step(1)
            if step % save_frequency == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                reverse_eq_new.append(new_pos)
                reverse_eq_old.append(old_pos)

        # Reverse work (1 -> 0)
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
        reverse_works_main.append(reverse_works)

        # Save output
        # TODO: Assume initially we want the trajectories to understand when something wrong/weird happens.
        # Save works
        forward_work_path = os.path.join(ctx.shared, f"forward_{phase}.npy")
        reverse_work_path = os.path.join(ctx.shared, f"reverse_{phase}.npy")
        with open(forward_work_path, 'wb') as out_file:
            np.save(out_file, forward_works_main)
        with open(reverse_work_path, 'wb') as out_file:
            np.save(out_file, reverse_works_main)

        # TODO: Do we need to save the trajectories?
        # Save trajs
        # trajectory paths
        forward_eq_old_path = os.path.join(ctx.shared, f"forward_eq_old_{phase}.npy")
        forward_eq_new_path = os.path.join(ctx.shared, f"forward_eq_new_{phase}.npy")
        forward_neq_old_path = os.path.join(ctx.shared, f"forward_neq_old_{phase}.npy")
        forward_neq_new_path = os.path.join(ctx.shared, f"forward_neq_new_{phase}.npy")
        reverse_eq_new_path = os.path.join(ctx.shared, f"reverse_eq_new_{phase}.npy")
        reverse_eq_old_path = os.path.join(ctx.shared, f"reverse_eq_old_{phase}.npy")
        reverse_neq_old_path = os.path.join(ctx.shared, f"reverse_neq_old_{phase}.npy")
        reverse_neq_new_path = os.path.join(ctx.shared, f"reverse_neq_new_{phase}.npy")

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
        }


class ResultUnit(ProtocolUnit):
    """
    The idea here is to get the results from the SimulationUnit and analyze/process them using an adaptation of
    what's in https://github.com/zhang-ivy/perses_protein_mutations/blob/master/code/31_rest_over_protocol/17_analyze_neq_switching_for_sukrit.ipynb
    """

    @staticmethod
    def _execute(ctx, *, phase, simulations, **inputs):
        import numpy as np
        import pymbar
        # TODO: This can take the settings and process a debug flag, and populate all the paths for trajectories as needed
        # Load the works from shared serialized objects
        forward_work = np.load(simulations[0]['forward_work'])
        reverse_work = np.load(simulations[0]['reverse_work'])
        free_energy, error = pymbar.bar.BAR(forward_work, reverse_work)

        return {"DDG": free_energy,
                "dDDG": error,
                "paths": {"forward_work": simulations[0]['forward_work'],
                          "reverse_work": simulations[0]['reverse_work']},
                }


class NonEquilibriumCyclingResult(ProtocolResult):

    def get_estimate(self):
        ...

    def get_uncertainty(self):
        ...

    def get_rate_of_convergence(self):
        ...


class NonEquilibriumCycling(Protocol):

    _results_cls = NonEquilibriumCyclingResult
    _supported_engines = ['openmm']

    @classmethod
    def _default_settings(cls):
        return {}

    # NOTE: create method should be really fast, since it would be running in the work units not the clients!!
    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[ComponentMapping] = None,
        extend_from: Optional[ProtocolDAGResult] = None,
    ) -> List[ProtocolUnit]:

        # Handle parameters
        # if mapping is None:
        #     raise ValueError("`mapping` is required for this Protocol")
        # if 'ligand' not in mapping:
        #     raise ValueError("'ligand' must be specified in `mapping` dict")
        # if extend_from:
        #     raise NotImplementedError("Can't extend simulations yet")

        # inputs to `ProtocolUnit.__init__` should either be `Gufe` objects
        # or JSON-serializable objects
        sim = SimulationUnit(state_a=stateA, state_b=stateB, mapping=mapping, settings=self.settings)

        end = ResultUnit(phase="solvent", name="result", simulations=[sim], settings=self.settings)

        return [sim, end]

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> Dict[str, Any]:

        outputs = []
        for pdr in protocol_dag_results:
            for pur in pdr.protocol_unit_results:
                if pur.name == "gather":
                    outputs.append(pur.data)

        return dict(data=outputs)


# testing example
# for informational purposes
# probably use this to develop tests in perses.tests.protocols.test_nonequilibrium_cycling.py
def protocol_dag(self, solvated_ligand, vacuum_ligand):
    protocol = NonEquilibriumCycling(settings=None)
    dag = protocol.create(
        stateA=solvated_ligand, stateB=vacuum_ligand, name="a dummy run"
    )
    
    # execute DAG locally, in-process
    dagresult: ProtocolDAGResult = execute(dag)

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

from openmmtools.utils import get_fastest_platform

from perses.app.relative_setup import RelativeFEPSetup
from perses.annihilation.relative import HybridTopologyFactory


# Global parameters for simulation
# TODO: Should we encode these in a ProtocolSettings object?
platform_name = get_fastest_platform().getName()
save_freq_eq = 1
save_freq_neq = 2

# TIER IV Settings - Specific for Relative FE in perses and Openmm
# Define lambda functions - for HTF
x = 'lambda'
DEFAULT_ALCHEMICAL_FUNCTIONS = {
    'lambda_sterics_core': x,
    'lambda_electrostatics_core': x,
    'lambda_sterics_insert': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    'lambda_sterics_delete': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    'lambda_electrostatics_insert': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    'lambda_electrostatics_delete': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    'lambda_bonds': x,
    'lambda_angles': x,
    'lambda_torsions': x
}


class SimulationUnit(ProtocolUnit):
    """
    Monolithic unit for simulation. It runs NEQ switching simulation from chemical systems and stores the
    work computed in numpy-formatted files, to be analyzed by another unit.
    """

    # TODO: This is not the best way to check for this, maybe we just want a couple of ifs instead of catching exceptions
    def _check_state_receptor(self, state):
        """
        Check receptor is found in the state, returning its topology and positions if found.
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

    def _execute(self, ctx, *, state_a, state_b, mapping, simulation_parameters, **inputs):
        """
        Execute the simulation part of the Nonequilibrium switching protocol using GUFE objects.

        Parameters
        ----------
        ctx: gufe.protocols.protocolunit.Context
        """
        # needed imports
        import numpy as np
        import openmm
        from openmm import unit
        from openmmtools.integrators import PeriodicNonequilibriumIntegrator
        from perses.utils.openeye import generate_unique_atom_names

        # Get receptor and ligands from states
        # NOTE: This assumes both states have same receptor
        # TODO: Check state_a protein and state_b protein are the same. Maybe in the Protocol._create method (as early as possible)
        # Check first state for receptor if not get receptor from second one
        receptor_top, receptor_pos = self._check_state_receptor(state_a)
        if not receptor_top:
            receptor_top, receptor_pos = self._check_state_receptor(state_b)

        # Get ligands
        ligand_a_component = state_a.components['ligand']
        ligand_b_component = state_b.components['ligand']
        ligand_a = ligand_a_component.to_openeye()
        ligand_b = ligand_b_component.to_openeye()

        # Generating unique atom names for ligands -- openmmforcefields needs them
        ligand_a = generate_unique_atom_names(ligand_a)
        ligand_b = generate_unique_atom_names(ligand_b)

        # Setting up forcefields and phases
        # TODO: These should be extracted from the ProtocolSettings information
        forcefield_files = [
            "amber/ff14SB.xml",
            "amber/tip3p_standard.xml",
            "amber/tip3p_HFE_multivalent.xml",
            "amber/phosaa10.xml",
        ]
        small_molecule_forcefield = 'openff-2.0.0'

        phase = "vacuum"  # This is probably taken from the ChemicalSystem or ProtocolSettings

        # TODO: What do we actually expect from the mapping? Index using what reference?
        # Get the ligand mapping from ComponentMapping object
        # ligand_mapping = mapping['ligand']

        # Interactive debugging
        # import pdb
        # pdb.set_trace()
        # Setup relative FE calculation
        fe_setup = RelativeFEPSetup(
            old_ligand=ligand_a,
            new_ligand=ligand_b,
            receptor=receptor_top,
            receptor_positions=receptor_pos,
            forcefield_files=forcefield_files,
            small_molecule_forcefield=small_molecule_forcefield,
            phases=phase,
            # transformation_atom_map=ligand_mapping,  # Handle atom mapping between systems
        )

        topology_proposals = fe_setup.topology_proposals
        old_positions = fe_setup.old_positions
        new_positions = fe_setup.new_positions

        # Generate Hybrid Topology Factory - Vanilla HTF
        htf = HybridTopologyFactory(
            topology_proposal=topology_proposals[phase],
            current_positions=old_positions[phase],
            new_positions=new_positions[phase],
            # TODO: I think the following should be extracted from the ProtocolSettings
            softcore_LJ_v2=True,
            interpolate_old_and_new_14s=False,
        )

        system = htf.hybrid_system
        positions = htf.hybrid_positions

        # Set up integrator
        timestep = 4.0 * unit.femtosecond
        temperature = 300 * unit.kelvin
        neq_splitting = 'V R H O R V'
        nsteps_eq = 2
        nsteps_neq = 32
        integrator = PeriodicNonequilibriumIntegrator(DEFAULT_ALCHEMICAL_FUNCTIONS,
                                                      nsteps_eq,
                                                      nsteps_neq,
                                                      neq_splitting,
                                                      timestep=timestep,
                                                      temperature=temperature)

        # Set up context
        platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name in ['CUDA', 'OpenCL']:
            platform.setPropertyDefaultValue('Precision', 'mixed')
        if platform_name in ['CUDA']:
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        context.setPositions(positions)

        # Minimize
        openmm.LocalEnergyMinimizer.minimize(context)

        # Equilibrate
        context.setVelocitiesToTemperature(temperature)

        # Prepare objects to store data
        forward_works_main, reverse_works_main = list(), list()
        forward_eq_old, forward_eq_new, forward_neq_old, forward_neq_new = list(), list(), list(), list()
        reverse_eq_new, reverse_eq_old, reverse_neq_old, reverse_neq_new = list(), list(), list(), list()

        # Equilibrium (lambda = 0)
        for step in range(nsteps_eq):
            integrator.step(1)
            # Store positions and works
            if step % save_freq_eq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                forward_eq_old.append(old_pos)
                forward_eq_new.append(new_pos)

        # Run neq
        # Forward (0 -> 1)
        forward_works = [integrator.get_protocol_work(dimensionless=True)]
        for fwd_step in range(nsteps_neq):
            integrator.step(1)
            forward_works.append(integrator.get_protocol_work(dimensionless=True))
            if fwd_step % save_freq_neq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                forward_neq_old.append(old_pos)
                forward_neq_new.append(new_pos)
        forward_works_main.append(forward_works)

        # Equilibrium (lambda = 1)
        for step in range(nsteps_eq):
            integrator.step(1)
            if step % save_freq_eq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                reverse_eq_new.append(new_pos)
                reverse_eq_old.append(old_pos)

        # Reverse work (1 -> 0)
        reverse_works = [integrator.get_protocol_work(dimensionless=True)]
        for rev_step in range(nsteps_neq):
            integrator.step(1)
            reverse_works.append(integrator.get_protocol_work(dimensionless=True))
            if rev_step % save_freq_neq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                old_pos = np.asarray(htf.old_positions(pos))
                new_pos = np.asarray(htf.new_positions(pos))
                reverse_neq_old.append(old_pos)
                reverse_neq_new.append(new_pos)
        reverse_works_main.append(reverse_works)

        # Save output
        # TODO: Assume initially we want the trajectories to understand when something wrong/weird happens.
        # TODO: We need a single API point where the analysis is performed to get DDG values for the NEQ works.
        # Save works
        forward_work_path = os.path.join(ctx.shared, f"forward_{phase}.npy")
        reverse_work_path = os.path.join(ctx.shared, f"reverse_{phase}.npy")
        with open(forward_work_path, 'wb') as out_file:
            np.save(out_file, forward_works_main)
        with open(reverse_work_path, 'wb') as out_file:
            np.save(out_file, reverse_works_main)

        # TODO: Do we need to save the trajectories?
        # Save trajs
        with open(os.path.join(ctx.shared, f"forward_eq_old_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(forward_eq_old))
        with open(os.path.join(ctx.shared, f"forward_eq_new_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(forward_eq_new))
        with open(os.path.join(ctx.shared, f"reverse_eq_new_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(reverse_eq_new))
        with open(os.path.join(ctx.shared, f"reverse_eq_old_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(reverse_eq_old))
        with open(os.path.join(ctx.shared, f"forward_neq_old_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(forward_neq_old))
        with open(os.path.join(ctx.shared, f"forward_neq_new_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(forward_neq_new))
        with open(os.path.join(ctx.shared, f"reverse_neq_old_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(reverse_neq_old))
        with open(os.path.join(ctx.shared, f"reverse_neq_new_{phase}.npy"), 'wb') as out_file:
            np.save(out_file, np.array(reverse_neq_new))

        return {
            'forward_work': forward_work_path,
            'reverse_work': reverse_work_path,
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
        simulations[0]['forward_work']  # We could also do it this way (convenience)
        forward_work_path = os.path.join(ctx.shared, f"forward_{phase}.npy")
        reverse_work_path = os.path.join(ctx.shared, f"reverse_{phase}.npy")
        forward_work = np.load(forward_work_path)
        reverse_work = np.load(reverse_work_path)
        free_energy, error = pymbar.bar.BAR(forward_work, reverse_work)

        return {"DDG": free_energy, "dDDG": error, "paths": works_paths}


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
        if mapping is None:
            raise ValueError("`mapping` is required for this Protocol")
        if 'ligand' not in mapping:
            raise ValueError("'ligand' must be specified in `mapping` dict")
        if extend_from:
            raise NotImplementedError("Can't extend simulations yet")

        # inputs to `ProtocolUnit.__init__` should either be `Gufe` objects
        # or JSON-serializable objects
        sim = SimulationUnit(state_a=stateA, state_b=stateB, mapping=mapping, simulation_parameters=self.settings)

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

from typing import Optional, Iterable, List, Dict, Any, Union, Type

import os
import pathlib

import networkx as nx

from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import Mapping
from gufe.protocols import (
    Protocol,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
    Context,
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
outdir_path = 'output/'

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
    @staticmethod
    def _execute(ctx, *, state_a, state_b, simulation_parameters, **inputs):
        # needed imports
        import numpy as np
        import openmm
        from openmm import unit
        from openmmtools.integrators import PeriodicNonequilibriumIntegrator

        # Get receptor and ligands from states
        # TODO: Assuming here both states have same receptor
        #  and/or that state_a is the one with the receptor we need
        # TODO: Check state_a protein and state_b protein are the same. Maybe in the Protocol._create method (as early as possible)
        # TODO: We also need to check if there is a protein because it can be just the solvent phase (check Richard's PR https://github.com/OpenFreeEnergy/openfe/pull/142/files)
        receptor = state_a.components['protein']
        ligand_a = state_a.components['ligand']
        ligand_b = state_b.components['ligand']

        # Setting up forcefields and phases
        # TODO: These should be extracted from the ProtocolSettings information
        forcefield_files = [
            "amber/ff14SB.xml",
            "amber/tip3p_standard.xml",
            "amber/tip3p_HFE_multivalent.xml",
            "amber/phosaa10.xml",
        ]
        small_molecule_forcefield = 'openff-2.0.0'
        phases = ["complex", "solvent"]

        # Setup relative FE calculation
        # FIXME: We need to be able to setup an FEP simulation with just a single phase. Think about SolventFEPSetup or ComplexFEPSetup, etc.
        fe_setup = RelativeFEPSetup(
            receptor=receptor.to_openmm_topology(),
            receptor_positions=receptor.to_openmm_positions(),
            old_ligand=ligand_a.to_openeye(),
            new_ligand=ligand_b.to_openeye(),
            forcefield_files=forcefield_files,
            small_molecule_forcefield=small_molecule_forcefield,
            phases=phases,
        )

        # Manually extracting objects for different phases - TODO: This could be done in a better way
        topology_proposals = {'complex': fe_setup.complex_topology_proposal,
                              'solvent': fe_setup.solvent_topology_proposal}
        old_positions = {'complex': fe_setup.complex_old_positions, 'solvent': fe_setup.solvent_old_positions}
        new_positions = {'complex': fe_setup.complex_new_positions, 'solvent': fe_setup.solvent_new_positions}
        forward_neglected_angle_terms = {'complex': fe_setup._complex_forward_neglected_angles,
                                         'solvent': fe_setup._solvent_forward_neglected_angles}
        reverse_neglected_angle_terms = {'complex': fe_setup._complex_reverse_neglected_angles,
                                         'solvent': fe_setup._solvent_reverse_neglected_angles}

        for phase in phases:
            # Generate Hybrid Topology Factory - Vanilla HTF
            htf = HybridTopologyFactory(
                topology_proposal=topology_proposals[phase],
                current_positions=old_positions[phase],
                new_positions=new_positions[phase],
                # TODO: I think the following should be extracted from the ProtocolSettings
                neglected_new_angle_terms=forward_neglected_angle_terms[phase],
                neglected_old_angle_terms=reverse_neglected_angle_terms[phase],
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
            forward_works_master, reverse_works_master = list(), list()
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
            # TODO: We can just run the required steps before saving the output, without having to check EVERY step.
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
            forward_works_master.append(forward_works)

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
            reverse_works_master.append(reverse_works)

            # Save output
            # TODO: Assume initially we want the trajectories to understand when something wrong/weird happens.
            # TODO: We need a single API point where the analysis is performed to get DDG values for the NEQ works.
            # create output directory if it does not exist
            out_path = pathlib.Path(outdir_path)
            out_path.mkdir(parents=True, exist_ok=True)
            # Save works
            with open(os.path.join(out_path, f"forward_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, forward_works_master)
            with open(os.path.join(out_path, f"reverse_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, reverse_works_master)

            # Save trajs
            with open(os.path.join(out_path, f"forward_eq_old_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(forward_eq_old))
            with open(os.path.join(out_path, f"forward_eq_new_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(forward_eq_new))
            with open(os.path.join(out_path, f"reverse_eq_new_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(reverse_eq_new))
            with open(os.path.join(out_path, f"reverse_eq_old_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(reverse_eq_old))
            with open(os.path.join(out_path, f"forward_neq_old_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(forward_neq_old))
            with open(os.path.join(out_path, f"forward_neq_new_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(forward_neq_new))
            with open(os.path.join(out_path, f"reverse_neq_old_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(reverse_neq_old))
            with open(os.path.join(out_path, f"reverse_neq_new_{phase}.npy"), 'wb') as out_file:
                np.save(out_file, np.array(reverse_neq_new))

        return dict()


class GatherUnit(ProtocolUnit):
    @staticmethod
    def _execute(ctx, *, simulations, **inputs):

        return dict()

        #output = [r.data for r in dependency_results]
        #output.append("assembling_results")

        #return dict(
        #    data=output,
        #)


class NonEquilibriumCyclingResult(ProtocolResult):

    def get_estimate(self):
        ...

    def get_uncertainty(self):
        ...

    def get_rate_of_convergence(self):
        ...


class NonEquilibriumCyclingProtocol(Protocol):

    _results_cls = NonEquilibriumCyclingResult
    _supported_engines = ['openmm']

    @classmethod
    def _default_settings(cls):
        return {}

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Mapping] = None,
        extend_from: Optional[ProtocolDAGResult] = None,
    ) -> List[ProtocolUnit]:

        # we generate a linear DAG here, since OpenMM performs nonequilibrium
        # cycling in a single simulation
        genhtop = GenerateHybridTopology(
            name="the beginning",
            settings=self.settings,
            stateA=stateA,
            stateB=stateB,
            mapping=mapping,
            start=extend_from,
            some_dict={'a': 2, 'b': 12})

        # inputs to `ProtocolUnit.__init__` should either be `Gufe` objects
        # or JSON-serializable objects
        sim = SimulationUnit(self.settings, initialization=genhtop)

        end = GatherUnit(self.settings, name="gather", simulations=[sim])

        return [genhtop, sim, end]

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
    protocol = NonEquilibriumCyclingProtocol(settings=None)
    dag = protocol.create(
        stateA=solvated_ligand, stateB=vacuum_ligand, name="a dummy run"
    )
    
    # execute DAG locally, in-process
    dagresult: ProtocolDAGResult = execute(dag)

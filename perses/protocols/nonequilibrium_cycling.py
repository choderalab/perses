from typing import Optional, Iterable, List, Dict, Any, Union, Type

import networkx as nx

from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import Mapping
from gufe.protocols import (
    Protocol,
    ProtocolDAG,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
    ProtocolUnitResult,
    Context,
    execute
)

#from gufe.protocols.settings import ...

from perses.app.relative_setup import RelativeFEPSetup
from perses.annihilation.relative import HybridTopologyFactory


class GenerateAlchemicalFactory(ProtocolUnit):
    # NOT really intented to carry a lot of state data (having its own constructor is dangerous!)

    @staticmethod
    def setup_fep(protein_file: str, ligands_file: Union[str, List[str]], old_ligand_index: int, new_ligand_index: int,
                  protein_forcefield_files: List[str], small_molecule_forcefield: str,
                  phases: List[str]) -> RelativeFEPSetup:
        # Setup Relative FEP simulation
        # old_ligand_index = 0
        # new_ligand_index = 1
        # ligands_file = 'Tyk2_ligands_shifted.sdf'
        # protein_file = 'Tyk2_protein.pdb'
        # forcefield_files = [
        #     "amber/ff14SB.xml",
        #     "amber/tip3p_standard.xml",
        #     "amber/tip3p_HFE_multivalent.xml",
        #     "amber/phosaa10.xml",
        # ]
        # small_molecule_forcefield = 'openff-2.0.0'
        # phases = ["complex", "solvent"]
        fe_setup = RelativeFEPSetup(
            ligand_input=ligands_file,
            old_ligand_index=old_ligand_index,
            new_ligand_index=new_ligand_index,
            protein_pdb_filename=protein_file,
            forcefield_files=protein_forcefield_files,
            small_molecule_forcefield=small_molecule_forcefield,
            phases=phases,
        )
        return fe_setup

    @staticmethod
    def generate_htf_object(fe_setup: RelativeFEPSetup, phase: str) -> HybridTopologyFactory:
        topology_proposal = getattr(fe_setup, f"{phase}_topology_proposal")
        current_positions = getattr(fe_setup, f"{phase}_old_positions")
        new_positions = getattr(fe_setup, f"{phase}_new_positions")
        forward_neglected_angle_terms = getattr(fe_setup, f"_{phase}_forward_neglected_angles")
        reverse_neglected_angle_terms = getattr(fe_setup, f"_{phase}_reverse_neglected_angles")

        htf = HybridTopologyFactory(
            topology_proposal=topology_proposal,
            current_positions=current_positions,
            new_positions=new_positions,
            neglected_new_angle_terms=forward_neglected_angle_terms,
            neglected_old_angle_terms=reverse_neglected_angle_terms,
            softcore_LJ_v2=True,
            interpolate_old_and_new_14s=False,
        )

        return htf

    @staticmethod
    def _execute(ctx: Context, *, settings, stateA, stateB, mapping, start, **inputs):

        # generate an OpenMM Topology
        openmmtop: "OpenMMTopology" = perses_generate_topology_lol()

        # serialize openmmtop to disk
        hybrid_top_path = ctx.dag_scratch / 'hybrid-topology.pkl'
        openmmtop.serialize(hybrid_top_path)

        # the return dict should be JSON-serializable
        return {'hybrid-topology-path': hybrid_top_path}

        #return dict(
        #    data="initialized",
        #)


class SimulationUnit(ProtocolUnit):
    @staticmethod
    def _execute(ctx, *, initialization, **inputs):

        hybrid_top_path = initialization['hybrid-topology-path']

        # deserialize hybrid topology

        return dict()


        #output = [r.data for r in dependency_results]
        #output.append("running_md_{}".format(self._kwargs["window"]))

        #return dict(
        #    data=output,
        #    window=self._kwargs["window"],  # extra attributes allowed
        #)


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

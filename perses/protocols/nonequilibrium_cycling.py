from typing import Optional, Iterable, List, Dict, Any

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


class GenerateHybridTopology(ProtocolUnit):
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

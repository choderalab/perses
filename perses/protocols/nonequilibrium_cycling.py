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
)
#from gufe.protocols.settings import ...


class InitializeUnit(ProtocolUnit):
    def _execute(self, dependency_results):

        return dict()

        #return dict(
        #    data="initialized",
        #)


class SimulationUnit(ProtocolUnit):
    def _execute(self, dependency_results):

        return dict()


        #output = [r.data for r in dependency_results]
        #output.append("running_md_{}".format(self._kwargs["window"]))

        #return dict(
        #    data=output,
        #    window=self._kwargs["window"],  # extra attributes allowed
        #)


class GatherUnit(ProtocolUnit):
    def _execute(self, dependency_results):

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
    def get_default_settings(cls):
        return {}

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Mapping] = None,
        extend_from: Optional[ProtocolDAGResult] = None,
    ) -> nx.DiGraph:

        # we generate a linear DAG here, since OpenMM performs nonequilibrium
        # cycling in a single simulation
        start = InitializeUnit(
            self.settings,
            stateA=stateA,
            stateB=stateB,
            mapping=mapping,
            start=extend_from,
        )

        # TODO: evaluate if there is a benefit to doing each sequence (eq, neq,
        # eq, neq) or each full cycle as a separate node in the DAG;
        # would come at the cost of more serialization/deserialization
        sim = SimulationUnit(self.settings)

        end = GatherUnit(self.settings, name="gather")

        dag = nx.DiGraph()
        dag.add_edge(sim, start)
        dag.add_edge(end, sim)

        return dag

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> Dict[str, Any]:

        outputs = []
        for pdr in protocol_dag_results:
            for pur in pdr.protocol_unit_results:
                if pur.name == "gather":
                    outputs.append(pur.data)

        return dict(data=outputs)


    def _validate(self):
        ...


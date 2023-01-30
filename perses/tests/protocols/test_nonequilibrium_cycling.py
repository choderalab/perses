import pytest
from perses.protocols import NonEquilibriumCyclingProtocol
from gufe.protocols.protocoldag import ProtocolDAGResult, execute_DAG
from gufe.protocols.protocolunit import ProtocolUnitResult
from gufe.protocols.protocolunit import ProtocolUnitFailure


class TestNonEquilibriumCycling:

    @pytest.fixture
    def protocol_short(self, short_settings):
        return NonEquilibriumCyclingProtocol(settings=short_settings)

    @pytest.fixture
    def protocol_dag(self, protocol_short, benzene_vacuum_system, toluene_vacuum_system, mapping_benzene_toluene):
        dag = protocol_short.create(
            stateA=benzene_vacuum_system, stateB=toluene_vacuum_system, name="Short vacuum transformation",
            mapping=mapping_benzene_toluene
        )
        dagresult: ProtocolDAGResult = execute_DAG(dag)

        return protocol_short, dag, dagresult

    def test_dag_execute(self, protocol_dag):
        protocol, dag, dagresult = protocol_dag

        assert dagresult.ok()

        # the FinishUnit will always be the last to execute
        finishresult = dagresult.protocol_unit_results[-1]
        assert finishresult.name == "result"

    def test_terminal_units(self, protocol_dag):
        prot, dag, res = protocol_dag

        finals = res.terminal_protocol_unit_results

        assert len(finals) == 1
        assert isinstance(finals[0], ProtocolUnitResult)
        assert finals[0].name == 'result'

    def test_dag_execute_failure(self, protocol_dag_broken):
        # TODO: Lets create a mapping that doesn't make sense for the ligand transformation. This should raise an error
        protocol, dag, dagfailure = protocol_dag_broken

        assert not dagfailure.ok()
        assert isinstance(dagfailure, ProtocolDAGResult)

        failed_units = dagfailure.protocol_unit_failures

        assert len(failed_units) == 1
        assert failed_units[0].name == "problem child"

        # parse exception arguments
        assert failed_units[0].exception[1][1]['data'] == "lol"
        assert isinstance(failed_units[0], ProtocolUnitFailure)

        succeeded_units = dagfailure.protocol_unit_results

        assert len(succeeded_units) > 0

    def test_dag_execute_failure_raise_error(self, protocol_short, benzene_vacuum_system, toluene_vacuum_system,
                                             broken_mapping):
        """Executes a bad setup of a protocol DAG which has an incorrect mapping"""
        dag = protocol_short.create(
            stateA=benzene_vacuum_system, stateB=toluene_vacuum_system, name="a broken dummy run",
            mapping=broken_mapping
        )

        with pytest.raises(AssertionError):
            execute_DAG(dag, raise_error=True)

    def test_create_execute_gather(self, protocol_short, benzene_vacuum_system, toluene_vacuum_system,
                                   mapping_benzene_toluene):
        """
        Perform 20 independent simulations of the NEQ cycling protocol and gather the results
        """

        n_simulations = 20
        results = []
        for i in range(n_simulations):
            protocol_dag = protocol_short.create(
                stateA=benzene_vacuum_system,
                stateB=toluene_vacuum_system,
                mapping=mapping_benzene_toluene,
                name=f"{i}"
            )
            protocol_dag_result = execute_DAG(protocol_dag)
            results.append(protocol_dag_result)
        # gather aggregated results of interest
        protocolresult = protocol_short.gather(results)

        # TODO: use the convergence criteria we've been using in perses (DDG < 6*dDDG)
        assert len(protocolresult.data['logs']) == 1
        assert len(protocolresult.data['logs'][0]) == 21 + 1

        assert protocolresult.get_estimate() == 105336

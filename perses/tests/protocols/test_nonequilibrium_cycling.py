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
    def protocol_short_multiple_cycles(self, short_settings_multiple_cycles):
        return NonEquilibriumCyclingProtocol(settings=short_settings_multiple_cycles)

    @pytest.fixture
    def protocol_dag_result(self, protocol_short, benzene_vacuum_system, toluene_vacuum_system,
                            mapping_benzene_toluene):
        dag = protocol_short.create(
            stateA=benzene_vacuum_system, stateB=toluene_vacuum_system, name="Short vacuum transformation",
            mapping=mapping_benzene_toluene
        )
        dagresult: ProtocolDAGResult = execute_DAG(dag)

        return protocol_short, dag, dagresult

    @pytest.fixture
    def protocol_dag(self, protocol_short_multiple_cycles, benzene_vacuum_system, toluene_vacuum_system,
                     mapping_benzene_toluene):
        dag = protocol_short_multiple_cycles.create(
            stateA=benzene_vacuum_system, stateB=toluene_vacuum_system, name="Short vacuum transformation",
            mapping=mapping_benzene_toluene
        )

        return protocol_short_multiple_cycles, dag

    @pytest.fixture
    def protocol_dag_broken(self, protocol_short, benzene_vacuum_system, toluene_vacuum_system, broken_mapping):
        dag = protocol_short.create(
            stateA=benzene_vacuum_system, stateB=toluene_vacuum_system, name="Broken vacuum transformation",
            mapping=broken_mapping
        )
        # Don't raise the error for getting ProtocolResult
        dagresult: ProtocolDAGResult = execute_DAG(dag, raise_error=False)

        return protocol_short, dag, dagresult

    def test_dag_execute(self, protocol_dag_result):
        protocol, dag, dagresult = protocol_dag_result

        assert dagresult.ok()

        # the FinishUnit will always be the last to execute
        finishresult = dagresult.protocol_unit_results[-1]
        assert finishresult.name == "result"

    def test_terminal_units(self, protocol_dag_result):
        prot, dag, res = protocol_dag_result

        finals = res.terminal_protocol_unit_results

        assert len(finals) == 1
        assert isinstance(finals[0], ProtocolUnitResult)
        assert finals[0].name == 'result'

    def test_dag_execute_failure(self, protocol_dag_broken):
        protocol, dag, dagfailure = protocol_dag_broken

        assert not dagfailure.ok()
        assert isinstance(dagfailure, ProtocolDAGResult)

        failed_units = dagfailure.protocol_unit_failures

        assert len(failed_units) == 1
        assert isinstance(failed_units[0], ProtocolUnitFailure)

    def test_dag_execute_failure_raise_error(self, protocol_short, benzene_vacuum_system, toluene_vacuum_system,
                                             broken_mapping):
        """Executes a bad setup of a protocol DAG which has an incorrect mapping"""
        dag = protocol_short.create(
            stateA=benzene_vacuum_system, stateB=toluene_vacuum_system, name="a broken dummy run",
            mapping=broken_mapping
         )

        # tries to access an atom index that does not exist
        with pytest.raises(IndexError):
            execute_DAG(dag, raise_error=True)

    def test_create_execute_gather(self, protocol_dag):
        """
        Perform 20 independent simulations of the NEQ cycling protocol and gather the results.

        This is done by using 4 replicates of the protocol with 5 simulation units each.
        """

        results = []
        n_replicates = 4
        for _ in range(n_replicates):
            protocol, dag = protocol_dag
            dagresult = execute_DAG(dag)
            results.append(dagresult)
        # gather aggregated results of interest
        protocolresult = protocol.gather(results)
        fe_estimate = protocolresult.get_estimate()
        fe_error = protocolresult.get_uncertainty(1000)
        print(f"Free energy = {fe_estimate} +/- {fe_error}")

        # TODO: use the convergence criteria we've been using in perses (DDG < 6*dDDG)

        # 1. Run without errors nan
        # 2. Can we get an estimate that is not NAN
        # 3. Do toluene to toluene with turning it in another carbon and it should result in 0 FE estimate
        #     - We can use the same mapping of benzene to toluene but use it in toluene to toluene
        # 4. We could also generate a plot with the forward and reverse works and visually check the results.
        raise NotImplementedError

    # TODO: Potentially setup (not run) a protein-ligand system


#############################################################################
# HYBRID SYSTEM SAMPLERS
#############################################################################

from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol

from openmmtools.multistate import sams, replicaexchange
from openmmtools import cache, utils
from perses.dispersed.utils import configure_platform
from openmmtools import cache
cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())
from openmmtools.states import *
from perses.dispersed.utils import create_endstates

import numpy as np
import copy

import logging
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger = logging.getLogger("multistate")


class HybridCompatibilityMixin(object):
    """
    Mixin that allows the MultistateSampler to accommodate the situation where
    unsampled endpoints have a different number of degrees of freedom.
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        self._hybrid_factory = hybrid_factory
        super(HybridCompatibilityMixin, self).__init__(*args, **kwargs)

    def setup(self, n_states, temperature, storage_file, minimisation_steps=100,
              n_replicas=None, lambda_schedule=None,
              lambda_protocol=LambdaProtocol(), endstates=True, high_temperature = None, lambda_endstate = 0.):
        """
        Args:
            n_states : int
                number of alchemical states
            temperature : simtk.unit.Quantity() compatible with unit.kelvin
                target temperature
            storage_file : str
                storage file that will be written to
            minimisation_steps : int
                number of minimization steps for each replica
            n_replicas : int
                number of replicas that will be simulated
            lambda_schedule : iterable, default None
                the schedule of the master lambda; if None, use np.linspace(0,1,n_states)
            lambda_protocol : perses.annihilation.lambda_protocol.LambdaProtocol
                the schedule of the enslaved lambdas in the alchemical transformation
            endstates : bool, default True
                whether to compute unsampled endstate MBAR contributions
            high_temperature : simtk.unit.Quantity() compatible with unit.kelvin, default None
                the maximum temperature for REST implementation;
                if not None, then REST2 is implemented at the `lambda_endstate` value with n_states _without_ the lambda_schedule
            lambda_endstate : float, default 0.
                the lambda endstate at which REST2 will be implemented;
                if high_temperature is None, this is not use.
        """


        from perses.dispersed import feptasks
        if render_full_protocol:
            raise Exception(f"coupling the full alchemical protocol to the temperature protocol is not supported at the moment")

        hybrid_system = self._factory.hybrid_system

        positions = self._factory.hybrid_positions
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(hybrid_system)

        thermostate = ThermodynamicState(hybrid_system, temperature=temperature)
        compound_thermodynamic_state = CompoundThermodynamicState(thermostate, composable_states=[lambda_zero_alchemical_state])

        thermodynamic_state_list = []
        sampler_state_list = []

        context_cache = cache.ContextCache()

        if n_replicas is None:
            _logger.info(f'n_replicas not defined, setting to match n_states, {n_states}')
            n_replicas = n_states
        elif n_replicas > n_states:
            _logger.warning(f'More sampler states: {n_replicas} requested greater than number of states: {n_states}. Setting n_replicas to n_states: {n_states}')
            n_replicas = n_states

        if high_temperature is not None:
            _logger.info(f"conducting REST2")
            assert lambda_endstate is not None
            assert high_temperature > temperature
            assert n_replicas == n_states
            temperature_schedule = [temperature + (high_temperature - temperature)*np.exp(float(i)/float(n_states-1)) for i in range(n_states)]
            lambda_schedule = np.array([lambda_endstate]*n_states)
        else:
            # TODO this feels like it should be somewhere else... just not sure where. Maybe into lambda_protocol
            if lambda_schedule is None:
                lambda_schedule = np.linspace(0.,1.,n_states)
            else:
                assert (len(lambda_schedule) == n_states) , 'length of lambda_schedule must match the number of states, n_states'
                assert (lambda_schedule[0] == 0.), 'lambda_schedule must start at 0.'
                assert (lambda_schedule[-1] == 1.), 'lambda_schedule must end at 1.'
                difference = np.diff(lambda_schedule)
                assert ( all(i >= 0. for i in difference ) ), 'lambda_schedule must be monotonicly increasing'

        #starting with the initial positions generated py geometry.py
        sampler_state =  SamplerState(positions, box_vectors=hybrid_system.getDefaultPeriodicBoxVectors())
        for lambda_val in lambda_schedule:
            compound_thermodynamic_state_copy = copy.deepcopy(compound_thermodynamic_state)
            compound_thermodynamic_state_copy.set_alchemical_parameters(lambda_val,lambda_protocol)
            thermodynamic_state_list.append(compound_thermodynamic_state_copy)

             # now generating a sampler_state for each thermodyanmic state, with relaxed positions
            context, context_integrator = context_cache.get_context(compound_thermodynamic_state_copy)
            feptasks.minimize(compound_thermodynamic_state_copy,sampler_state)
            sampler_state_list.append(copy.deepcopy(sampler_state))

        reporter = storage_file

        # making sure number of sampler states equals n_replicas
        if len(sampler_state_list) != n_replicas:
            # picking roughly evenly spaced sampler states
            # if n_replicas == 1, then it will pick the first in the list
            idx = np.round(np.linspace(0, len(sampler_state_list) - 1, n_replicas)).astype(int)
            sampler_state_list = [state for i,state in enumerate(sampler_state_list) if i in idx]

        assert len(sampler_state_list) == n_replicas

        if endstates:
            # generating unsampled endstates
            _logger.info('Generating unsampled endstates.')
            unsampled_dispersion_endstates = create_endstates(copy.deepcopy(thermodynamic_state_list[0]), copy.deepcopy(thermodynamic_state_list[-1]))
            self.create(thermodynamic_states=thermodynamic_state_list, sampler_states=sampler_state_list,
                    storage=reporter, unsampled_thermodynamic_states=unsampled_dispersion_endstates)
        else:
            self.create(thermodynamic_states=thermodynamic_state_list, sampler_states=sampler_state_list,
                        storage=reporter)

class HybridSAMSSampler(HybridCompatibilityMixin, sams.SAMSSampler):
    """
    SAMSSampler that supports unsampled end states with a different number of positions
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        super(HybridSAMSSampler, self).__init__(*args, hybrid_factory=hybrid_factory, **kwargs)
        self._factory = hybrid_factory


class HybridRepexSampler(HybridCompatibilityMixin, replicaexchange.ReplicaExchangeSampler):
    """
    ReplicaExchangeSampler that supports unsampled end states with a different number of positions
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        super(HybridRepexSampler, self).__init__(*args, hybrid_factory=hybrid_factory, **kwargs)
        self._factory = hybrid_factory

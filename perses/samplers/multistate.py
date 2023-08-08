#############################################################################
# HYBRID SYSTEM SAMPLERS
#############################################################################

from openmm import unit

from openmmtools.multistate import sams, replicaexchange
from openmmtools.states import CompoundThermodynamicState, SamplerState, ThermodynamicState
from perses.dispersed.utils import create_endstates_from_real_systems
from openmmtools.constants import kB
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, RESTCapableRelativeAlchemicalState, LambdaProtocol, RESTCapableLambdaProtocol

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

    # TODO: Should this overload the create() method from parent instead of being setup()?
    def setup(self, n_states, temperature, storage_file, minimisation_steps=100,
              n_replicas=None, lambda_schedule=None,
              lambda_protocol=None, endstates=True, t_max=None):
        """
        Set up the simulation with the specified parameters.

        Parameters:
        -----------
        n_states : int
            The number of alchemical states to simulate.
        temperature : openmm.unit.Quantity
            The temperature of the simulation in Kelvin.
        storage_file : str
            The path to the storage file to store the simulation results.
        minimisation_steps : int, optional
            The number of minimisation steps to perform before simulation. Default is 100.
        n_replicas : int, optional
            The number of replicas for replica exchange. If not specified, it will be set to `n_states`.
        lambda_schedule : array-like, optional
            The schedule of lambda values for the alchemical states. Default is a linear schedule from 0 to 1.
        lambda_protocol : object, optional
            The lambda protocol object that defines the alchemical transformation protocol. Default is None.
        endstates : bool, optional
            Whether to generate unsampled endstates. Default is True.
        t_max : openmm.unit.Quantity, optional
            The maximum temperature for REST scaling. Default is None.

        Raises:
        -------
        ValueError
            If the hybrid factory name is not supported.

        Returns:
        --------
        None
        """
        from perses.dispersed import feptasks

        # Retrieve class name, hybrid system, and hybrid positions
        factory_name = self._hybrid_factory.__class__.__name__
        hybrid_system = self._hybrid_factory.hybrid_system
        positions = self._hybrid_factory.hybrid_positions

        # Create alchemical state and lambda protocol
        if factory_name == 'HybridTopologyFactory':
            lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(hybrid_system)
            lambda_protocol = LambdaProtocol() if lambda_protocol is None else lambda_protocol

        elif factory_name == 'RESTCapableHybridTopologyFactory':
            lambda_zero_alchemical_state = RESTCapableRelativeAlchemicalState.from_system(hybrid_system)
            lambda_protocol = RESTCapableLambdaProtocol() if lambda_protocol is None else lambda_protocol

            # Default to current temperature if t_max is not specified (no REST scaling)
            if t_max is None:
                t_max = temperature

            # Set beta_0 and beta_m
            beta_0 = 1 / (kB * temperature)
            beta_m = 1 / (kB * t_max)
        else:
            raise ValueError(f"{factory_name} not supported")

        # Create reference compound thermodynamic state
        thermostate = ThermodynamicState(hybrid_system, temperature=temperature)
        compound_thermodynamic_state = CompoundThermodynamicState(thermostate, composable_states=[lambda_zero_alchemical_state])

        thermodynamic_state_list = []
        sampler_state_list = []

        if n_replicas is None:
            _logger.info(f'n_replicas not defined, setting to match n_states, {n_states}')
            n_replicas = n_states
        elif n_replicas > n_states:
            _logger.warning(f'More sampler states: {n_replicas} requested greater than number of states: {n_states}. Setting n_replicas to n_states: {n_states}')
            n_replicas = n_states

        # TODO this feels like it should be somewhere else... just not sure where. Maybe into lambda_protocol
        if lambda_schedule is None:
            lambda_schedule = np.linspace(0.,1.,n_states)
        else:
            assert (len(lambda_schedule) == n_states) , 'length of lambda_schedule must match the number of states, n_states'
            assert (lambda_schedule[0] == 0.), 'lambda_schedule must start at 0.'
            assert (lambda_schedule[-1] == 1.), 'lambda_schedule must end at 1.'
            difference = np.diff(lambda_schedule)
            assert ( all(i >= 0. for i in difference ) ), 'lambda_schedule must be monotonicly increasing'

        # Create sampler state (with positions generated by the geometry engine)
        sampler_state = SamplerState(positions, box_vectors=hybrid_system.getDefaultPeriodicBoxVectors())

        for lambda_val in lambda_schedule:
           # Create a compound thermodynamic for lambda_val and set alchemical parameters
            compound_thermodynamic_state_copy = copy.deepcopy(compound_thermodynamic_state)
            if factory_name == 'HybridTopologyFactory':
                compound_thermodynamic_state_copy.set_alchemical_parameters(lambda_val,lambda_protocol)
            elif factory_name == 'RESTCapableHybridTopologyFactory':
                compound_thermodynamic_state_copy.set_alchemical_parameters(lambda_val, beta_0, beta_m, lambda_protocol)
            thermodynamic_state_list.append(compound_thermodynamic_state_copy)

            # Generate a sampler_state for each thermodynamic state
            feptasks.minimize(compound_thermodynamic_state_copy, sampler_state, max_iterations=minimisation_steps)
            sampler_state_list.append(copy.deepcopy(sampler_state))

        reporter = storage_file

        # Make sure number of sampler states equals n_replicas
        if len(sampler_state_list) != n_replicas:
            # Picking roughly evenly spaced sampler states
            # If n_replicas == 1, then it will pick the first in the list
            idx = np.round(np.linspace(0, len(sampler_state_list) - 1, n_replicas)).astype(int)
            sampler_state_list = [state for i,state in enumerate(sampler_state_list) if i in idx]

        assert len(sampler_state_list) == n_replicas

        if endstates:
            # Generating unsampled endstates
            _logger.info('Generating unsampled endstates.')
            unsampled_dispersion_endstates = create_endstates_from_real_systems(self._hybrid_factory)
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

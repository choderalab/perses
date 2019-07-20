################################################################################
# HYBRID SYSTEM SAMPLERS
################################################################################

from perses.annihilation.lambda_protocol import RelativeAlchemicalState

from openmmtools.multistate import sams, replicaexchange
from openmmtools import cache
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState, group_by_compatibility

import numpy as np
import copy

class HybridCompatibilityMixin(object):
    """
    Mixin that allows the MultistateSampler to accommodate the situation where unsampled endpoints
    have a different number of degrees of freedom.
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        self._hybrid_factory = hybrid_factory
        super(HybridCompatibilityMixin, self).__init__(*args, **kwargs)

    def _compute_replica_energies(self, replica_id):
        """Compute the energy for the replica in every ThermodynamicState."""
        # Initialize replica energies for each thermodynamic state.
        energy_thermodynamic_states = np.zeros(self.n_states)
        energy_unsampled_states = np.zeros(len(self._unsampled_states))

        # Retrieve sampler state associated to this replica.
        sampler_state = self._sampler_states[replica_id]

        # Determine neighborhood
        state_index = self._replica_thermodynamic_states[replica_id]
        neighborhood = self._neighborhood(state_index)
        # Only compute energies over neighborhoods
        energy_neighborhood_states = energy_thermodynamic_states[neighborhood]  # Array, can be indexed like this
        neighborhood_thermodynamic_states = [self._thermodynamic_states[n] for n in neighborhood]  # List

        # Compute energy for all thermodynamic states.
        for idx, (energies, states) in enumerate([(energy_neighborhood_states, neighborhood_thermodynamic_states),
                                                  (energy_unsampled_states, self._unsampled_states)]):
            # Group thermodynamic states by compatibility.
            compatible_groups, original_indices = group_by_compatibility(states)

            # Are we treating the unsampled states? if so, idx will be one:
            if idx == 1:
                unsampled_state = True
            else:
                unsampled_state = False

            # Compute the reduced potentials of all the compatible states.
            for compatible_group, state_indices in zip(compatible_groups, original_indices):
                # Get the context, any Integrator works.
                context, integrator = cache.global_context_cache.get_context(compatible_group[0])

                # Are we trying to compute a potential at an unsampled (different number of particles) state?
                if unsampled_state:
                    if state_indices[0] == 0:
                        positions = self._hybrid_factory.old_positions(sampler_state.positions)
                    elif state_indices[0] == 1:
                        positions = self._hybrid_factory.new_positions(sampler_state.positions)
                    else:
                        raise ValueError("This mixin isn't defined for more than two unsampled states")

                    box_vectors = sampler_state.box_vectors

                    context.setPositions(positions)
                    context.setPeriodicBoxVectors(*box_vectors)
                else:
                    # Update positions and box vectors. We don't need
                    # to set Context velocities for the potential.
                    sampler_state.apply_to_context(context, ignore_velocities=True)

                # Compute and update the reduced potentials.
                compatible_energies = ThermodynamicState.reduced_potential_at_states(context, compatible_group)
                for energy_idx, state_idx in enumerate(state_indices):
                    energies[state_idx] = compatible_energies[energy_idx]

        # Return the new energies.
        return energy_neighborhood_states, energy_unsampled_states


class HybridSAMSSampler(HybridCompatibilityMixin, sams.SAMSSampler):
    """
    SAMSSampler that supports unsampled end states with a different number of positions
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        super(HybridSAMSSampler, self).__init__(*args, hybrid_factory=hybrid_factory, **kwargs)
        self._factory = hybrid_factory

    def setup(self, n_states, temperature, storage_file, minimisation_steps=100,lambda_schdeule=None):

        from openmmtools.integrators import FIREMinimizationIntegrator

        hybrid_system = self._factory.hybrid_system

        initial_hybrid_positions = self._factory.hybrid_positions
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(hybrid_system)


        thermostate = ThermodynamicState(hybrid_system, temperature=temperature)
        compound_thermodynamic_state = CompoundThermodynamicState(thermostate, composable_states=[lambda_zero_alchemical_state])

        thermodynamic_state_list = []
        sampler_state_list = []

        integrator = FIREMinimizationIntegrator()
        context_cache = cache.ContextCache()

        if lambda_schdeule is None:
            lambda_schdeule = np.linspace(0.,1.,n_states)
        else:
            assert (len(lambda_schdeule) == n_states) , 'length of lambda_schdeule must match the number of states, n_states'
            assert (lambda_schdeule[0] == 0.), 'lambda_schdeule must start at 0.'
            assert (lambda_schdeule[-1] == 1.), 'lambda_schdeule must end at 1.'
            difference = np.diff(lambda_schdeule)
            assert ( all(i >= 0. for i in difference ) ), 'lambda_schdeule must be monotonicly increasing'

        #starting with the initial positions generated py geometry.py
        positions = initial_hybrid_positions
        for lambda_val in lambda_schdeule:
            compound_thermodynamic_state_copy = copy.deepcopy(compound_thermodynamic_state)
            compound_thermodynamic_state_copy.set_alchemical_parameters(lambda_val)
            thermodynamic_state_list.append(compound_thermodynamic_state_copy)

            # now generating a sampler_state for each thermodyanmic state, with relaxed positions
            context, context_integrator = context_cache.get_context(compound_thermodynamic_state_copy,
                                                        integrator)
            # set the positions to the end-point of the previous lambda window
            context.setPositions(positions)

            context_integrator.step(minimisation_steps)

            state = context.getState(getPositions=True)
            minimized_hybrid_positions = copy.deepcopy(state.getPositions())

            sampler_state = SamplerState(minimized_hybrid_positions, box_vectors=hybrid_system.getDefaultPeriodicBoxVectors())
            sampler_state_list.append(sampler_state)
            # save the positions for the next iteration
            positions = minimized_hybrid_positions

        #nonalchemical_thermodynamic_states = [
        #    ThermodynamicState(self._factory._old_system, temperature=temperature),
        #    ThermodynamicState(self._factory._new_system, temperature=temperature)]


        reporter = storage_file

        self.create(thermodynamic_states=thermodynamic_state_list, sampler_states=sampler_state_list,
                    #            storage=reporter, unsampled_thermodynamic_states=nonalchemical_thermodynamic_states)
                    storage=reporter, unsampled_thermodynamic_states=None)


class HybridRepexSampler(HybridCompatibilityMixin, replicaexchange.ReplicaExchangeSampler):
    """
    ReplicaExchangeSampler that supports unsampled end states with a different number of positions
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        super(HybridRepexSampler, self).__init__(*args, hybrid_factory=hybrid_factory, **kwargs)
        self._factory = hybrid_factory

    def setup(self, n_states, temperature, storage_file, minimisation_steps=100,lambda_schdeule=None):

        from openmmtools.integrators import FIREMinimizationIntegrator

        hybrid_system = self._factory.hybrid_system

        initial_hybrid_positions = self._factory.hybrid_positions
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(hybrid_system)


        thermostate = ThermodynamicState(hybrid_system, temperature=temperature)
        compound_thermodynamic_state = CompoundThermodynamicState(thermostate, composable_states=[lambda_zero_alchemical_state])

        thermodynamic_state_list = []
        sampler_state_list = []

        integrator = FIREMinimizationIntegrator()
        context_cache = cache.ContextCache()

        if lambda_schdeule is None:
            lambda_schdeule = np.linspace(0.,1.,n_states)
        else:
            assert (len(lambda_schdeule) == n_states) , 'length of lambda_schdeule must match the number of states, n_states'
            assert (lambda_schdeule[0] == 0.), 'lambda_schdeule must start at 0.'
            assert (lambda_schdeule[-1] == 1.), 'lambda_schdeule must end at 1.'
            difference = np.diff(lambda_schdeule)
            assert ( all(i >= 0. for i in difference ) ), 'lambda_schdeule must be monotonicly increasing'

        #starting with the initial positions generated py geometry.py
        positions = initial_hybrid_positions
        for lambda_val in lambda_schdeule:
            compound_thermodynamic_state_copy = copy.deepcopy(compound_thermodynamic_state)
            compound_thermodynamic_state_copy.set_alchemical_parameters(lambda_val)
            thermodynamic_state_list.append(compound_thermodynamic_state_copy)

            # now generating a sampler_state for each thermodyanmic state, with relaxed positions
            context, context_integrator = context_cache.get_context(compound_thermodynamic_state_copy,
                                                        integrator)
            # set the positions to the end-point of the previous lambda window
            context.setPositions(positions)

            context_integrator.step(minimisation_steps)

            state = context.getState(getPositions=True)
            minimized_hybrid_positions = copy.deepcopy(state.getPositions())

            sampler_state = SamplerState(minimized_hybrid_positions, box_vectors=hybrid_system.getDefaultPeriodicBoxVectors())
            sampler_state_list.append(sampler_state)
            # save the positions for the next iteration
            positions = minimized_hybrid_positions

        nonalchemical_thermodynamic_states = [
            ThermodynamicState(self._factory._old_system, temperature=temperature),
            ThermodynamicState(self._factory._new_system, temperature=temperature)]


        reporter = storage_file

        self.create(thermodynamic_states=thermodynamic_state_list, sampler_states=sampler_state_list,
                    storage=reporter, unsampled_thermodynamic_states=nonalchemical_thermodynamic_states)
        #            storage=reporter, unsampled_thermodynamic_states=None)

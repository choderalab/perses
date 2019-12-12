#############################################################################
# HYBRID SYSTEM SAMPLERS
#############################################################################

from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol

from openmmtools.multistate import sams, replicaexchange
from openmmtools import cache
from openmmtools.states import *

import numpy as np
import copy

import logging
logger = logging.getLogger(__name__)


class HybridCompatibilityMixin(object):
    """
    Mixin that allows the MultistateSampler to accommodate the situation where
    unsampled endpoints have a different number of degrees of freedom.
    """

    def __init__(self, *args, hybrid_factory=None, **kwargs):
        self._hybrid_factory = hybrid_factory
        super(HybridCompatibilityMixin, self).__init__(*args, **kwargs)

    def setup(self, n_states, temperature, storage_file, minimisation_steps=100,lambda_schedule=None,lambda_protocol=LambdaProtocol(),endstates=True,expanded_cutoff=None):

        from perses.dispersed import feptasks

        hybrid_system = self._factory.hybrid_system

        positions = self._factory.hybrid_positions
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(hybrid_system)

        thermostate = ThermodynamicState(hybrid_system, temperature=temperature)
        compound_thermodynamic_state = CompoundThermodynamicState(thermostate, composable_states=[lambda_zero_alchemical_state])

        thermodynamic_state_list = []
        sampler_state_list = []

        context_cache = cache.ContextCache()

        context, _ = context_cache.get_context(compound_thermodynamic_state)
        platform = context.getPlatform()
        logger.info('Setting the platform precision to mixed')
        platform.setPropertyDefaultValue('Precision','mixed')


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

        if endstates:
            # generating unsampled endstates
            logger.info('Generating unsampled endstates.')
            unsampled_endstates = [thermodynamic_state_list[0],thermodynamic_state_list[-1]] # taking the first and last states of the alchemical protocol
            if expanded_cutoff is None:
                logger.warning('expanded_cutoff MUST be defined to use unsampled endstates, proceeding WITHOUT endstates.')
                endstates=False
            
             # changing the non-bonded method for the unsampled endstates
            unsampled_dispersion_endstates = []
            for master_lambda,endstate in zip([0.,1.],unsampled_endstates):
                context, context_integrator = context_cache.get_context(endstate)
                dispersion_system = context.getSystem()
                box_vectors = hybrid_system.getDefaultPeriodicBoxVectors()
                dimensions = [x[i] for i,x in enumerate(box_vectors)]
                minimum_length = min(dimensions)
                assert ( expanded_cutoff < minimum_length ), "Expanded cutoff of the unsampled endstates cannot be larger than the shortest dimension of the system"
                for force in dispersion_system.getForces():
                    # expanding the cutoff for both the NonbondedForce and CustomNonbondedForce
                    if 'CustomNonbondedForce' in force.__class__.__name__:
                        force.setCutoffDistance(expanded_cutoff)
                    # use long range correction for the customnonbonded force
                    if force.__class__.__name__ == 'CustomNonbondedForce':
                        force.setUseLongRangeCorrection(True)
                    # setting the default GlobalParameters for both end states, so that the long-range dispersion correction is correctly computed
                    if force.__class__.__name__ in ['NonbondedForce','CustomNonbondedForce','CustomBondForce','CustomAngleForce','CustomTorsionForce']:
                        for parameter_index in range(force.getNumGlobalParameters()):
                            # finding alchemical global parameters
                            if force.getGlobalParameterName(parameter_index)[0:7] == 'lambda_':
                                force.setGlobalParameterDefaultValue(parameter_index, master_lambda)
                unsampled_dispersion_endstates.append(ThermodynamicState(dispersion_system, temperature=temperature))

        reporter = storage_file

        if endstates:
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

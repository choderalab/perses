#############################################################################
# HYBRID SYSTEM SAMPLERS
#############################################################################

from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol

from openmmtools.multistate import sams, replicaexchange
from openmmtools import cache, utils
from perses.dispersed.utils import configure_platform
cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())
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

    def setup(self, n_states, temperature, storage_file, minimisation_steps=100,lambda_schedule=None,lambda_protocol=LambdaProtocol(),endstates=True):

        from perses.dispersed import feptasks

        hybrid_system = self._factory.hybrid_system

        positions = self._factory.hybrid_positions
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(hybrid_system)

        thermostate = ThermodynamicState(hybrid_system, temperature=temperature)
        compound_thermodynamic_state = CompoundThermodynamicState(thermostate, composable_states=[lambda_zero_alchemical_state])

        thermodynamic_state_list = []
        sampler_state_list = []

        context_cache = cache.ContextCache()

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

            # For the unsampled endstates:
            # - move all alchemical atom LJ parameters from CustomNonbondedForce back to NonbondedForce
            # - delete CustomNonbondedForce (since it is no longer needed)
            # - set PME tolerance to 1e-5 of better
            # - enable LJPME to handle long-range dispersion correction in physically reasonable manner
            unsampled_dispersion_endstates = list()
            for master_lambda, endstate in zip([0.,1.],unsampled_endstates):
                # Get a copy of the system
                dispersion_system = endstate.get_system()
                energy_unit = unit.kilocalories_per_mole
                # Find the NonbondedForce (there must be only one)
                forces = { force.__class__.__name__ : force for force in dispersion_system.getForces() }
                # Set NonbondedForce to use LJPME
                forces['NonbondedForce'].setNonbondedMethod(openmm.NonbondedForce.LJPME)
                # Set tight PME tolerance
                TIGHT_PME_TOLERANCE = 1.0e-5
                forces['NonbondedForce'].setEwaldErrorTolerance(TIGHT_PME_TOLERANCE)
                # Move alchemical LJ sites from CustomNonbondedForce back to NonbondedForce
                for particle_index in range(forces['NonbondedForce'].getNumParticles()):
                    charge, sigma, epsilon = forces['NonbondedForce'].getParticleParameters(particle_index)
                    sigmaA, epsilonA, sigmaB, epsilonB, unique_old, unique_new = forces['CustomNonbondedForce'].getParticleParameters(particle_index)
                    if (epsilon/energy_unit == 0.0) and ((epsilonA > 0.0) or (epsilonB > 0.0)):
                        sigma = (1-master_lambda)*sigmaA + master_lambda*sigmaB
                        epsilon = (1-master_lambda)*epsilonA + master_lambda*epsilonB
                        nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)
                # Delete the CustomNonbondedForce since we have moved all alchemical particles out of it
                for force_index, force in dispersion_system.getForces():
                    if force.__class__.__name__ == 'CustomNonbondedForce':
                        custom_nonbonded_force_index = force_index
                        break
                system.removeForce(custom_nonbonded_force_index)
                # Set all parameters to master lambda
                for force_index, force in system.getForces():
                    if hasattr(force, 'getNumGlobalParameters'):
                        for parameter_index in range(force.getNumGlobalParameters()):
                            if force.getGlobalParameterName(parameter_index)[0:7] == 'lambda_':
                                force.setGlobalParameterDefaultValue(parameter_index, master_lambda)
                # Store the unsampled endstate
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
        self._real_endstates = real_endstates

    def setup(self, n_states, temperature, storage_file, minimisation_steps=100,lambda_schedule=None,real_endstates=False,lambda_protocol=LambdaProtocol()):

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

         # adding unsampled endstates, wether these are real systems, with a different number of atoms to the hybrid, or hybrid systems with larger cutoffs
        if self._real_endstates == True:
            logger.debug("Simulating endstates that represent the real system. This can increase the variance of the resulting energies")
            unsampled_endstates = [
                ThermodynamicState(self._factory._old_system, temperature=temperature),
                ThermodynamicState(self._factory._new_system, temperature=temperature)]
        else:
            unsampled_endstates = [thermodynamic_state_list[0],thermodynamic_state_list[-1]] # taking the first and last states of the alchemical protocol

         # changing the non-bonded method for the unsampled endstates
        unsampled_dispersion_endstates = []
        for master_lambda,endstate in zip([0.,1.],unsampled_endstates):
            context, context_integrator = context_cache.get_context(endstate)
            dispersion_system = context.getSystem()
            box_vectors = hybrid_system.getDefaultPeriodicBoxVectors()
            dimensions = [x[i] for i,x in enumerate(box_vectors)]
            minimum_length = min(dimensions)
            for force in dispersion_system.getForces():
                # expanding the cutoff for both the NonbondedForce and CustomNonbondedForce
                if 'NonbondedForce' in force.__class__.__name__:
                    force.setCutoffDistance((minimum_length._value/2.) - 0.5)
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

        self.create(thermodynamic_states=thermodynamic_state_list, sampler_states=sampler_state_list,
                    storage=reporter, unsampled_thermodynamic_states=unsampled_dispersion_endstates)

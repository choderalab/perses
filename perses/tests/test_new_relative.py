###########################################
# IMPORTS
###########################################
import copy
from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from openeye import oechem
from openmmtools.constants import kB
from openmmtools import alchemy, states

from perses.tests.utils import generate_vacuum_topology_proposal
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.annihilation.new_relative import HybridTopologyFactory

from perses.annihilation.lambda_protocol import RelativeAlchemicalState
import copy
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState
from openmmtools import integrators

#############################################
# CONSTANTS
#############################################
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01
ENERGY_THRESHOLD = 1e-1
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CPU")

"""
Test to assess whether the HybridTopologyFactory is bookkeeping all energies at
lambda = 0, 1 states as compared to nonalchemical endstates given a single transformation.

The following tests will be conducted...
    1. a solvated small molecule transformation (current_mol = 'toluene', proposed_mol = '1,2-bis(trifluoromethyl) benzene') will be generated; it will be asserted that the nonalchemical zero state is equal to the alchemical zero state less the forward valence energy.
       It will also be asserted that the alchemical one state is equal to the nonalchemical one state less  the reverse valence energy.
"""


def energy_bookkeeping(top_proposal, old_positions):
    """
    This function returns the energy difference between the lambda = 0, 1 nonalchemical and alchemical states, corrected for the new and old valence energies.
    """
    from perses.tests.utils import generate_vacuum_topology_proposal
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    from perses.annihilation.new_relative import HybridTopologyFactory
    from openmmtools import cache
    from openmmtools import alchemy, states
    from perses.annihilation.lambda_protocol import RelativeAlchemicalState
    import copy
    from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState

    #define the geometry engine and conduct forward/backward proposal
    _geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=100, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)
    _new_positions, _lp = _geometry_engine.propose(top_proposal, old_positions, beta)
    _lp_rev = _geometry_engine.logp_reverse(top_proposal, _new_positions, old_positions, beta)

    #define the HybridTopologyFactory
    HTF = HybridTopologyFactory(top_proposal, old_positions, _new_positions)

    #define alchemical states and set parameters
    hybrid_system = HTF.hybrid_system
    lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(hybrid_system)
    lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

    lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
    lambda_one_alchemical_state.set_alchemical_parameters(1.0)

    # Create thermodynamic states for the nonalchemical endpoints
    thermodynamic_state = ThermodynamicState(hybrid_system, temperature = temperature)

    nonalchemical_thermodynamic_states = {
        0: ThermodynamicState(top_proposal.old_system, temperature=temperature),
        1: ThermodynamicState(top_proposal.new_system, temperature=temperature)}

    # Now create the compound states with different alchemical states
    hybrid_thermodynamic_states = {0: CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_zero_alchemical_state]),
                                   1: CompoundThermodynamicState(copy.deepcopy(thermodynamic_state), composable_states=[lambda_one_alchemical_state])}

    #build the cache
    context_cache = cache.ContextCache(capacity = 1)

    #get nonalchemical zero thermodynamic state
    nonalch_zero_context, context_integrator = context_cache.get_context(nonalchemical_thermodynamic_states[0], integrators.LangevinIntegrator(temperature=temperature))
    nonalch_zero_context.setPositions(old_positions)
    nonalch_zero_rp = SamplerState.from_context(nonalch_zero_context).potential_energy*beta

    #get nonalchemical one thermodynamic state
    nonalch_one_context, context_integrator = context_cache.get_context(nonalchemical_thermodynamic_states[1], integrators.LangevinIntegrator(temperature=temperature))
    nonalch_one_context.setPositions(_new_positions)
    nonalch_one_rp = SamplerState.from_context(nonalch_one_context).potential_energy*beta

    #get alchemical zero thermodynamic state
    alch_zero_context, context_integrator = context_cache.get_context(hybrid_thermodynamic_states[0], integrators.LangevinIntegrator(temperature=temperature))
    alch_zero_context.setPositions(HTF._hybrid_positions)
    alch_zero_rp = SamplerState.from_context(alch_zero_context).potential_energy*beta

    #get alchemical zero thermodynamic state
    alch_one_context, context_integrator = context_cache.get_context(hybrid_thermodynamic_states[1], integrators.LangevinIntegrator(temperature=temperature))
    alch_one_context.setPositions(HTF._hybrid_positions)
    alch_one_rp = SamplerState.from_context(alch_one_context).potential_energy*beta

    #valence energy definitions
    forward_added_valence_energy = _geometry_engine.forward_final_context_reduced_potential - _geometry_engine.forward_atoms_with_positions_reduced_potential
    reverse_subtracted_valence_energy = _geometry_engine.reverse_final_context_reduced_potential - _geometry_engine.reverse_atoms_with_positions_reduced_potential

    return nonalch_zero_rp - alch_zero_rp + forward_added_valence_energy, nonalch_one_rp - alch_one_rp + reverse_subtracted_valence_energy

def test_HybridTopologyFactory(current_mol = 'toluene', proposed_mol = '1,2-bis(trifluoromethyl) benzene'):

    #Just test the solvated system
    from perses.tests.utils import generate_solvated_hybrid_test_topology
    top_proposal, old_positions = generate_solvated_hybrid_test_topology(current_mol_name = current_mol, proposed_mol_name = proposed_mol, propose_geometry = False)

    #but first turn off the dispersion correction
    top_proposal._old_system.getForce(3).setUseDispersionCorrection(False)
    top_proposal._new_system.getForce(3).setUseDispersionCorrection(False)

    zero_rp_diff, one_rp_diff = energy_bookkeeping(top_proposal, old_positions)
    assert abs(zero_rp_diff) < ENERGY_THRESHOLD, f"The zero state alchemical and nonalchemical energy absolute difference {zero_rp_diff} is greater than the threshold of {ENERGY_THRESHOLD}."
    assert abs(one_rp_diff) < ENERGY_THRESHOLD, f"The one state alchemical and nonalchemical energy absolute difference {one_rp_diff} is greater than the threshold of {ENERGY_THRESHOLD}."

    return zero_rp_diff, one_rp_diff

# zero_rp_diff, one_rp_diff = test_HybridTopologyFactory()
# print(zero_rp_diff)
# print(one_rp_diff)

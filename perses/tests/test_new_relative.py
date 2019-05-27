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


istravis = os.environ.get('TRAVIS', None) == 'true'
#############################################
# CONSTANTS
#############################################
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01
ENERGY_THRESHOLD = 1e-1
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CPU")

def test_HybridTopologyFactory_energies(current_mol = 'toluene', proposed_mol = '1,2-bis(trifluoromethyl) benzene'):
    """
    Test whether the difference in the nonalchemical zero and alchemical zero states is the forward valence energy.  Also test for the one states.
    """
    from perses.tests.utils import generate_solvated_hybrid_test_topology, generate_endpoint_thermodynamic_states
    import openmmtools.cache as cache

    #Just test the solvated system
    top_proposal, old_positions, _ = generate_solvated_hybrid_test_topology(current_mol_name = current_mol, proposed_mol_name = proposed_mol, propose_geometry = False)

    #remove the dispersion correction
    top_proposal._old_system.getForce(3).setUseDispersionCorrection(False)
    top_proposal._new_system.getForce(3).setUseDispersionCorrection(False)


    # run geometry engine to generate old and new positions
    _geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=100, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)
    _new_positions, _lp = _geometry_engine.propose(top_proposal, old_positions, beta)
    _lp_rev = _geometry_engine.logp_reverse(top_proposal, _new_positions, old_positions, beta)

    # make the hybrid system, reset the CustomNonbondedForce cutoff
    HTF = HybridTopologyFactory(top_proposal, old_positions, _new_positions)
    hybrid_system = HTF.hybrid_system
    nonalch_zero, nonalch_one, alch_zero, alch_one = generate_endpoint_thermodynamic_states(hybrid_system, top_proposal)

    # compute reduced energies
    #for the nonalchemical systems...
    attrib_list = [(nonalch_zero, old_positions, top_proposal._old_system.getDefaultPeriodicBoxVectors()),
                    (alch_zero, HTF._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    (alch_one, HTF._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    (nonalch_one, _new_positions, top_proposal._new_system.getDefaultPeriodicBoxVectors())]

    rp_list = []
    for (state, pos, box_vectors) in attrib_list:
        context, integrator = cache.global_context_cache.get_context(state)
        samplerstate = SamplerState(positions = pos, box_vectors = box_vectors)
        samplerstate.apply_to_context(context)
        rp = state.reduced_potential(context)
        rp_list.append(rp)

    #valence energy definitions
    forward_added_valence_energy = _geometry_engine.forward_final_context_reduced_potential - _geometry_engine.forward_atoms_with_positions_reduced_potential
    reverse_subtracted_valence_energy = _geometry_engine.reverse_final_context_reduced_potential - _geometry_engine.reverse_atoms_with_positions_reduced_potential

    nonalch_zero_rp, alch_zero_rp, alch_one_rp, nonalch_one_rp = rp_list[0], rp_list[1], rp_list[2], rp_list[3]
    # print(f"Difference between zeros: {nonalch_zero_rp - alch_zero_rp}; forward added: {forward_added_valence_energy}")
    # print(f"Difference between ones: {nonalch_zero_rp - alch_zero_rp}; forward added: {forward_added_valence_energy}")

    assert abs(nonalch_zero_rp - alch_zero_rp + forward_added_valence_energy) < ENERGY_THRESHOLD, f"The zero state alchemical and nonalchemical energy absolute difference {abs(nonalch_zero_rp - alch_zero_rp + forward_added_valence_energy)} is greater than the threshold of {ENERGY_THRESHOLD}."
    assert abs(nonalch_one_rp - alch_one_rp + reverse_subtracted_valence_energy) < ENERGY_THRESHOLD, f"The one state alchemical and nonalchemical energy absolute difference {abs(nonalch_one_rp - alch_one_rp + reverse_subtracted_valence_energy)} is greater than the threshold of {ENERGY_THRESHOLD}."

    print(f"Abs difference in zero alchemical vs nonalchemical systems: {abs(nonalch_zero_rp - alch_zero_rp + forward_added_valence_energy)}")
    print(f"Abs difference in one alchemical vs nonalchemical systems: {abs(nonalch_one_rp - alch_one_rp + reverse_subtracted_valence_energy)}")

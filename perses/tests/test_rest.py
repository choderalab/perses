###########################################
# IMPORTS
###########################################
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from simtk import unit, openmm
from perses.tests.utils import compute_potential_components
from openmmtools.constants import kB
from openmmtools import cache, utils
from perses.dispersed import feptasks
from perses.dispersed.feptasks import minimize
from perses.dispersed.utils import configure_platform
from perses.annihilation.rest import RESTTopologyFactory
from perses.annihilation.lambda_protocol import RESTState
import numpy as np
from perses.tests.test_topology_proposal import generate_atp, generate_dipeptide_top_pos_sys
from openmmtools.testsystems import AlanineDipeptideVacuum
import itertools

cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())

#############################################
# CONSTANTS
#############################################
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CPU")


# def compare_energy_components(rest_system, other_system, positions, platform=REFERENCE_PLATFORM):
#     """
#     Get energy components of a given system
#     """
#     platform = configure_platform(platform)
#
#     # Create thermodynamic state and sampler state for non-rest system
#     thermostate_other = ThermodynamicState(system=other_system, temperature=temperature)
#     sampler_state_other = SamplerState(positions, box_vectors=other_system.getDefaultPeriodicBoxVectors())
#
#     # Create context for non-rest system
#     context_cache = cache.ContextCache()
#     context_other, context_other_integrator = context_cache.get_context(thermostate_other)
#
#     # Minimize
#     minimize(thermostate_other, sampler_state_other)
#     sampler_state_other.apply_to_context(context_other)
#
#     # Get energy components for non-rest system
#     components_other = compute_potential_components(context_other, beta=beta)
#
#     # Create thermodynamic state for rest_system
#     thermostate_rest = ThermodynamicState(system=rest_system, temperature=temperature)
#
#     # Create context for rest system
#     context_cache = cache.ContextCache()
#     context_rest, context_rest_integrator = context_cache.get_context(thermostate_rest)
#     sampler_state_other.apply_to_context(context_rest)
#
#     # Get energy components for rest system
#     components_rest = compute_potential_components(context_rest, beta=beta)
#
#     # Check that bond, angle, and torsion energies match
#     for other, rest in zip(components_other[:3], components_rest[:3]):
#         assert np.isclose([other[1]], [rest[1]]), f"The energies do not match for the {other[0]}: {other[1]} (other system) vs. {rest[1]} (REST system)"
#
#     # Check that nonbonded energies
#     nonbonded_other = components_other[3][1]
#     nonbonded_rest = components_rest[3][1] + components_rest[4][1] + components_rest[5][1]
#     assert np.isclose([nonbonded_other], [nonbonded_rest]), f"The energies do not match for the NonbondedForce: {nonbonded_other} (other system) vs. {nonbonded_rest} (REST system)"
#
#
# def test_bookkeeping():
#     """
#     Given the default Global variables, do energy component bookkeeping with the REST system and the original system.
#     Specifically, the first CustomBondForce must match the HarmonicBondForce
#     (same with the second and third forces with the HarmonicAngle and PeriodicTorsionForces).
#     The sum of the NonbondedForce, the CustomNonbondedForce, and the CustomBondForce (last one) must be equal to the
#     energy of the original system's NonbondedForce.
#     """
#
#     # Create vanilla system for alanine dipeptide
#     ala = AlanineDipeptideVacuum()
#     system = ala.system
#     positions = ala.positions
#
#     # Create REST system for alanine dipeptide
#     system.removeForce(4)
#     factory = RESTTopologyFactory(system, solute_region=list(range(6, 16)))
#     REST_system = factory.REST_system
#
#     compare_energy_components(REST_system, system, positions)
#
#     # Create repartitioned hybrid system for lambda 0 endstate for alanine dipeptide
#     atp, system_generator = generate_atp(phase='vacuum')
#     htf = generate_dipeptide_top_pos_sys(atp.topology,
#                                          new_res='THR',
#                                          system=atp.system,
#                                          positions=atp.positions,
#                                          system_generator=system_generator,
#                                          conduct_htf_prop=True,
#                                          repartitioned=True,
#                                          endstate=0,
#                                          validate_endstate_energy=False)
#
#     # Create REST-ified hybrid system for alanine dipeptide
#     factory = RESTTopologyFactory(htf.hybrid_system, solute_region=list(range(6, 16)))
#     REST_system = factory.REST_system
#
#     compare_energy_components(REST_system, htf.hybrid_system, htf.hybrid_positions)


def compare_energies(system, positions, T_min, T):

    # Compute energy for REST system
    # Create REST system
    factory = RESTTopologyFactory(system, solute_region=list(range(22)))
    REST_system = factory.REST_system

    # Create thermodynamic state
    lambda_zero_alchemical_state = RESTState.from_system(REST_system)
    thermostate = ThermodynamicState(REST_system, temperature=T_min)
    compound_thermodynamic_state = CompoundThermodynamicState(thermostate,
                                                              composable_states=[lambda_zero_alchemical_state])

    # Set alchemical parameters
    beta_0 = 1 / (kB * T_min)
    beta_m = 1 / (kB * T)
    compound_thermodynamic_state.set_alchemical_parameters(beta_0, beta_m)

    # Minimize and save energy
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = compound_thermodynamic_state.create_context(integrator)
    context.setPositions(positions)
    sampler_state = SamplerState.from_context(context)
    REST_energy = compound_thermodynamic_state.reduced_potential(sampler_state)

    # Compute energy for non-RESTified system
    # Determine regions and scaling factors
    solute = list(range(22))
    solvent = list(range(22, system.getNumParticles()))
    solute_scaling = beta_m / beta_0
    inter_scaling = np.sqrt(beta_m / beta_0)

    # Scale the terms in the bond force appropriately
    bond_force = system.getForce(0)
    for bond in range(bond_force.getNumBonds()):
        p1, p2, length, k = bond_force.getBondParameters(bond)
        if p1 in solute and p2 in solute:
            bond_force.setBondParameters(bond, p1, p2, length, k * solute_scaling)
        elif (p1 in solute and p2 in solvent) or (p1 in solvent and p2 in solute):
            bond_force.setBondParameters(bond, p1, p2, length, k * inter_scaling)

    # Scale the terms in the angle force appropriately
    angle_force = system.getForce(1)
    for angle_index in range(angle_force.getNumAngles()):
        p1, p2, p3, angle, k = angle_force.getAngleParameters(angle_index)
        if p1 in solute and p2 in solute and p3 in solute:
            angle_force.setAngleParameters(angle_index, p1, p2, p3, angle, k * solute_scaling)
        elif set([p1, p2, p3]).intersection(set(solute)) != set() and set([p1, p2, p3]).intersection(
                set(solvent)) != set():
            angle_force.setAngleParameters(angle_index, p1, p2, p3, angle, k * inter_scaling)

    # Scale the terms in the torsion force appropriately
    torsion_force = system.getForce(2)
    for torsion_index in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(torsion_index)
        if p1 in solute and p2 in solute and p3 in solute and p4 in solute:
            torsion_force.setTorsionParameters(torsion_index, p1, p2, p3, p4, periodicity, phase, k * solute_scaling)
        elif set([p1, p2, p3, p4]).intersection(set(solute)) != set() and set([p1, p2, p3, p4]).intersection(
                set(solvent)) != set():
            torsion_force.setTorsionParameters(torsion_index, p1, p2, p3, p4, periodicity, phase, k * inter_scaling)

    # Scale the exceptions in the nonbonded force appropriately
    nb_force = system.getForce(3)
    for nb_index in range(nb_force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(nb_index)
        if p1 in solute and p2 in solute:
            nb_force.setExceptionParameters(nb_index, p1, p2, solute_scaling * chargeProd, sigma, solute_scaling * epsilon)
        elif (p1 in solute and p2 in solvent) or (p1 in solvent and p2 in solute):
            nb_force.setExceptionParameters(nb_index, p1, p2, inter_scaling * chargeProd, sigma, inter_scaling * epsilon)

    # Scale nonbonded interactions for solute-solute region by adding exceptions for all pairs of atoms
    exception_pairs = [tuple(sorted([nb_force.getExceptionParameters(nb_index)[0], nb_force.getExceptionParameters(nb_index)[1]])) for nb_index in range(nb_force.getNumExceptions())]
    solute_pairs = set([tuple(sorted(pair)) for pair in list(itertools.product(solute, solute))])
    for pair in list(solute_pairs):
        p1 = pair[0]
        p2 = pair[1]
        p1_charge, p1_sigma, p1_epsilon = nb_force.getParticleParameters(p1)
        p2_charge, p2_sigma, p2_epsilon = nb_force.getParticleParameters(p2)
        if p1 != p2:
            if pair not in exception_pairs:
                nb_force.addException(p1, p2, p1_charge * p2_charge * solute_scaling, 0.5 * (p1_sigma + p2_sigma),
                                      np.sqrt(p1_epsilon * p2_epsilon) * solute_scaling)

    # Scale nonbonded interactions for inter region by adding exceptions for all pairs of atoms
    for pair in list(itertools.product(solute, solvent)):
        p1 = pair[0]
        p2 = int(pair[1])  # otherwise, will be a numpy int
        p1_charge, p1_sigma, p1_epsilon = nb_force.getParticleParameters(p1)
        p2_charge, p2_sigma, p2_epsilon = nb_force.getParticleParameters(p2)
        nb_force.addException(p1, p2, p1_charge * p2_charge * inter_scaling, 0.5 * (p1_sigma + p2_sigma), np.sqrt(p1_epsilon * p2_epsilon) * inter_scaling)

    # Get energy
    thermostate = ThermodynamicState(system, temperature=T_min)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = thermostate.create_context(integrator)
    context.setPositions(positions)
    sampler_state = SamplerState.from_context(context)
    nonREST_energy = thermostate.reduced_potential(sampler_state)

    assert np.isclose([REST_energy], [nonREST_energy]), f"The energy of the REST system ({REST_energy}) does not match " \
                                                        f"that of the non-REST system with terms manually scaled according to REST2({nonREST_energy})."

def test_energy_scaling():
    """
        Test whether the energy of a REST-ified system is equal to the energy of the system with terms manually scaled by
        the same factor as is used in REST.  T_min is 298 K and the thermodynamic state has temperature 600 K.
    """

    # Set temperatures
    T_min = 298.0 * unit.kelvin
    T = 600 * unit.kelvin

    # # Create vanilla system for alanine dipeptide
    # ala = AlanineDipeptideVacuum()
    # system = ala.system
    # system.removeForce(4)
    # positions = ala.positions
    #
    # # Check energy scaling
    # compare_energies(system, positions, T_min, T)

    # # Create repartitioned hybrid system for lambda 0 endstate for alanine dipeptide
    # atp, system_generator = generate_atp(phase='vacuum')
    # htf = generate_dipeptide_top_pos_sys(atp.topology,
    #                                      new_res='THR',
    #                                      system=atp.system,
    #                                      positions=atp.positions,
    #                                      system_generator=system_generator,
    #                                      conduct_htf_prop=True,
    #                                      repartitioned=True,
    #                                      endstate=0,
    #                                      validate_endstate_energy=False)
    # # Check energy scaling
    # compare_energies(htf.hybrid_system, htf.hybrid_positions, T_min, T)

    # Create vanilla system for alanine dipeptide in solvent
    # from openmmtools.testsystems import AlanineDipeptideExplicit
    # ala = AlanineDipeptideExplicit()
    # system = ala.system
    # system.removeForce(4)
    # positions = ala.positions
    from simtk.openmm import app
    from openmmforcefields.generators import SystemGenerator

    ala = AlanineDipeptideVacuum()
    forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    barostat = openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 298 * unit.kelvin, 50)
    system_generator = SystemGenerator(forcefields=forcefield_files,
                                       barostat=barostat,
                                       forcefield_kwargs={'removeCMMotion': False,
                                                          'ewaldErrorTolerance': 1e-4,
                                                          'constraints': app.HBonds,
                                                          'hydrogenMass': 4 * unit.amus},
                                       periodic_forcefield_kwargs={'nonbondedMethod': app.PME},
                                       small_molecule_forcefield='gaff-2.11',
                                       nonperiodic_forcefield_kwargs=None,
                                       molecules=None,
                                       cache=None)
    modeller = app.Modeller(ala.topology, ala.positions)
    modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=9 * unit.angstroms, ionicStrength=0.15 * unit.molar)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()

    # Canonicalize the solvated positions: turn tuples into np.array
    positions = unit.quantity.Quantity(
        value=np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]),
        unit=unit.nanometers)
    system = system_generator.create_system(solvated_topology)

    # Check energy scaling
    compare_energies(system, positions, T_min, T)
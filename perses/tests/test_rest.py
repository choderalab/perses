###########################################
# IMPORTS
###########################################
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from simtk import unit, openmm
from perses.tests.utils import compute_potential_components
from openmmtools.constants import kB
from perses.dispersed.utils import configure_platform
from perses.annihilation.rest import RESTTopologyFactory
from perses.annihilation.lambda_protocol import RESTState
import numpy as np
from perses.tests.test_topology_proposal import generate_atp, generate_dipeptide_top_pos_sys
from openmmtools.testsystems import AlanineDipeptideVacuum, AlanineDipeptideExplicit
import itertools

#############################################
# CONSTANTS
#############################################
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CPU")


def compare_energy_components(rest_system, other_system, positions, platform=REFERENCE_PLATFORM):
    """
    Get energy components of a given system
    """
    platform = configure_platform(platform)

    # Create thermodynamic state and sampler state for non-rest system
    thermostate_other = ThermodynamicState(system=other_system, temperature=temperature)

    # Create context for non-rest system
    integrator_other = openmm.VerletIntegrator(1.0*unit.femtosecond)
    context_other = thermostate_other.create_context(integrator_other)
    context_other.setPositions(positions)

    # Get energy components for non-rest system
    components_other = [component[1] for component in compute_potential_components(context_other, beta=beta)]

    # Create thermodynamic state for rest_system
    thermostate_rest = ThermodynamicState(system=rest_system, temperature=temperature)

    # Create context forrest system
    integrator_rest = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context_rest = thermostate_rest.create_context(integrator_rest)
    context_rest.setPositions(positions)

    # Get energy components for rest system
    components_rest = [component[1] for component in compute_potential_components(context_rest, beta=beta)]

    # Check that bond, angle, and torsion energies match
    for other, rest in zip(components_other[:3], components_rest[:3]):
        assert np.isclose([other], [rest]), f"The energies do not match for the {other[0]}: {other[1]} (other system) vs. {rest[1]} (REST system)"

    # Check that nonbonded energies match
    print(components_rest)
    nonbonded_other = np.array(components_other[3:]).sum()
    nonbonded_rest = np.array(components_rest[3:]).sum()
    assert np.isclose([nonbonded_other], [nonbonded_rest]), f"The energies do not match for the NonbondedForce: {nonbonded_other} (other system) vs. {nonbonded_rest} (REST system)"


def test_bookkeeping():
    """
    Given the default Global variables, do energy component bookkeeping with the REST system and the original system.
    Specifically, the first CustomBondForce must match the HarmonicBondForce
    (same with the second and third forces with the HarmonicAngle and PeriodicTorsionForces).
    The sum of the NonbondedForce, the CustomNonbondedForce, and the CustomBondForce (last one) must be equal to the
    energy of the original system's NonbondedForce.
    """

    ## CASE 1: alanine dipeptide in vacuum
    # Create vanilla system
    ala = AlanineDipeptideVacuum()
    system = ala.system
    positions = ala.positions

    # Create REST system
    system.removeForce(4)
    res1 = list(ala.topology.residues())[1]
    rest_atoms = [atom.index for atom in res1.atoms()]
    factory = RESTTopologyFactory(system, solute_region=rest_atoms)
    REST_system = factory.REST_system

    # Compare energy components
    compare_energy_components(REST_system, system, positions)

    ## CASE 2: alanine dipeptide in solvent
    # Create vanilla system
    ala = AlanineDipeptideExplicit()
    system = ala.system
    positions = ala.positions

    # Create REST system
    system.removeForce(4)
    res1 = list(ala.topology.residues())[1]
    rest_atoms = [atom.index for atom in res1.atoms()]
    factory = RESTTopologyFactory(system, solute_region=rest_atoms, use_dispersion_correction=True)
    REST_system = factory.REST_system

    # Compare energy components
    compare_energy_components(REST_system, system, positions)

    ## CASE 3: alanine dipeptide in solvent with repartitioned hybrid system
    # Create repartitioned hybrid system for lambda 0 endstate
    atp, system_generator = generate_atp(phase='solvent')
    htf = generate_dipeptide_top_pos_sys(atp.topology,
                                         new_res='THR',
                                         system=atp.system,
                                         positions=atp.positions,
                                         system_generator=system_generator,
                                         conduct_htf_prop=True,
                                         generate_repartitioned_hybrid_topology_factory=True,
                                         endstate=0,
                                         validate_endstate_energy=False)

    # Create REST-ified hybrid system
    res1 = list(htf.hybrid_topology.residues)[1]
    rest_atoms = [atom.index for atom in list(res1.atoms)]
    factory = RESTTopologyFactory(htf.hybrid_system, solute_region=rest_atoms, use_dispersion_correction=True)
    REST_system = factory.REST_system

    # Compare energy components
    compare_energy_components(REST_system, htf.hybrid_system, htf.hybrid_positions)


def compare_energies(REST_system, other_system, positions, rest_atoms, T_min, T):

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
    solute = rest_atoms
    solvent = [i for i in range(other_system.getNumParticles()) if i not in solute]
    solute_scaling = beta_m / beta_0
    inter_scaling = np.sqrt(beta_m / beta_0)

    # Scale the terms in the bond force appropriately
    bond_force = other_system.getForce(0)
    for bond in range(bond_force.getNumBonds()):
        p1, p2, length, k = bond_force.getBondParameters(bond)
        if p1 in solute and p2 in solute:
            bond_force.setBondParameters(bond, p1, p2, length, k * solute_scaling)
        elif (p1 in solute and p2 in solvent) or (p1 in solvent and p2 in solute):
            bond_force.setBondParameters(bond, p1, p2, length, k * inter_scaling)

    # Scale the terms in the angle force appropriately
    angle_force = other_system.getForce(1)
    for angle_index in range(angle_force.getNumAngles()):
        p1, p2, p3, angle, k = angle_force.getAngleParameters(angle_index)
        if p1 in solute and p2 in solute and p3 in solute:
            angle_force.setAngleParameters(angle_index, p1, p2, p3, angle, k * solute_scaling)
        elif set([p1, p2, p3]).intersection(set(solute)) != set() and set([p1, p2, p3]).intersection(
                set(solvent)) != set():
            angle_force.setAngleParameters(angle_index, p1, p2, p3, angle, k * inter_scaling)

    # Scale the terms in the torsion force appropriately
    torsion_force = other_system.getForce(2)
    for torsion_index in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(torsion_index)
        if p1 in solute and p2 in solute and p3 in solute and p4 in solute:
            torsion_force.setTorsionParameters(torsion_index, p1, p2, p3, p4, periodicity, phase, k * solute_scaling)
        elif set([p1, p2, p3, p4]).intersection(set(solute)) != set() and set([p1, p2, p3, p4]).intersection(
                set(solvent)) != set():
            torsion_force.setTorsionParameters(torsion_index, p1, p2, p3, p4, periodicity, phase, k * inter_scaling)

    # Scale the exceptions in the nonbonded force appropriately
    nb_force = other_system.getForce(3)
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
        if tuple(sorted(pair)) not in exception_pairs:
            nb_force.addException(p1, p2, p1_charge * p2_charge * inter_scaling, 0.5 * (p1_sigma + p2_sigma), np.sqrt(p1_epsilon * p2_epsilon) * inter_scaling)

    # Get energy
    thermostate = ThermodynamicState(other_system, temperature=T_min)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = thermostate.create_context(integrator)
    context.setPositions(positions)
    sampler_state = SamplerState.from_context(context)
    nonREST_energy = thermostate.reduced_potential(sampler_state)

    assert REST_energy - nonREST_energy < 1, f"The energy of the REST system ({REST_energy}) does not match " \
                                                        f"that of the non-REST system with terms manually scaled according to REST2({nonREST_energy})."

def test_energy_scaling():
    """
        Test whether the energy of a REST-ified system is equal to the energy of the system with terms manually scaled by
        the same factor as is used in REST.  T_min is 298 K and the thermodynamic state has temperature 600 K.
    """

    # Set temperatures
    T_min = 298.0 * unit.kelvin
    T = 600 * unit.kelvin

    ## CASE 1: alanine dipeptide in vacuum
    # Create vanilla system
    ala = AlanineDipeptideVacuum()
    system = ala.system
    positions = ala.positions

    # Create REST system
    system.removeForce(4)
    res1 = list(ala.topology.residues())[1]
    rest_atoms = [atom.index for atom in res1.atoms()]
    factory = RESTTopologyFactory(system, solute_region=rest_atoms)
    REST_system = factory.REST_system

    # Check energy scaling
    compare_energies(REST_system, system, positions, rest_atoms, T_min, T)

    ## CASE 2: alanine dipeptide in solvent
    # Create vanilla system
    ala = AlanineDipeptideExplicit()
    system = ala.system
    positions = ala.positions

    # Create REST system
    system.removeForce(4)
    res1 = list(ala.topology.residues())[1]
    rest_atoms = [atom.index for atom in res1.atoms()]
    factory = RESTTopologyFactory(system, solute_region=rest_atoms, use_dispersion_correction=True)
    REST_system = factory.REST_system

    # Check energy scaling
    compare_energies(REST_system, system, positions, rest_atoms, T_min, T)

    ## CASE 3: alanine dipeptide in solvent with repartitioned hybrid system
    # Create repartitioned hybrid system for lambda 0 endstate
    atp, system_generator = generate_atp(phase='solvent')
    htf = generate_dipeptide_top_pos_sys(atp.topology,
                                         new_res='THR',
                                         system=atp.system,
                                         positions=atp.positions,
                                         system_generator=system_generator,
                                         conduct_htf_prop=True,
                                         generate_repartitioned_hybrid_topology_factory=True,
                                         endstate=0,
                                         validate_endstate_energy=False)
    system = htf.hybrid_system
    system.removeForce(0) # Remove barostat

    # Create REST-ified hybrid system
    res1 = list(htf.hybrid_topology.residues)[1]
    rest_atoms = [atom.index for atom in list(res1.atoms)]
    factory = RESTTopologyFactory(system, solute_region=rest_atoms, use_dispersion_correction=True)
    REST_system = factory.REST_system

    # Check energy scaling
    compare_energies(REST_system, system, htf.hybrid_positions, rest_atoms, T_min, T)

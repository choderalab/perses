###########################################
# IMPORTS
###########################################
from openmmtools.states import SamplerState, ThermodynamicState
from simtk import unit, openmm
from perses.tests.utils import compute_potential_components
from openmmtools.constants import kB
from perses.dispersed.feptasks import minimize
from perses.dispersed.utils import configure_platform
from perses.annihilation.rest import RESTTopologyFactory
import numpy as np

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
    sampler_state_other = SamplerState(positions, box_vectors=other_system.getDefaultPeriodicBoxVectors())

    # Create context for non-rest system
    integrator_other = openmm.VerletIntegrator(1.0)
    context_other = thermostate_other.create_context(integrator_other, platform=platform)

    # Minimize
    minimize(thermostate_other, sampler_state_other)
    sampler_state_other.apply_to_context(context_other)

    # Get energy components for non-rest system
    components_other = compute_potential_components(context_other, beta=beta)

    # Create thermodynamic state for rest_system
    thermostate_rest = ThermodynamicState(system=rest_system, temperature=temperature)

    # Create context for rest system
    integrator_rest = openmm.VerletIntegrator(1.0)
    platform = configure_platform(platform)
    context_rest = thermostate_rest.create_context(integrator_rest, platform=platform)
    sampler_state_other.apply_to_context(context_rest)

    # Get energy components for rest system
    components_rest = compute_potential_components(context_rest, beta=beta)
    print(components_rest)

    # Check that bond, angle, and torsion energies match
    for other, rest in zip(components_other[:3], components_rest[:3]):
        assert np.isclose([other[1]], [rest[1]]), f"The energies do not match for the {other[0]}: {other[1]} (other system) vs. {rest[1]} (REST system)"

    # Check that nonbonded energies
    nonbonded_other = components_other[3][1]
    nonbonded_rest = components_rest[3][1] + components_rest[4][1] + components_rest[5][1]
    assert np.isclose([nonbonded_other], [nonbonded_rest]), f"The energies do not match for the NonbondedForce: {nonbonded_other} (other system) vs. {nonbonded_rest} (REST system)"


def test_bookkeeping():
    """
    Given the default Global variables, do energy component bookkeeping with the REST system and the original system.
    Specifically, the first CustomBondForce must match the HarmonicBondForce
    (same with the second and third forces with the HarmonicAngle and PeriodicTorsionForces).
    The sum of the NonbondedForce, the CustomNonbondedForce, and the CustomBondForce (last one) must be equal to the
    energy of the original system's NonbondedForce.
    """
    from perses.tests.test_topology_proposal import generate_atp, generate_dipeptide_top_pos_sys
    from openmmtools.testsystems import AlanineDipeptideVacuum

    # Create vanilla system for alanine dipeptide
    ala = AlanineDipeptideVacuum()
    system = ala.system
    positions = ala.positions

    # Create REST system for alanine dipeptide
    system.removeForce(4)
    factory = RESTTopologyFactory(system, solute_region=list(range(6, 16)))
    REST_system = factory.REST_system

    compare_energy_components(REST_system, system, positions)

    # Create repartitioned hybrid system for lambda 0 endstate for alanine dipeptide
    atp, system_generator = generate_atp(phase='vacuum')
    htf = generate_dipeptide_top_pos_sys(atp.topology,
                                         new_res='THR',
                                         system=atp.system,
                                         positions=atp.positions,
                                         system_generator=system_generator,
                                         conduct_htf_prop=True,
                                         repartitioned=True,
                                         endstate=0,
                                         validate_endstate_energy=False)

    # Create REST-ified hybrid system for alanine dipeptide
    factory = RESTTopologyFactory(htf.hybrid_system, solute_region=list(range(6, 16)))
    REST_system = factory.REST_system

    compare_energy_components(REST_system, htf.hybrid_system, htf.hybrid_positions)

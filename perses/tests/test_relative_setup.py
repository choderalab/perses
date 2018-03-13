import numpy as np
from simtk import openmm, unit
import simtk.openmm.app as app
import openeye.oechem as oechem
from perses.dispersed import relative_setup, feptasks
from openmmtools import states, alchemy, testsystems, integrators
import pickle

def generate_example_waterbox_states(temperature=300.0*unit.kelvin, pressure=1.0*unit.atmosphere):
    """
    This is a convenience function to generate a CompoundThermodynamicState and SamplerState to use in other tests.
    Here, we generate an alchemical water box
    """
    #get the water box testsystem
    water_ts = testsystems.AlchemicalWaterBox()
    system = water_ts.system
    positions = water_ts.positions

    #construct the openmmtools objects for it
    sampler_state = states.SamplerState(positions, box_vectors=system.getDefaultPeriodicBoxVectors())
    thermodynamic_state = states.ThermodynamicState(system, temperature=temperature, pressure=pressure)

    #make an alchemical state
    alchemical_state = alchemy.AlchemicalState.from_system(system)
    alchemical_state.set_alchemical_parameters(0.0)

    #make a compound thermodynamic state
    cpd_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, [alchemical_state])

    return cpd_thermodynamic_state, sampler_state, water_ts.topology

def test_run_nonequilibrium_switching_move():
    """
    Test that the NonequilibriumSwitchingMove changes lambda from 0 to 1 in multiple iterations
    """
    n_iterations = 5
    cpd_thermodynamic_state, sampler_state, topology = generate_example_waterbox_states()

    #make a BAOAB integrator for use with the run_protocol module
    integrator = integrators.AlchemicalNonequilibriumLangevinIntegrator(splitting="V R O H O R V", nsteps_neq=100)

    #make the EquilibriumResult object that will be used to initialize the protocol runs:
    eq_result = feptasks.EquilibriumResult(0.0, sampler_state)
    
    #run the run_protocol task n_iterations times, checking that the context is correctly handled.
    for i in range(n_iterations):
        #serialize the integrator and deserialize to simulate transmission over the network.
        dumped_integrator = pickle.dumps(integrator)
        deserialized_integrator = pickle.loads(dumped_integrator)
        ne_result = feptasks.run_protocol(eq_result, cpd_thermodynamic_state, deserialized_integrator, topology, 10)

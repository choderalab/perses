import numpy as np
import os
from pkg_resources import resource_filename
from simtk import openmm, unit
import simtk.openmm.app as app
import openeye.oechem as oechem
from perses.dispersed import relative_setup, feptasks
import mdtraj as md
from openmmtools import states, alchemy, testsystems, integrators, cache
import pickle
import yaml

default_forward_functions = {
        'lambda_sterics' : 'lambda',
        'lambda_electrostatics' : 'lambda',
    }


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
    
    md_topology = md.Topology.from_openmm(topology)

    #make the EquilibriumResult object that will be used to initialize the protocol runs:
    eq_result = feptasks.EquilibriumResult(0.0, sampler_state)
    
    #run the NE switching move task n_iterations times, checking that the context is correctly handled.
    for i in range(n_iterations):
        ne_move = feptasks.NonequilibriumSwitchingMove(default_forward_functions, splitting="V R O H R V", temperature=300.0*unit.kelvin, nsteps_neq=10, top=md_topology, work_save_interval=10)

        integrator = ne_move._integrator

        context, integrator = cache.global_context_cache.get_context(cpd_thermodynamic_state, integrator)
        
        assert context.getParameter("lambda_sterics") == 0.0
        assert integrator.getGlobalVariableByName("lambda") == 0.0
        ne_move.apply(cpd_thermodynamic_state, sampler_state)

        #check that the value changed to 1.0 for all parameters
        assert context.getParameter("lambda_sterics") == 1.0
        assert integrator.getGlobalVariableByName("lambda") == 1.0

def test_run_cdk2_iterations():
    """
    Ensure that we can instantiate and run the cdk2 ligands in vacuum
    """
    setup_directory = resource_filename("perses", "data/cdk2-example")
    os.chdir(setup_directory)
    yaml_filename = "cdk2_setup.yaml"
    yaml_file = open(yaml_filename, "r")
    setup_options = yaml.load(yaml_file)
    yaml_file.close()
    setup_options['solvate'] = False
    setup_options['phase'] = 'solvent'
    setup_options['scheduler_address'] = None

    fe_setup, ne_fep = relative_setup.run_setup(setup_options)

    ne_fep.run(n_iterations=1)

if __name__=="__main__":
    test_run_cdk2_iterations()
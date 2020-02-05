import numpy as np
import os
from pkg_resources import resource_filename
from simtk import unit
from perses.dispersed import feptasks
from perses.app import relative_setup, setup_relative_calculation
import mdtraj as md
from openmmtools import states, alchemy, testsystems, cache
import yaml
from unittest import skipIf
running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

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

# TODO fails as integrator not bound to context
@skipIf(running_on_github_actions, "Skip analysis test on GH Actions.  Currently broken")
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
        ne_move = feptasks.NonequilibriumSwitchingMove(default_forward_functions, splitting="V R O H R V", temperature=300.0*unit.kelvin, nsteps_neq=10, timestep=1.0*unit.femtoseconds, top=md_topology, work_save_interval=10)

        integrator = ne_move._integrator

        context, integrator = cache.global_context_cache.get_context(cpd_thermodynamic_state, integrator)

        assert context.getParameter("lambda_sterics") == 0.0
        assert integrator.getGlobalVariableByName("lambda") == 0.0
        ne_move.apply(cpd_thermodynamic_state, sampler_state)

        #check that the value changed to 1.0 for all parameters
        assert context.getParameter("lambda_sterics") == 1.0
        assert integrator.getGlobalVariableByName("lambda") == 1.0

        
@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip slow test on TRAVIS.")
def test_run_cdk2_iterations_neq():
    """
    Ensure that we can instantiate and run a nonequilibrium relative free energy calculation for the cdk2 ligands in vacuum
    """
    setup_directory = resource_filename("perses", "data/cdk2-example")
    os.chdir(setup_directory)
    n_iterations = 2

    yaml_filename = "cdk2_setup_neq.yaml"
    yaml_file = open(yaml_filename, "r")
    setup_options = yaml.safe_load(yaml_file)
    yaml_file.close()

    if not os.path.exists(setup_options['trajectory_directory']):
        os.makedirs(setup_options['trajectory_directory'])

    setup_options['solvate'] = False
    setup_options['scheduler_address'] = None

    length_of_protocol = setup_options['n_steps_ncmc_protocol']
    write_interval = setup_options['n_steps_per_move_application']

    n_work_values_per_iteration = length_of_protocol // write_interval

    setup_dict = setup_relative_calculation.run_setup(setup_options)

    setup_dict['ne_fep']['solvent'].run(n_iterations=n_iterations)

    #now check that the correct number of iterations was written out:
    os.chdir(setup_options['trajectory_directory'])
    import glob

    #for the verification of work writing, we add one to the work dimension, since the first work value is always zero

    #check for lambda zero
    lambda_zero_filenames = glob.glob("*0.cw.npy")
    lambda_zero_npy = np.stack([np.load(filename) for filename in lambda_zero_filenames])
    assert np.shape(lambda_zero_npy) == (n_iterations, n_work_values_per_iteration+1)

    #check for lambda one
    lambda_one_filenames = glob.glob("*1.cw.npy")
    lambda_one_npy = np.stack([np.load(filename) for filename in lambda_one_filenames])
    assert np.shape(lambda_one_npy) == (n_iterations, n_work_values_per_iteration+1)

@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip slow test on TRAVIS.")
def test_run_cdk2_iterations_repex():
    """
    Ensure that we can instantiate and run a repex relative free energy calculation the cdk2 ligands in vacuum
    """
    # Enter a temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Move to temporary directory
        os.chdir(tmpdirname)
        print(f'Running example in temporary directory: {tmpdirname}')

        # Setup directory
        setup_directory = resource_filename("perses", "data/cdk2-example")

        # Get options
        from perses.app.setup_relative_calculation import getSetupOptions
        yaml_filename = os.path.join(setup_directory, "cdk2_setup_repex.yaml")
        setup_options = getSetupOptions(yaml_filename)

        # DEBUG: Print traceback for any UserWarnings
        show_warning_stacktraces = False
        if show_warning_stacktraces:
            import traceback
            import warnings
            _old_warn = warnings.warn
            def warn(*args, **kwargs):
                tb = traceback.extract_stack()
                _old_warn(*args, **kwargs)
                print("".join(traceback.format_list(tb)[:-1]))
            warnings.warn = warn

        # Update options
        #setup_options['solvate'] = False
        #setup_options['n_cycles'] = 2
        setup_options['scheduler_address'] = None
        for parameter in ['protein_pdb', 'ligand_file', 'small_molecule_parameters_cache']:
            setup_options[parameter] = os.path.join(setup_directory, setup_options[parameter])
        for parameter in ['trajectory_directory', 'trajectory_prefix', 'save_setup_pickle_as']:
            setup_options[parameter] = os.path.join(tmpdirname, setup_options[parameter])

        #length_of_protocol = setup_options['n_steps_ncmc_protocol']
        #write_interval = setup_options['n_steps_per_move_application']
        #n_work_values_per_iteration = length_of_protocol // write_interval

        # Run setup
        setup_dict = setup_relative_calculation.run_setup(setup_options)
        setup_dict['hybrid_samplers']['solvent'].run(n_iterations=n_iterations)

        # TODO: Check output


if __name__=="__main__":
    test_run_cdk2_iterations_repex()

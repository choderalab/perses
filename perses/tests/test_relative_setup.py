import os
import shutil

from pkg_resources import resource_filename
from simtk import unit
from perses.dispersed import feptasks
from perses.app import setup_relative_calculation
import mdtraj as md
from openmmtools import states, alchemy, testsystems, cache
from unittest import skipIf
import pytest

from perses.tests.utils import enter_temp_directory

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


def test_parsed_yaml_generation():
    """
    Test input yaml options are correctly parsed into output yaml options file. Including extra metadata (timestamp and
    ligands names).
    """
    import yaml
    from perses.app.setup_relative_calculation import getSetupOptions, _generate_parsed_yaml
    with enter_temp_directory():
        base_dir = resource_filename(
            "perses",
            os.path.join("data", "Tyk2_ligands_example"),
        )
        input_yaml_file = os.path.join(base_dir, "tyk2_0_3.yaml")  # Get yaml path from perses data directory
        # Read the contents of input YAML file
        with open(input_yaml_file) as input_file:
            input_yaml_data = yaml.load(input_file, Loader=yaml.FullLoader)
        # generate setup options from input file
        setup_options = getSetupOptions(input_yaml_file)
        # Copy the ligand file -- needed for getting ligands names
        shutil.copy(os.path.join(base_dir, 'Tyk2_ligands_shifted.sdf'), ".")
        # generate parsed yaml file from setup options
        parsed_yaml_file_path = _generate_parsed_yaml(setup_options=setup_options, input_yaml_file_path=input_yaml_file)

        # Make sure keys in input exist in parsed yaml file
        with open(parsed_yaml_file_path) as parsed_file:
            parsed_yaml_data = yaml.load(parsed_file, Loader=yaml.FullLoader)
        input_keys_set = set(input_yaml_data.keys())
        parsed_keys_set = set(parsed_yaml_data.keys())
        assert input_keys_set.issubset(parsed_keys_set), "Input yaml file options are not a subset of the parsed yaml" \
                                                         " file."
        # Also check that the metadata keys are added (timestamp and ligands names)
        metadata_keys_set = {'timestamp', 'old_ligand_name', 'new_ligand_name'}
        assert metadata_keys_set.issubset(parsed_keys_set), \
            f"Metadata keys {metadata_keys_set} not found in parsed keys."


# TODO fails as integrator not bound to context
#@skipIf(running_on_github_actions, "Skip analysis test on GH Actions.  Currently broken")
@pytest.mark.skip(reason="Skip analysis test on GH Actions.  Currently broken")
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


#def test_run_cdk2_iterations_neq():
#    """
#    Ensure that we can instantiate and run a nonequilibrium relative free energy calculation for the cdk2 ligands in vacuum
#    """
#    setup_directory = resource_filename("perses", "data/cdk2-example")
#    os.chdir(setup_directory) # WARNING: DON'T CHANGE THE WORKING DIRECTORY BECAUSE IT WILL BREAK SUBSEQUENT TESTS
#    n_iterations = 2
#
#    yaml_filename = "cdk2_setup_neq.yaml"
#    from perses.app.setup_relative_calculation import getSetupOptions
#    setup_options = getSetupOptions(yaml_filename)
#
#    if not os.path.exists(setup_options['trajectory_directory']):
#        os.makedirs(setup_options['trajectory_directory'])
#
#    setup_options['solvate'] = False
#    setup_options['scheduler_address'] = None
#
#    length_of_protocol = setup_options['n_steps_ncmc_protocol']
#    write_interval = setup_options['n_steps_per_move_application']
#
#    n_work_values_per_iteration = length_of_protocol // write_interval
#
#    setup_dict = setup_relative_calculation.run_setup(setup_options)
#
#    setup_dict['ne_fep']['solvent'].run(n_iterations=n_iterations)
#
#    #now check that the correct number of iterations was written out:
#    os.chdir(setup_options['trajectory_directory']) # WARNING: DON'T CHANGE THE WORKING DIRECTORY BECAUSE IT WILL BREAK SUBSEQUENT TESTS
#    import glob
#
#    #for the verification of work writing, we add one to the work dimension, since the first work value is always zero
#
#    #check for lambda zero
#    lambda_zero_filenames = glob.glob("*0.cw.npy")
#    lambda_zero_npy = np.stack([np.load(filename) for filename in lambda_zero_filenames])
#    assert np.shape(lambda_zero_npy) == (n_iterations, n_work_values_per_iteration+1)
#
#    #check for lambda one
#    lambda_one_filenames = glob.glob("*1.cw.npy")
#    lambda_one_npy = np.stack([np.load(filename) for filename in lambda_one_filenames])
#    assert np.shape(lambda_one_npy) == (n_iterations, n_work_values_per_iteration+1)

#@skipIf(running_on_github_actions, "Skip analysis test on GH Actions. SLOW")
@pytest.mark.skip(reason="Skip analysis test on GH Actions. SLOW")
def test_run_cdk2_iterations_repex():
    """
    Ensure that we can instantiate and run a repex relative free energy calculation the cdk2 ligands in vacuum
    """
    # Enter a temporary directory
    from perses.tests.utils import enter_temp_directory
    with enter_temp_directory() as tmpdirname:
        # Move to temporary directory
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
        for parameter in ['protein_pdb', 'ligand_file']:
            setup_options[parameter] = os.path.join(setup_directory, setup_options[parameter])
        for parameter in ['trajectory_directory', 'save_setup_pickle_as']:
            setup_options[parameter] = os.path.join(tmpdirname, setup_options[parameter])

        #length_of_protocol = setup_options['n_steps_ncmc_protocol']
        #write_interval = setup_options['n_steps_per_move_application']
        #n_work_values_per_iteration = length_of_protocol // write_interval

        # Run setup
        setup_dict = setup_relative_calculation.run_setup(setup_options)
        setup_dict['hybrid_samplers']['solvent'].run(n_iterations=n_iterations)

        # TODO: Check output

#@skipIf(running_on_github_actions, "Skip analysis test on GH Actions. SLOW")
@pytest.mark.skip(reason="Skip analysis test on GH Actions. SLOW")
def test_run_bace_spectator():
    """
    Ensure that we can instantiate and run a repex relative free energy calculation the cdk2 ligands in vacuum
    """
    # Enter a temporary directory
    from perses.tests.utils import enter_temp_directory
    with enter_temp_directory() as tmpdirname:
        # Move to temporary directory
        print(f'Running example in temporary directory: {tmpdirname}')

        # Setup directory
        setup_directory = resource_filename("perses", "data/bace-example")
        print(f'Setup directory : {setup_directory}')

        # Get options
        from perses.app.setup_relative_calculation import getSetupOptions
        yaml_filename = os.path.join(setup_directory, "bace_setup.yaml")
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

        setup_options['scheduler_address'] = None
        for parameter in ['protein_pdb', 'ligand_file']:
            setup_options[parameter] = os.path.join(setup_directory, setup_options[parameter])
            # only one spectator
            setup_options['spectators'] = [ os.path.join(setup_directory, setup_options['spectators'][0])]
        for parameter in ['trajectory_directory', 'trajectory_prefix', 'save_setup_pickle_as']:
            setup_options[parameter] = os.path.join(tmpdirname, setup_options[parameter])


        # Run setup
        n_iterations = 2
        setup_dict = setup_relative_calculation.run_setup(setup_options)
        setup_dict['hybrid_samplers']['complex'].run(n_iterations=n_iterations)

        # test that there is TLA in the complex system
        found_tla = False
        for res in setup_dict['hybrid_topology_factories']['complex'].hybrid_topology.residues:
            if res.name == 'TLA':
                found_tla = True
        assert found_tla == True, 'Spectator TLA not in old topology'


def test_host_guest_deterministic_geometries():
    """
    execute the `RelativeFEPSetup` with geometries specified for a host-guest relative free energy pair
    """
    from perses.app.relative_setup import RelativeFEPSetup

    # Setup directory
    ligand_sdf = resource_filename("perses", "data/given-geometries/ligands.sdf")
    host_pdb = resource_filename("perses", "data/given-geometries/receptor.pdb")

    setup = RelativeFEPSetup(
                 ligand_input = ligand_sdf,
                 old_ligand_index=0,
                 new_ligand_index=1,
                 forcefield_files = ['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml'],
                 phases = ['complex', 'solvent', 'vacuum'],
                 protein_pdb_filename=host_pdb,
                 receptor_mol2_filename=None,
                 pressure=1.0 * unit.atmosphere,
                 temperature=300.0 * unit.kelvin,
                 solvent_padding=9.0 * unit.angstroms,
                 ionic_strength=0.15 * unit.molar,
                 hmass=3*unit.amus,
                 neglect_angles=False,
                 map_strength='default',
                 atom_expr=None,
                 bond_expr=None,
                 anneal_14s=False,
                 small_molecule_forcefield='gaff-2.11',
                 small_molecule_parameters_cache=None,
                 trajectory_directory=None,
                 trajectory_prefix=None,
                 spectator_filenames=None,
                 nonbonded_method = 'PME',
                 complex_box_dimensions=None,
                 solvent_box_dimensions=None,
                 remove_constraints=False,
                 use_given_geometries = True
                 )

def test_relative_setup_charge_change():
    """
    execute `RelativeFEPSetup` in solvent/complex phase on a charge change and assert that the modified new system and old system charge difference is zero.
    also assert endstate validation.
    """
    from perses.app.relative_setup import RelativeFEPSetup
    import numpy as np
    # Setup directory
    ligand_sdf = resource_filename("perses", "data/bace-example/Bace_ligands_shifted.sdf")
    host_pdb = resource_filename("perses", "data/bace-example/Bace_protein.pdb/receptor.pdb")

    setup = RelativeFEPSetup(
                 ligand_input = ligand_sdf,
                 old_ligand_index=0,
                 new_ligand_index=12,
                 forcefield_files = ['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml'],
                 phases = ['solvent', 'vacuum'],
                 protein_pdb_filename=host_pdb,
                 receptor_mol2_filename=None,
                 pressure=1.0 * unit.atmosphere,
                 temperature=300.0 * unit.kelvin,
                 solvent_padding=9.0 * unit.angstroms,
                 ionic_strength=0.15 * unit.molar,
                 hmass=4*unit.amus,
                 neglect_angles=False,
                 map_strength='default',
                 atom_expr=None,
                 bond_expr=None,
                 anneal_14s=False,
                 small_molecule_forcefield='gaff-2.11',
                 small_molecule_parameters_cache=None,
                 trajectory_directory=None,
                 trajectory_prefix=None,
                 spectator_filenames=None,
                 nonbonded_method = 'PME',
                 complex_box_dimensions=None,
                 solvent_box_dimensions=None,
                 remove_constraints=False,
                 use_given_geometries = False
                 )

    # sum all of the charges of topology.
    """strictly speaking, this is redundant because endstate validation is done in the `RelativeFEPSetup`"""
    old_nbf = [force for force in setup._solvent_topology_proposal._old_system.getForces() if force.__class__.__name__ == 'NonbondedForce'][0]
    new_nbf = [force for force in setup._solvent_topology_proposal._new_system.getForces() if force.__class__.__name__ == 'NonbondedForce'][0]
    old_system_charge_sum = sum([old_nbf.getParticleParameters(i)[0].value_in_unit_system(unit.md_unit_system) for i in range(old_nbf.getNumParticles())])
    new_system_charge_sum = sum([old_nbf.getParticleParameters(i)[0].value_in_unit_system(unit.md_unit_system) for i in range(new_nbf.getNumParticles())])
    charge_diff = int(old_system_charge_sum - new_system_charge_sum)
    assert np.isclose(charge_diff, 0), f"charge diff is {charge_diff} but should be zero."


def test_relative_setup_solvent_padding():
    """
    Check that the user inputted solvent_padding argument to `RelativeFEPSetup` actually changes the padding for the solvent phase
    """
    from perses.app.relative_setup import RelativeFEPSetup
    import numpy as np

    input_solvent_padding = 1.7 * unit.nanometers
    smiles_filename = resource_filename("perses", os.path.join("data", "test.smi"))
    fe_setup = RelativeFEPSetup(
        ligand_input=smiles_filename,
        old_ligand_index=0,
        new_ligand_index=1,
        forcefield_files=["amber14/tip3p.xml"],
        small_molecule_forcefield="gaff-2.11",
        phases=["solvent"],
        solvent_padding=input_solvent_padding)
    assert input_solvent_padding == fe_setup._padding, f"Input solvent padding, {input_solvent_padding}, is different from setup object solvent padding, {fe_setup._padding}."


# TODO: parametrize test to load ligands from list of files as well
def test_relative_fep_setup_from_files():
    """Test Relative Free Energy Perturbation setup from files"""
    from perses.app.relative_setup import RelativeFEPSetup
    receptor_file = ""
    ligands_file = ""
    old_ligand_index = ""
    new_ligand_index = ""
    forcefield_files = ""
    phases = ""

    fep_setup = RelativeFEPSetup.from_files(
        receptor_file,
        ligands_file,
        old_ligand_index,
        new_ligand_index,
        forcefield_files=forcefield_files,
        phases=phases,
    )

    # Check that the receptor and ligands are as expected
    assert fep_setup._receptor == omm_receptor
    assert fep_setup._ligand_offmol_old == off_old_ligand
    assert fep_setup._ligand_offmol_new == off_new_ligand

    return NotImplementedError


def test_relative_fep_setup_init():
    """Test initialization of RelativeFEPSetup object from openff molecules and openmm receptor."""
    from openmm.app import PDBFile
    from perses.app.relative_setup import RelativeFEPSetup
    from perses.utils.openeye import createOEMolFromSDF
    # Load receptor from PDB
    pdb_file = resource_filename("perses", os.path.join("data", "Tyk2_ligands_example", "Tyk2_protein.pdb"))
    omm_pdb = PDBFile(pdb_file)
    omm_top = omm_pdb.topology
    omm_pos = omm_pdb.positions
    # Load sdf file with ligands
    sdf_file = resource_filename("perses", os.path.join("data", "Tyk2_ligands_example", "Tyk2_ligands_shifted.sdf"))
    # Load old ligand from sdf -- first molecule in sdf file
    old_ligand = createOEMolFromSDF(sdf_file, index=0)
    # Load new ligand from sdf -- second molecule in sdf file
    old_ligand = createOEMolFromSDF(sdf_file, index=0)
    # Feed receptor and ligands to RelativeFEPSetup
    fe_setup = RelativeFEPSetup(omm_top, old_ligand, new_ligand, receptor_positions=omm_pos)
    return NotImplementedError
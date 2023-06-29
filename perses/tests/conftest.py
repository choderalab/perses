import pytest
import yaml


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu_ci: mark test as useful to run on GPU")
    config.addinivalue_line("markers", "gpu_needed: mark test as GPU required")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")

    if not config.getoption("--runslow"):
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

@pytest.fixture
def in_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield

@pytest.fixture
def input_yaml_template():
    input_yaml = """
                atom_selection: null
                checkpoint_interval: 5
                fe_type: repex
                forcefield_files:
                - amber/ff14SB.xml
                - amber/tip3p_standard.xml
                - amber/tip3p_HFE_multivalent.xml
                - amber/phosaa10.xml
                n_cycles: 10
                n_equilibration_iterations: 10
                n_states: 3
                n_steps_per_move_application: 50
                new_ligand_index: 15
                old_ligand_index: 14
                phases:
                - vacuum
                pressure: 1.0
                save_setup_pickle_as: fesetup_hbonds.pkl
                small_molecule_forcefield: openff-2.0.0
                solvent_padding: 9.0
                temperature: 300.0
                timestep: 4.0
                trajectory_directory: cdk2_repex_hbonds
                trajectory_prefix: cdk2
                """
    return input_yaml

@pytest.fixture
def input_template_obj_default_selection(input_yaml_template):
    input_obj = yaml.safe_load(input_yaml_template)
    return input_obj


@pytest.fixture
def input_template_not_water_selection(input_template_obj_default_selection):
    input_template_obj_default_selection["atom_selection"] = "not water"
    return input_template_obj_default_selection
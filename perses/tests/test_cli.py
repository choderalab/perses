import os

import pytest
import yaml
from click.testing import CliRunner
from pkg_resources import resource_filename

from perses.app.cli import cli


@pytest.fixture
def default_input_yaml_template_cli():
    test_yaml = """
    protein_pdb: Tyk2_protein.pdb
    ligand_file: Tyk2_ligands_shifted.sdf
    old_ligand_index: 0
    new_ligand_index: 3
    forcefield_files:
      - amber/ff14SB.xml
      - amber/tip3p_standard.xml
      - amber/tip3p_HFE_multivalent.xml
      - amber/phosaa10.xml
    small_molecule_forcefield: openff-2.0.0
    pressure: 1
    temperature: 300
    solvent_padding: 9
    atom_expression:
      - IntType
    bond_expession:
      - DefaultBonds
    n_steps_per_move_application: 1
    fe_type: repex
    checkpoint_interval: 10
    n_cycles: 1
    n_states: 3
    n_equilibration_iterations: 0
    trajectory_directory: temp/offlig10to24
    trajectory_prefix: out
    atom_selection: not water
    phases:
      - solvent
      - vacuum
    timestep: 4
    h_constraints: true
    """
    return test_yaml

@pytest.fixture
def input_template_obj_cli(default_input_yaml_template_cli):
    input_obj = yaml.safe_load(default_input_yaml_template_cli)
    return input_obj

@pytest.fixture
def input_template_obj_tip5p_cli(input_template_obj_cli):
    """Input yaml options changing the solvent model to spce."""
    input_template_obj_cli["solvent_model"] = "spce"
    return input_template_obj_cli


@pytest.mark.parametrize("input_params", ["input_template_obj_cli",
                                          "input_template_obj_tip5p_cli"])
def test_dummy_cli_with_override(input_params, in_tmpdir, request):
    runner = CliRunner()
    yaml_doc = request.getfixturevalue(input_params)
    with runner.isolated_filesystem():
        with open("test.yaml", "w") as f:
            yaml.dump(yaml_doc, f)

        protein_pdb = resource_filename(
            "perses", os.path.join("data", "Tyk2_ligands_example", "Tyk2_protein.pdb")
        )
        ligand_file = resource_filename(
            "perses",
            os.path.join("data", "Tyk2_ligands_example", "Tyk2_ligands_shifted.sdf"),
        )
        result = runner.invoke(
            cli,
            [
                "--yaml",
                "test.yaml",
                "--override",
                f"protein_pdb:{protein_pdb}",
                "--override",
                f"ligand_file:{ligand_file}",
            ],
        )
        assert result.exit_code == 0


@pytest.mark.skipif(
    not os.environ.get("GITHUB_ACTIONS", None),
    reason="This test needs API keys from AWS to work",
)
def test_s3_yaml_read(in_tmpdir):
    runner = CliRunner()

    with runner.isolated_filesystem():
        protein_pdb = resource_filename(
            "perses", os.path.join("data", "Tyk2_ligands_example", "Tyk2_protein.pdb")
        )
        ligand_file = resource_filename(
            "perses",
            os.path.join("data", "Tyk2_ligands_example", "Tyk2_ligands_shifted.sdf"),
        )
        result = runner.invoke(
            cli,
            [
                "--yaml",
                "s3://perses-testing/s3_test.yaml",
                "--override",
                f"protein_pdb:{protein_pdb}",
                "--override",
                f"ligand_file:{ligand_file}",
            ],
        )
        assert result.exit_code == 0

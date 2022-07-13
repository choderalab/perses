import os
import subprocess

from click.testing import CliRunner
from pkg_resources import resource_filename

from perses.app.cli import cli

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
checkpoint_interval: 50
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


def test_dummy_cli_with_override(in_tmpdir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.yaml", "w") as f:
            f.write(test_yaml)

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
                "test.yaml",
                "--override",
                f"protein_pdb:{protein_pdb}",
                "--override",
                f"ligand_file:{ligand_file}",
            ],
        )
        assert result.exit_code == 0

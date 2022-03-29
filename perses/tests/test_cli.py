from click.testing import CliRunner
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
  - complex
  - solvent
  - vacuum
timestep: 4
h_constraints: true
"""


def test_dummy_cli(in_tmpdir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.yaml", "w") as f:
            f.write(test_yaml)

        result = runner.invoke(cli, ["--yaml", "test.yaml"])
        print(result)
        print(result.output)
        assert result.exit_code == 0

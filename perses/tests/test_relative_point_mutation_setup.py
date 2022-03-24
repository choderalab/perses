def test_PointMutationExecutor():
    from pkg_resources import resource_filename
    from simtk import unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor

    pdb_filename = resource_filename("perses", "data/ala_vacuum.pdb")
    PointMutationExecutor(
        pdb_filename,
        "1",
        "2",
        "ASP",
        ionic_strength=0.15 * unit.molar,
        flatten_torsions=True,
        flatten_exceptions=True,
        conduct_endstate_validation=False,
    )


def test_PointMutationExecutor_endstate_validation():
    from pkg_resources import resource_filename
    from simtk import unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor

    pdb_filename = resource_filename("perses", "data/ala_vacuum.pdb")
    PointMutationExecutor(
        pdb_filename,
        "1",
        "2",
        "ASP",
        ionic_strength=0.15 * unit.molar,
        flatten_torsions=False,
        flatten_exceptions=False,
        conduct_endstate_validation=True,
    )

def test_PointMutationExecutor_solvated():
    from pkg_resources import resource_filename
    import os
    import tempfile
    from openmm import app, unit

    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from perses.tests.test_topology_proposal import generate_atp

    with tempfile.TemporaryDirectory() as temp_dir:
        ala, system_generator = generate_atp(phase='solvent')
        app.PDBFile.writeFile(ala.topology, ala.positions, open(os.path.join(temp_dir, "ala_solvated.pdb"), "w"), keepIds=True)

        PointMutationExecutor(
            os.path.join(temp_dir, "ala_solvated.pdb"),
            "1",
            "2",
            "ASP",
            solvate=False,
            flatten_torsions=False,
            flatten_exceptions=False,
            conduct_endstate_validation=False
        )



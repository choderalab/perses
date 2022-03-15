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

def test_PointMutationExecutor():
    from pkg_resources import resource_filename
    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from simtk import unit
    pdb_filename = resource_filename('perses', 'ala_vacuum.pdb')
    solvent_delivery = PointMutationExecutor("ala_vacuum.pdb",
                            '1',
                            '2',
                            'ASP',
                            ionic_strength=0.15*unit.molar,
                            flatten_torsions=True,
                            flatten_exceptions=True,
                            conduct_endstate_validation=False
                           )

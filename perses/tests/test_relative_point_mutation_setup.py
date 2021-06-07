def test_PointMutationExecutor():
    from perses.app.relative_point_mutation_setup import PointMutationExecutor
    from simtk import unit
    solvent_delivery = PointMutationExecutor("ala_vacuum.pdb",
                            '1',
                            '2',
                            'ASP',
                            ionic_strength=0.15*unit.molar,
                            flatten_torsions=True,
                            flatten_exceptions=True,
                            conduct_endstate_validation=False
                           )

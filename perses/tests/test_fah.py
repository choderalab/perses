"""
Tests for folding at home generator suite in perses

TODO:
* Write tests

"""


def test_pipeline_small_molecule():
        from pkg_resources import resource_filename
        from perses.app.fah_generator import run_neq_fah_setup
        ligand_file = resource_filename('perses', 'data/bace-example/Bace_ligands_shifted.sdf')
        protein_file = resource_filename('perses', 'data/bace-example/Bace_protein.pdb')

        forcefield_files = ['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml']

        run_neq_fah_setup(ligand_file, 0, 1, forcefield_files,'RUN0',protein_pdb=protein_file,phases=['complex','solvent'],phase_project_ids={'complex':10000,'solvent':10001})

        ## now test that it breaks if we don't match the phase_project_ids properly

def test_pipeline_small_molecule_solvent():
        from pkg_resources import resource_filename
        from perses.app.fah_generator import run_neq_fah_setup
        ligand_file = resource_filename('perses', 'data/bace-example/Bace_ligands_shifted.sdf')

        forcefield_files = ['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml']

        run_neq_fah_setup(ligand_file, 0, 1, forcefield_files,'RUN0',phases=['solvent','vacuum'],phase_project_ids={'solvent':10000,'vacuum':10001})


def test_pipeline_protein():
    # write a test that calls run() for protein mutations
    return


def test_core_file():
    """ Checks that a core.xml file is written
    """
    import tempfile
    from perses.app.fah_generator import make_core_file
    import os
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')

    make_core_file(100, 1, 1, directory=tmpdir)
    assert os.path.exists(f'{tmpdir}/core.xml')


def test_neq_integrator():
    return

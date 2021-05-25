"""
Tests for folding at home generator suite in perses

TODO:
* Write tests

"""

import os
from unittest import skipIf
running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

#default arguments for tests
DEFAULT_PLATFORM_NAME='CPU'
DEFAULT_PROJ_ID = {'complex': 1000, 'solvent': 1001, 'vacuum': 1002}

#num_equilibration_iterations=1000,
#num_equilibration_steps_per_iteration=250

DEFAULT_EQ_ITERATIONS = 2
DEFAULT_EQ_STEPS_PER_ITERATION = 25
DEFAULT_N_STEPS_EQ = 250
DEFAULT_N_STEPS_NEQ = 250


forcefield_files = ['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml']


@skipIf(running_on_github_actions, "Skip slow test on GH Actions")
def test_pipeline_small_molecule():
    from pkg_resources import resource_filename
    from perses.app.fah_generator import run_neq_fah_setup
    ligand_file = resource_filename('perses', 'data/bace-example/Bace_ligands_shifted.sdf')
    protein_file = resource_filename('perses', 'data/bace-example/Bace_protein.pdb')

    forcefield_files = ['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml']

    projs = {'complex':DEFAULT_PROJ_ID['complex'],'solvent':DEFAULT_PROJ_ID['solvent']}

    run_neq_fah_setup(ligand_file, 0, 1,
                      forcefield_files,
                      'RUN0',
                      protein_pdb=protein_file,
                      phases=['complex','solvent'],
                      phase_project_ids=projs,
                      platform_name = DEFAULT_PLATFORM_NAME,
                      num_equilibration_iterations = DEFAULT_EQ_ITERATIONS,
                      num_equilibration_steps_per_iteration = DEFAULT_EQ_STEPS_PER_ITERATION)

    ## now test that it breaks if we don't match the phase_project_ids properly

    #remove the folders
    for val in projs.values():
        os.system(f"rm -r {val}")

#@skipIf(running_on_github_actions, "Skipping for now as it's currently failing to generate state file -- needs debugging")
def test_pipeline_small_molecule_solvent():
    from pkg_resources import resource_filename
    from perses.app.fah_generator import run_neq_fah_setup
    ligand_file = resource_filename('perses', 'data/cdk2-example/CDK2_ligands.sdf')

    projs = {'vacuum':DEFAULT_PROJ_ID['vacuum'],'solvent':DEFAULT_PROJ_ID['solvent']}

    run_neq_fah_setup(ligand_file, 0, 1,
                      forcefield_files,
                      'RUN0',
                      phases=['solvent','vacuum'],
                      phase_project_ids=projs,
                      platform_name = DEFAULT_PLATFORM_NAME,
                      num_equilibration_iterations = DEFAULT_EQ_ITERATIONS,
                      num_equilibration_steps_per_iteration = DEFAULT_EQ_STEPS_PER_ITERATION)
    #remove the folders
    for val in projs.values():
        os.system(f"rm -r {val}")

@skipIf(running_on_github_actions, "Skip slow test on GH Actions")
def test_pipeline_protein():
    from pkg_resources import resource_filename
    from perses.app.fah_generator import run_neq_fah_setup
    yaml_filename = resource_filename('perses', 'data/barstar-mutation/mutant.yaml')

    import yaml
    yaml_file = open(yaml_filename, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()

    ligand_file = resource_filename('perses', 'data/barstar-mutation/mmc2_barnase.pdb')
    protein_file = resource_filename('perses', 'data/barstar-mutation/mmc2_barstar.pdb')
    # need to replace ligand and protein location in file

    setup_options['phase_project_ids'] = {'complex':'temp-complex','apo':'temp-apo'}
    setup_options['protein_kwargs']['ligand_file'] = ligand_file
    setup_options['protein_kwargs']['protein_filename'] = protein_file
    setup_options['phases'] = ['complex']
    #TODO: add platform name option to test

    run_neq_fah_setup(**setup_options)


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
    # TODO write this; not strictly necessary since it is already tested in openmmtools
    return

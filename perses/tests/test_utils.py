"""
Test util functions

"""

__author__ = 'John D. Chodera'

import os
istravis = os.environ.get('TRAVIS', None) == 'true'


# functions testing perses.utils.data
def test_get_data_filename(datafile='data/gaff2.xml'):
    """
    Checks that function returns real path

    Parameters
    ----------
    datafile : str, default 'data/gaff2.xml'

    """
    from perses.utils.data import get_data_filename
    import os

    path = get_data_filename(datafile)

    assert os.path.exists(path), "Either path to datafile is broken, or datafile does not exist"




# functions testing perses.utils.openeye
def test_createOEMolFromSMILES(smiles='CC', title='MOL'):
    """

    :param smiles:
    :param title:
    :return:
    """
    from perses.utils.openeye import createOEMolFromSMILES
    molecule = createOEMolFromSMILES(smiles,title)

    # checking that it has returned an OEMol with a non-zero number of atoms
    assert (molecule.NumAtoms() > 0), "createOEMolFromSMILES has returned an empty molecule"

    # checking that the OEMol has been correctly titled
    assert (molecule.GetTitle() == title), "createOEMolFromSMILES has not titled OEMol object correctly"
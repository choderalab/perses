"""
Tests for folding at home generator suite in perses

TODO:
* Write tests

"""


def test_core_file():
    """ Checks that a core.xml file is written
    """
    import tempfile
    from perses.app.fah_generator import make_core_file
    import os
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')

    make_core_file(100, 1, 1, directory=tmpdir)
    assert os.path.exists(f'{tmpdir}/core.xml')

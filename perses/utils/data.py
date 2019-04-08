"""

Tools for managing datafiles in perses

"""

__author__ = 'John D. Chodera'


def get_data_filename(relative_path):
    """get the full path to one of the reference files shipped for testing

    in the source distribution, these files are in ``perses/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    relative_path : str
        name of the file to load (with respect to the openmoltools folder).

    Returns
    -------
    Absolute path to file

    """
    from pkg_resources import resource_filename
    import os

    fn = resource_filename('perses', relative_path)

    if not os.path.exists(fn):
        raise valueerror("sorry! %s does not exist. if you just added it, you'll have to re-install" % fn)

    return fn


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

def generate_gaff_xml():
    """
    Return a file-like object for `gaff.xml`
    """
    #TODO this function isn't used anywhere
    from openmoltools import amber
    gaff_dat_filename = amber.find_gaff_dat()

    # Generate ffxml file contents for parmchk-generated frcmod output.
    leaprc = StringIO("parm = loadamberparams %s" % gaff_dat_filename)
    import parmed
    params = parmed.amber.AmberParameterSet.from_leaprc(leaprc)
    params = parmed.openmm.OpenMMParameterSet.from_parameterset(params)
    citations = """\
    Wang, J., Wang, W., Kollman P. A.; Case, D. A. "Automatic atom type and bond type perception in molecular mechanical calculations". Journal of Molecular Graphics and Modelling , 25, 2006, 247260.
    Wang, J., Wolf, R. M.; Caldwell, J. W.;Kollman, P. A.; Case, D. A. "Development and testing of a general AMBER force field". Journal of Computational Chemistry, 25, 2004, 1157-1174.
    """
    ffxml = str()
    gaff_xml = StringIO(ffxml)
    provenance=dict(OriginalFile='gaff.dat', Reference=citations)
    params.write(gaff_xml, provenance=provenance)

    return gaff_xml

def forcefield_directory():
    """
    Return the forcefield directory for the additional forcefield files like gaff.xml

    Returns
    -------
    forcefield_directory_name : str
        Directory where OpenMM can find additional forcefield files

    """
    #TODO this function isn't used anywhere
    forcefield_directory_name = resource_filename("perses", "data")
    return forcefield_directory_name

def load_smi(smi_file,index=None):
    """
    loads list of smiles from a text file. Will return the i-th smiles in file if index is provided, where index
    starts at zero.

    Parameters
    ----------
    smi_file : str
        file name containing strings
    index : None or int, default None
        index of smiles to return. If not provided, list of smiles is returned

    Returns
    --------
    smiles : string, or list of strings
        depending on number of smiles in file, or if index is provided
    """
    with open(smi_file) as f:
        smiless = f.read().splitlines()
    if index is None:
        return smiless
    else:
        smiles = smiless[index]
        return smiles
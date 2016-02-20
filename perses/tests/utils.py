"""
Unit tests for NCMC switching engine.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from functools import partial
from pkg_resources import resource_filename
from openeye import oechem
if sys.version_info >= (3, 0):
    from io import StringIO
    from subprocess import getstatusoutput
else:
    from cStringIO import StringIO
    from commands import getstatusoutput

################################################################################
# CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

################################################################################
# UTILITIES
################################################################################]

# TODO: Move some of these utility routines to openmoltools.

def show_topology(topology):
    output = ""
    for atom in topology.atoms():
        output += "%8d %5s %5s %3s: bonds " % (atom.index, atom.name, atom.residue.id, atom.residue.name)
        for bond in atom.residue.bonds():
            if bond[0] == atom:
                output += " %8d" % bond[1].index
            if bond[1] == atom:
                output += " %8d" % bond[0].index
        output += '\n'
    print(output)

def extractPositionsFromOEMOL(molecule):
    positions = unit.Quantity(np.zeros([molecule.NumAtoms(), 3], np.float32), unit.angstroms)
    coords = molecule.GetCoords()
    for index in range(molecule.NumAtoms()):
        positions[index,:] = unit.Quantity(coords[index], unit.angstroms)
    return positions

def createOEMolFromIUPAC(iupac_name='bosutinib'):
    from openeye import oechem, oeiupac, oeomega

    # Create molecule.
    mol = oechem.OEMol()
    oeiupac.OEParseIUPACName(mol, iupac_name)
    mol.SetTitle(iupac_name)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(mol, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(mol)

    # Create atom names.
    oechem.OETriposAtomNames(mol)

    # Assign geometry
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)
    omega(mol)

    return mol

def createOEMolFromSMILES(smiles='CC', title='MOL'):
    """
    Generate an oemol with a geometry
    """
    from openeye import oechem, oeiupac, oeomega

    # Create molecule
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)

    # Set title.
    mol.SetTitle(title)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(mol, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(mol)

    # Create atom names.
    oechem.OETriposAtomNames(mol)

    # Assign geometry
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)
    omega(mol)

    return mol

def oemol_to_omm_ff(oemol, molecule_name):
    from perses.rjmc import topology_proposal
    from openmoltools import forcefield_generators
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])
    topology = forcefield_generators.generateTopologyFromOEMol(oemol)
    system = system_generator.build_system(topology)
    positions = extractPositionsFromOEMOL(oemol)
    return system, positions, topology



def generate_gaff_xml():
    """
    Return a file-like object for `gaff.xml`
    """
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

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    In the source distribution, these files are in ``perses/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the openmoltools folder).

    """

    fn = resource_filename('perses', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

def createSystemFromIUPAC(iupac_name):
    """
    Create an openmm system out of an oemol

    Parameters
    ----------
    iupac_name : str
        IUPAC name

    Returns
    -------
    molecule : openeye.OEMol
        OEMol molecule
    system : openmm.System object
        OpenMM system
    positions : [n,3] np.array of floats
        Positions
    topology : openmm.app.Topology object
        Topology
    """

    # Create OEMol
    molecule = createOEMolFromIUPAC(iupac_name)

    # Generate a topology.
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    topology = generateTopologyFromOEMol(molecule)

    # Initialize a forcefield with GAFF.
    # TODO: Fix path for `gaff.xml` since it is not yet distributed with OpenMM
    from simtk.openmm.app import ForceField
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    forcefield = ForceField(gaff_xml_filename)

    # Generate template and parameters.
    from openmoltools.forcefield_generators import generateResidueTemplate
    [template, ffxml] = generateResidueTemplate(molecule)

    # Register the template.
    forcefield.registerResidueTemplate(template)

    # Add the parameters.
    forcefield.loadFile(StringIO(ffxml))

    # Create the system.
    system = forcefield.createSystem(topology)

    # Extract positions
    positions = extractPositionsFromOEMOL(molecule)

    return (molecule, system, positions, topology)

def get_atoms_with_undefined_stereocenters(molecule, verbose=False):
    """
    Return list of atoms with undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to check.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    TODO
    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Returns
    -------
    atoms : list of openeye.oechem.OEAtom
        List of atoms with undefined stereochemistry.

    """
    from openeye.oechem import OEAtomStereo_Undefined, OEAtomStereo_Tetrahedral
    undefined_stereocenters = list()
    for atom in molecule.GetAtoms():
        chiral = atom.IsChiral()
        stereo = OEAtomStereo_Undefined
        if atom.HasStereoSpecified(OEAtomStereo_Tetrahedral):
            v = list()
            for nbr in atom.GetAtoms():
                v.append(nbr)
            stereo = atom.GetStereo(v, OEAtomStereo_Tetrahedral)

        if chiral and (stereo == OEAtomStereo_Undefined):
            undefined_stereocenters.append(atom)
            if verbose:
                print("Atom %d (%s) of molecule '%s' has undefined stereochemistry (chiral=%s, stereo=%s)." % (atom.GetIdx(), atom.GetName(), molecule.GetTitle(), str(chiral), str(stereo)))

    return undefined_stereocenters

def has_undefined_stereocenters(molecule, verbose=False):
    """
    Check if given molecule has undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to check.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    TODO
    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Returns
    -------
    result : bool
        True if molecule has undefined stereocenters.

    Examples
    --------
    Enumerate undefined stereocenters
    >>> smiles = "[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]"
    >>> from openeye.oechem import OEGraphMol, OESmilesToMol
    >>> molecule = OEGraphMol()
    >>> OESmilesToMol(molecule, smiles)
    >>> print has_undefined_stereocenters(smiles)
    True

    """
    atoms = get_atoms_with_undefined_stereocenters(molecule, verbose=verbose)
    if len(atoms) > 0:
        return True

    return False

def enumerate_undefined_stereocenters(molecule, verbose=False):
    """
    Check if given molecule has undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule whose stereocenters are to be expanded.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    Returns
    -------
    molecules : list of OEMol
        The molecules with fully defined stereocenters.

    TODO
    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Examples
    --------
    Enumerate undefined stereocenters
    >>> smiles = "[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]"
    >>> from openeye.oechem import OEGraphMol, OESmilesToMol
    >>> molecule = OEGraphMol()
    >>> OESmilesToMol(molecule, smiles)
    >>> molecules = enumerate_undefined_stereocenters(smiles)
    >>> len(molecules)
    2

    """
    from openeye.oechem import OEAtomStereo_RightHanded, OEAtomStereo_LeftHanded, OEAtomStereo_Tetrahedral
    from itertools import product

    molecules = list()
    atoms = get_atoms_with_undefined_stereocenters(molecule, verbose=verbose)
    for stereocenters in product([OEAtomStereo_RightHanded, OEAtomStereo_LeftHanded], repeat=len(atoms)):
        for (index,atom) in enumerate(atoms):
            neighbors = list()
            for neighbor in atom.GetAtoms():
                neighbors.append(neighbor)
            atom.SetStereo(neighbors, OEAtomStereo_Tetrahedral, stereocenters[index])
        molecules.append(molecule.CreateCopy())

    return molecules

def sanitizeSMILES(smiles_list, mode='drop', verbose=False):
    """
    Sanitize set of SMILES strings by ensuring all are canonical isomeric SMILES.
    Duplicates are also removed.

    Parameters
    ----------
    smiles_list : iterable of str
        The set of SMILES strings to sanitize.
    mode : str, optional, default='drop'
        When a SMILES string that does not correspond to canonical isomeric SMILES is found, select the action to be performed.
        'exception' : raise an `Exception`
        'drop' : drop the SMILES string
        'expand' : expand all stereocenters into multiple molecules
    verbose : bool, optional, default=False
        If True, print verbose output.

    Returns
    -------
    sanitized_smiles_list : list of str
         Sanitized list of canonical isomeric SMILES strings.

    Examples
    --------

    Sanitize a simple list.
    >>> smiles_list = ['CC', 'CCC', '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]']

    Throw an exception if undefined stereochemistry is present.
    >>> sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='exception')
    Traceback (most recent call last):
      ...
    Exception: Molecule '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]' has undefined stereocenters

    Drop molecules iwth undefined stereochemistry.
    >>> sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='drop')
    >>> len(sanitized_smiles_list)
    2

    Expand molecules iwth undefined stereochemistry.
    >>> sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='expand')
    >>> len(sanitized_smiles_list)
    4

    """
    from openeye.oechem import OEGraphMol, OESmilesToMol, OECreateIsoSmiString
    sanitized_smiles_set = set()
    for smiles in smiles_list:
        molecule = OEGraphMol()
        OESmilesToMol(molecule, smiles)

        if verbose:
            molecule.SetTitle(smiles)
            oechem.OETriposAtomNames(molecule)

        if has_undefined_stereocenters(molecule, verbose=verbose):
            if mode == 'drop':
                continue
            elif mode == 'exception':
                raise Exception("Molecule '%s' has undefined stereocenters" % smiles)
            elif mode == 'expand':
                molecules = enumerate_undefined_stereocenters(molecule, verbose=verbose)
                for molecule in molecules:
                    isosmiles = OECreateIsoSmiString(molecule)
                    sanitized_smiles_set.add(isosmiles)
        else:
            # Convert to OpenEye's canonical isomeric SMILES.
            isosmiles = OECreateIsoSmiString(molecule)
            sanitized_smiles_set.add(isosmiles)

    sanitized_smiles_list = list(sanitized_smiles_set)
    return sanitized_smiles_list

def test_sanitizeSMILES():
    """
    Test SMILES sanitization.
    """
    smiles_list = ['CC', 'CCC', '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]']

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='drop')
    if len(sanitized_smiles_list) != 2:
        raise Exception("Molecules with undefined stereochemistry are not being properly dropped (size=%d)." % len(sanitized_smiles_list))

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='expand')
    if len(sanitized_smiles_list) != 4:
        raise Exception("Molecules with undefined stereochemistry are not being properly expanded (size=%d)." % len(sanitized_smiles_list))

    # Check that all molecules can be round-tripped.
    from openeye.oechem import OEGraphMol, OESmilesToMol, OECreateIsoSmiString
    for smiles in sanitized_smiles_list:
        molecule = OEGraphMol()
        OESmilesToMol(molecule, smiles)
        isosmiles = OECreateIsoSmiString(molecule)
        if (smiles != isosmiles):
            raise Exception("Molecule '%s' was not properly round-tripped (result was '%s')" % (smiles, isosmiles))


def check_system(system):
    """
    Check OpenMM System object for pathologies, like duplicate atoms in torsions.

    Parameters
    ----------
    system : simtk.openmm.System

    """
    forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
    force = forces['PeriodicTorsionForce']
    for index in range(force.getNumTorsions()):
        [i, j, k, l, periodicity, phase, barrier] = force.getTorsionParameters(index)
        if len(set([i,j,k,l])) < 4:
            # TODO: Serialize system.xml on exceptions.
            msg  = 'Torsion index %d of self._topology_proposal.new_system has duplicate atoms: %d %d %d %d\n' % (index,i,j,k,l)
            msg += 'Serialzed system to system.xml for inspection.\n'
            from simtk.openmm import XmlSerializer
            serialized_system = XmlSerializer.serialize(system)
            outfile = open('system.xml', 'w')
            outfile.write(serialized_system)
            outfile.close()
            raise Exception(msg)

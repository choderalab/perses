"""
Unit tests for NCMC switching engine.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################
import copy
from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from functools import partial
from pkg_resources import resource_filename
from perses.rjmc import geometry
from perses.rjmc.topology_proposal import SystemGenerator, TopologyProposal, SmallMoleculeSetProposalEngine
from openeye import oechem
if sys.version_info >= (3, 0):
    from io import StringIO
    from subprocess import getstatusoutput
else:
    from cStringIO import StringIO
    from commands import getstatusoutput
from openmmtools.constants import kB
from openmmtools import alchemy, states

################################################################################
# CONSTANTS
################################################################################

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

################################################################################
# UTILITIES
################################################################################]

# TODO: Move some of these utility routines to openmoltools.

class NaNException(Exception):
    pass

def quantity_is_finite(quantity):
    """
    Check that elements in quantity are all finite.

    Parameters
    ----------
    quantity : simtk.unit.Quantity
        The quantity to check

    Returns
    -------
    is_finite : bool
        If quantity is finite, returns True; otherwise False.

    """
    if np.any( np.isnan( np.array(quantity / quantity.unit) ) ):
        return False
    return True

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

def giveOpenmmPositionsToOEMOL(positions, molecule):
    assert molecule.NumAtoms() == len(positions)
    coords = molecule.GetCoords()
    for key in coords.keys(): # openmm in nm, openeye in A
        coords[key] = (positions[key][0]/unit.angstrom,positions[key][1]/unit.angstrom,positions[key][2]/unit.angstrom)
    molecule.SetCoords(coords)

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

    # Create bond types
    oechem.OETriposBondTypeNames(mol)

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

def compare_at_lambdas(context, functions):
    """
    Compare the energy components at all lambdas = 1 and 0.
    """

    #first, set all lambdas to 0
    for parm in functions.keys():
        context.setParameter(parm, 0.0)

    energy_components_0 = compute_potential_components(context)

    for parm in functions.keys():
        context.setParameter(parm, 1.0)

    energy_components_1 = compute_potential_components(context)

    print("-----------------------")
    print("Energy components at lambda=0")

    for i in range(len(energy_components_0)):
        name, value = energy_components_0[i]
        print("%s\t%s" % (name, str(value)))

    print("-----------------------")
    print("Energy components at lambda=1")

    for i in range(len(energy_components_1)):
        name, value = energy_components_1[i]
        print("%s\t%s" % (name, str(value)))

    print("------------------------")

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

def forcefield_directory():
    """
    Return the forcefield directory for the additional forcefield files like gaff.xml

    Returns
    -------
    forcefield_directory_name : str
        Directory where OpenMM can find additional forcefield files
    """
    forcefield_directory_name = resource_filename("perses", "data")
    return forcefield_directory_name

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
    system = forcefield.createSystem(topology, removeCMMotion=False)

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

def smiles_to_oemol(smiles_string, title="MOL"):
    """
    Convert the SMILES string into an OEMol

    Returns
    -------
    oemols : np.array of type object
    array of oemols
    """
    from openeye import oechem, oeomega
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles_string)
    mol.SetTitle(title)
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega(mol)
    return mol

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
    OESMILES_OPTIONS = oechem.OESMILESFlag_ISOMERIC | oechem.OESMILESFlag_Hydrogens  ## IVY
    for smiles in smiles_list:
        molecule = OEGraphMol()
        OESmilesToMol(molecule, smiles)

        oechem.OEAddExplicitHydrogens(molecule)

        if verbose:
            molecule.SetTitle(smiles)
            oechem.OETriposAtomNames(molecule)

        if has_undefined_stereocenters(molecule, verbose=verbose):
            if mode == 'drop':
                if verbose:
                    print("Dropping '%s' due to undefined stereocenters." % smiles)
                continue
            elif mode == 'exception':
                raise Exception("Molecule '%s' has undefined stereocenters" % smiles)
            elif mode == 'expand':
                if verbose:
                    print('Expanding stereochemistry:')
                    print('original: %s', smiles)
                molecules = enumerate_undefined_stereocenters(molecule, verbose=verbose)
                for molecule in molecules:
                    # isosmiles = OECreateIsoSmiString(molecule) ## IVY
                    # sanitized_smiles_set.add(isosmiles) ## IVY

                    smiles_string = oechem.OECreateSmiString(molecule, OESMILES_OPTIONS)  ## IVY
                    # smiles_string = oechem.OEMolToSmiles(molecule)
                    sanitized_smiles_set.add(smiles_string)  ## IVY
                    if verbose: print('expanded: %s', smiles_string)
        else:
            # Convert to OpenEye's canonical isomeric SMILES.
            # isosmiles = OECreateIsoSmiString(molecule)
            smiles_string = oechem.OECreateSmiString(molecule, OESMILES_OPTIONS) ## IVY
            # smiles_string = oechem.OEMolToSmiles(molecule)
            sanitized_smiles_set.add(smiles_string) ## IVY

    sanitized_smiles_list = list(sanitized_smiles_set)
    return sanitized_smiles_list


def render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map, width=1200, height=1200):
    """
    Render the atom mapping to a PDF file.

    Parameters
    ----------
    filename : str
        The PDF filename to write to.
    molecule1 : openeye.oechem.OEMol
        Initial molecule
    molecule2 : openeye.oechem.OEMol
        Final molecule
    new_to_old_atom_map : dict of int
        new_to_old_atom_map[molecule2_atom_index] is the corresponding molecule1 atom index
    width : int, optional, default=1200
        Width in pixels
    height : int, optional, default=1200
        Height in pixels

    """
    from openeye import oechem

    # Make copies of the input molecules
    molecule1, molecule2 = oechem.OEGraphMol(molecule1), oechem.OEGraphMol(molecule2)

    oechem.OEGenerate2DCoordinates(molecule1)
    oechem.OEGenerate2DCoordinates(molecule2)

    old_atoms_1 = [atom for atom in molecule1.GetAtoms()]
    old_atoms_2 = [atom for atom in molecule2.GetAtoms()]

    # Add both to an OEGraphMol reaction
    rmol = oechem.OEGraphMol()
    rmol.SetRxn(True)
    def add_molecule(mol):
        # Add atoms
        new_atoms = list()
        old_to_new_atoms = dict()
        for old_atom in mol.GetAtoms():
            new_atom = rmol.NewAtom(old_atom.GetAtomicNum())
            new_atoms.append(new_atom)
            old_to_new_atoms[old_atom] = new_atom
        # Add bonds
        for old_bond in mol.GetBonds():
            rmol.NewBond(old_to_new_atoms[old_bond.GetBgn()], old_to_new_atoms[old_bond.GetEnd()], old_bond.GetOrder())
        return new_atoms, old_to_new_atoms

    [new_atoms_1, old_to_new_atoms_1] = add_molecule(molecule1)
    [new_atoms_2, old_to_new_atoms_2] = add_molecule(molecule2)

    # Label reactant and product
    for atom in new_atoms_1:
        atom.SetRxnRole(oechem.OERxnRole_Reactant)
    for atom in new_atoms_2:
        atom.SetRxnRole(oechem.OERxnRole_Product)

    # Label mapped atoms
    index =1
    for (index2, index1) in new_to_old_atom_map.items():
        new_atoms_1[index1].SetMapIdx(index)
        new_atoms_2[index2].SetMapIdx(index)
        index += 1
    # Set up image options
    from openeye import oedepict
    itf = oechem.OEInterface()
    oedepict.OEConfigureImageOptions(itf)
    ext = oechem.OEGetFileExtension(filename)
    if not oedepict.OEIsRegisteredImageFile(ext):
        raise Exception('Unknown image type for filename %s' % filename)
    ofs = oechem.oeofstream()
    if not ofs.open(filename):
        raise Exception('Cannot open output file %s' % filename)

    # Setup depiction options
    oedepict.OEConfigure2DMolDisplayOptions(itf, oedepict.OE2DMolDisplaySetup_AromaticStyle)
    opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
    oedepict.OESetup2DMolDisplayOptions(opts, itf)
    opts.SetBondWidthScaling(True)
    opts.SetAtomPropertyFunctor(oedepict.OEDisplayAtomMapIdx())
    opts.SetAtomColorStyle(oedepict.OEAtomColorStyle_WhiteMonochrome)

    # Depict reaction with component highlights
    oechem.OEGenerate2DCoordinates(rmol)
    rdisp = oedepict.OE2DMolDisplay(rmol, opts)

    colors = [c for c in oechem.OEGetLightColors()]
    highlightstyle = oedepict.OEHighlightStyle_BallAndStick
    #common_atoms_and_bonds = oechem.OEAtomBondSet(common_atoms)
    oedepict.OERenderMolecule(ofs, ext, rdisp)
    ofs.close()


def canonicalize_SMILES(smiles_list):
    """Ensure all SMILES strings end up in canonical form.
    Stereochemistry must already have been expanded.
    SMILES strings are converted to a OpenEye Topology and back again.
    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings
    Returns
    -------
    canonical_smiles_list : list of str
        List of SMILES strings, after canonicalization.
    """

    # Round-trip each molecule to a Topology to end up in canonical form
    from openmoltools.forcefield_generators import generateOEMolFromTopologyResidue, generateTopologyFromOEMol
    from openeye import oechem
    canonical_smiles_list = list()
    for smiles in smiles_list:
        molecule = smiles_to_oemol(smiles)
        topology = generateTopologyFromOEMol(molecule)
        residues = [ residue for residue in topology.residues() ]
        new_molecule = generateOEMolFromTopologyResidue(residues[0])
        new_smiles = oechem.OECreateIsoSmiString(new_molecule)
        canonical_smiles_list.append(new_smiles)
    return canonical_smiles_list

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

def describe_oemol(mol):
    """
    Render the contents of an OEMol to a string.

    Parameters
    ----------
    mol : OEMol
        Molecule to describe

    Returns
    -------
    description : str
        The description
    """
    description = ""
    description += "ATOMS:\n"
    for atom in mol.GetAtoms():
        description += "%8d %5s %5d\n" % (atom.GetIdx(), atom.GetName(), atom.GetAtomicNum())
    description += "BONDS:\n"
    for bond in mol.GetBonds():
        description += "%8d %8d\n" % (bond.GetBgnIdx(), bond.GetEndIdx())
    return description

def compute_potential(system, positions, platform=None):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    system : simtk.openmm.System
        The system object to check.
    positions : simtk.unit.Quantity of size (natoms,3) with units compatible with nanometers
        The positions to check.
    platform : simtk.openmm.Platform, optional, default=none
        If specified, this platform will be used.

    """
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    if platform is not None:
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    context.setPositions(positions)
    context.applyConstraints(integrator.getConstraintTolerance())
    potential = context.getState(getEnergy=True).getPotentialEnergy()
    del context, integrator
    if np.isnan(potential / unit.kilocalories_per_mole):
        raise NaNException("Potential energy is NaN")
    return potential

def compute_potential_components(context):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    context : simtk.openmm.Context
        The context from which to extract, System, parameters, and positions.

    """
    # Make a deep copy of the system.
    import copy
    system = context.getSystem()
    system = copy.deepcopy(system)
    # Get positions.
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Get Parameters
    parameters = context.getParameters()
    # Segregate forces.
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
    # Create new Context.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for (parameter, value) in parameters.items():
        context.setParameter(parameter, value)
    energy_components = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        forcename = force.__class__.__name__
        groups = 1<<index
        potential = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        energy_components.append((forcename, potential))
    del context, integrator
    return energy_components

def smiles_to_topology(smiles):
    """
    Convert a SMILES string to an OpenMM
    Topology

    Parameters
    ----------
    smiles : str
        smiles to be made into topology

    Returns
    -------
    topology : simtk.openmm.topology.app
        topology of the smiles
    mol : OEMol
        OEMol with explicit hydrogens
    """
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
    topology = generateTopologyFromOEMol(mol)
    return topology, mol

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
            msg  = 'Torsion index %d of self._topology_proposal.new_system has duplicate atoms: %d %d %d %d\n' % (index,i,j,k,l)
            msg += 'Serialzed system to system.xml for inspection.\n'
            from simtk.openmm import XmlSerializer
            serialized_system = XmlSerializer.serialize(system)
            outfile = open('system.xml', 'w')
            outfile.write(serialized_system)
            outfile.close()
            raise Exception(msg)

def generate_endpoint_thermodynamic_states(system: openmm.System, topology_proposal: TopologyProposal):
    """
    Generate endpoint thermodynamic states for the system

    Parameters
    ----------
    system : openmm.System
        System object corresponding to thermodynamic state
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        TopologyProposal representing transformation

    Returns
    -------
    nonalchemical_zero_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda zero endpoint
    nonalchemical_one_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda one endpoint
    lambda_zero_thermodynamic_state : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda zero
    lambda_one_thermodynamic_State : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda one
    """
    #create the thermodynamic state
    from perses.annihilation.lambda_protocol import RelativeAlchemicalState

    lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(system)
    lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

    #ensure their states are set appropriately
    lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
    lambda_one_alchemical_state.set_alchemical_parameters(1.0)

    check_system(system)

    #create the base thermodynamic state with the hybrid system
    thermodynamic_state = states.ThermodynamicState(system, temperature=temperature)

    #Create thermodynamic states for the nonalchemical endpoints
    nonalchemical_zero_thermodynamic_state = states.ThermodynamicState(topology_proposal.old_system, temperature=temperature)
    nonalchemical_one_thermodynamic_state = states.ThermodynamicState(topology_proposal.new_system, temperature=temperature)

    #Now create the compound states with different alchemical states
    lambda_zero_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_zero_alchemical_state])
    lambda_one_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_one_alchemical_state])

    return nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state


def generate_vacuum_topology_proposal(current_mol_name="benzene", proposed_mol_name="toluene", forcefield_kwargs=None, system_generator_kwargs=None):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet from two IUPAC molecule names.

    Constraints are added to the system by default. To override this, set ``forcefield_kwargs = None``.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule
    forcefield_kwargs : dict, optional, default=None
        Additional arguments to ForceField in addition to
        'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff
    system_generator_kwargs : dict, optional, default=None
        Dict passed onto SystemGenerator

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators
    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    gaff_filename = get_data_filename('data/gaff.xml')
    default_forcefield_kwargs = {'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff, 'constraints' : app.HBonds}
    forcefield_kwargs = default_forcefield_kwargs.update(forcefield_kwargs) if (forcefield_kwargs is not None) else default_forcefield_kwargs
    system_generator_kwargs = system_generator_kwargs if (system_generator_kwargs is not None) else dict()
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'],
        forcefield_kwargs=forcefield_kwargs,
        **system_generator_kwargs)

    old_oemol = createOEMolFromIUPAC(current_mol_name)
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    old_topology = generateTopologyFromOEMol(old_oemol)
    old_positions = extractPositionsFromOEMOL(old_oemol)
    old_smiles = oechem.OEMolToSmiles(old_oemol)
    old_system = system_generator.build_system(old_topology)

    new_oemol = createOEMolFromIUPAC(proposed_mol_name)
    new_smiles = oechem.OEMolToSmiles(new_oemol)

    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [old_smiles, new_smiles], system_generator, residue_name=current_mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(old_system, old_topology, current_mol=old_oemol, proposed_mol=new_oemol)

    # show atom mapping
    filename = str(current_mol_name)+str(proposed_mol_name)+'.pdf'
    render_atom_mapping(filename, old_oemol, new_oemol, topology_proposal.new_to_old_atom_map)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, old_positions, beta)

    # DEBUG: Zero out bonds and angles for one system
    #print('Zeroing bonds of old system')
    #for force in topology_proposal.old_system.getForces():
    #    if force.__class__.__name__ == 'HarmonicAngleForce':
    #        for index in range(force.getNumAngles()):
    #            p1, p2, p3, angle, K = force.getAngleParameters(index)
    #            K *= 0.0
    #            force.setAngleParameters(index, p1, p2, p3, angle, K)
    #    if False and force.__class__.__name__ == 'HarmonicBondForce':
    #        for index in range(force.getNumBonds()):
    #            p1, p2, r0, K = force.getBondParameters(index)
    #            K *= 0.0
    #            force.setBondParameters(index, p1, p2, r0, K)

    # DEBUG : Box vectors
    #box_vectors = np.eye(3) * 100 * unit.nanometers
    #topology_proposal.old_system.setDefaultPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])
    #topology_proposal.new_system.setDefaultPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])

    return topology_proposal, old_positions, new_positions

def generate_solvated_hybrid_test_topology(current_mol_name="naphthalene", proposed_mol_name="benzene"):
    """
    Generate a test solvated topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    current_mol, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(current_mol_name)
    proposed_mol = createOEMolFromIUPAC(proposed_mol_name)

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    modeller = app.Modeller(top_old, pos_old)
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()
    solvated_system = forcefield.createSystem(solvated_topology, nonbondedMethod=app.PME, removeCMMotion=False)
    barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, temperature, 50)

    solvated_system.addForce(barostat)

    gaff_filename = get_data_filename('data/gaff.xml')

    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], barostat=barostat, forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.PME})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=current_mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, solvated_topology)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, solvated_positions, beta)

    return topology_proposal, solvated_positions, new_positions

def generate_vacuum_hostguest_proposal(current_mol_name="B2", proposed_mol_name="MOL"):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators
    from openmmtools import testsystems

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    host_guest = testsystems.HostGuestVacuum()
    unsolv_old_system, pos_old, top_old = host_guest.system, host_guest.positions, host_guest.topology
    ligand_topology = [res for res in top_old.residues()]
    current_mol = forcefield_generators.generateOEMolFromTopologyResidue(ligand_topology[1]) # guest is second residue in topology
    proposed_mol = createOEMolFromSMILES('C1CC2(CCC1(CC2)C)C')

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=current_mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old, current_mol=current_mol, proposed_mol=proposed_mol)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, pos_old, beta)

    return topology_proposal, pos_old, new_positions

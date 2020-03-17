"""

Utility functions for handing small molecules

"""

__author__ = 'John D. Chodera'

import numpy as np

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
    from openeye import oechem
    from openeye.oechem import OEGraphMol, OESmilesToMol, OECreateIsoSmiString
    from perses.tests.utils import has_undefined_stereocenters, enumerate_undefined_stereocenters
    sanitized_smiles_set = set()
    OESMILES_OPTIONS = oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_ISOMERIC | oechem.OESMILESFlag_Hydrogens  ## IVY
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
                    smiles_string = oechem.OECreateSmiString(molecule, OESMILES_OPTIONS)  ## IVY
                    sanitized_smiles_set.add(smiles_string)  ## IVY
                    if verbose: print('expanded: %s', smiles_string)
        else:
            # Convert to OpenEye's canonical isomeric SMILES.
            smiles_string = oechem.OECreateSmiString(molecule, OESMILES_OPTIONS) ## IVY
            sanitized_smiles_set.add(smiles_string) ## IVY

    sanitized_smiles_list = list(sanitized_smiles_set)

    return sanitized_smiles_list

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
    from perses.utils.openeye import smiles_to_oemol
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

def show_topology(topology):
    """
    Outputs bond atoms and bonds in topology object

    Paramters
    ---------
    topology : Topology object
    """
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

def render_single_molecule(filename, molecule, width=1200, height=600):
    """
    simple function to create an oemol image

    Arguments
    ---------
    filename : str
        The PDF filename to write to.
    molecule : openeye.oechem.OEMol
        molecule
    width : int, optional, default=1200
        Width in pixels
    height : int, optional, default=1200
        Height in pixels
    """
    from openeye import oechem, oedepict
    oedepict.OEPrepareDepiction(molecule)
    oedepict.OERenderMolecule(filename, molecule)

def render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map, width=1200, height=600):
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
    from openeye import oechem, oedepict

    # Make copies of the input molecules
    # making a copy resets the atom indices, so the new_to_old_atom_map has to be remapped with the new, zero-indexed indices
    molecule1_indices = [atom.GetIdx() for atom in molecule1.GetAtoms()]
    molecule2_indices = [atom.GetIdx() for atom in molecule2.GetAtoms()]

    molecule1, molecule2 = oechem.OEGraphMol(molecule1), oechem.OEGraphMol(molecule2)

    molecule1_indices_new = [atom.GetIdx() for atom in molecule1.GetAtoms()]
    molecule2_indices_new = [atom.GetIdx() for atom in molecule2.GetAtoms()]

    modified_map_1 = {old: new for new, old in zip(molecule1_indices_new, molecule1_indices)}
    modified_map_2 = {old: new for new, old in zip(molecule2_indices_new, molecule2_indices)}
    new_to_old_atom_map = {modified_map_2[key]: modified_map_1[val] for key, val in new_to_old_atom_map.items()}


    oechem.OEGenerate2DCoordinates(molecule1)
    oechem.OEGenerate2DCoordinates(molecule2)

    # Add both to an OEGraphMol reaction
    rmol = oechem.OEGraphMol()
    rmol.SetRxn(True)
    def add_molecule(mol):
        # Add atoms
        new_atoms = list()
        old_to_new_atoms = dict()
        for old_atom in mol.GetAtoms():
            new_atom = rmol.NewAtom(old_atom.GetAtomicNum())
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
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

    core1 = oechem.OEAtomBondSet()
    core2 = oechem.OEAtomBondSet()
    # add all atoms to the set
    core1.AddAtoms(new_atoms_1)
    core2.AddAtoms(new_atoms_2)
    # Label mapped atoms
    core_change = oechem.OEAtomBondSet()
    index =1
    for (index2, index1) in new_to_old_atom_map.items():
        new_atoms_1[index1].SetMapIdx(index)
        new_atoms_2[index2].SetMapIdx(index)
        # now remove the atoms that are core, so only uniques are highlighted
        core1.RemoveAtom(new_atoms_1[index1])
        core2.RemoveAtom(new_atoms_2[index2])
        if new_atoms_1[index1].GetAtomicNum() != new_atoms_2[index2].GetAtomicNum():
            # this means the element type is changing
            core_change.AddAtom(new_atoms_1[index1])
            core_change.AddAtom(new_atoms_2[index2])
        index += 1
    # Set up image options
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

    if core1.NumAtoms() != 0:
        oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEPink),oedepict.OEHighlightStyle_Stick, core1)
    if core2.NumAtoms() != 0:
        oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEPurple),oedepict.OEHighlightStyle_Stick, core2)
    if core_change.NumAtoms() != 0:
        oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEGreen),oedepict.OEHighlightStyle_Stick, core_change)
    oedepict.OERenderMolecule(ofs, ext, rdisp)
    ofs.close()


def generate_ligands_figure(molecules,figsize=None,filename='ligands.png'):
    """ Plot an image with all of the ligands passed in

    Parameters
    ----------
    molecules : list
        list of openeye.oemol objects
    figsize : list or tuple
        list or tuple of len() == 2 of the horizontal and vertical lengths of image
    filename : string
        name of file to save the image

    Returns
    -------

    """
    from openeye import oechem,oedepict

    to_draw = []
    for lig in molecules:
        oedepict.OEPrepareDepiction(lig)
        to_draw.append(oechem.OEGraphMol(lig))

    dim = int(np.ceil(len(to_draw)**0.5))

    if figsize is None:
        x_len = 1000*dim
        y_len = 500*dim
        image = oedepict.OEImage(x_len, y_len)
    else:
        assert ( len(figsize) == 2 ), "figsize arguement should be a tuple or list of length 2"
        image = oedepict.OEImage(figsize[0],figsize[1])

    rows, cols = dim, dim
    grid = oedepict.OEImageGrid(image, rows, cols)

    opts = oedepict.OE2DMolDisplayOptions(grid.GetCellWidth(), grid.GetCellHeight(), oedepict.OEScale_AutoScale)

    minscale = float("inf")
    for mol in to_draw:
        minscale = min(minscale, oedepict.OEGetMoleculeScale(mol, opts))
    #     print(mol.GetTitle())

    opts.SetScale(minscale)
    for idx, cell in enumerate(grid.GetCells()):
        mol = to_draw[idx]
        disp = oedepict.OE2DMolDisplay(mol, opts)
        oedepict.OERenderMolecule(cell, disp)

    oedepict.OEWriteImage(filename, image)

    return

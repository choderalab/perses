from openmoltools import forcefield_generators
from openeye import oechem


def normalize_molecule(mol):
    # Assign aromaticity.
    oechem.OEAssignAromaticFlags(mol, oechem.OEAroModelOpenEye)

    # Add hydrogens.
    oechem.OEAddExplicitHydrogens(mol)

    # Check for any missing atom names, if found reassign all of them.
    if any([atom.GetName() == '' for atom in mol.GetAtoms()]):
        oechem.OETriposAtomNames(mol)

    ofs = oechem.oemolostream('out.mol2')
    ofs.SetFormat(oechem.OEFormat_MOL2H)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()

    return mol


def generate_forcefield(molecule_file, outfile):

    ifs = oechem.oemolistream()
    ifs.open(molecule_file)

    # get the list of molecules
    mol_list = [normalize_molecule(oechem.OEMol(mol)) for mol in ifs.GetOEMols()]

    # TODO: HORRIBLE HACK ; WILL DIE AT > 999 RESIDUES!
    for idx, mol in enumerate(mol_list):
        mol.SetTitle("%03d" % idx)

    ffxml = forcefield_generators.generateForceFieldFromMolecules(mol_list, normalize=False)

    with open(outfile, 'w') as output_file:
        output_file.write(ffxml)


if __name__=="__main__":
    import sys
    infile_name = sys.argv[1]
    outfile_name = sys.argv[2]

    generate_forcefield(infile_name, outfile_name)

import openeye.oechem as oechem
import pdbfixer
from simtk.openmm import app

def read_and_write_protein(protein_filename: str, protein_output_filename: str):
    """
    Read in and write back out a protein file using PDBFixer. This works to correct the issues in the Schrodinger JACS
    set protein files.

    Parameters
    ----------
    protein_filename : str
        the protein pdb filename
    protein_output_filename : str
        the name of the protein pdb file to use for output
    """
    fixer = pdbfixer.PDBFixer(filename=protein_filename)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    #extract topology and positions
    topology = fixer.topology
    positions = fixer.positions

    #open the output file
    protein_outfile = open(protein_output_filename, 'w')

    #write out the protein
    app.PDBFile.writeFile(topology, positions, file=protein_outfile)

    #close the output file
    protein_outfile.close()

def convert_molecules_to_mol2(sdf_filename: str, mol2_output_filename: str):
    """
    Convert the input set of molecules in sdf format to a mol2 file.

    Parameters
    ----------
    sdf_filename : str
        The name of the original SDF format molecule file
    mol2_output_filename : str
        The name of the output mol2 format molecule file
    """
    #open an input stream
    istream = oechem.oemolistream()
    istream.open(sdf_filename)

    #open an output stream
    ostream = oechem.oemolostream()
    ostream.open(mol2_output_filename)
    ostream.SetFormat(oechem.OEFormat_MOL2)

    #read in the molecules, give tripos atom names, and write them back out
    for molecule in istream.GetOEMols():
        outmol = oechem.OEMol(molecule)
        oechem.OETriposAtomNames(outmol)
        oechem.OEWriteMolecule(ostream, outmol)

    #close the streams:
    istream.close()
    ostream.close()


if __name__=="__main__":
    read_and_write_protein("MCL1_protein.pdb", "MCL1_fixed.pdb")
    convert_molecules_to_mol2("MCL1_ligands.sdf", "MCL1_ligands.mol2")
from openmoltools import forcefield_generators
from openeye import oechem

def generate_forcefield(molecule_file, outfile):

    ifs = oechem.oemolistream()
    ifs.open(molecule_file)

    # get the list of molecules
    mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]

    for idx, mol in enumerate(mol_list):
        mol.SetTitle("MOL{}".format(idx))

    ffxml = forcefield_generators.generateForceFieldFromMolecules(mol_list)

    with open(outfile, 'w') as output_file:
        output_file.write(ffxml)

if __name__=="__main__":
    import sys
    infile_name = sys.argv[1]
    outfile_name = sys.argv[2]

    generate_forcefield(infile_name, outfile_name)

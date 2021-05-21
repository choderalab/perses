#!/usr/bin/env python
"""
Generate ffxml files for ligands in a multi-mole mol2.
"""


mol2_filename = 'Imatinib-epik-charged.mol2'
ffxml_filename = 'Imatinib-epik-charged.ffxml'

# Read mol2 file containing protonation states and extract canonical isomeric SMILES from this.
print("Reading molecules")
from openeye import oechem
ifs = oechem.oemolistream(mol2_filename)
mol = oechem.OEMol()
molecules = list()
while oechem.OEReadMolecule(ifs, mol):
    molecules.append(oechem.OEMol(mol))

print("Generating forcefield parameters...")
from openmoltools.forcefield_generators import generateForceFieldFromMolecules
ffxml = generateForceFieldFromMolecules(molecules, ignoreFailures=False, generateUniqueNames=True)

print("Writing forcefield to '%s'..." % ffxml_filename)
outfile = open(ffxml_filename, 'w')
outfile.write(ffxml)
outfile.close()

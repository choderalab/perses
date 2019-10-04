#!/usr/bin/env python
from openeye.oechem import *

ifs = oemolistream('imidazole/imidazole-epik-charged.mol2')
ofs = oemolostream('imidazol.pdb')

mol = OEGraphMol()

while OEReadMolecule(ifs, mol):
    OEWriteMolecule(ofs, mol)

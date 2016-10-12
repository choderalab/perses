grep BNZ 181L.pdb | grep HETATM > ligand.pdb
pdbfixer 181L.pdb --keep-heterogens=none --add-atoms=heavy --ph=7.0 --replace-nonstandard --output=receptor.pdbfixer.pdb
python /Users/grinawap/anaconda/lib/python3.5/site-packages/openeye/examples/oechem/convert.py ligand.pdb ligand.mol2h
python /Users/grinawap/anaconda/lib/python3.5/site-packages/openeye/examples/oechem/convert.py ligand.mol2h ligand_h.pdb
cat receptor.pdbfixer.pdb ligand_h.pdb > t4complex.pdb

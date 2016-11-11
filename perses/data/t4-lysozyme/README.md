# Preparation of T4 Affinity Test System

This folder contains files used to prepare the T4AffinityTestSystem

## Manifest

* `181L.pdb` - Initial starting PDB file for the complex of T4 and Benzene
* `clean-t4-ligands.py` - File to take the set of SMILES originally in the L99A-binders.txt and L99A-nonbinders.txt files
    and output a list of smiles whose proposal graphs form a connected graph; that is, there are no unreachable parts of the
    proposal graph. The output is `clean_smiles_t4_networkx.txt`
* 
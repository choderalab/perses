"""
Clean the T4 ligands to include only valid smiles that can be proposed.
"""

import openeye.oechem as oechem
import openeye.oeomega as oeomega
import openeye.oequacpac as oequacpac
import networkx as nx
from perses.rjmc import topology_proposal
import logging

ATOM_OPTS = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember | oechem.OEExprOpts_HvyDegree | oechem.OEExprOpts_AtomicNumber
BOND_OPTS = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember

RAW_FILENAME_BINDERS = "../L99A-binders.txt"
RAW_FILENAME_NONBINDERS = "../L99A-non-binders.txt"

def read_smiles(filename):
    import csv
    molecules = list()
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in csvreader:
            name = row[0]
            smiles = row[1]
            reference = row[2]
            molecules.append(smiles)
    return molecules

def get_raw_canonical_smiles():
    """
     Verify that the smiles strings here can be turned into molecules, and have conformations
    and charges generated

    Returns
    -------
    canonical_smiles : list of str
        list of canonical smiles strings of the ligands that can be made
    failed_ligands : list of str
        list of ligands that cannot be made
    """
    binders = read_smiles(RAW_FILENAME_BINDERS)
    nonbinders = read_smiles(RAW_FILENAME_NONBINDERS)
    ligands = binders+nonbinders
    canonical_smiles = []
    failed_ligands = []
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetStrictStereo(False)
    for ligand_smiles in ligands:
        try:
            mol = oechem.OEMol()
            oechem.OESmilesToMol(mol, ligand_smiles)
            oechem.OEAddExplicitHydrogens(mol)
            omega(mol)
            oequacpac.OEAssignPartialCharges(mol, oequacpac.OECharges_AM1BCC)
            canonical_smiles.append(oechem.OECreateCanSmiString(mol))
        except:
            failed_ligands.append(ligand_smiles)
            logging.warning("Failed %s" % ligand_smiles)
    return canonical_smiles, failed_ligands

def get_acceptable_smiles():
    canonical_smiles, failed_ligands = get_raw_canonical_smiles()
    clean_smiles, removed_smiles = topology_proposal.SmallMoleculeSetProposalEngine.clean_molecule_list(canonical_smiles, ATOM_OPTS, BOND_OPTS)
    return clean_smiles, list(removed_smiles)+failed_ligands

if __name__=="__main__":
    clean_smiles, rejected_smiles = get_acceptable_smiles()

    clean_smiles_file = open("clean_smiles_t4_networkx.txt",'w')
    for smiles_group in clean_smiles:
        clean_smiles_file.writelines(["GROUP"])
        clean_smiles_file.writelines("\n".join(smiles_group))
        clean_smiles_file.close()

    rejected_smiles_file = open("rejected_smiles_t4_nx.txt", 'w')
    rejected_smiles_file.writelines("\n".join(rejected_smiles))
    rejected_smiles_file.close()
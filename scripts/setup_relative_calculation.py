import openeye.oechem as oechem
import rdkit.Chem as chem
import yaml
from perses.rjmc import topology_proposal, geometry
from perses.annihilation import relative
from perses.tests import utils
import simtk.unit as unit
from openmmtools.constants import kB
import simtk.openmm.app as app
from openmoltools import forcefield_generators
import copy
import numpy as np


def load_mol_file(mol_filename):
    """
    Utility function to load
    Parameters
    ----------
    mol_filename : str
        The name of the molecule file. Must be supported by openeye.

    Returns
    -------
    mol : oechem.OEMol
        the list of molecules as OEMols
    """
    ifs = oechem.oemolistream()
    ifs.open(mol_filename)
    #get the list of molecules
    mol_list = [mol for mol in ifs.GetOEMols()]
    #we'll always take the first for now
    return mol_list

def load_receptor_pdb(pdb_filename):
    """
    Load the receptor by PDB filename.

    Parameters
    ----------
    pdb_filename
    """
    f = open(pdb_filename, 'r')
    pdbfile = app.PDBFile(f)
    return pdbfile.topology, pdbfile.positions

def solvate_topology(topology, positions, forcefield, padding=9.0*unit.angstrom):
    """
    Solvate the given topology to run an explicit solvent simulation.
    Parameters
    ----------
    topology

    Returns
    -------

    """
    modeller = app.Modeller(topology, positions)
    modeller.addSolvent(forcefield, padding=padding, model='tip3p')
    topology = modeller.getTopology()
    positions = modeller.getPositions()

    return topology, positions

def _find_mol_start_index(topology):
        """
        Find the starting index of the molecule in the topology.
        Throws an exception if resname is not present.

        Parameters
        ----------
        topology : app.Topology object
            The topology containing the molecule

        Returns
        -------
        mol_start_idx : int
            start index of the molecule
        mol_length : int
            the number of atoms in the molecule
        """
        resname = "MOL"
        mol_residues = [res for res in topology.residues() if res.name==resname]
        if len(mol_residues)!=1:
            raise ValueError("There must be exactly one residue with a specific name in the topology. Found %d residues with name '%s'" % (len(mol_residues), resname))
        mol_residue = mol_residues[0]
        atoms = list(mol_residue.atoms())
        mol_start_idx = atoms[0].index
        return mol_start_idx, len(list(atoms))


def append_topology(destination_topology, source_topology, exclude_residue_name=None):
    """
    Add the source OpenMM Topology to the destination Topology.

    Parameters
    ----------
    destination_topology : simtk.openmm.app.Topology
        The Topology to which the contents of `source_topology` are to be added.
    source_topology : simtk.openmm.app.Topology
        The Topology to be added.
    exclude_residue_name : str, optional, default=None
        If specified, any residues matching this name are excluded.

    """
    newAtoms = {}
    for chain in source_topology.chains():
        newChain = destination_topology.addChain(chain.id)
        for residue in chain.residues():
            if (residue.name == exclude_residue_name):
                continue
            newResidue = destination_topology.addResidue(residue.name, newChain, residue.id)
            for atom in residue.atoms():
                newAtom = destination_topology.addAtom(atom.name, atom.element, newResidue, atom.id)
                newAtoms[atom] = newAtom
    for bond in source_topology.bonds():
        if (bond[0].residue.name==exclude_residue_name) or (bond[1].residue.name==exclude_residue_name):
            continue
        # TODO: Preserve bond order info using extended OpenMM API
        destination_topology.addBond(newAtoms[bond[0]], newAtoms[bond[1]])

def _build_new_topology(current_receptor_topology, oemol_proposed):
    """
    Construct a new topology
    Parameters
    ----------
    oemol_proposed : oechem.OEMol object
        the proposed OEMol object
    current_receptor_topology : app.Topology object
        The current topology without the small molecule

    Returns
    -------
    new_topology : app.Topology object
        A topology with the receptor and the proposed oemol
    mol_start_index : int
        The first index of the small molecule
    """
    oemol_proposed.SetTitle("MOL")
    mol_topology = forcefield_generators.generateTopologyFromOEMol(oemol_proposed)
    new_topology = app.Topology()
    append_topology(new_topology, current_receptor_topology)
    append_topology(new_topology, mol_topology)
    # Copy periodic box vectors.
    if current_receptor_topology._periodicBoxVectors != None:
        new_topology._periodicBoxVectors = copy.deepcopy(current_receptor_topology._periodicBoxVectors)

    return new_topology

def prepare_topology_proposal():
    from perses.tests.utils import get_data_filename
    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml', 'amber99sbildn.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)
    oemol_list = load_mol_file("p38_ligands.sdf")
    mol_a = oemol_list[0]
    mol_b = oemol_list[1]

    modeller = combine_mol_with_receptor(mol_a, "p38_protein.pdb")
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    topology = modeller.getTopology()
    positions = modeller.getPositions()
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME)



def combine_mol_with_receptor(mol, receptor_pdb_filename):
    from perses.tests.utils import extractPositionsFromOEMOL
    receptor_top, receptor_pos = load_receptor_pdb(receptor_pdb_filename)
    n_atoms_receptor = receptor_top.getNumAtoms()
    unsolvated_receptor_mol = _build_new_topology(receptor_top, mol)
    mol_positions = extractPositionsFromOEMOL(mol)

    #find positions of molecule in receptor topology:
    mol_start_index, len_mol = _find_mol_start_index(unsolvated_receptor_mol)

    #copy positions to new position array
    new_positions = unit.Quantity(value=np.zeros([n_atoms_receptor+len_mol, 3]), unit=unit.nanometer)
    new_positions[:mol_start_index, :] = receptor_pos
    new_positions[mol_start_index:, :] = mol_positions

    modeller = app.Modeller(unsolvated_receptor_mol, new_positions)

    return modeller
if __name__=="__main__":
    pass
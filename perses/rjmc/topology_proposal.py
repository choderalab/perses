"""
This file contains the base classes for topology proposals
"""

from simtk import unit

from simtk import openmm
from simtk.openmm import app

from collections import namedtuple
import copy
import warnings
import logging
import itertools
import json
import os
import openeye.oechem as oechem
import numpy as np
import openeye.oeomega as oeomega
import tempfile
import networkx as nx
from openmoltools import forcefield_generators
import openeye.oegraphsim as oegraphsim
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.storage import NetCDFStorageView
from io import StringIO
import openmoltools
import base64
import logging
import time
import progressbar
from typing import List, Dict
try:
    from subprocess import getoutput  # If python 3
except ImportError:
    from commands import getoutput  # If python 2

################################################################################
# CONSTANTS
################################################################################

OESMILES_OPTIONS = oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_ISOMERIC | oechem.OESMILESFlag_Hydrogens
#OESMILES_OPTIONS = oechem.OESMILESFlag_ISOMERIC | oechem.OESMILESFlag_Hydrogens

#DEFAULT_ATOM_EXPRESSION = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_Hybridization #| oechem.OEExprOpts_EqAromatic | oechem.OEExprOpts_EqHalogen | oechem.OEExprOpts_RingMember | oechem.OEExprOpts_EqCAliphaticONS
#DEFAULT_BOND_EXPRESSION = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember

DEFAULT_ATOM_EXPRESSION = oechem.OEExprOpts_DefaultAtoms
DEFAULT_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds
################################################################################
# LOGGER
################################################################################

_logger = logging.getLogger("proposal_engine")

################################################################################
# UTILITIES
################################################################################

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
    if exclude_residue_name is None:
        exclude_residue_name = "   " #something with 3 characters that is never a residue name
    newAtoms = {}
    for chain in source_topology.chains():
        new_chain = destination_topology.addChain(chain.id)
        for residue in chain.residues():
            # TODO: should we use complete residue names?
            if (residue.name[:3] == exclude_residue_name[:3]):
                continue
            new_residue = destination_topology.addResidue(residue.name, new_chain, residue.id)
            for atom in residue.atoms():
                new_atom = destination_topology.addAtom(atom.name, atom.element, new_residue, atom.id)
                new_atoms[atom] = new_atom
    for bond in source_topology.bonds():
        if (bond[0].residue.name[:3] == exclude_residue_name[:3]) or (bond[1].residue.name[:3] == exclude_residue_name[:3]):
            continue
        order = bond.order
        destination_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]], order=order)

def deepcopy_topology(source_topology):
    """
    Drop-in replacement for copy.deepcopy(topology) that fixes backpointer issues.

    Parameters
    ----------
    source_topology : simtk.openmm.app.Topology
        The Topology to be added.

    """
    topology = app.Topology()
    append_topology(topology, source_topology)
    return topology

def has_h_mapped(atommap, mola: oechem.OEMol, molb: oechem.OEMol):
    for a_atom, b_atom in atommap.items():
        if mola.GetAtom(oechem.OEHasAtomIdx(a_atom)).GetAtomicNum() == 1 or molb.GetAtom(oechem.OEHasAtomIdx(b_atom)).GetAtomicNum() == 1:
            return True

    return False

class SmallMoleculeAtomMapper(object):
    """
    This is a utility class for generating and retrieving sets of atom maps between molecules using OpenEye.
    It additionally verifies that all atom maps lead to valid proposals, as well as checking that the graph of
    proposals is not disconnected.
    """

    def __init__(self, list_of_smiles: List[str], atom_match_expression: int=None, bond_match_expression: int=None, prohibit_hydrogen_mapping: bool=True):

        self._unique_noncanonical_smiles_list = list(set(list_of_smiles))
        self._oemol_dictionary = self._initialize_oemols(self._unique_noncanonical_smiles_list)
        self._unique_smiles_list = list(self._oemol_dictionary.keys())

        self._n_molecules = len(self._unique_smiles_list)

        self._prohibit_hydrogen_mapping = prohibit_hydrogen_mapping

        if atom_match_expression is None:
            self._atom_expr = DEFAULT_ATOM_EXPRESSION
        else:
            self._atom_expr = atom_match_expression

        if bond_match_expression is None:
            self._bond_expr = DEFAULT_BOND_EXPRESSION
        else:
            self._bond_expr = bond_match_expression

        self._molecules_mapped = False
        self._molecule_maps = {}
        self._failed_molecule_maps = {}

        self._proposal_matrix = np.zeros([self._n_molecules, self._n_molecules])
        self._proposal_matrix_generated = False
        self._constraints_checked = False

    def map_all_molecules(self):
        """
        Run the atom mapping routines to get all atom maps. This automatically preserves only maps that contain enough torsions to propose.
        It does not ensure that constraints do not change--use verify_constraints to check that property. This method is idempotent--running it a second
        time will have no effect.
        """

        if self._molecules_mapped:
            _logger.info("The molecules have already been mapped. Returning.")
            return

        with progressbar.ProgressBar(max_value=self._n_molecules*(self._n_molecules-1)/2.0) as bar:

            current_index = 0

            for molecule_smiles_pair in itertools.combinations(self._oemol_dictionary.keys(), 2):
                molecule_pair = tuple(self._oemol_dictionary[molecule] for molecule in molecule_smiles_pair)

                self._molecule_maps[molecule_smiles_pair] = []
                self._failed_molecule_maps[molecule_smiles_pair] = []

                atom_matches, failed_atom_matches = self._map_atoms(molecule_pair[0], molecule_pair[1])

                for atom_match in atom_matches:
                    self._molecule_maps[molecule_smiles_pair].append(atom_match)

                for failed_atom_match in failed_atom_matches:
                    self._failed_molecule_maps[molecule_smiles_pair].append(failed_atom_match)

                current_index += 1
                bar.update(current_index)

        self._molecules_mapped = True

    def get_atom_maps(self, smiles_A: str, smiles_B: str) -> List[Dict]:
        """
        Given two canonical smiles strings, get the atom maps.

        Arguments
        ---------
        smiles_A : str
            Canonical smiles for the first molecule (keys)
        smiles_B : str
            Canonical smiles for the second molecule (values)

        Returns
        -------
        atom_maps : list of dict
            List of map of molecule_A_atom : molecule_B_atom
        """
        try:
            atom_maps = self._molecule_maps[(smiles_A, smiles_B)]
            return atom_maps
        except KeyError:
            try:
                atom_maps = self._molecule_maps[(smiles_B, smiles_A)]
                output_atom_maps = []
                for atom_map in atom_maps:
                    reversed_map = {value: key for key, value in atom_map.items()}
                    output_atom_maps.append(reversed_map)
                return output_atom_maps
            except KeyError as e:
                print("The requested pair was not found. Ensure you are using canonicalized smiles.")
                raise e

    def generate_and_check_proposal_matrix(self):
        """
        Generate a proposal matrix and check it for connectivity. Note that if constraints have not been checked, this may produce
        a proposal matrix that makes proposals changing constraint lengths.
        """

        if not self._constraints_checked:
            _logger.warn("Constraints have not been checked. Building proposal matrix, but it might result in error.")
            _logger.warn("Call constraint_check() with an appropriate system generator to ensure this does not happen.")

        proposal_matrix = self._create_proposal_matrix()
        adjacency_matrix = proposal_matrix > 0.0
        graph = nx.from_numpy_array(adjacency_matrix)

        if not nx.is_connected(graph):
            _logger.warn("The graph of proposals is not connected! Some molecules will be unreachable.")

        self._proposal_matrix = proposal_matrix

    def _create_proposal_matrix(self) -> np.array:
        """
        In RJ calculations, we propose based on how many atoms are in common between molecules. This routine checks that the graph of proposals cannot
        be separated. In calculating the proposal matrix, we use the min(n_atom_mapped) when there are multiple maps.

        Returns
        -------
        normalized_proposal_matrix : np.array of float
            The proposal matrix
        """
        proposal_matrix = np.zeros([self._n_molecules, self._n_molecules])

        for smiles_pair, atom_maps in self._molecule_maps.items():

            #retrieve the smiles strings of these molecules
            molecule_A = smiles_pair[0]
            molecule_B = smiles_pair[1]

            #retrieve the indices of these molecules from the list of smiles
            molecule_A_idx = self._unique_smiles_list.index(molecule_A)
            molecule_B_idx = self._unique_smiles_list.index(molecule_B)

            #if there are no maps, we can't propose
            if len(atom_maps) == 0:
                proposal_matrix[molecule_A_idx, molecule_B_idx] = 0.0
                proposal_matrix[molecule_B_idx, molecule_A_idx] = 0.0
                continue

            #get a list of the number of atoms mapped for each map
            number_of_atoms_in_maps = [len(atom_map.keys()) for atom_map in atom_maps]

            unnormalized_proposal_probability = float(min(number_of_atoms_in_maps))


            proposal_matrix[molecule_A_idx, molecule_B_idx] = unnormalized_proposal_probability
            proposal_matrix[molecule_B_idx, molecule_A_idx] = unnormalized_proposal_probability

        #normalize the proposal_matrix:

        #First compute the normalizing constants by summing the rows
        normalizing_constants = np.sum(proposal_matrix, axis=1)

        #If any normalizing constants are zero, that means that the molecule is completely unproposable:
        if np.any(normalizing_constants==0.0):
            where_zero = np.where(normalizing_constants==0.0)[0]
            failed_molecules = []
            for zero_idx in where_zero:
                failed_molecules.append(self._unique_smiles_list[zero_idx])

            print("The following molecules are unproposable:\n")
            print(failed_molecules)
            raise ValueError("Some molecules could not be proposed. Make sure the atom mapping criteria do not completely exclude a molecule.")

        normalized_proposal_matrix = proposal_matrix / normalizing_constants[:, np.newaxis]

        return normalized_proposal_matrix

    @staticmethod
    def _canonicalize_smiles(mol: oechem.OEMol) -> str:
        """
        Convert an oemol into canonical isomeric smiles

        Parameters
        ----------
        mol : oechem.OEmol
            OEMol for molecule
        Returns
        -------
        iso_can_smiles : str
            OpenEye isomeric canonical smiles corresponding to the input
        """
        iso_can_smiles = oechem.OECreateSmiString(mol, OESMILES_OPTIONS)
        return iso_can_smiles

    def _map_atoms(self, moleculeA: oechem.OEMol, moleculeB: oechem.OEMol, exhaustive: bool=True) -> List[Dict]:
        """
        Run the mapping on the two input molecules. This will return a list of atom maps.
        This is an internal method that is only intended to be used by other methods of this class.

        Arguments
        ---------
        moleculeA : oechem.OEMol
            The first oemol of the pair
        moleculeB : oechem.OEMol
            The second oemol of the pair
        exhaustive: bool, default True
            Whether to use an exhaustive procedure for enumerating MCS matches. Default True, but for large molecules,
            may be prohibitively slow.

        Returns
        -------
        atom_matches: list of dict
            This returns a list of dictionaries, where each dictionary is a map of the form {molA_atom: molB_atom}.
            Atom maps with less than 3 mapped atoms, or where the map is not sufficient to begin a geometry proposal
            will be returned separately.
        failed_atom_matches : list of dict
            This is a list of atom maps that cannot be used for geometry proposals. It is returned for debugging purposes.
        """

        oegraphmol_current = oechem.OEGraphMol(moleculeA) # pattern molecule
        oegraphmol_proposed = oechem.OEGraphMol(moleculeB) # target molecule

        if exhaustive:
            mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
        else:
            mcs = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)

        mcs.Init(oegraphmol_current, self._atom_expr, self._bond_expr)

        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())

        #only use unique matches
        unique = True

        matches = [m for m in mcs.Match(oegraphmol_proposed, unique)]

        #if there are no matches at all, we return two empty lists.
        if not matches:
            return [], []

        atom_matches = []
        failed_atom_matches = []

        for match in matches:
            #if there are less than 3 mapped atoms, it can't be used for geometry proposals.
            #Continue without recording it.
            if match.NumAtoms() < 3:
                continue

            #extract the match as a dictionary.
            a_to_b_atom_map = {}
            for matchpair in match.GetAtoms():
                a_index = matchpair.pattern.GetIdx()
                b_index = matchpair.target.GetIdx()

                #if we aren't allowing hydrogen maps, we need to ensure that neither mapped atom is a hydrogen
                if self._prohibit_hydrogen_mapping:
                    if matchpair.pattern.GetAtomicNum() == 1 or matchpair.target.GetAtomicNum() == 1:
                        continue
                    else:
                        a_to_b_atom_map[a_index] = b_index
                else:
                    a_to_b_atom_map[a_index] = b_index
            #Even if there are at least three atoms mapped, it is possible that the geometry proposal still cannot proceed
            #An example of this would be mapping H-C-H -- There will be no topological torsions to use for the proposal.
            if self._valid_match(moleculeA, moleculeB, a_to_b_atom_map):
                atom_matches.append(a_to_b_atom_map)
            else:
                failed_atom_matches.append(a_to_b_atom_map)

        return atom_matches, failed_atom_matches

    def _valid_match(self, moleculeA: oechem.OEMol, moleculeB: oechem.OEMol, a_to_b_mapping: Dict[int, int]) -> bool:
        """
        Check that the map can allow for a geometry proposal. Essentially, this amounts to ensuring that there exists
        a starting topological torsion. Examples of cases where this would not exist would include:
        H-C-H, Cl-C-Cl, CH3, etc.

        Arguments
        ---------
        moleculeA : oechem.OEMol
            The first molecule in the mapping
        moleculeB: oechem.OEMol
            The second molecule used in the mapping
        b_to_a_mapping : dict
            The mapping from molecule B to molecule A

        Returns
        -------
        is_valid : bool
            Whether the mapping can be used to generate a geometry proposal
        """
        graphA = self._mol_to_graph(moleculeA)
        graphB = self._mol_to_graph(moleculeB)

        #if both of these are good to make a map, we can make the map
        return self._can_make_proposal(graphA, a_to_b_mapping.keys()) and self._can_make_proposal(graphB, a_to_b_mapping.values())

    def _can_make_proposal(self, graph: nx.Graph, mapped_atoms: List) -> bool:
        """
        Check whether a given setup (molecule graph along with mapped atoms) can be proposed.

        Arguments
        ---------
        graph: nx.Graph
            The molecule represented as a NetworkX graph
        mapped_atoms : list
            The list of atoms that have been mapped

        Returns
        -------
        can_make_proposal : bool
            Whether this map permits the GeometryEngine to make a proposal
        """

        proposable = False
        total_atoms = set(range(graph.number_of_nodes()))
        unmapped_atoms = total_atoms - set(mapped_atoms)
        mapped_atoms_set = set(mapped_atoms)

        #find the set of atoms that are unmapped, but on the boundary with those that are mapped
        boundary_atoms = nx.algorithms.node_boundary(graph, unmapped_atoms, mapped_atoms)

        #now check if there is atom 3 hops away and has a position. Since we are starting with boundary
        #atoms, there will never be a case where there is an atom with positions 3 hops away but no torsion
        #A ring might cause this artifact, but there would be a torsion in that case.
        for atom in boundary_atoms:
            shortest_paths = nx.algorithms.shortest_path_length(graph, source=atom)
            for other_atom, distance in shortest_paths.items():
                if distance == 3 and other_atom in mapped_atoms:
                    #find all shortest paths to the other atom. if any of them have all atoms with positions, it can be proposed
                    shortest_path = nx.shortest_path(graph, source=atom, target=other_atom)
                    if len(mapped_atoms_set.intersection(shortest_path)) == 3:
                        proposable = True

        return proposable

    def _mol_to_graph(self, molecule: oechem.OEMol) -> nx.Graph:
        """
        Convert an OEMol to a networkx graph for analysis

        Arguments
        ---------
        molecule : oechem.OEMol
            Molecule to convert to a graph

        Returns
        -------
        g : nx.Graph
            NetworkX graph representing the molecule
        """
        g = nx.Graph()
        for atom in molecule.GetAtoms():
            g.add_node(atom.GetIdx())
        for bond in molecule.GetBonds():
            g.add_edge(bond.GetBgnIdx(), bond.GetEndIdx(), bond=bond)

        return g

    def _initialize_oemols(self, list_of_smiles: List[str]) -> Dict[str, oechem.OEMol]:
        """
        Initialize the set of OEMols that we will use to construct the atom map

        Arguments
        ---------
        list_of_smiles : list of str
            list of smiles strings to use

        Returns
        -------
        dict_of_oemol : dict of oechem.OEmol
            dict of canonical_smiles : oechem.OEMol
        """
        list_of_mols = []
        for smiles in list_of_smiles:
            mol = oechem.OEMol()
            oechem.OESmilesToMol(mol, smiles)
            oechem.OEAddExplicitHydrogens(mol)
            list_of_mols.append(mol)

        #now write out all molecules and read them back in:
        molecule_string = SmallMoleculeAtomMapper.molecule_library_to_string(list_of_mols)
        dict_of_oemol = SmallMoleculeAtomMapper.molecule_library_from_string(molecule_string)

        return dict_of_oemol

    def get_oemol_from_smiles(self, smiles_string: str) -> oechem.OEMol:
        """
        Get the OEMol corresponding to the smiles string requested. This method exists
        to avoid having atom order rearranged by regeneration from smiles.

        Arguments
        ---------
        smiles_string : str
            The smiles string for which to retrieve the OEMol. Only pre-existing OEMols are allowed

        Returns
        -------
        mol : oechem.OEMol
            The OEMol corresponding to the requested smiles string.
        """
        return self._oemol_dictionary[smiles_string]

    def get_smiles_index(self, smiles: str) -> int:
        """
        Get the index of the smiles in question

        Arguments
        ---------
        smiles : str
            Canonicalized smiles string to retrieve molecule

        Returns
        -------
        mol_index : int
            Index of molecule in list
        """
        mol_index = self._unique_smiles_list.index(smiles)
        return mol_index

    @staticmethod
    def molecule_library_from_string(molecule_string: str) -> Dict[str, oechem.OEMol]:
        """
        Given a library of molecules in a mol2-format string, return a dictionary that maps
        the respective canonical smiles to the oemol object.

        Parameters
        ----------
        molecule_string : str
            The string containing the molecule data

        Returns
        -------
        molecule_dictionary : dict of str: oemol
            Dictionary mapping smiles strings to OEMols

        """

        ifs = oechem.oemolistream()
        ifs.SetFormat(oechem.OEFormat_OEB)
        ifs.openstring(molecule_string)

        molecule_dictionary = {}
        for mol in ifs.GetOEMols():
            copied_mol = oechem.OEMol(mol)
            canonical_smiles = SmallMoleculeAtomMapper._canonicalize_smiles(copied_mol)
            molecule_dictionary[canonical_smiles] = copied_mol

        ifs.close()

        return molecule_dictionary

    @staticmethod
    def molecule_library_to_string(molecule_library: List[oechem.OEMol]) -> str:
        """
        Given a list of oechem.OEMol objects, write all of them in mol2 format to a string

        Parameters
        ----------
        molecule_library : list of oechem.OEMol
            molecules to write to string

        Returns
        -------
        molecule_string : str
            String containing molecules in mol2 format.
        """

        ofs = oechem.oemolostream()
        ofs.SetFormat(oechem.OEFormat_OEB)
        ofs.openstring()

        for mol in molecule_library:
            oechem.OEWriteMolecule(ofs, mol)

        molecule_string = ofs.GetString()

        ofs.close()

        return molecule_string

    def to_json(self) -> str:
        """
        Write out this class to JSON. This saves all information (including built molecules and maps, if present)

        Returns
        -------
        json_str : str
            JSON string representing this class
        """
        json_dict = {}

        #first, save all the things that are not too difficult to put into JSON
        json_dict['molecule_maps'] = {"_".join(smiles_names): maps for smiles_names, maps in self._molecule_maps.items()}
        json_dict['molecules_mapped'] = self._molecules_mapped
        json_dict['failed_molecule_maps'] = {"_".join(smiles_names): maps for smiles_names, maps in self._failed_molecule_maps.items()}
        json_dict['constraints_checked'] = self._constraints_checked
        json_dict['bond_expr'] = self._bond_expr
        json_dict['atom_expr'] = self._atom_expr
        json_dict['proposal_matrix'] = self._proposal_matrix.tolist()
        json_dict['proposal_matrix_generated'] = self._proposal_matrix_generated
        json_dict['unique_smiles_list'] = self._unique_smiles_list

        #now we have to convert the OEMols to a string format that preserves the atom ordering in order to save them
        #We will use the multi-molecule mol2 scheme, but instead of saving to a file, we will save to a string.

        molecule_bytes = SmallMoleculeAtomMapper.molecule_library_to_string(self._oemol_dictionary.values())

        base64_molecule_bytes = base64.b64encode(molecule_bytes)

        json_dict['molecules'] = base64_molecule_bytes.decode()

        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, json_string: str):
        """
        Restore this class from a saved JSON file.

        Arguments
        ---------
        json_string : str
            The JSON string representing the serialized class

        Returns
        -------
        atom_mapper : SmallMoleculeAtomMapper
            An instance of the SmallMoleculeAtomMapper
        """
        json_dict = json.loads(json_string)

        #first let's read in all the molecules that were saved as a string:
        molecule_bytes = json_dict['molecules'].encode()
        molecule_string = base64.b64decode(molecule_bytes)
        molecule_dictionary = SmallMoleculeAtomMapper.molecule_library_from_string(molecule_string)

        bond_expr = json_dict['bond_expr']
        atom_expr = json_dict['atom_expr']
        smiles_list = json_dict['unique_smiles_list']

        map_to_ints = lambda maps: [{int(key): int(value) for key, value in map.items()} for map in maps]

        mapper = cls(smiles_list, atom_match_expression=atom_expr, bond_match_expression=bond_expr)

        mapper._molecule_maps = {tuple(smiles for smiles in key.split("_")) : map_to_ints(maps) for key, maps in json_dict['molecule_maps'].items()}
        mapper._molecules_mapped = json_dict['molecules_mapped']
        mapper._failed_molecule_maps = {tuple(smiles for smiles in key.split("_")) : maps for key, maps in json_dict['failed_molecule_maps'].items()}
        mapper._constraints_checked = json_dict['constraints_checked']
        mapper._proposal_matrix = np.array(json_dict['proposal_matrix'])
        mapper._proposal_matrix_generated = json_dict['proposal_matrix_generated']
        mapper._oemol_dictionary = molecule_dictionary

        return mapper

    @property
    def proposal_matrix(self):
        return self._proposal_matrix

    @property
    def n_molecules(self):
        return self._n_molecules

    @property
    def smiles_list(self):
        return self._unique_smiles_list



from perses.rjmc.geometry import NoTorsionError
class TopologyProposal(object):
    """
    This is a container class with convenience methods to access various objects needed
    for a topology proposal

    Arguments
    ---------
    new_topology : simtk.openmm.Topology object
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.Topology object
        openmm Topology of the current system
    old_system : simtk.openmm.System object
        openm System of the current state
    logp_proposal : float
        contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
    new_to_old_atom_map : dict
        {new_atom_idx : old_atom_idx} map for the two systems
    old_alchemical_atoms : list, optional, default=None
        List of all atoms in old system that are being transformed.
        If None, all atoms are assumed to be part of the alchemical region.
    chemical_state_key : str
        The current chemical state (unique)
    metadata : dict
        additional information of interest about the state

    Properties
    ----------
    new_topology : simtk.openmm.Topology object
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.Topology object
        openmm Topology of the current system
    old_system : simtk.openmm.System object
        openm System of the current state
    old_positions : [n, 3] np.array, Quantity
        positions of the old system
    logp_proposal : float
        contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
    new_to_old_atom_map : dict
        {new_atom_idx : old_atom_idx} map for the two systems
    old_to_new_atom_map : dict
        {old_atom_idx : new_atom_idx} map for the two systems
    new_alchemical_atoms : list
        List of all atoms in new system that are being transformed
    new_environment_atoms : list
        List of all atoms in new system that are not transformed, just mapped
    old_alchemical_atoms : list
        List of all atoms in old system that are being transformed
    old_environment_atoms : list
        List of all atoms in old system that are not transformed, just mapped
    unique_new_atoms : list of int
        List of indices of the unique new atoms
    unique_old_atoms : list of int
        List of indices of the unique old atoms
    natoms_new : int
        Number of atoms in the new system
    natoms_old : int
        Number of atoms in the old system
    old_chemical_state_key : str
        The previous chemical state key
    new_chemical_state_key : str
        The proposed chemical state key
    metadata : dict
        additional information of interest about the state
    """

    def __init__(self,
                 new_topology=None, new_system=None,
                 old_topology=None, old_system=None,
                 logp_proposal=None,
                 new_to_old_atom_map=None, old_alchemical_atoms=None,
                 old_chemical_state_key=None, new_chemical_state_key=None,
                 metadata=None):

        if new_chemical_state_key is None or old_chemical_state_key is None:
            raise ValueError("chemical_state_keys must be set.")
        self._new_topology = new_topology
        self._new_system = new_system
        self._old_topology = old_topology
        self._old_system = old_system
        self._logp_proposal = logp_proposal
        self._new_chemical_state_key = new_chemical_state_key
        self._old_chemical_state_key = old_chemical_state_key
        self._new_to_old_atom_map = new_to_old_atom_map
        self._old_to_new_atom_map = {old_atom : new_atom for new_atom, old_atom in new_to_old_atom_map.items()}
        self._unique_new_atoms = list(set(range(self._new_topology.getNumAtoms()))-set(self._new_to_old_atom_map.keys()))
        self._unique_old_atoms = list(set(range(self._old_topology.getNumAtoms()))-set(self._new_to_old_atom_map.values()))
        self._old_alchemical_atoms = set(old_alchemical_atoms) if (old_alchemical_atoms is not None) else {atom for atom in range(old_system.getNumParticles())}
        self._new_alchemical_atoms = set(self._old_to_new_atom_map.values()).union(self._unique_new_atoms)
        self._old_environment_atoms = set(range(old_system.getNumParticles())) - self._old_alchemical_atoms
        self._new_environment_atoms = set(range(new_system.getNumParticles())) - self._new_alchemical_atoms
        self._metadata = metadata

    @property
    def new_topology(self):
        return self._new_topology
    @property
    def new_system(self):
        return self._new_system
    @property
    def old_topology(self):
        return self._old_topology
    @property
    def old_system(self):
        return self._old_system
    @property
    def logp_proposal(self):
        return self._logp_proposal
    @property
    def new_to_old_atom_map(self):
        return self._new_to_old_atom_map
    @property
    def old_to_new_atom_map(self):
        return self._old_to_new_atom_map
    @property
    def unique_new_atoms(self):
        return self._unique_new_atoms
    @property
    def unique_old_atoms(self):
        return self._unique_old_atoms
    @property
    def new_alchemical_atoms(self):
        return list(self._new_alchemical_atoms)
    @property
    def old_alchemical_atoms(self):
        return list(self._old_alchemical_atoms)
    @property
    def new_environment_atoms(self):
        return list(self._new_environment_atoms)
    @property
    def old_environment_atoms(self):
        return list(self._old_environment_atoms)
    @property
    def n_atoms_new(self):
        return self._new_system.getNumParticles()
    @property
    def n_atoms_old(self):
        return self._old_system.getNumParticles()
    @property
    def new_chemical_state_key(self):
        return self._new_chemical_state_key
    @property
    def old_chemical_state_key(self):
        return self._old_chemical_state_key
    @property
    def metadata(self):
        return self._metadata

class ProposalEngine(object):
    """
    This defines a type which, given the requisite metadata, can produce Proposals (namedtuple)
    of new topologies.

    Arguments
    --------
    system_generator : SystemGenerator
        The SystemGenerator to use to generate new System objects for proposed Topology objects
    proposal_metadata : dict
        Contains information necessary to initialize proposal engine
    verbose : bool, optional, default=False
        If True, print verbose debugging output

    Properties
    ----------
    chemical_state_list : list of str
         a list of all the chemical states that this proposal engine may visit.
    """

    def __init__(self, system_generator, proposal_metadata=None, always_change=True, verbose=False):
        self._system_generator = system_generator
        self.verbose = verbose
        self._always_change = always_change

    def propose(self, current_system, current_topology, current_metadata=None):
        """
        Base interface for proposal method.

        Arguments
        ---------
        current_system : simtk.openmm.System object
            The current system object
        current_topology : simtk.openmm.app.Topology object
            The current topology
        current_metadata : dict
            Additional metadata about the state
        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        return TopologyProposal(new_topology=app.Topology(), old_topology=app.Topology(), old_system=current_system, old_chemical_state_key="C", new_chemical_state_key="C", logp_proposal=0.0, new_to_old_atom_map={0 : 0}, metadata={'molecule_smiles' : 'CC'})

    def compute_state_key(self, topology):
        """
        Compute the corresponding state key of a given topology,
        according to this proposal engine's scheme.

        Parameters
        ----------
        topology : app.Topology
            the topology in question

        Returns
        -------
        chemical_state_key : str
            The chemical_state_key
        """
        pass

    @property
    def chemical_state_list(self):
        raise NotImplementedError("This ProposalEngine does not expose a list of possible chemical states.")

class PolymerProposalEngine(ProposalEngine):
    """
    """

    # TODO: Eliminate 'verbose' option in favor of logging

    def __init__(self, system_generator, chain_id, proposal_metadata=None, verbose=False, always_change=True, aggregate=False):
        """
        Create a polymer proposal engine

        Parameters
        ----------
        system_generator : SystemGenerator
            The SystemGenerator to use to generate perturbed systems
        chain_id : str
            The chain identifier in the Topology object to be mutated
        proposal_metadata : dict, optional, default=None
            Any metadata to be maintained
        verbose : bool, optional, default=False
            If True, will generate verbose output
        always_change : bool, optional, default=True
            If True, will not propose self transitions
        aggregate : bool, optional, default=False
            ???????

        """
        super(PolymerProposalEngine,self).__init__(system_generator, proposal_metadata=proposal_metadata, verbose=verbose, always_change=always_change)
        self._chain_id = chain_id # Chain ID to be mutated
        self._aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL']  # Note this does not include PRO since there's a problem with OpenMM's template DEBUG
        self._aggregate = aggregate # ?????????

    def propose(self, current_system, current_topology, current_metadata=None):
        """
        Generate a TopologyProposal

        Arguments
        ---------
        current_system : simtk.openmm.System object
            The current system object
        current_topology : simtk.openmm.app.Topology object
            The current topology
        current_metadata : dict -- OPTIONAL
        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """

        # old_topology : simtk.openmm.app.Topology
        old_topology = app.Topology()
        append_topology(old_topology, current_topology)

        # new_topology : simtk.openmm.app.Topology
        new_topology = app.Topology()
        append_topology(new_topology, current_topology)

        # Check that old_topology and old_system have same number of atoms.
        old_system = current_system
        old_topology_natoms = old_topology.getNumAtoms()  # number of topology atoms
        old_system_natoms = old_system.getNumParticles()
        if old_topology_natoms != old_system_natoms:
            msg = 'PolymerProposalEngine: old_topology has %d atoms, while old_system has %d atoms' % (old_topology_natoms, old_system_natoms)
            raise Exception(msg)

        # metadata : dict, key = 'chain_id' , value : str
        metadata = current_metadata
        if metadata is None:
            metadata = dict()
        # old_chemical_state_key : str
        old_chemical_state_key = self.compute_state_key(old_topology)

        # index_to_new_residues : dict, key : int (index) , value : str (three letter name of proposed residue)
        index_to_new_residues, metadata = self._choose_mutant(old_topology, metadata)

        # residue_map : list(tuples : simtk.openmm.app.topology.Residue (existing residue), str (three letter name of proposed residue))
        residue_map = self._generate_residue_map(old_topology, index_to_new_residues)


        for (res, new_name) in residue_map:
            if res.name == new_name:
                del(index_to_new_residues[res.index])
        if len(index_to_new_residues) == 0:
            atom_map = dict()
            for atom in old_topology.atoms():
                atom_map[atom.index] = atom.index
            if self.verbose: print('PolymerProposalEngine: No changes to topology proposed, returning old system and topology')
            topology_proposal = TopologyProposal(new_topology=old_topology, new_system=old_system, old_topology=old_topology, old_system=old_system, old_chemical_state_key=old_chemical_state_key, new_chemical_state_key=old_chemical_state_key, logp_proposal=0.0, new_to_old_atom_map=atom_map)
            return topology_proposal

        elif len(index_to_new_residues) > 1:
            raise Exception("Attempting to mutate more than one residue at once: ", index_to_new_residues, " The geometry engine cannot handle this.")

        # Add modified_aa property to residues in old topology
        for res in old_topology.residues():
            res.modified_aa = True if res.index in index_to_new_residues.keys() else False

        # Identify differences between old topology and proposed changes
        # excess_atoms : list(simtk.openmm.app.topology.Atom) atoms from existing residue not in new residue
        # excess_bonds : list(tuple (simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)) bonds from existing residue not in new residue
        # missing_bonds : list(tuple (simtk.openmm.app.topology._TemplateAtomData, simtk.openmm.app.topology._TemplateAtomData)) bonds from new residue not in existing residue
        excess_atoms, excess_bonds, missing_atoms, missing_bonds = self._identify_differences(old_topology, residue_map)

        # Delete excess atoms and bonds from old topology
        excess_atoms_bonds = excess_atoms + excess_bonds
        new_topology = self._delete_atoms(old_topology, excess_atoms_bonds)

        # Add missing atoms and bonds to new topology
        new_topology = self._add_new_atoms(new_topology, missing_atoms, missing_bonds, residue_map)

        # index_to_new_residues : dict, key : int (index) , value : str (three letter name of proposed residue)
        atom_map = self._construct_atom_map(residue_map, old_topology, index_to_new_residues, new_topology)

        # new_chemical_state_key : str
        new_chemical_state_key = self.compute_state_key(new_topology)
        # new_system : simtk.openmm.System

        # Copy periodic box vectors from current topology
        new_topology.setPeriodicBoxVectors(current_topology.getPeriodicBoxVectors())

        # Build system
        new_system = self._system_generator.build_system(new_topology)

        # Adjust logp_propose based on HIS presence
        his_residues = ['HID', 'HIE']
        old_residue = residue_map[0][0]
        proposed_residue = residue_map[0][1]
        if old_residue.name in his_residues and proposed_residue not in his_residues:
            logp_propose = math.log(2)
        elif old_residue.name not in his_residues and proposed_residue in his_residues:
            logp_propose = math.log(0.5)
        else:
            logp_propose = 0.0

        # Create TopologyProposal.
        topology_proposal = TopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=old_system, old_chemical_state_key=old_chemical_state_key, new_chemical_state_key=new_chemical_state_key, logp_proposal=logp_propose, new_to_old_atom_map=atom_map)

        # Check that old_topology and old_system have same number of atoms.
        old_topology_natoms = old_topology.getNumAtoms()  # number of topology atoms
        old_system_natoms = old_system.getNumParticles()
        if old_topology_natoms != old_system_natoms:
            msg = 'PolymerProposalEngine: old_topology has %d atoms, while old_system has %d atoms' % (old_topology_natoms, old_system_natoms)
            raise Exception(msg)

        # Check that new_topology and new_system have same number of atoms.
        new_topology_natoms = new_topology.getNumAtoms()  # number of topology atoms
        new_system_natoms = new_system.getNumParticles()
        if new_topology_natoms != new_system_natoms:
            msg = 'PolymerProposalEngine: new_topology has %d atoms, while new_system has %d atoms' % (new_topology_natoms, new_system_natoms)
            raise Exception(msg)

        # Check to make sure no out-of-bounds atoms are present in new_to_old_atom_map
        natoms_old = topology_proposal.old_system.getNumParticles()
        natoms_new = topology_proposal.new_system.getNumParticles()
        if not set(topology_proposal.new_to_old_atom_map.values()).issubset(range(natoms_old)):
            msg = "Some new atoms in TopologyProposal.new_to_old_atom_map are not in span of new atoms (1..%d):\n" % natoms_new
            msg += str(topology_proposal.new_to_old_atom_map)
            raise Exception(msg)
        if not set(topology_proposal.new_to_old_atom_map.keys()).issubset(range(natoms_new)):
            msg = "Some new atoms in TopologyProposal.new_to_old_atom_map are not in span of old atoms (1..%d):\n" % natoms_new
            msg += str(topology_proposal.new_to_old_atom_map)
            raise Exception(msg)

        return topology_proposal

    def _choose_mutant(self, topology, metadata):
        index_to_new_residues = dict()
        return index_to_new_residues, metadata

    def _generate_residue_map(self, topology, index_to_new_residues):
        """
        generates list to reference residue instance to be edited, because topology.residues() cannot be referenced directly by index

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        Returns
        -------
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue)
        """
        # residue_map : list(tuples : simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue))
        # r : simtk.openmm.app.topology.Residue, r.index : int, 0-indexed
        residue_map = [(r, index_to_new_residues[r.index]) for r in topology.residues() if r.index in index_to_new_residues]
        return residue_map

    def _delete_excess_atoms(self, topology, residue_map):
        """
        delete excess atoms from old residues and identify new atoms for new residues

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue)
        Returns
        -------
        topology : simtk.openmm.app.Topology
            extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        missing_atoms : dict
            key : simtk.openmm.app.topology.Residue
            value : list(simtk.openmm.app.topology._TemplateAtomData)
        """
        # delete excess atoms from old residues and identify new atoms for new residues
        # delete_atoms : list(simtk.openmm.app.topology.Atom) atoms from existing residue not in new residue
        delete_atoms = list()
        # missing_atoms : dict, key : simtk.openmm.app.topology.Residue, value : list(simtk.openmm.app.topology._TemplateAtomData)
        missing_atoms = dict()
        # residue : simtk.openmm.app.topology.Residue (existing residue)
        # replace_with : str (three letter residue name of proposed residue)
        for k, (residue, replace_with) in enumerate(residue_map):
            # chain_residues : list(simtk.openmm.app.topology.Residue) all residues in chain ==> why
            chain_residues = list(residue.chain.residues())
            if residue == chain_residues[0]:
                replace_with = 'N'+replace_with
                residue_map[k] = (residue, replace_with)
            if residue == chain_residues[-1]:
                replace_with = 'C'+replace_with
                residue_map[k] = (residue, replace_with)
            # template : simtk.openmm.app.topology._TemplateData
            template = self._templates[replace_with]
            # standard_atoms : set of unique atom names within new residue : str
            standard_atoms = set(atom.name for atom in template.atoms)
            # template_atoms : list(simtk.openmm.app.topology._TemplateAtomData) atoms in new residue
            template_atoms = list(template.atoms)
            # atom_names : set of unique atom names within existing residue : str
            atom_names = set(atom.name for atom in residue.atoms())
            # atom : simtk.openmm.app.topology.Atom in existing residue
            for atom in residue.atoms(): # shouldn't remove hydrogen
                if atom.name not in standard_atoms:
                    delete_atoms.append(atom)
#            if residue == chain_residues[0]: # this doesn't apply?
#                template_atoms = [atom for atom in template_atoms if atom.name not in ('P', 'OP1', 'OP2')]
            # missing : list(simtk.openmm.app.topology._TemplateAtomData) atoms in new residue not found in existing residue
            missing = list()
            # atom : simtk.openmm.app.topology._TemplateAtomData atoms in new residue
            for atom in template_atoms:
                if atom.name not in atom_names:
                    missing.append(atom)
            # BUG : error if missing = 0?
            if len(missing) > 0:
                missing_atoms[residue] = missing
        # topology : simtk.openmm.app.Topology extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        topology = self._to_delete(topology, delete_atoms)
        topology = self._to_delete_bonds(topology, residue_map)
        topology._numAtoms = len(list(topology.atoms()))

        return(topology, missing_atoms)

    def _to_delete(self, topology, delete_atoms):
        """
        remove instances of atoms and corresponding bonds from topology

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        delete_atoms : list(simtk.openmm.app.topology.Atom)
            atoms from existing residue not in new residue
        Returns
        -------
        topology : simtk.openmm.app.Topology
            extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        """
        # delete_atoms : list(simtk.openmm.app.topology.Atom) atoms from existing residue not in new residue
        # atom : simtk.openmm.app.topology.Atom
        for atom in delete_atoms:
            atom.residue._atoms.remove(atom)
            for bond in topology._bonds:
                if atom in bond:
                    topology._bonds = list(filter(lambda a: a != bond, topology._bonds))
        # topology : simtk.openmm.app.Topology extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        return topology

    def _to_delete_bonds(self, topology, residue_map):
        """
        Remove any bonds between atoms in both new and old residue that do not belong in new residue
        (e.g. breaking the ring in PRO)
        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue)
        Returns
        -------
        topology : simtk.openmm.app.Topology
            extra atoms and bonds from old residue have been deleted, missing atoms in new residue not yet added
        """

        for residue, replace_with in residue_map:
            # template : simtk.openmm.app.topology._TemplateData
            template = self._templates[replace_with]

            old_res_bonds = list()
            # bond : tuple(simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)
            for atom1, atom2 in topology._bonds:
                if atom1.residue == residue or atom2.residue == residue:
                    old_res_bonds.append((atom1.name, atom2.name))
            # make a list of bonds that should exist in new residue
            # template_bonds : list(tuple(str (atom name), str (atom name))) bonds in template
            template_bonds = [(template.atoms[bond[0]].name, template.atoms[bond[1]].name) for bond in template.bonds]
            # add any bonds that exist in template but not in new residue
            for bond in old_res_bonds:
                if bond not in template_bonds and (bond[1],bond[0]) not in template_bonds:
                    topology._bonds = list(filter(lambda a: a != bond, topology._bonds))
                    topology._bonds = list(filter(lambda a: a != (bond[1],bond[0]), topology._bonds))
        return topology

    def _add_new_atoms(self, topology, missing_atoms, residue_map):
        """
        add new atoms (and corresponding bonds) to new residues

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
            extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        missing_atoms : dict
            key : simtk.openmm.app.topology.Residue
            value : list(simtk.openmm.app.topology._TemplateAtomData)
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue, str (three letter residue name of new residue)
        Returns
        -------
        topology : simtk.openmm.app.Topology
            new residue has all correct atoms for desired mutation
        """
        # add new atoms to new residues
        # topology : simtk.openmm.app.Topology extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        # missing_atoms : dict, key : simtk.openmm.app.topology.Residue, value : list(simtk.openmm.app.topology._TemplateAtomData)
        # residue_map : list(tuples : simtk.openmm.app.topology.Residue (old residue), str (three letter residue name of new residue))

        # new_atoms : list(simtk.openmm.app.topology.Atom) atoms that have been added to new residue
        new_atoms = list()
        # k : int
        # residue_ent : tuple(simtk.openmm.app.topology.Residue (old residue), str (three letter residue name of new residue))
        for k, residue_ent in enumerate(residue_map):
            # residue : simtk.openmm.app.topology.Residue (old residue) BUG : wasn't this editing the residue in place what is old and new map
            residue = residue_ent[0]
            # replace_with : str (three letter residue name of new residue)
            replace_with = residue_ent[1]
            # directly edit the simtk.openmm.app.topology.Residue instance
            residue.name = replace_with
            # load template to compare bonds
            # template : simtk.openmm.app.topology._TemplateData
            template = self._templates[replace_with]
            # add each missing atom
            # atom : simtk.openmm.app.topology._TemplateAtomData
            try:
                for atom in missing_atoms[residue]:
                    # new_atom : simtk.openmm.app.topology.Atom
                    new_atom = topology.addAtom(atom.name, atom.element, residue)
                    # new_atoms : list(simtk.openmm.app.topology.Atom)
                    new_atoms.append(new_atom)
            except KeyError:
                pass
            # make a dictionary to map atom names in new residue to atom object
            # new_res_atoms : dict, key : str (atom name) , value : simtk.openmm.app.topology.Atom
            new_res_atoms = dict()
            # atom : simtk.openmm.app.topology.Atom
            for atom in residue.atoms():
                # atom.name : str
                new_res_atoms[atom.name] = atom
            # make a list of bonds already existing in new residue
            # new_res_bonds : list(tuple(str (atom name), str (atom name))) bonds between atoms both within new residue
            new_res_bonds = list()
            # bond : tuple(simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)
            for bond in topology._bonds:
                if bond[0].residue == residue and bond[1].residue == residue:
                    new_res_bonds.append((bond[0].name, bond[1].name))
            # make a list of bonds that should exist in new residue
            # template_bonds : list(tuple(str (atom name), str (atom name))) bonds in template
            template_bonds = [(template.atoms[bond[0]].name, template.atoms[bond[1]].name) for bond in template.bonds]
            # add any bonds that exist in template but not in new residue
            for bond in new_res_bonds:
                if bond not in template_bonds and (bond[1],bond[0]) not in template_bonds:
                    bonded_0 = new_res_atoms[bond[0]]
                    bonded_1 = new_res_atoms[bond[1]]
                    topology._bonds.remove((bonded_0, bonded_1))
            for bond in template_bonds:
                if bond not in new_res_bonds and (bond[1],bond[0]) not in new_res_bonds:
                    # new_bonded_0 : simtk.openmm.app.topology.Atom
                    new_bonded_0 = new_res_atoms[bond[0]]
                    # new_bonded_1 : simtk.openmm.app.topology.Atom
                    new_bonded_1 = new_res_atoms[bond[1]]
                    topology.addBond(new_bonded_0, new_bonded_1)
        topology._numAtoms = len(list(topology.atoms()))

        # add new bonds to the new residues
        return topology

    def _construct_atom_map(self, residue_map, old_topology, index_to_new_residues, new_topology):
        # atom_map : dict, key : int (index of atom in old topology) , value : int (index of same atom in new topology)
        atom_map = dict()

        # atoms with an old_index attribute should be mapped
        # k : int
        # atom : simtk.openmm.app.topology.Atom
        def match_backbone(old_residue, new_residue, atom_name):
            """
            Forcibly including CA and N in the map even if they don't meet
            matching criteria
            """
            found_old_atom = False
            for atom in old_residue.atoms():
                if atom.name == atom_name:
                    old_atom = atom
                    found_old_atom = True
                    break
            assert found_old_atom
            found_new_atom = False
            for atom in new_residue.atoms():
                if atom.name == atom_name:
                    new_atom = atom
                    found_new_atom = True
                    break
            assert found_new_atom
            return new_atom.index, old_atom.index

        modified_residues = dict()
        old_residues = dict()
        for map_entry in residue_map:
            modified_residues[map_entry[0].index] = map_entry[0]
        for residue in old_topology.residues():
            if residue.index in index_to_new_residues.keys():
                old_residues[residue.index] = residue
        for k, atom in enumerate(new_topology.atoms()):
            atom.index=k
            if atom.residue in modified_residues.values():
                continue
            try:
                atom_map[atom.index] = atom.old_index
            except AttributeError:
                pass
        for index in index_to_new_residues.keys():
            old_oemol_res = FFAllAngleGeometryEngine._oemol_from_residue(old_residues[index])
            new_oemol_res = FFAllAngleGeometryEngine._oemol_from_residue(modified_residues[index])
            _ , local_atom_map = self._get_mol_atom_matches(old_oemol_res, new_oemol_res)

            for backbone_name in ['CA','N']:
                new_index, old_index = match_backbone(old_residues[index], modified_residues[index], backbone_name)
                local_atom_map[new_index] = old_index

            atom_map.update(local_atom_map)

        return atom_map

    def _get_mol_atom_matches(self, current_molecule, proposed_molecule):
        """
        Given two molecules, returns the mapping of atoms between them.

        Arguments
        ---------
        current_molecule : openeye.oechem.oemol object
             The current molecule in the sampler
        proposed_molecule : openeye.oechem.oemol object
             The proposed new molecule

        Returns
        -------
        matches : list of match
            list of the matches between the molecules
        """
        oegraphmol_current = oechem.OEGraphMol(current_molecule)
        oegraphmol_proposed = oechem.OEGraphMol(proposed_molecule)
        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)

        atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember | oechem.OEExprOpts_Degree | oechem.OEExprOpts_AtomicNumber
        bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember
        mcs.Init(oegraphmol_current, atomexpr, bondexpr)

        def forcibly_matched(mcs, proposed, atom_name):
            old_atom = list(mcs.GetPattern().GetAtoms(atom_name))
            assert len(old_atom) == 1
            old_atom = old_atom[0]

            new_atom = list(proposed.GetAtoms(atom_name))
            assert len(new_atom) == 1
            new_atom = new_atom[0]
            return old_atom, new_atom

        force_matches = list()
        for matched_atom_name in ['C','O']:
            force_matches.append(oechem.OEHasAtomName(matched_atom_name))

        for force_match in force_matches:
            old_atom, new_atom = forcibly_matched(mcs, oegraphmol_proposed, force_match)
            this_match = oechem.OEMatchPairAtom(old_atom, new_atom)
            assert mcs.AddConstraint(this_match)

        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        unique = True
        matches = [m for m in mcs.Match(oegraphmol_proposed, unique)]
        if len(matches)==0:
            from perses.tests.utils import describe_oemol
            msg = 'No matches found in _get_mol_atom_matches.\n'
            msg += '\n'
            msg += 'oegraphmol_current:\n'
            msg += describe_oemol(oegraphmol_current)
            msg += '\n'
            msg += 'oegraphmol_proposed:\n'
            msg += describe_oemol(oegraphmol_proposed)
            raise Exception(msg)
        match = np.random.choice(matches)
        new_to_old_atom_map = {}
        for matchpair in match.GetAtoms():
            old_index = matchpair.pattern.GetData("topology_index")
            new_index = matchpair.target.GetData("topology_index")
            if old_index < 0 or new_index < 0:
                continue
            new_to_old_atom_map[new_index] = old_index
        return matches, new_to_old_atom_map


    def compute_state_key(self, topology):
        for chain in topology.chains():
            if chain.id == self._chain_id:
                break
        chemical_state_key = ''
        for (index, res) in enumerate(chain._residues):
            if (index > 0):
                chemical_state_key += '-'
            chemical_state_key += res.name

        return chemical_state_key

class PointMutationEngine(PolymerProposalEngine):
    """
    A ProposalEngine for allowing point mutations of a biopolymer component of the system to be specified.

    Note: When instantiating this class, be sure to include cap residues ('ACE', 'NME') in the input topology.
    Otherwise, mutating the first residue will result in a residue template mismatch.
    See openmmtools/data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb for reference.

    Parameters
    ----------
    wildtype_topology : openmm.app.Topology
        OpenMM Topology containing wild-type biopolymer
    system_generator : SystemGenerator
        System generator to use for generating parameterized System objects
    chain_id : str
        id of the chain to mutate
        (using the first chain with the id, if there are multiple)
    proposal_metadata : dict -- OPTIONAL
        Contains information necessary to initialize proposal engine
    max_point_mutants : int -- OPTIONAL
        default = None
    residues_allowed_to_mutate : list(str) -- OPTIONAL
        default = None
        Contains residue ids
        If not specified, engine assumes all residues (except ACE and NME caps) may be mutated.
    allowed_mutations : list(tuple) -- OPTIONAL
        default = None
        ('residue id to mutate','desired mutant residue name (3-letter code)')
        Example:
            Desired systems are wild type T4 lysozyme, T4 lysozyme L99A, and T4 lysozyme L99A/M102Q
            allowed_mutations = [
                ('99','ALA'),
                ('102','GLN')
            ]
        If this is not specified, the engine will propose a random amino acid at a random location.
    always_change : Boolean -- OPTIONAL, default True
        Have the proposal engine always propose another mutation
        If allowed_mutations is not specified, always_change will require ALL
        point mutations to be different
        The proposal engine will choose number of locations specified by
        max_point_mutants, and will require all of those residues to change
        eg: if old topology included L99A and M102Q, the new proposal cannot
            include L99A OR M102Q
        (This is only relevant in cases where max_point_mutants > 1)
    aggregate : Boolean -- OPTIONAL
        default = False
        Have the proposal engine aggregate mutations.
        If aggregate is set to False, the engine will undo mutants in the current topology such that the next proposal
        if the WT state
        If aggregate is set to True, the engine will not undo the mutants from the current topology, thereby allowing
        each proposal to contain multiple mutations.
    """

    def __init__(self, wildtype_topology, system_generator, chain_id, proposal_metadata=None, max_point_mutants=None, residues_allowed_to_mutate=None, allowed_mutations=None, verbose=False, always_change=True):
        super(PointMutationEngine,self).__init__(system_generator, chain_id, proposal_metadata=proposal_metadata, verbose=verbose, always_change=always_change)

        # Check that provided topology has specified chain.
        chain_ids_in_topology = [ chain.id for chain in wildtype_topology.chains() ]
        if chain_id not in chain_ids_in_topology:
            raise Exception("Specified chain_id '%s' not found in provided wildtype_topology. Choices are: %s" % (chain_id, str(chain_ids_in_topology)))

        self._wildtype = wildtype_topology
        self._max_point_mutants = max_point_mutants
        self._ff = system_generator.forcefield
        self._templates = self._ff._templates
        self._residues_allowed_to_mutate = residues_allowed_to_mutate
        if allowed_mutations is not None:
            for mutation in allowed_mutations:
                mutation.sort()
        self._allowed_mutations = allowed_mutations
        if proposal_metadata is None:
            proposal_metadata = dict()
        self._metadata = proposal_metadata
        if max_point_mutants is None and allowed_mutations is None:
            raise Exception("Must specify either max_point_mutants or allowed_mutations.")
        if max_point_mutants is not None and allowed_mutations is not None:
            warnings.warn("PointMutationEngine: max_point_mutants and allowed_mutations were both specified -- max_point_mutants will be ignored")

    def _choose_mutant(self, topology, metadata):
        chain_id = self._chain_id
        old_key = self.compute_state_key(topology)
        index_to_new_residues = self._undo_old_mutants(topology, chain_id, old_key)
        if self._allowed_mutations is not None:
            allowed_mutations = self._allowed_mutations
            index_to_new_residues = self._choose_mutation_from_allowed(topology, chain_id, allowed_mutations, index_to_new_residues, old_key)
        else:
            # index_to_new_residues : dict, key : int (index) , value : str (three letter residue name)
            index_to_new_residues = self._propose_mutations(topology, chain_id, index_to_new_residues, old_key)
        # metadata['mutations'] : list(str (three letter WT residue name - index - three letter MUT residue name) )
        metadata['mutations'] = self._save_mutations(topology, index_to_new_residues)

        return index_to_new_residues, metadata

    def _undo_old_mutants(self, topology, chain_id, old_key):
        index_to_new_residues = dict()
        if old_key == 'WT':
            return index_to_new_residues
        for chain in topology.chains():
            if chain.id == chain_id:
                break
        residue_id_to_index = {residue.id : residue.index for residue in chain._residues}
        for mutant in old_key.split('-'):
            old_res = mutant[:3]
            residue_id = mutant[3:-3]
            new_res = mutant[-3:]
            index_to_new_residues[residue_id_to_index[residue_id]] = old_res
        return index_to_new_residues

    def _choose_mutation_from_allowed(self, topology, chain_id, allowed_mutations, index_to_new_residues, old_key):
        """
        Used when allowed mutations have been specified
        Assume (for now) uniform probability of selecting each specified mutant

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        chain_id : str
        allowed_mutations : list(list(tuple))
            list of allowed mutant states; each entry in the list is a list because multiple mutations may be desired
            tuple : (str, str) -- residue id and three-letter amino acid code of desired mutant

        Returns
        -------
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        """

        # chain : simtk.openmm.app.topology.Chain
        chain_found = False
        for anychain in topology.chains():
            if anychain.id == chain_id:
                chain = anychain
                chain_found = True
                break
        if not chain_found:
            chains = [ chain.id for chain in topology.chains() ]
            raise Exception("Chain '%s' not found in Topology. Chains present are: %s" % (chain_id, str(chains)))
        residue_id_to_index = {residue.id : residue.index for residue in chain._residues}
        # location_prob : np.array, probability value for each residue location (uniform)
        if self._always_change:
            location_prob = np.array([1.0/len(allowed_mutations) for i in range(len(allowed_mutations)+1)])
            if old_key == 'WT':
                location_prob[len(allowed_mutations)] = 0.0
            else:
                current_mutation = list()
                for mutant in old_key.split('-'):
                    residue_id = mutant[3:-3]
                    new_res = mutant[-3:]
                    current_mutation.append((residue_id,new_res))
                current_mutation.sort()
                location_prob[allowed_mutations.index(current_mutation)] = 0.0
        else:
            location_prob = np.array([1.0/(len(allowed_mutations)+1.0) for i in range(len(allowed_mutations)+1)])
        proposed_location = np.random.choice(range(len(allowed_mutations)+1), p=location_prob)
        if proposed_location == len(allowed_mutations):
            # choose WT
            pass
        else:
            for residue_id, residue_name in allowed_mutations[proposed_location]:
                # original_residue : simtk.openmm.app.topology.Residue
                original_residue = chain._residues[residue_id_to_index[residue_id]]
                if original_residue.name in ['HID','HIE']:
                    original_residue.name = 'HIS'
                if original_residue.name == residue_name:
                    continue
                # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
                index_to_new_residues[residue_id_to_index[residue_id]] = residue_name
                if residue_name == 'HIS':
                    his_state = ['HIE','HID']
                    his_prob = np.array([0.5 for i in range(len(his_state))])
                    his_choice = np.random.choice(range(len(his_state)),p=his_prob)
                    index_to_new_residues[residue_id_to_index[residue_id]] = his_state[his_choice]
                # DEBUG
                if self.verbose: print('Proposed mutation: %s %s %s' % (original_residue.name, residue_id, residue_name))

        # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
        return index_to_new_residues

    def _propose_mutations(self, topology, chain_id, index_to_new_residues, old_key):
        """
        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        chain_id : str
        index_to_new_residues : dict
            contains information to mutate back to WT as starting point for new mutants
        old_key : str
            chemical_state_key for old topology

        Returns
        -------
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        """
        # this shouldn't be here
        aminos = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
        # chain : simtk.openmm.app.topology.Chain
        chain_found = False
        for anychain in topology.chains():
            if anychain.id == chain_id:
                chain = anychain
                if self._residues_allowed_to_mutate is None:
                    # num_residues : int
                    num_residues = len(chain._residues)
                    chain_residues = chain._residues
                chain_found = True
                residue_id_to_index = {residue.id : residue.index for residue in chain._residues}
                break
        if not chain_found:
            chains = [ chain.id for chain in topology.chains() ]
            raise Exception("Chain '%s' not found in Topology. Chains present are: %s" % (chain_id, str(chains)))
        if self._residues_allowed_to_mutate is not None:
            num_residues = len(self._residues_allowed_to_mutate)
            chain_residues = self._mutable_residues(chain)
        # location_prob : np.array, probability value for each residue location (uniform)
        location_prob = np.array([1.0/num_residues for i in range(num_residues)])
        if self._always_change:
            residue_id_to_index = {residue.id : residue.index for residue in chain._residues}
            current_mutation = dict()
            if old_key != 'WT':
                for mutant in old_key.split('-'):
                    residue_id = mutant[3:-3]
                    new_res = mutant[-3:]
                    current_mutation[residue_id] = new_res

        for i in range(self._max_point_mutants):
            # proposed_location : int, index of chosen entry in location_prob
            proposed_location = np.random.choice(range(num_residues), p=location_prob)
            # original_residue : simtk.openmm.app.topology.Residue
            original_residue = chain_residues[proposed_location]
            if original_residue.name in ['HIE','HID']:
                original_residue.name = 'HIS'
            # amino_prob : np.array, probability value for each amino acid option (uniform)
            if self._always_change:
                amino_prob = np.array([1.0/(len(aminos)-1) for l in range(len(aminos))])
                amino_prob[aminos.index(original_residue.name)] = 0.0
            else:
                amino_prob = np.array([1.0/(len(aminos)) for k in range(len(aminos))])
            # proposed_amino_index : int, index of three letter residue name in aminos list
            proposed_amino_index = np.random.choice(range(len(aminos)), p=amino_prob)
            # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
            if self._residues_allowed_to_mutate is not None:
                proposed_location = residue_id_to_index[self._residues_allowed_to_mutate[proposed_location]]
            index_to_new_residues[proposed_location] = aminos[proposed_amino_index]
            if aminos[proposed_amino_index] == 'HIS':
                his_state = ['HIE','HID']
                his_prob = np.array([0.5 for j in range(len(his_state))])
                his_choice = np.random.choice(range(len(his_state)),p=his_prob)
                index_to_new_residues[proposed_location] = his_state[his_choice]
        return index_to_new_residues

    def _mutable_residues(self, chain):
        chain_residues = [residue for residue in chain._residues if residue.id in self._residues_allowed_to_mutate]
        return chain_residues

    def _save_mutations(self, topology, index_to_new_residues):
        """
        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        Returns
        -------
        mutations : list(str)
            XXX-##-XXX
            three letter WT residue name - id - three letter MUT residue name
            id is based on the protein sequence NOT zero-indexed

        """
        return [r.name+'-'+str(r.id)+'-'+index_to_new_residues[r.index] for r in topology.residues() if r.index in index_to_new_residues]

    def compute_state_key(self, topology):
        chemical_state_key = ''
        wildtype = self._wildtype
        for anychain in topology.chains():
            if anychain.id == self._chain_id:
                chain = anychain
                break
        for anywt_chain in wildtype.chains():
            if anywt_chain.id == self._chain_id:
                wt_chain = anywt_chain
                break
        for wt_res, res in zip(wt_chain._residues, chain._residues):
            if wt_res.name != res.name:
                if chemical_state_key:
                    chemical_state_key+='-'
                chemical_state_key+= str(wt_res.name)+str(res.id)+str(res.name)
        if not chemical_state_key:
            chemical_state_key = 'WT'
        return chemical_state_key

class PeptideLibraryEngine(PolymerProposalEngine):
    """

    Arguments
    --------
    system_generator : SystemGenerator
    library : list of strings
        each string is a 1-letter-code list of amino acid sequence
    chain_id : str
        id of the chain to mutate
        (using the first chain with the id, if there are multiple)
    proposal_metadata : dict -- OPTIONAL
        Contains information necessary to initialize proposal engine
    """

    def __init__(self, system_generator, library, chain_id, proposal_metadata=None, verbose=False, always_change=True):
        super(PeptideLibraryEngine,self).__init__(system_generator, chain_id, proposal_metadata=proposal_metadata, verbose=verbose, always_change=always_change)
        self._library = library
        self._ff = system_generator.forcefield
        self._templates = self._ff._templates

    def _choose_mutant(self, topology, metadata):
        """
        Used when library of pepide sequences has been provided
        Assume (for now) uniform probability of selecting each peptide

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        chain_id : str
        allowed_mutations : list(list(tuple))
            list of allowed mutant states; each entry in the list is a list because multiple mutations may be desired
            tuple : (str, str) -- residue id and three-letter amino acid code of desired mutant

        Returns
        -------
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        metadata : dict
            has not been altered
        """
        library = self._library

        index_to_new_residues = dict()

        # chain : simtk.openmm.app.topology.Chain
        chain_id = self._chain_id
        chain_found = False
        for chain in topology.chains():
            if chain.id == chain_id:
                chain_found = True
                break
        if not chain_found:
            chains = [ chain.id for chain in topology.chains() ]
            raise Exception("Chain '%s' not found in Topology. Chains present are: %s" % (chain_id, str(chains)))
        # location_prob : np.array, probability value for each residue location (uniform)
        location_prob = np.array([1.0/len(library) for i in range(len(library))])
        proposed_location = np.random.choice(range(len(library)), p=location_prob)
        for residue_index, residue_one_letter in enumerate(library[proposed_location]):
            # original_residue : simtk.openmm.app.topology.Residue
            original_residue = chain._residues[residue_index]
            residue_name = self._one_to_three_letter_code(residue_one_letter)
            if original_residue.name == residue_name:
                continue
            # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
            index_to_new_residues[residue_index] = residue_name
            if residue_name == 'HIS':
                his_state = ['HIE','HID']
                his_prob = np.array([0.5 for i in range(len(his_state))])
                his_choice = np.random.choice(range(len(his_state)),p=his_prob)
                index_to_new_residues[residue_index] = his_state[his_choice]

        # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
        return index_to_new_residues, metadata

    def _one_to_three_letter_code(self, residue_one_letter):
        three_letter_code = {
            'A' : 'ALA',
            'C' : 'CYS',
            'D' : 'ASP',
            'E' : 'GLU',
            'F' : 'PHE',
            'G' : 'GLY',
            'H' : 'HIS',
            'I' : 'ILE',
            'K' : 'LYS',
            'L' : 'LEU',
            'M' : 'MET',
            'N' : 'ASN',
            'P' : 'PRO',
            'Q' : 'GLN',
            'R' : 'ARG',
            'S' : 'SER',
            'T' : 'THR',
            'V' : 'VAL',
            'W' : 'TRP',
            'Y' : 'TYR'
        }
        return three_letter_code[residue_one_letter]

class SystemGenerator(object):
    """
    This is a utility class to generate OpenMM Systems from
    topology objects.

    Parameters
    ----------
    forcefields_to_use : list of string
        List of the names of ffxml files that will be used in system creation.
    forcefield_kwargs : dict of arguments to createSystem, optional
        Allows specification of various aspects of system creation.
    metadata : dict, optional
        Metadata associated with the SystemGenerator.
    use_antechamber : bool, optional, default=True
        If True, will add the GAFF residue template generator.
    barostat : MonteCarloBarostat, optional, default=None
        If provided, a matching barostat will be added to the generated system.
    """

    def __init__(self, forcefields_to_use, forcefield_kwargs=None, metadata=None, use_antechamber=True, barostat=None):
        self._forcefield_xmls = forcefields_to_use
        self._forcefield_kwargs = forcefield_kwargs if forcefield_kwargs is not None else {}
        self._forcefield = app.ForceField(*self._forcefield_xmls)
        if use_antechamber:
            self._forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)
        if 'removeCMMotion' not in self._forcefield_kwargs:
            self._forcefield_kwargs['removeCMMotion'] = False
        self._barostat = None
        if barostat is not None:
            pressure = barostat.getDefaultPressure()
            if hasattr(barostat, 'getDefaultTemperature'):
                temperature = barostat.getDefaultTemperature()
            else:
                temperature = barostat.getTemperature()
            frequency = barostat.getFrequency()
            self._barostat = (pressure, temperature, frequency)

    def getForceField(self):
        """
        Return the associated ForceField object.

        Returns
        -------
        forcefield : simtk.openmm.app.ForceField
            The current ForceField object.
        """
        return self._forcefield

    def build_system(self, new_topology):
        """
        Build a system from the new_topology, adding templates
        for the molecules in oemol_list

        Parameters
        ----------
        new_topology : simtk.openmm.app.Topology object
            The topology of the system

        Returns
        -------
        new_system : openmm.System
            A system object generated from the topology
        """
        _logger.info('Generating System...')
        timer_start = time.time()

        try:
            system = self._forcefield.createSystem(new_topology, **self._forcefield_kwargs)
        except Exception as e:
            from simtk import unit
            nparticles = sum([1 for atom in new_topology.atoms()])
            positions = unit.Quantity(np.zeros([nparticles,3], np.float32), unit.angstroms)
            # Write PDB file of failed topology
            from simtk.openmm.app import PDBFile
            outfile = open('BuildSystem-failure.pdb', 'w')
            pdbfile = PDBFile.writeFile(new_topology, positions, outfile)
            outfile.close()
            msg = str(e)
            import traceback
            msg += traceback.format_exc(e)
            msg += "\n"
            msg += "PDB file written as 'BuildSystem-failure.pdb'"
            raise Exception(msg)

        # Add barostat if requested.
        if self._barostat is not None:
            MAXINT = np.iinfo(np.int32).max
            barostat = openmm.MonteCarloBarostat(*self._barostat)
            seed = np.random.randint(MAXINT)
            barostat.setRandomNumberSeed(seed)
            system.addForce(barostat)

        # DEBUG: See if any torsions have duplicate atoms.
        #from perses.tests.utils import check_system
        #check_system(system)

        _logger.info('System generation took %.3f s' % (time.time() - timer_start))

        return system

    @property
    def ffxmls(self):
        return self._forcefield_xmls

    @property
    def forcefield(self):
        return self._forcefield

class SmallMoleculeSetProposalEngine(ProposalEngine):
    """
    This class proposes new small molecules from a prespecified set. It uses
    uniform proposal probabilities, but can be extended. The user is responsible
    for providing a list of smiles that can be interconverted! The class includes
    extra functionality to assist with that (it is slow).

    Parameters
    ----------
    list_of_smiles : list of string
        list of smiles that will be sampled
    system_generator : SystemGenerator object
        SystemGenerator initialized with the appropriate forcefields
    residue_name : str
        The name that will be used for small molecule residues in the topology
    proposal_metadata : dict
        metadata for the proposal engine
    storage : NetCDFStorageView, optional, default=None
        If specified, write statistics to this storage
    """

    def __init__(self, list_of_smiles, system_generator, residue_name='MOL',
                 atom_expr=None, bond_expr=None, proposal_metadata=None, storage=None,
                 always_change=True, atom_map=None):

        # Default atom and bond expressions for MCSS
        self.atom_expr = atom_expr or DEFAULT_ATOM_EXPRESSION
        self.bond_expr = bond_expr or DEFAULT_BOND_EXPRESSION
        self._allow_ring_breaking = True # allow ring breaking

        # Canonicalize all SMILES strings
        self._smiles_list = [SmallMoleculeSetProposalEngine.canonicalize_smiles(smiles) for smiles in set(list_of_smiles)]

        self._n_molecules = len(self._smiles_list)

        self._residue_name = residue_name
        self._generated_systems = dict()
        self._generated_topologies = dict()
        self._matches = dict()

        self._storage = None
        if storage is not None:
            self._storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        self._probability_matrix = self._calculate_probability_matrix(self._smiles_list)

        self._atom_map = atom_map

        super(SmallMoleculeSetProposalEngine, self).__init__(system_generator, proposal_metadata=proposal_metadata, always_change=always_change)

    def propose(self, current_system, current_topology, current_mol=None, proposed_mol=None, current_metadata=None):
        """
        Propose the next state, given the current state

        Parameters
        ----------
        current_system : openmm.System object
            the system of the current state
        current_topology : app.Topology object
            the topology of the current state
        current_metadata : dict
            dict containing current smiles as a key
        current_mol : OEMol, optional, default=None
            If specified, use this OEMol instead of converting from topology
        proposed_mol : OEMol, optional, default=None
            If specified, use this OEMol instead of converting from topology

        Returns
        -------
        proposal : TopologyProposal object
           topology proposal object
        """
        # Determine SMILES string for current small molecule
        if current_mol is None:
            current_mol_smiles, current_mol = self._topology_to_smiles(current_topology)
        else:
            # TODO: Make sure we're using canonical mol to smiles conversion
            current_mol_smiles = oechem.OEMolToSmiles(current_mol)

        # Remove the small molecule from the current Topology object
        current_receptor_topology = self._remove_small_molecule(current_topology)

        # Find the initial atom index of the small molecule in the current topology
        old_mol_start_index, len_old_mol = self._find_mol_start_index(current_topology)

        # Determine atom indices of the small molecule in the current topology
        old_alchemical_atoms = range(old_mol_start_index, len_old_mol)

        # Select the next molecule SMILES given proposal probabilities
        if proposed_mol is None:
            proposed_mol_smiles, proposed_mol, logp_proposal = self._propose_molecule(current_system, current_topology, current_mol_smiles)
        else:
            # TODO: Make sure we're using canonical mol to smiles conversion
            proposed_mol_smiles = oechem.OEMolToSmiles(current_mol)
            logp_proposal = 0.0

        # Build the new Topology object, including the proposed molecule
        new_topology = self._build_new_topology(current_receptor_topology, proposed_mol)
        new_mol_start_index, len_new_mol = self._find_mol_start_index(new_topology)

        # Generate an OpenMM System from the proposed Topology
        new_system = self._system_generator.build_system(new_topology)

        # Determine atom mapping between old and new molecules
        if not self._atom_map:
            mol_atom_map = self._get_mol_atom_map(current_mol, proposed_mol, atom_expr=self.atom_expr,
                                                  bond_expr=self.bond_expr, verbose=self.verbose,
                                                  allow_ring_breaking=self._allow_ring_breaking)
        else:
            mol_atom_map = self._atom_map

        # Adjust atom mapping indices for the presence of the receptor
        adjusted_atom_map = {}
        for (key, value) in mol_atom_map.items():
            adjusted_atom_map[key+new_mol_start_index] = value + old_mol_start_index

        # Incorporate atom mapping of all environment atoms
        old_mol_offset = len_old_mol
        for i in range(new_mol_start_index):
            if i >= old_mol_start_index:
                old_idx = i + old_mol_offset
            else:
                old_idx = i
            adjusted_atom_map[i] = old_idx

        # Create the TopologyProposal onbject
        proposal = TopologyProposal(logp_proposal=logp_proposal, new_to_old_atom_map=adjusted_atom_map,
            old_topology=current_topology, new_topology=new_topology,
            old_system=current_system, new_system=new_system,
            old_alchemical_atoms=old_alchemical_atoms,
            old_chemical_state_key=current_mol_smiles, new_chemical_state_key=proposed_mol_smiles)

        ndelete = proposal.old_system.getNumParticles() - len(proposal.old_to_new_atom_map.keys())
        ncreate = proposal.new_system.getNumParticles() - len(proposal.old_to_new_atom_map.keys())
        _logger.info('Proposed transformation would delete %d atoms and create %d atoms.' % (ndelete, ncreate))

        return proposal

    @staticmethod
    def canonicalize_smiles(smiles):
        """
        Convert a SMILES string into canonical isomeric smiles

        Parameters
        ----------
        smiles : str
            Any valid SMILES for a molecule

        Returns
        -------
        iso_can_smiles : str
            OpenEye isomeric canonical smiles corresponding to the input
        """
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        oechem.OEAddExplicitHydrogens(mol)
        iso_can_smiles = oechem.OECreateSmiString(mol, OESMILES_OPTIONS)
        return iso_can_smiles

    def _topology_to_smiles(self, topology):
        """
        Get the smiles string corresponding to a specific residue in an
        OpenMM Topology

        Parameters
        ----------
        topology : app.Topology
            The topology containing the molecule of interest

        Returns
        -------
        smiles_string : string
            an isomeric canonicalized SMILES string representing the molecule
        oemol : oechem.OEMol object
            molecule
        """
        molecule_name = self._residue_name

        matching_molecules = [res for res in topology.residues() if res.name[:3] == molecule_name[:3]]  # Find residue in topology by searching for residues with name "MOL"
        if len(matching_molecules) != 1:
            raise ValueError("More than one residue with the same name!")
        mol_res = matching_molecules[0]
        oemol = forcefield_generators.generateOEMolFromTopologyResidue(mol_res)
        smiles_string = oechem.OECreateSmiString(oemol, OESMILES_OPTIONS)
        final_smiles_string = smiles_string
        return final_smiles_string, oemol

    def compute_state_key(self, topology):
        """
        Given a topology, come up with a state key string.
        For this class, the state key is an isomeric canonical SMILES.

        Parameters
        ----------
        topology : app.Topology object
            The topology object in question.

        Returns
        -------
        chemical_state_key : str
            isomeric canonical SMILES

        """
        chemical_state_key, _ = self._topology_to_smiles(topology)
        return chemical_state_key

    def _find_mol_start_index(self, topology):
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
        resname = self._residue_name
        mol_residues = [res for res in topology.residues() if res.name[:3]==resname[:3]]  # Find the residue by searching for residues with "MOL"
        if len(mol_residues)!=1:
            raise ValueError("There must be exactly one residue with a specific name in the topology. Found %d residues with name '%s'" % (len(mol_residues), resname))
        mol_residue = mol_residues[0]
        atoms = list(mol_residue.atoms())
        mol_start_idx = atoms[0].index
        return mol_start_idx, len(list(atoms))

    def _build_new_topology(self, current_receptor_topology, oemol_proposed):
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
        _logger.info('Building new Topology object...')
        timer_start = time.time()

        oemol_proposed.SetTitle(oemol_proposed.GetTitle())
        mol_topology = forcefield_generators.generateTopologyFromOEMol(oemol_proposed)
        new_topology = app.Topology()
        append_topology(new_topology, current_receptor_topology)
        append_topology(new_topology, mol_topology)
        # Copy periodic box vectors.
        if current_receptor_topology._periodicBoxVectors != None:
            new_topology._periodicBoxVectors = copy.deepcopy(current_receptor_topology._periodicBoxVectors)

        _logger.info('Topology generation took %.3f s' % (time.time() - timer_start))

        return new_topology

    def _remove_small_molecule(self, topology):
        """
        This method removes the small molecule parts of a topology from
        a protein+small molecule complex based on the name of the
        small molecule residue

        Parameters
        ----------
        topology : app.Topology
            Topology with the appropriate residue name.

        Returns
        -------
        receptor_topology : app.Topology
            Topology without small molecule
        """
        receptor_topology = app.Topology()
        append_topology(receptor_topology, topology, exclude_residue_name=self._residue_name)
        # Copy periodic box vectors.
        if topology._periodicBoxVectors != None:
            receptor_topology._periodicBoxVectors = copy.deepcopy(topology._periodicBoxVectors)
        return receptor_topology

    @staticmethod
    def _get_mol_atom_map(current_molecule, proposed_molecule, atom_expr=None, bond_expr=None, verbose=False, allow_ring_breaking=True):
        """
        Given two molecules, returns the mapping of atoms between them using the match with the greatest number of atoms

        Arguments
        ---------
        current_molecule : openeye.oechem.oemol object
             The current molecule in the sampler
        proposed_molecule : openeye.oechem.oemol object
             The proposed new molecule
        allow_ring_breaking : bool, optional, default=True
             If False, will check to make sure rings are not being broken or formed.

        Returns
        -------
        matches : list of match
            list of the matches between the molecules
        """
        _logger.info('Generating atom map...')
        timer_start = time.time()

        atom_expr = atom_expr or DEFAULT_ATOM_EXPRESSION
        bond_expr = bond_expr or DEFAULT_BOND_EXPRESSION

        oegraphmol_current = oechem.OEGraphMol(current_molecule) # pattern molecule
        oegraphmol_proposed = oechem.OEGraphMol(proposed_molecule) # target molecule
        #mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)
        mcs.Init(oegraphmol_current, atom_expr, bond_expr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        unique = True
        matches = [m for m in mcs.Match(oegraphmol_proposed, unique)]

        def enumerate_cycle_basis(molecule):
            """Enumerate a closed cycle basis of bonds in molecule.

            This uses cycle_basis from NetworkX:
            https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis

            Parameters
            ----------
            molecule : OEMol
                The molecule for a closed cycle basis of Bonds is to be identified

            Returns
            -------
            bond_cycle_basis : list of list of OEBond
                bond_cycle_basis[cycle_index] is a list of OEBond objects that define a cycle in the basis
                You can think of these as the minimal spanning set of ring systems to check.
            """
            import networkx as nx
            g = nx.Graph()
            for atom in molecule.GetAtoms():
                g.add_node(atom.GetIdx())
            for bond in molecule.GetBonds():
                g.add_edge(bond.GetBgnIdx(), bond.GetEndIdx(), bond=bond)
            bond_cycle_basis = list()
            for cycle in nx.cycle_basis(g):
                bond_cycle = list()
                for i in range(len(cycle)):
                    atom_index_1 = cycle[i]
                    atom_index_2 = cycle[(i+1)%len(cycle)]
                    edge = g.edges[atom_index_1,atom_index_2]
                    bond = edge['bond']
                    bond_cycle.append(bond)
                bond_cycle_basis.append(bond_cycle)
            return bond_cycle_basis

        def enumerate_ring_bonds(molecule, ring_membership, ring_index):
            """Enumerate OEBond objects in ring."""
            for bond in molecule.GetBonds():
                if (ring_membership[bond.GetBgnIdx()] == ring_index) and (ring_membership[bond.GetEndIdx()] == ring_index):
                    yield bond

        def breaks_rings_in_transformation(molecule1, molecule2, atom_map):
            """Return True if the transformation from molecule1 to molecule2 breaks rings.

            Parameters
            ----------
            molecule1 : OEMol
                Initial molecule whose rings are to be checked for not being broken
            molecule2 : OEMol
                Final molecule
            atom_map : dict of OEAtom : OEAtom
                atom_map[molecule1_atom] is the corresponding molecule2 atom
            """
            for cycle in enumerate_cycle_basis(molecule1):
                for bond in cycle:
                    # All bonds in this cycle must also be present in molecule2
                    if not ((bond.GetBgn() in atom_map) and (bond.GetEnd() in atom_map)):
                        return True # there are no corresponding atoms in molecule2
                    if not atom_map[bond.GetBgn()].GetBond(atom_map[bond.GetEnd()]):
                        return True # corresponding atoms have no bond in molecule2
            return False # no rings in molecule1 are broken in molecule2

        def preserves_rings(match):
            """Returns True if the transformation allows ring systems to be broken or created."""
            pattern_atoms = { atom.GetIdx() : atom for atom in oegraphmol_current.GetAtoms() }
            target_atoms = { atom.GetIdx() : atom for atom in oegraphmol_proposed.GetAtoms() }

            pattern_to_target_map = { pattern_atoms[matchpair.pattern.GetIdx()] : target_atoms[matchpair.target.GetIdx()] for matchpair in match.GetAtoms() }
            if breaks_rings_in_transformation(oegraphmol_current, oegraphmol_proposed, pattern_to_target_map):
                return False

            target_to_pattern_map = { target_atoms[matchpair.target.GetIdx()] : pattern_atoms[matchpair.pattern.GetIdx()] for matchpair in match.GetAtoms() }
            if breaks_rings_in_transformation(oegraphmol_proposed, oegraphmol_current, target_to_pattern_map):
                return False

            return True

        if allow_ring_breaking is False:
            # Filter the matches to remove any that allow ring breaking
            matches = [m for m in matches if preserves_rings(m)]

        if not matches:
            return {}
        match = max(matches, key=lambda m: m.NumAtoms())
        new_to_old_atom_map = {}
        for matchpair in match.GetAtoms():
            old_index = matchpair.pattern.GetIdx()
            new_index = matchpair.target.GetIdx()

            old_atom = current_molecule.GetAtom(oechem.OEHasAtomIdx(old_index))
            new_atom = proposed_molecule.GetAtom(oechem.OEHasAtomIdx(new_index))

            #Check if a hydrogen was mapped to a non-hydroden (basically the xor of is_h_a and is_h_b)
            if (old_atom.GetAtomicNum() == 1) != (new_atom.GetAtomicNum() == 1):
                continue

            # If the above is not true, then they are either both hydrogens or both not hydrogens
            elif old_atom.GetAtomicNum() == 1:
                bond = list(old_atom.GetBonds())[0] # There is only one for hydrogen
                bgn = bond.GetBgn()
                end = bond.GetEnd()

                # Is this atom the beginning of the bond?
                if bgn.GetIdx() == old_index:
                    other_atom = end
                else:
                    other_atom = bgn

                new_bond = list(new_atom.GetBonds())[0]
                new_bgn = new_bond.GetBgn()
                new_end = new_bond.GetEnd()

                if new_bgn.GetIdx() == new_index:
                    new_other_atom = new_end
                else:
                    new_other_atom = new_bgn

                if new_other_atom.GetAtomicNum() != other_atom.GetAtomicNum():
                    continue


            new_to_old_atom_map[new_index] = old_index

        _logger.info('Atom map took %.3f s' % (time.time() - timer_start))
        return new_to_old_atom_map

    def _propose_molecule(self, system, topology, molecule_smiles, exclude_self=False):
        """
        Propose a new molecule given the current molecule.

        The current scheme uses a probability matrix computed via _calculate_probability_matrix.

        Arguments
        ---------
        system : simtk.openmm.System object
            The current system
        topology : simtk.openmm.app.Topology object
            The current topology
        positions : [n, 3] np.ndarray of floats (Quantity nm)
            The current positions of the system
        molecule_smiles : string
            The current molecule smiles
        exclude_self : bool, optional, default=True
            If True, exclude self-transitions

        Returns
        -------
        proposed_mol_smiles : str
             The SMILES of the proposed molecule
        mol : oechem.OEMol
            The next molecule to simulate
        logp_proposal : float
            contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
            log [P(Mold | Mnew) / P(Mnew | Mold)]
        """
        # Compute contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
        # log [P(Mold | Mnew) / P(Mnew | Mold)]

        # Retrieve the current molecule index
        try:
            current_smiles_idx = self._smiles_list.index(molecule_smiles)
        except ValueError as e:
            msg = "Current SMILES string '%s' not found in canonical molecule set.\n"
            msg += "Molecule set: %s" % self._smiles_list
            raise Exception(msg)

        # Propose a new molecule
        molecule_probabilities = self._probability_matrix[current_smiles_idx, :]
        proposed_smiles_idx = np.random.choice(range(len(self._smiles_list)), p=molecule_probabilities)
        reverse_probability = self._probability_matrix[proposed_smiles_idx, current_smiles_idx]
        forward_probability = molecule_probabilities[proposed_smiles_idx]
        proposed_smiles = self._smiles_list[proposed_smiles_idx]
        logp = np.log(reverse_probability) - np.log(forward_probability)
        from perses.tests.utils import smiles_to_oemol
        proposed_mol = smiles_to_oemol(proposed_smiles, "MOL_%d" %proposed_smiles_idx)
        return proposed_smiles, proposed_mol, logp

    def _calculate_probability_matrix(self, molecule_smiles_list):
        """
        Calculate the matrix of probabilities of choosing A | B
        based on normalized MCSS overlap. Does not check for torsions!
        Parameters
        ----------
        molecule_smiles_list : list of str
            list of molecules to be potentially selected

        Returns
        -------
        probability_matrix : [n, n] np.ndarray
            probability_matrix[Mold, Mnew] is the probability of choosing molecule Mnew given the current molecule is Mold

        """
        n_smiles = len(molecule_smiles_list)
        probability_matrix = np.zeros([n_smiles, n_smiles])
        for i in range(n_smiles):
            for j in range(i):
                current_mol = oechem.OEMol()
                proposed_mol = oechem.OEMol()
                oechem.OESmilesToMol(current_mol, molecule_smiles_list[i])
                oechem.OESmilesToMol(proposed_mol, molecule_smiles_list[j])
                atom_map = self._get_mol_atom_map(current_mol, proposed_mol, atom_expr=self.atom_expr, bond_expr=self.bond_expr)
                if not atom_map:
                    n_atoms_matching = 0
                    continue
                n_atoms_matching = len(atom_map.keys())
                probability_matrix[i, j] = n_atoms_matching
                probability_matrix[j, i] = n_atoms_matching
        #normalize the rows:
        for i in range(n_smiles):
            row_sum = np.sum(probability_matrix[i, :])
            try:
                probability_matrix[i, :] /= row_sum
            except ZeroDivisionError:
                print("One molecule is completely disconnected!")
                raise

        if self._storage:
            self._storage.write_object('molecule_smiles_list', molecule_smiles_list)
            self._storage.write_array('probability_matrix', probability_matrix)

        return probability_matrix

    @property
    def chemical_state_list(self):
         return self._smiles_list

    @staticmethod
    def clean_molecule_list(smiles_list, atom_opts, bond_opts):
        """
        A utility function to determine which molecules can be proposed
        from any other molecule.

        Parameters
        ----------
        smiles_list
        atom_opts
        bond_opts

        Returns
        -------
        safe_smiles
        removed_smiles
        """
        from perses.tests.utils import smiles_to_topology
        import itertools
        from perses.rjmc.geometry import ProposalOrderTools
        from perses.tests.test_geometry_engine import oemol_to_openmm_system
        safe_smiles = set()
        smiles_pairs = set()
        smiles_set = set(smiles_list)

        for mol1, mol2 in itertools.combinations(smiles_list, 2):
            smiles_pairs.add((mol1, mol2))

        for smiles_pair in smiles_pairs:
            topology_1, mol1 = smiles_to_topology(smiles_pair[0])
            topology_2, mol2 = smiles_to_topology(smiles_pair[1])
            try:
                sys1, pos1, top1 = oemol_to_openmm_system(mol1)
                sys2, pos2, top2 = oemol_to_openmm_system(mol2)
            except:
                continue
            new_to_old_atom_map = SmallMoleculeSetProposalEngine._get_mol_atom_map(mol1, mol2, atom_expr=atom_opts, bond_expr=bond_opts)
            if not new_to_old_atom_map:
                continue
            top_proposal = TopologyProposal(new_topology=top2, old_topology=top1, new_system=sys2, old_system=sys1, new_to_old_atom_map=new_to_old_atom_map, new_chemical_state_key='e', old_chemical_state_key='w')
            proposal_order = ProposalOrderTools(top_proposal)
            try:
                forward_order = proposal_order.determine_proposal_order(direction='forward')
                reverse_order = proposal_order.determine_proposal_order(direction='reverse')
                safe_smiles.add(smiles_pair[0])
                safe_smiles.add(smiles_pair[1])
                _logger.info("Adding %s and %s" % (smiles_pair[0], smiles_pair[1]))
            except NoTorsionError:
                pass
        removed_smiles = smiles_set.difference(safe_smiles)
        return safe_smiles, removed_smiles

class PremappedSmallMoleculeSetProposalEngine(SmallMoleculeSetProposalEngine):
    """
    This proposal engine uses the SmallMoleculeAtomMapper to have all atoms premapped and the proposal distribution pre-formed and checked.
    It is intended to be substantially faster, as well as more robust (having excluded mappings that would not lead to a valid geometry proposal)
    """

    def __init__(self, atom_mapper: SmallMoleculeAtomMapper, system_generator: SystemGenerator, residue_name: str="MOL", storage: NetCDFStorageView=None):
        self._atom_mapper = atom_mapper
        self._atom_mapper.map_all_molecules()
        self._atom_mapper.generate_and_check_proposal_matrix()
        self._proposal_matrix = self._atom_mapper.proposal_matrix

        self._n_molecules = self._atom_mapper.n_molecules

        super(PremappedSmallMoleculeSetProposalEngine, self).__init__(self._atom_mapper.smiles_list, system_generator, residue_name=residue_name,
                 proposal_metadata=None, storage=storage,
                 always_change=True)

    def propose(self, current_system, current_topology, proposed_mol=None, map_index=None, current_metadata=None):
        """
        Propose the next state, given the current state

        Parameters
        ----------
        current_system : openmm.System object
            the system of the current state
        current_topology : app.Topology object
            the topology of the current state
        proposed_mol : oechem.OEMol, optional
            the molecule to propose. If None, choose randomly based on the current molecule
        map_index : int, default None
            The index of the atom map to use. If None, choose randomly. Otherwise, use map idx of map_index mod n_maps
        current_metadata : dict
            dict containing current smiles as a key

        Returns
        -------
        proposal : TopologyProposal object
           topology proposal object
        """
        # Determine SMILES string for current small molecule
        current_mol_smiles, current_mol = self._topology_to_smiles(current_topology)

        # Remove the small molecule from the current Topology object
        current_receptor_topology = self._remove_small_molecule(current_topology)

        # Find the initial atom index of the small molecule in the current topology
        old_mol_start_index, len_old_mol = self._find_mol_start_index(current_topology)

        # Determine atom indices of the small molecule in the current topology
        old_alchemical_atoms = range(old_mol_start_index, len_old_mol)

        # Select the next molecule SMILES given proposal probabilities
        current_mol_index = self._atom_mapper.get_smiles_index(current_mol_smiles)

        #If we aren't specifying a proposed molecule, then randomly propose one:
        if proposed_mol is None:
            #get probability vector for proposal
            proposal_probability = self._proposal_matrix[current_mol_index, :]

            #propose next index
            proposed_index = np.random.choice(range(self._n_molecules), p=proposal_probability)

            #proposal logp
            proposed_logp = np.log(proposal_probability[proposed_index])

            #reverse proposal logp
            reverse_logp = np.log(self._proposal_matrix[proposed_index, current_mol_index])

            #logp overall of proposal
            logp_proposal = reverse_logp - proposed_logp

            #get the oemol corresponding to the proposed molecule:
            proposed_mol_smiles = self._atom_mapper.smiles_list[proposed_index]
            proposed_mol = self._atom_mapper.get_oemol_from_smiles(proposed_mol_smiles)

        else:
            logp_proposal = 0.0
            proposed_mol_smiles = oechem.OECreateSmiString(proposed_mol, OESMILES_OPTIONS)

        #You will get a weird error if you don't assign atom names.
        oechem.OETriposAtomNames(proposed_mol)

        # Build the new Topology object, including the proposed molecule
        new_topology = self._build_new_topology(current_receptor_topology, proposed_mol)
        new_mol_start_index, len_new_mol = self._find_mol_start_index(new_topology)

        # Generate an OpenMM System from the proposed Topology
        new_system = self._system_generator.build_system(new_topology)

        # Determine atom mapping between old and new molecules
        mol_atom_maps = self._atom_mapper.get_atom_maps(current_mol_smiles, proposed_mol_smiles)

        #If no map index is given, just randomly choose one:
        if map_index is None:
            mol_atom_map = np.random.choice(mol_atom_maps)

        #Otherwise, pick the map whose index is the specified index mod n_maps
        else:
            n_atom_maps = len(mol_atom_maps)
            index_to_choose = map_index % n_atom_maps
            mol_atom_map = mol_atom_maps[index_to_choose]

        # Adjust atom mapping indices for the presence of the receptor
        adjusted_atom_map = {}
        for (key, value) in mol_atom_map.items():
            adjusted_atom_map[key+new_mol_start_index] = value + old_mol_start_index

        # Incorporate atom mapping of all environment atoms
        old_mol_offset = len_old_mol
        for i in range(new_mol_start_index):
            if i >= old_mol_start_index:
                old_idx = i + old_mol_offset
            else:
                old_idx = i
            adjusted_atom_map[i] = old_idx

        # Create the TopologyProposal onbject
        proposal = TopologyProposal(logp_proposal=logp_proposal, new_to_old_atom_map=adjusted_atom_map,
            old_topology=current_topology, new_topology=new_topology,
            old_system=current_system, new_system=new_system,
            old_alchemical_atoms=old_alchemical_atoms,
            old_chemical_state_key=current_mol_smiles, new_chemical_state_key=proposed_mol_smiles)

        ndelete = proposal.old_system.getNumParticles() - len(proposal.old_to_new_atom_map.keys())
        ncreate = proposal.new_system.getNumParticles() - len(proposal.old_to_new_atom_map.keys())
        _logger.info('Proposed transformation would delete %d atoms and create %d atoms.' % (ndelete, ncreate))

        return proposal


class TwoMoleculeSetProposalEngine(SmallMoleculeSetProposalEngine):
    """
    Dummy proposal engine that always chooses the new molecule of the two, but uses the base class's atom mapping
    functionality.
    """

    def __init__(self, old_mol, new_mol, system_generator, residue_name='MOL', atom_expr=None, bond_expr=None, proposal_metadata=None, storage=None, always_change=True, atom_map=None):
        self._old_mol_smiles = oechem.OECreateSmiString(old_mol, OESMILES_OPTIONS)
        self._new_mol_smiles = oechem.OECreateSmiString(new_mol, OESMILES_OPTIONS)
        self._old_mol = old_mol
        self._new_mol = new_mol

        super(TwoMoleculeSetProposalEngine, self).__init__([self._old_mol_smiles, self._new_mol_smiles], system_generator, residue_name=residue_name, atom_expr=atom_expr, bond_expr=bond_expr, atom_map=atom_map)

        self._allow_ring_breaking = False # don't allow ring breaking

    def _propose_molecule(self, system, topology, molecule_smiles, exclude_self=False):
        return self._new_mol_smiles, self._new_mol, 0.0

class NullProposalEngine(SmallMoleculeSetProposalEngine):
    """
    Base class for NaphthaleneProposalEngine and ButantProposalEngine
    Not intended for use on its own

    """
    def __init__(self, system_generator, residue_name="MOL", atom_expr=None, bond_expr=None, proposal_metadata=None, storage=None, always_change=True):
        super(NullProposalEngine, self).__init__(list(), system_generator, residue_name=residue_name)
        self._fake_states = ["A","B"]
        self.smiles = ''

    def propose(self, current_system, current_topology, current_metadata=None):
        """
        Custom proposal for NaphthaleneTestSystem will switch from current naphthalene
        "state" (either 'naphthalene-A' or 'naphthalene-B') to the other

        This proposal engine can only be used with input topology of
        naphthalene, and propose() will first confirm naphthalene is residue "MOL"
        The topology may have water but should not have any other
        carbon-containing residue.

        The "new" system and topology are deep copies of the old, but the atom_map
        is custom defined to only match one of the two rings.

        Arguments:
        ----------
        current_system : openmm.System object
            the system of the current state
        current_topology : app.Topology object
            the topology of the current state
        current_metadata : dict, OPTIONAL
            Not implemented

        Returns:
        -------
        proposal : TopologyProposal object
           topology proposal object

        """
        given_smiles = super(NullProposalEngine, self).compute_state_key(current_topology)
        if self.smiles != given_smiles:
            raise(Exception("{0} can only be used with {1} topology; smiles of given topology is: {2}".format(type(self), self.smiles, given_smiles)))

        old_key = current_topology._state_key
        new_key = [key for key in self._fake_states if key != old_key][0]

        new_topology = app.Topology()
        append_topology(new_topology, current_topology)
        new_topology._state_key = new_key
        new_system = copy.deepcopy(current_system)
        atom_map = self._make_skewed_atom_map(current_topology)
        proposal = TopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=current_topology, old_system=current_system, logp_proposal=0.0,
                                                 new_to_old_atom_map=atom_map, old_chemical_state_key=old_key, new_chemical_state_key=new_key)
        return proposal

    def _make_skewed_atom_map(topology):
        return dict()

    def compute_state_key(self, topology):
        """
        For this test system, the topologies for the two states are
        identical; therefore a custom attribute `_state_key` has
        been added to the topology itself, to track whether a switch
        has been accepted

        compute_state_key will return topology._state_key rather than
        SMILES, because SMILES are identical in the two states.
        """
        return topology._state_key


class NaphthaleneProposalEngine(NullProposalEngine):
    """
    Custom ProposalEngine to use with NaphthaleneTestSystem defines two "states"
    of naphthalene, identified 'naphthalene-A' and 'naphthalene-B', which are
    tracked by adding a custom _state_key attribute to the topology

    Generates TopologyProposal from naphthalene to naphthalene, only matching
    one ring such that geometry must rebuild the other

    Can only be used with input topology of naphthalene

    Constructor Arguments:
        system_generator, SystemGenerator object
            SystemGenerator initialized with the appropriate forcefields
        residue_name, OPTIONAL,  str
            Default = "MOL"
            The name that will be used for small molecule residues in the topology
        atom_expr, OPTIONAL, oechem.OEExprOpts
            Default is None
            Currently not implemented -- would dictate how match is defined
        bond_expr, OPTIONAL, oechem.OEExprOpts
            Default is None
            Currently not implemented -- would dictate how match is defined
        proposal_metadata, OPTIONAL, dict
            Default is None
            metadata for the proposal engine
        storage, OPTIONAL, NetCDFStorageView
            Default is None
            If specified, write statistics to this storage layer
        always_change, OPTIONAL, bool
            Default is True
            Currently not implemented -- will always behave as True
            The proposal will always be from the current "state" to the other
            Self proposals will never be made
    """
    def __init__(self, system_generator, residue_name="MOL", atom_expr=None, bond_expr=None, proposal_metadata=None, storage=None, always_change=True):
        super(NaphthaleneProposalEngine, self).__init__(system_generator, residue_name=residue_name, atom_expr=atom_expr, bond_expr=bond_expr, proposal_metadata=proposal_metadata, storage=storage, always_change=always_change)
        self._fake_states = ["naphthalene-A", "naphthalene-B"]
        self.smiles = 'c1ccc2ccccc2c1'

    def _make_skewed_atom_map(self, topology):
        """
        Custom definition for the atom map between naphthalene and naphthalene

        If a regular atom map was constructed (via oechem.OEMCSSearch), all
        atoms would be matched, and the geometry engine would have nothing
        to add.  This method manually finds one of the two naphthalene rings
        and matches it to itself, rotated 180degrees.  The second ring and
        all hydrogens will have to be repositioned by the geometry engine.

        The rings are identified by finding how many other carbons each
        carbon in the topology is bonded to.  This will be 2 for all carbons
        in naphthalene except the two carbon atoms shared by both rings.
        Starting with these 2 shared atoms, a ring is traced by following
        bonds between adjacent carbons until returning to the shared carbons.

        Arguments:
        ----------
        topology : app.Topology object
            topology of naphthalene
            Only one topology is needed, because current and proposed are
            identical
        Returns:
        --------
        atom_map : dict
            maps the atom indices of carbons from one ring back to themselves
        """
        # make dict of carbons in topology to list of carbons they're bonded to
        carbon_bonds = dict()
        for atom in topology.atoms():
            if atom.element == app.element.carbon:
                carbon_bonds[atom] = set()
        for bond in topology.bonds():
            if bond[0].element == app.element.carbon and bond[1].element == app.element.carbon:
                carbon_bonds[bond[0]].add(bond[1])
                carbon_bonds[bond[1]].add(bond[0])

        # identify the rings by finding 2 middle carbons then tracing their bonds
        find_ring = list()
        for carbon, bounds in carbon_bonds.items():
            if len(bounds) == 3:
                find_ring.append(carbon)
        assert len(find_ring) == 2

        add_next = [carbon for carbon in carbon_bonds[find_ring[1]] if carbon not in find_ring]
        while len(add_next) > 0:
            find_ring.append(add_next[0])
            add_next = [carbon for carbon in carbon_bonds[add_next[0]] if carbon not in find_ring]
        assert len(find_ring) == 6

        # map the ring flipped 180
        atom_map = {find_ring[i].index : find_ring[(i+3)%6].index for i in range(6)}
        return atom_map

class ButaneProposalEngine(NullProposalEngine):
    """
    Custom ProposalEngine to use with ButaneTestSystem defines two "states"
    of butane, identified 'butane-A' and 'butane-B', which are
    tracked by adding a custom _state_key attribute to the topology

    Generates TopologyProposal from butane to butane, only matching
    two carbons such that geometry must rebuild the others

    Can only be used with input topology of butane

    Constructor Arguments:
        system_generator, SystemGenerator object
            SystemGenerator initialized with the appropriate forcefields
        residue_name, OPTIONAL,  str
            Default = "MOL"
            The name that will be used for small molecule residues in the topology
        atom_expr, OPTIONAL, oechem.OEExprOpts
            Default is None
            Currently not implemented -- would dictate how match is defined
        bond_expr, OPTIONAL, oechem.OEExprOpts
            Default is None
            Currently not implemented -- would dictate how match is defined
        proposal_metadata, OPTIONAL, dict
            Default is None
            metadata for the proposal engine
        storage, OPTIONAL, NetCDFStorageView
            Default is None
            If specified, write statistics to this storage layer
        always_change, OPTIONAL, bool
            Default is True
            Currently not implemented -- will always behave as True
            The proposal will always be from the current "state" to the other
            Self proposals will never be made
    """
    def __init__(self, system_generator, residue_name="MOL", atom_expr=None, bond_expr=None, proposal_metadata=None, storage=None, always_change=True):
        super(ButaneProposalEngine, self).__init__(system_generator, residue_name=residue_name, atom_expr=atom_expr, bond_expr=bond_expr, proposal_metadata=proposal_metadata, storage=storage, always_change=always_change)
        self._fake_states = ["butane-A", "butane-B"]
        self.smiles = 'CCCC'

    def _make_skewed_atom_map_old(self, topology):
        """
        Custom definition for the atom map between butane and butane

        (disabled)

        MAP:

                   H - C - CH- C - C - H
          H - C - CH - C - CH- H

        If a regular atom map was constructed (via oechem.OEMCSSearch), all
        atoms would be matched, and the geometry engine would have nothing
        to add.  This method manually finds two of the four carbons and
        matches them to each other, rotated 180degrees.  The other two carbons
        and hydrogens will have to be repositioned by the geometry engine.

        The two carbons are chosen by finding the first carbon-carbon bond in
        the topology. Because geometry needs at least 3 atoms in the atom_map,
        one H from each carbon is also found and mapped to the other.

        Arguments:
        ----------
        topology : app.Topology object
            topology of butane
            Only one topology is needed, because current and proposed are
            identical
        Returns:
        --------
        atom_map : dict
            maps the atom indices of 2 carbons to each other
        """
        for bond in topology.bonds():
            if all([atom.element == app.element.carbon for atom in bond]):
                ccbond = bond
                break
        for bond in topology.bonds():
            if ccbond[0] in bond and any([atom.element == app.element.hydrogen for atom in bond]):
                hydrogen0 = [atom for atom in bond if atom.element == app.element.hydrogen][0]
            if ccbond[1] in bond and any([atom.element == app.element.hydrogen for atom in bond]):
                hydrogen1 = [atom for atom in bond if atom.element == app.element.hydrogen][0]

        atom_map = {
            ccbond[0].index : ccbond[1].index,
            ccbond[1].index : ccbond[0].index,
            hydrogen0.index : hydrogen1.index,
            hydrogen1.index : hydrogen0.index,
        }
        return atom_map

    def _make_skewed_atom_map(self, topology):
        """
        Custom definition for the atom map between butane and butane

        MAP:

        C - C - C - C
            C - C - C - C

        Arguments:
        ----------
        topology : app.Topology object
            topology of butane
            Only one topology is needed, because current and proposed are
            identical
        Returns:
        --------
        atom_map : dict
            maps the atom indices of 2 carbons to each other
        """

        carbons = [ atom for atom in topology.atoms() if (atom.element == app.element.carbon) ]
        neighbors = { carbon : set() for carbon in carbons }
        for (atom1,atom2) in topology.bonds():
            if (atom1.element == app.element.carbon) and (atom2.element == app.element.carbon):
                neighbors[atom1].add(atom2)
                neighbors[atom2].add(atom1)
        end_carbons = list()
        for carbon in carbons:
            if len(neighbors[carbon]) == 1:
                end_carbons.append(carbon)

        # Extract linear chain of carbons
        carbon_chain = list()
        last_carbon = end_carbons[0]
        carbon_chain.append(last_carbon)
        finished = False
        while (not finished):
            next_carbons = list(neighbors[last_carbon].difference(carbon_chain))
            if len(next_carbons) == 0:
                finished = True
            else:
                carbon_chain.append(next_carbons[0])
                last_carbon = next_carbons[0]

        atom_map = {
            carbon_chain[0].index : carbon_chain[2].index,
            carbon_chain[1].index : carbon_chain[1].index,
            carbon_chain[2].index : carbon_chain[0].index,
        }
        return atom_map

class PropaneProposalEngine(NullProposalEngine):
    """
    Custom ProposalEngine to use with PropaneTestSystem defines two "states"
    of butane, identified 'propane-A' and 'propane-B', which are
    tracked by adding a custom _state_key attribute to the topology

    Generates TopologyProposal from propane to propane, only matching
    one CH3 and the middle C such that geometry must rebuild the other

    Can only be used with input topology of propane

    Constructor Arguments:
        system_generator, SystemGenerator object
            SystemGenerator initialized with the appropriate forcefields
        residue_name, OPTIONAL,  str
            Default = "MOL"
            The name that will be used for small molecule residues in the topology
        atom_expr, OPTIONAL, oechem.OEExprOpts
            Default is None
            Currently not implemented -- would dictate how match is defined
        bond_expr, OPTIONAL, oechem.OEExprOpts
            Default is None
            Currently not implemented -- would dictate how match is defined
        proposal_metadata, OPTIONAL, dict
            Default is None
            metadata for the proposal engine
        storage, OPTIONAL, NetCDFStorageView
            Default is None
            If specified, write statistics to this storage layer
        always_change, OPTIONAL, bool
            Default is True
            Currently not implemented -- will always behave as True
            The proposal will always be from the current "state" to the other
            Self proposals will never be made
    """
    def __init__(self, system_generator, residue_name="MOL", atom_expr=None, bond_expr=None, proposal_metadata=None, storage=None, always_change=True):
        super(PropaneProposalEngine, self).__init__(system_generator, residue_name=residue_name, atom_expr=atom_expr, bond_expr=bond_expr, proposal_metadata=proposal_metadata, storage=storage, always_change=always_change)
        self._fake_states = ["propane-A", "propane-B"]
        self.smiles = 'CCC'

    def _make_skewed_atom_map(self, topology):
        """
        Custom definition for the atom map between propane and propane

        If a regular atom map was constructed (via oechem.OEMCSSearch), all
        atoms would be matched, and the geometry engine would have nothing
        to add.  This method manually finds CH3-CH2 and matches each atom to
        itself.  The other carbon and three hydrogens will have to be
        repositioned by the geometry engine.

        The two carbons are chosen by finding the first carbon-carbon bond in
        the topology. To minimize the atoms being rebuilt by geometry, all
        hydrogens from each of the selected carbons are also found and
        mapped to themselves.

        Arguments:
        ----------
        topology : app.Topology object
            topology of propane
            Only one topology is needed, because current and proposed are
            identical
        Returns:
        --------
        atom_map : dict
            maps the atom indices of CH3-CH2 to themselves
        """
        atom_map = dict()
        for bond in topology.bonds():
            if all([atom.element == app.element.carbon for atom in bond]):
                ccbond = bond
                break
        for bond in topology.bonds():
            if any([carbon in bond for carbon in ccbond]) and app.element.hydrogen in [atom.element for atom in bond]:
                atom_map[bond[0].index] = bond[0].index
                atom_map[bond[1].index] = bond[1].index
        return atom_map

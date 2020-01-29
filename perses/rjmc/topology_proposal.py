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


# TODO write a mapping-protocol class to handle these options

# weak requirements for mapping atoms == more atoms mapped, more in core
# atoms need to match in aromaticity. Same with bonds.
# maps ethane to ethene, CH3 to NH2, but not benzene to cyclohexane
WEAK_ATOM_EXPRESSION = oechem.OEExprOpts_EqAromatic | oechem.OEExprOpts_EqNotAromatic
WEAK_BOND_EXPRESSION = oechem.OEExprOpts_Aromaticity

# default atom expression, requires same aromaticitiy and hybridization
# bonds need to match in bond order
# ethane to ethene wouldn't map, CH3 to NH2 would map but CH3 to HC=O wouldn't
DEFAULT_ATOM_EXPRESSION = oechem.OEExprOpts_Hybridization | oechem.OEExprOpts_HvyDegree
DEFAULT_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

# strong requires same hybridization AND the same atom type
# bonds are same as default, require them to match in bond order
STRONG_ATOM_EXPRESSION = oechem.OEExprOpts_Hybridization  | oechem.OEExprOpts_HvyDegree | oechem.OEExprOpts_DefaultAtoms
STRONG_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

#specific to proteins
PROTEIN_ATOM_EXPRESSION = oechem.OEExprOpts_Hybridization | oechem.OEExprOpts_EqAromatic
PROTEIN_BOND_EXPRESSION = oechem.OEExprOpts_Aromaticity

################################################################################
# LOGGER
################################################################################

import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("proposal_generator")
_logger.setLevel(logging.WARNING)

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
    new_atoms = {}
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

    def __init__(self, list_of_smiles: List[str], map_strength: str='default', atom_match_expression: int=None, bond_match_expression: int=None, prohibit_hydrogen_mapping: bool=True):

        self._unique_noncanonical_smiles_list = list(set(list_of_smiles))
        self._oemol_dictionary = self._initialize_oemols(self._unique_noncanonical_smiles_list)
        self._unique_smiles_list = list(self._oemol_dictionary.keys())

        self._n_molecules = len(self._unique_smiles_list)

        self._prohibit_hydrogen_mapping = prohibit_hydrogen_mapping

        if atom_match_expression is None:
            _logger.debug(f'map_strength = {map_strength}')
            if map_strength == 'default':
                self._atom_expr = DEFAULT_ATOM_EXPRESSION
            elif map_strength == 'weak':
                self._atom_expr = WEAK_ATOM_EXPRESSION
            elif map_strength == 'strong':
                self._atom_expr = STRONG_ATOM_EXPRESSION
            else:
                _logger.warning(f"atom_match_expression not recognised, setting to default")
                self._atom_expr = DEFAULT_ATOM_EXPRESSION
        else:
            self._atom_expr = atom_expr
            _logger.info(f'Setting the atom expression to user defined: {atom_expr}')
            _logger.info('If map_strength has been set, it will be ignored')

        if bond_match_expression is None:
            if map_strength == 'default':
                self._bond_expr = DEFAULT_BOND_EXPRESSION
            elif map_strength == 'weak':
                self._bond_expr = WEAK_BOND_EXPRESSION
            elif map_strength == 'strong':
                self._bond_expr = STRONG_BOND_EXPRESSION
            else:
                _logger.warning(f"bond_match_expression not recognised, setting to default")
                self._bond_expr = DEFAULT_BOND_EXPRESSION
        else:
            self.bond_expr = bond_expr
            _logger.info(f'Setting the bond expression to user defined: {bond_expr}')
            _logger.info('If map_strength has been set, it will be ignored')

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
            oechem.OEAssignHybridization(molecule)
            oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)

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
    old_residue_name : str
        Name of the old residue
    new_residue_name : str
        Name of the new residue
    _old_networkx_residue : NetworkXMolecule
        networkx molecule of old residue
    _new_networkx_residue : NetworkXMolecule
        networkx molecule of new residue
    metadata : dict
        additional information of interest about the state
    """

    def __init__(self,
                 new_topology=None, new_system=None,
                 old_topology=None, old_system=None,
                 logp_proposal=None,
                 new_to_old_atom_map=None, old_alchemical_atoms=None,
                 old_chemical_state_key=None, new_chemical_state_key=None,
                 old_residue_name='MOL', new_residue_name='MOL',
                 old_networkx_residue = None,
                 new_networkx_residue = None,
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
        self._old_residue_name = old_residue_name
        self._new_residue_name = new_residue_name
        self._new_to_old_atom_map = new_to_old_atom_map
        self._old_to_new_atom_map = {old_atom : new_atom for new_atom, old_atom in new_to_old_atom_map.items()}
        self._unique_new_atoms = list(set(range(self._new_topology.getNumAtoms()))-set(self._new_to_old_atom_map.keys()))
        self._unique_old_atoms = list(set(range(self._old_topology.getNumAtoms()))-set(self._new_to_old_atom_map.values()))
        self._old_alchemical_atoms = set(old_alchemical_atoms) if (old_alchemical_atoms is not None) else {atom for atom in range(old_system.getNumParticles())}
        self._new_alchemical_atoms = set([self._old_to_new_atom_map[old_alch_atom] for old_alch_atom in self._old_alchemical_atoms if old_alch_atom in list(self._new_to_old_atom_map.values())]).union(set(self._unique_new_atoms))
        #self._new_alchemical_atoms = set(self._old_to_new_atom_map.values()).union(self._unique_new_atoms)
        self._old_environment_atoms = set(range(old_system.getNumParticles())) - self._old_alchemical_atoms
        self._new_environment_atoms = set(range(new_system.getNumParticles())) - self._new_alchemical_atoms
        self._metadata = metadata
        self._old_networkx_residue = old_networkx_residue
        self._new_networkx_residue = new_networkx_residue
        self._core_new_to_old_atom_map = {new_atom: old_atom for new_atom, old_atom in self._new_to_old_atom_map.items() if new_atom in self._new_alchemical_atoms and old_atom in self._old_alchemical_atoms}

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
    def old_residue_name(self):
        return self._old_residue_name
    @property
    def new_residue_name(self):
        return self._new_residue_name
    @property
    def metadata(self):
        return self._metadata
    @property
    def core_new_to_old_atom_map(self):
        return self._core_new_to_old_atom_map

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
    Base class for ProposalEngine implementations that modify polymer components of systems.

    This base class is not meant to be invoked directly.
    """

    # TODO: Eliminate 'verbose' option in favor of logging
    # TODO: Document meaning of 'aggregate'
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

        This base class is not meant to be invoked directly.
        """
        from perses.utils.smallmolecules import render_atom_mapping
        _logger.debug(f"Instantiating PolymerProposalEngine")
        super(PolymerProposalEngine,self).__init__(system_generator, proposal_metadata=proposal_metadata, verbose=verbose, always_change=always_change)
        self._chain_id = chain_id # chain identifier defining polymer to be modified
        self._aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL'] # common naturally-occurring amino acid names
                        # Note this does not include PRO since there's a problem with OpenMM's template DEBUG
        self._aggregate = aggregate # ?????????

    def propose(self,
                current_system,
                current_topology,
                current_metadata=None):
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
        topology_proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        local_atom_map_stereo_sidechain : dict
            chirality-corrected map of new_oemol_res to old_oemol_res
        old_oemol_res : openeye.oechem.oemol object
            oemol of the old residue sidechain
        new_oemol_res : openeye.oechem.oemol object
            oemol of the new residue sidechain

        """
        _logger.info(f"\tConducting polymer point mutation proposal...")
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
        _logger.debug(f"\tcomputing state key of old topology...")
        old_chemical_state_key = self.compute_state_key(old_topology)
        _logger.debug(f"\told chemical state key for chain {self._chain_id}: {old_chemical_state_key}")

        # index_to_new_residues : dict, key : int (index) , value : str (three letter name of proposed residue)
        _logger.debug(f"\tchoosing mutant...")
        index_to_new_residues, metadata = self._choose_mutant(old_topology, metadata)
        _logger.debug(f"\t\tindex to new residues: {index_to_new_residues}")

        # residue_map : list(tuples : simtk.openmm.app.topology.Residue (existing residue), str (three letter name of proposed residue))
        _logger.debug(f"\tgenerating residue map...")
        residue_map = self._generate_residue_map(old_topology, index_to_new_residues)
        _logger.debug(f"\t\tresidue map: {residue_map}")

        for (res, new_name) in residue_map:
            if res.name == new_name:
                #remove the index_to_new_residues entries where the topology is already mutated
                del(index_to_new_residues[res.index])
        if len(index_to_new_residues) == 0:
            _logger.debug(f"\t\tno mutation detected in this proposal; generating old proposal")
            atom_map = dict()
            for atom in old_topology.atoms():
                atom_map[atom.index] = atom.index
            if self.verbose: print('PolymerProposalEngine: No changes to topology proposed, returning old system and topology')
            topology_proposal = TopologyProposal(new_topology=old_topology, new_system=old_system, old_topology=old_topology, old_system=old_system, old_chemical_state_key=old_chemical_state_key, new_chemical_state_key=old_chemical_state_key, logp_proposal=0.0, new_to_old_atom_map=atom_map)
            return topology_proposal

        elif len(index_to_new_residues) > 1:
            raise Exception("Attempting to mutate more than one residue at once: ", index_to_new_residues, " The geometry engine cannot handle this.")

        chosen_res_index = list(index_to_new_residues.keys())[0]
        # Add modified_aa property to residues in old topology
        for res in old_topology.residues():
            res.modified_aa = True if res.index in index_to_new_residues.keys() else False

        _logger.debug(f"\tfinal index_to_new_residues: {index_to_new_residues}")
        _logger.debug(f"\tfinding excess and missing atoms/bonds...")
        # Identify differences between old topology and proposed changes
        # excess_atoms : list(simtk.openmm.app.topology.Atom) atoms from existing residue not in new residue
        # excess_bonds : list(tuple (simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)) bonds from existing residue not in new residue
        # missing_bonds : list(tuple (simtk.openmm.app.topology._TemplateAtomData, simtk.openmm.app.topology._TemplateAtomData)) bonds from new residue not in existing residue
        excess_atoms, excess_bonds, missing_atoms, missing_bonds = self._identify_differences(old_topology, residue_map)

        # Delete excess atoms and bonds from old topology
        excess_atoms_bonds = excess_atoms + excess_bonds
        _logger.debug(f"\t excess atoms bonds: {excess_atoms_bonds}")
        new_topology = self._delete_atoms(old_topology, excess_atoms_bonds)

        # Add missing atoms and bonds to new topology
        new_topology = self._add_new_atoms(new_topology, missing_atoms, missing_bonds, residue_map)

        # index_to_new_residues : dict, key : int (index) , value : str (three letter name of proposed residue)
        _logger.debug(f"\tconstructing atom map for TopologyProposal...")
        atom_map, old_res_to_oemol_map, new_res_to_oemol_map, local_atom_map_stereo_sidechain, old_oemol_res, new_oemol_res, old_oemol_res_copy, new_oemol_res_copy  = self._construct_atom_map(residue_map, old_topology, index_to_new_residues, new_topology)

        _logger.debug(f"\tadding indices of the 'C' backbone atom in the next residue and the 'N' atom in the previous")
        _logger.debug(f"\t{list(index_to_new_residues.keys())[0]}")
        extra_atom_map = self._find_adjacent_special_atoms(old_topology, new_topology, list(index_to_new_residues.keys())[0])

        #now to add all of the other residue atoms to the atom map...
        all_other_residues_new = [res for res in new_topology.residues() if res.index != list(index_to_new_residues.keys())[0]]
        all_other_residues_old = [res for res in old_topology.residues() if res.index != list(index_to_new_residues.keys())[0]]

        all_other_atoms_map = {}
        for res_new, res_old in zip(all_other_residues_new, all_other_residues_old):
            assert res_new.name == res_old.name, f"all other residue names do not match"
            all_other_atoms_map.update({atom_new.index: atom_old.index for atom_new, atom_old in zip(res_new.atoms(), res_old.atoms())})

        # new_chemical_state_key : str
        new_chemical_state_key = self.compute_state_key(new_topology)
        # new_system : simtk.openmm.System

        # Copy periodic box vectors from current topology
        new_topology.setPeriodicBoxVectors(current_topology.getPeriodicBoxVectors())

        # Build system
        new_system = self._system_generator.build_system(new_topology)

        #make constraint repairs
        atom_map = SmallMoleculeSetProposalEngine._constraint_repairs(atom_map, old_system, new_system, old_topology, new_topology)
        _logger.debug(f"\tafter constraint repairs, the atom map is as such: {atom_map}")

        _logger.debug(f"\tadding all env atoms to the atom map...")
        atom_map.update(all_other_atoms_map)

        old_res_names = [res.name for res in old_topology.residues() if res.index == list(index_to_new_residues.keys())[0]]
        assert len(old_res_names) == 1, f"no old res name match found"
        old_res_name = old_res_names[0]
        _logger.debug(f"\told res name: {old_res_name}")
        new_res_name = list(index_to_new_residues.values())[0]


        # Adjust logp_propose based on HIS presence
        # his_residues = ['HID', 'HIE']
        # old_residue = residue_map[0][0]
        # proposed_residue = residue_map[0][1]
        # if old_residue.name in his_residues and proposed_residue not in his_residues:
        #     logp_propose = math.log(2)
        # elif old_residue.name not in his_residues and proposed_residue in his_residues:
        #     logp_propose = math.log(0.5)
        # else:
        #     logp_propose = 0.0

        #we should be able to check the system to make sure that all of the core atoms

        # Create TopologyProposal.
        current_res = [res for res in current_topology.residues() if res.index == chosen_res_index][0]
        proposed_res = [res for res in new_topology.residues() if res.index == chosen_res_index][0]
        old_networkx_molecule = NetworkXMolecule(mol_oemol = old_oemol_res_copy, mol_residue = current_res, residue_to_oemol_map = old_res_to_oemol_map)
        new_networkx_molecule = NetworkXMolecule(mol_oemol = new_oemol_res_copy, mol_residue = proposed_res, residue_to_oemol_map = new_res_to_oemol_map)
        topology_proposal = TopologyProposal(logp_proposal = 0.,
                                             new_to_old_atom_map = atom_map,
                                             old_topology = old_topology,
                                             new_topology = new_topology,
                                             old_system = old_system,
                                             new_system = new_system,
                                             old_alchemical_atoms = [atom.index for atom in current_res.atoms()] + list(extra_atom_map.values()),
                                             old_chemical_state_key = old_chemical_state_key,
                                             new_chemical_state_key = new_chemical_state_key,
                                             old_residue_name = old_res_name,
                                             new_residue_name = new_res_name,
                                             old_networkx_residue = old_networkx_molecule,
                                             new_networkx_residue = new_networkx_molecule)

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

        #validate the old/new system matches
        # TODO: create more rigorous checks for this validation either in TopologyProposal or in the HybridTopologyFactory
        #assert PolymerProposalEngine.validate_core_atoms_with_system(topology_proposal)


        return topology_proposal, local_atom_map_stereo_sidechain, new_oemol_res, old_oemol_res

    def _find_adjacent_special_atoms(self, old_topology, new_topology, mutated_residue_index):
        """
        return the atom maps of the next residue C and N atoms in the new topology compared to the old topology

        Arguments
        ---------
        old_topology : simtk.openmm.app.Topology object
            topology of the old system
        new_topology : simtk.openmm.app.Topology object
            topology of the new object
        mutated_residue_index : int
            index of the residue being mutated

        Returns
        -------
        new_to_old_map : dict
            dict of extra C and N indices
        """
        #pull the correct chains
        chain_id = self._chain_id
        new_chain = [chain for chain in new_topology.chains() if chain.id == chain_id][0]
        old_chain = [chain for chain in old_topology.chains() if chain.id == chain_id][0]

        prev_res_index, next_res_index = mutated_residue_index - 1, mutated_residue_index + 1

        new_next_res = [res for res in new_chain.residues() if res.index == next_res_index][0]
        old_next_res = [res for res in old_chain.residues() if res.index == next_res_index][0]

        new_prev_res = [res for res in new_chain.residues() if res.index == prev_res_index][0]
        old_prev_res = [res for res in old_chain.residues() if res.index == prev_res_index][0]

        new_next_res_N_index = [atom.index for atom in new_next_res.atoms() if atom.name.replace(" ", "") == 'N']
        old_next_res_N_index = [atom.index for atom in old_next_res.atoms() if atom.name.replace(" ", "") == 'N']

        new_prev_res_C_index = [atom.index for atom in new_prev_res.atoms() if atom.name.replace(" ", "") == 'C']
        old_prev_res_C_index = [atom.index for atom in old_prev_res.atoms() if atom.name.replace(" ", "") == 'C']

        for _list in [new_next_res_N_index, old_next_res_N_index, new_prev_res_C_index, old_prev_res_C_index]:
            assert len(_list) == 1, f"atoms in the next or prev residue are not uniquely named"

        new_to_old_map = {new_next_res_N_index[0]: old_next_res_N_index[0],
                          new_prev_res_C_index[0]: old_prev_res_C_index[0]}

        return new_to_old_map




    def _choose_mutant(self, topology, metadata):
        """
        Dummy function in parent (PolymerProposalEngine) class to choose a mutant

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
            topology of the protein
        metadata : dict
            metadata associated with mutant choice

        Returns
        -------
        index_to_new_residues : dict
            dict of {index: new_residue}
        metadata : dict
            input metadata
        """
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

    def _identify_differences(self, topology, residue_map):
        """
        Identify excess atoms, excess bonds, missing atoms, and missing bonds.

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
            The original Topology object to be processed
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue)

        Returns
        -------
        excess_atoms : list(simtk.openmm.app.topology.Atom)
            atoms from existing residue not in new residue
        excess_bonds : list(tuple (simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom))
            bonds from existing residue not in new residue
        missing_bonds : list(tuple (simtk.openmm.app.topology._TemplateAtomData, simtk.openmm.app.topology._TemplateAtomData))
            bonds from new residue not in existing residue
        """
        # excess_atoms : list(simtk.openmm.app.topology.Atom) atoms from existing residue not in new residue
        excess_atoms = list()
        # excess_bonds : list(tuple (simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)) bonds from existing residue not in new residue
        excess_bonds = list()
        # missing_atoms : dict, key : simtk.openmm.app.topology.Residue, value : list(simtk.openmm.app.topology._TemplateAtomData)
        missing_atoms = dict()
        # missing_bonds : list(tuple (simtk.openmm.app.topology._TemplateAtomData, simtk.openmm.app.topology._TemplateAtomData)) bonds from new residue not in existing residue
        missing_bonds = list()

        # residue : simtk.openmm.app.topology.Residue (existing residue)
        for k, (residue, replace_with) in enumerate(residue_map):
            # Load residue template for residue to replace with
            if replace_with =='HIS':
                replace_with = 'HIE'
            template = self._templates[replace_with]

            # template_atom_names : dict, key : template atom index, value : template atom name
            template_atom_names = {}
            for atom in template.atoms:
                template_atom_names[template.getAtomIndexByName(atom.name)] = atom.name

            # template_atoms : list(simtk.openmm.app.topology._TemplateAtomData) atoms in new residue
            template_atoms = list(template.atoms)


            # template_bonds : dict, key : simtk.openmm.app.topology.Atom, value : str template atom name
            template_bonds = [(template_atom_names[bond[0]], template_atom_names[bond[1]]) for bond in template.bonds]
            # old_atom_names : set of unique atom names within existing residue : str
            old_atom_names = set(atom.name for atom in residue.atoms())

            # Make a list of atoms in the existing residue that are not in the new residue
            # atom : simtk.openmm.app.topology.Atom in existing residue
            for atom in residue.atoms():
                if atom.name not in template_atom_names.values():
                    excess_atoms.append(atom)

            # if residue == chain_residues[0]: # this doesn't apply?
            # template_atoms = [atom for atom in template_atoms if atom.name not in ('P', 'OP1', 'OP2')]

            # Make a list of atoms in the new residue that are not in the existing residue
            # missing : list(simtk.openmm.app.topology._TemplateAtomData) atoms in new residue not found in existing residue
            missing = list()
            # atom : simtk.openmm.app.topology._TemplateAtomData atoms in new residue
            for atom in template_atoms:
                if atom.name not in old_atom_names:
                    missing.append(atom)

            if len(missing) > 0:
                missing_atoms[residue] = missing
            else:
                missing_atoms[residue] = []

            # Make a dictionary to map atom names in old residue to atom object
            # old_atom_map : dict, key : str (atom name) , value : simtk.openmm.app.topology.Atom
            old_atom_map = dict()
            # atom : simtk.openmm.app.topology.Atom
            for atom in residue.atoms():
                # atom.name : str
                old_atom_map[atom.name] = atom

            # Make a dictionary to map atom names in new residue to atom object
            # new_atom_map : dict, key : str (atom name) , value : simtk.openmm.app.topology.Atom
            new_atom_map = dict()
            # atom : simtk.openmm.app.topology.Atom
            for atom in template_atoms:
                # atom.name : str
                new_atom_map[atom.name] = atom

            # Make a list of bonds already existing in new residue
            # old_bonds : list(tuple(str (atom name), str (atom name))) bonds between atoms both within old residue
            old_bonds = list()
            # bond : tuple(simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)
            for bond in topology.bonds():
                if bond[0].residue == residue and bond[1].residue == residue:
                    old_bonds.append((bond[0].name, bond[1].name))

            # Add any bonds that exist in old residue but not in template to excess_bonds
            for bond in old_bonds:
                if bond not in template_bonds and (bond[1], bond[0]) not in template_bonds:
                    excess_bonds.append((old_atom_map[bond[0]], old_atom_map[bond[1]]))

            # Add any bonds that exist in template but not in old residue to missing_bonds
            for bond in template_bonds:
                if bond not in old_bonds and (bond[1], bond[0]) not in old_bonds:
                    missing_bonds.append((new_atom_map[bond[0]], new_atom_map[bond[1]]))

        return(excess_atoms, excess_bonds, missing_atoms, missing_bonds)

    def _delete_atoms(self, topology, to_delete):
        """
        Delete excess atoms (and corresponding bonds) from specified topology

        Parameters
        ----------
        topology : simtk.openmm.app.Topology
            The Topology to be processed
        excess_atoms : list(simtk.openmm.app.topology.Atom)
            atoms from existing residue not in new residue
        excess_bonds : list(tuple (simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom))
            bonds from old residue not in new residue

        Returns
        -------
        topology : simtk.openmm.app.Topology
            extra atoms and bonds from old residue have been deleted, missing atoms and bottoms in new residue not yet added

        """
        new_topology = app.Topology()
        new_topology.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())

        # new_atoms : dict, key : simtk.openmm.app.topology.Atom, value : simtk.openmm.app.topology.Atom maps old atoms to the corresponding Atom in the new residue
        new_atoms = {}
        delete_set = set(to_delete)

        for chain in topology.chains():
            if chain not in delete_set:
                new_chain = new_topology.addChain(chain.id)
                for residue in chain.residues():
                    new_residue = new_topology.addResidue(residue.name, new_chain, residue.id)
                    for atom in residue.atoms():
                        if atom not in delete_set:
                            new_atom = new_topology.addAtom(atom.name, atom.element, new_residue, atom.id)
                            new_atom.old_index = atom.index
                            new_atoms[atom] = new_atom
        for bond in topology.bonds():
            if bond[0] in new_atoms and bond[1] in new_atoms:
                if bond not in delete_set and (bond[1], bond[0]) not in delete_set:
                    new_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]])
        return new_topology


    def _add_new_atoms(self, topology, missing_atoms, missing_bonds, residue_map):
        """
        Add new atoms (and corresponding bonds) to new residues
        Arguments
        ---------
        topology : simtk.openmm.app.Topology
            extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        missing_atoms : dict
            key : simtk.openmm.app.topology.Residue
            value : list(simtk.openmm.app.topology._TemplateAtomData)
        missing_bonds : list(tuple (simtk.openmm.app.topology._TemplateAtomData, simtk.openmm.app.topology._TemplateAtomData))
            bonds from new residue not in existing residue
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue, str (three letter residue name of new residue)
        Returns
        -------
        topology : simtk.openmm.app.Topology
            new residues have all correct atoms and bonds for desired mutation
        """

        new_topology = app.Topology()
        new_topology.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
        # new_atoms : dict, key : simtk.openmm.app.topology.Atom, value : simtk.openmm.app.topology.Atom maps old atoms to the corresponding Atom in the new residue
        new_atoms = {}
        # new_atom_names : dict, key : str new atom name, value : simtk.openmm.app.topology.Atom maps name of new atom to the corresponding Atom in the new residue (only contains map for missing residue)
        new_atom_names = {}
        # old_residues : list(simtk.openmm.app.topology.Residue)
        old_residues = [old.index for old, new in residue_map]
        for chain in topology.chains():
            new_chain = new_topology.addChain(chain.id)
            for residue in chain.residues():
                new_residue = new_topology.addResidue(residue.name, new_chain, residue.id)
                # Add modified property to residues in new topology
                new_residue.modified_aa = True if residue.index in old_residues else False
                # Copy over atoms from old residue to new residue
                for atom in residue.atoms():
                    # new_atom : simtk.openmm.app.topology.Atom
                    new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                    new_atom.old_index = atom.old_index
                    new_atoms[atom] = new_atom
                    if new_residue.modified_aa:
                        new_atom_names[new_atom.name] = new_atom
                # Check if old residue is in residue_map
                # old_residue : simtk.openmm.app.topology.Residue (old residue)
                # new_residue_name : str (three letter residue name of new residue)
                for i, (old_residue, new_residue_name) in enumerate(residue_map):
                    if self._is_residue_equal(residue, old_residue):
                        # Add missing atoms to new residue
                        # atom : simtk.openmm.app.topology._TemplateAtomData
                        for atom in missing_atoms[old_residue]:
                            new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                            new_atoms[atom] = new_atom
                            new_atom_names[new_atom.name] = new_atom
                        new_residue.name = residue_map[i][1]

        # Copy over bonds from topology to new topology
        for bond in topology.bonds():
            new_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]])

        for bond in missing_bonds:
            new_topology.addBond(new_atom_names[bond[0].name], new_atom_names[bond[1].name])

        return new_topology

    def _is_residue_equal(self, residue, other_residue):
        """
            Check if residue is equal to other_residue based on their names, indices, ids, and chain ids.
            Arguments
            ---------
            residue : simtk.openmm.app.topology.Residue
            other_residue : simtk.openmm.app.topology.Residue
            Returns
            -------
            boolean True if residues are equal, otherwise False
        """
        return residue.name == other_residue.name and residue.index == other_residue.index and residue.chain.id == other_residue.chain.id and residue.id == other_residue.id

    def _construct_atom_map(self,
                            residue_map,
                            old_topology,
                            index_to_new_residues,
                            new_topology):
        """
        Construct atom map (key: index to new residue, value: index to old residue) to supply as an argument to the TopologyProposal.

        Arguments
        ---------
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue, str (three letter residue name of new residue)
        old_topology : simtk.openmm.app.Topology
            topology of old system
        index_to_new_residues : dict
            key : int (index) , value : str (three letter name of proposed residue)
        new_topology : simtk.openmm.app.Topology
            topology of new system

        Returns
        -------
        adjusted_atom_map : dict, key: int (index
            new residues have all correct atoms and bonds for desired mutation
        old_res_to_oemol_map : dict
            key: int (index);  value: int (index)
        new_res_to_oemol_map : dict
            key: int (index);  value: int (index)
        local_atom_map_stereo_sidechain : dict
            chirality-corrected map of new_oemol_res to old_oemol_res
        old_oemol_res : openeye.oechem.oemol object
            copy of modified old oemol sidechain
        new_oemol_res : openeye.oechem.oemol object
            copy of modified new oemol sidechain
        old_oemol_res_copy : openeye.oechem.oemol object
            copy of modified old oemol
        new_oemol_res_copy : openeye.oechem.oemol object
            copy of modified new oemol
        """
        from perses.utils.openeye import createOEMolFromSDF
        from pkg_resources import resource_filename

        # atom_map : dict, key : int (index of atom in old topology) , value : int (index of same atom in new topology)
        atom_map = dict()

        # atoms with an old_index attribute should be mapped
        # k : int
        # atom : simtk.openmm.app.topology.Atom

        # old_to_new_residues : dict, key : str old residue name, key : simtk.openmm.app.topology.Residue new residue
        old_to_new_residues = {}
        for old_residue in old_topology.residues():
            for new_residue in new_topology.residues():
                if old_residue.index == new_residue.index:
                    #old_to_new_residues[old_residue.name] = new_residue
                    old_to_new_residues[old_residue] = new_residue
                    break
        #_logger.debug(f"\t\told_to_new_residues: {old_to_new_residues}")

        # modified_residues : dict, key : index of old residue, value : proposed residue
        modified_residues = dict()

        for map_entry in residue_map:
            old_residue = map_entry[0]
            modified_residues[old_residue.index] = old_to_new_residues[old_residue]
        _logger.debug(f"\t\tmodified residues: {modified_residues}")

        # old_residues : dict, key : index of old residue, value : old residue
        old_residues = dict()
        for residue in old_topology.residues():
            if residue.index in index_to_new_residues.keys():
                old_residues[residue.index] = residue
        _logger.debug(f"\t\t\told residues: {old_residues}")

        # Update atom map with atom mappings for residues that have been modified
        assert len(index_to_new_residues) == 1, f"index_to_new_residues is not of length 1"
        index = list(index_to_new_residues.keys())[0]
        #old_res = old_residues[index]
        old_res = old_residues[index]
        new_res = modified_residues[index]
        _logger.debug(f"\t\t\told res: {old_res.name}; new res: {new_res.name}")

        _logger.debug(f"\t\t\told topology res names: {[(atom.index, atom.name) for atom in old_res.atoms()]}")
        _logger.debug(f"\t\t\tnew topology res names: {[(atom.index, atom.name) for atom in new_res.atoms()]}")

        old_res_name = old_res.name
        new_res_name = new_res.name

        #make correction for HIS
        his_templates = ['HIE', 'HID']
        if old_res_name in his_templates:
            old_res_name = 'HIS'
        elif new_res_name in his_templates:
            new_res_name = 'HIS'
        else:
            pass

        old_oemol_res = createOEMolFromSDF(resource_filename('perses', os.path.join('data', 'amino_acid_templates', f"{old_res_name}.pdb")), add_hydrogens = True)
        new_oemol_res = createOEMolFromSDF(resource_filename('perses', os.path.join('data', 'amino_acid_templates', f"{new_res_name}.pdb")), add_hydrogens = True)


        #assert the names are unique:
        if not len(set([atom.GetName() for atom in old_oemol_res.GetAtoms()])) == len([atom.GetName() for atom in old_oemol_res.GetAtoms()]):
            _logger.warning(f"\t\t\tthe sidechain atoms in the old res are not uniquely named")
            return {}
        elif not len(set([atom.GetName() for atom in new_oemol_res.GetAtoms()])) == len([atom.GetName() for atom in new_oemol_res.GetAtoms()]):
            _logger.warning(f"\t\t\tthe sidechain atoms in the new res are not uniquely named")
            return {}

        #fix atom names (spaces and numbers before letters correction)
        for atom in old_oemol_res.GetAtoms():
            name_with_spaces = atom.GetName()
            name_without_spaces = name_with_spaces.replace(" ", "")
            if name_without_spaces[0].isdigit():
                name_without_spaces = name_without_spaces[1:] + name_without_spaces[0]
            atom.SetName(name_without_spaces)

        for atom in new_oemol_res.GetAtoms():
            name_with_spaces = atom.GetName()
            name_without_spaces = name_with_spaces.replace(" ", "")
            if name_without_spaces[0].isdigit():
                name_without_spaces = name_without_spaces[1:] + name_without_spaces[0]
            atom.SetName(name_without_spaces)

        old_oemol_res_copy = copy.deepcopy(old_oemol_res)
        new_oemol_res_copy = copy.deepcopy(new_oemol_res)

        _logger.debug(f"\t\t\told_oemol_res names: {[(atom.GetIdx(), atom.GetName()) for atom in old_oemol_res.GetAtoms()]}")
        _logger.debug(f"\t\t\tnew_oemol_res names: {[(atom.GetIdx(), atom.GetName()) for atom in new_oemol_res.GetAtoms()]}")

        #create bookkeeping dictionaries
        old_res_to_oemol_map = {atom.index: old_oemol_res.GetAtom(oechem.OEHasAtomName(atom.name)).GetIdx() for atom in old_res.atoms()}
        new_res_to_oemol_map = {atom.index: new_oemol_res.GetAtom(oechem.OEHasAtomName(atom.name)).GetIdx() for atom in new_res.atoms()}

        _logger.debug(f"\t\t\told_res_to_oemol_map: {old_res_to_oemol_map}")
        _logger.debug(f"\t\t\tnew_res_to_oemol_map: {new_res_to_oemol_map}")

        old_oemol_to_res_map = {val: key for key, val in old_res_to_oemol_map.items()}
        new_oemol_to_res_map = {val: key for key, val in new_res_to_oemol_map.items()}

        old_res_to_oemol_molecule_map = {atom.index: old_oemol_res.GetAtom(oechem.OEHasAtomName(atom.name)) for atom in old_res.atoms()}
        new_res_to_oemol_molecule_map = {atom.index: new_oemol_res.GetAtom(oechem.OEHasAtomName(atom.name)) for atom in new_res.atoms()}



        #initialize_the atom map
        local_atom_map = {}

        #now remove backbones in both molecules and map them separately
        backbone_atoms = ['C', 'CA', 'N', 'O', 'H', 'HA', "H'"]
        old_atoms_to_delete, new_atoms_to_delete = [], []
        for atom in new_oemol_res.GetAtoms():
            if atom.GetName() in backbone_atoms:
                try: #to get the backbone atom with the same naem in the old_oemol_res
                    old_corresponding_backbones = [_atom for _atom in old_oemol_res.GetAtoms() if _atom.GetName() == atom.GetName()]
                    if old_corresponding_backbones == []:
                        #this is an exception when the old oemol res is a glycine.  if this is the case, then we do not map HA2 or HA3
                        assert set(['HA2', 'HA3']).issubset([_atom.GetName() for _atom in old_oemol_res.GetAtoms()]), f"old oemol residue is not a GLY template"
                        #we have to map HA3 to HA (old, new)
                        old_corresponding_backbones = [_atom for _atom in old_oemol_res.GetAtoms() if _atom.GetName() == 'HA3' and atom.GetName() == 'HA']
                    assert len(old_corresponding_backbones) == 1, f"there can only be one corresponding backbone in the old molecule; corresponding backbones: {[atom.GetName() for atom in old_corresponding_backbones]}"
                    old_corresponding_backbone = old_corresponding_backbones[0]
                    if not atom.GetName() == "H'": #throw out the extra H
                        local_atom_map[atom.GetIdx()] = old_corresponding_backbone.GetIdx()
                    old_atoms_to_delete.append(old_corresponding_backbone)
                    new_atoms_to_delete.append(atom)
                    assert new_oemol_res.DeleteAtom(atom), f"failed to delete new_oemol atom {atom}"
                    assert old_oemol_res.DeleteAtom(old_corresponding_backbone), f"failed to delete old_oemol atom {old_corresponding_backbone}"
                except Exception as e:
                    raise Exception(f"failed to map the backbone separately: {e}")


        #now we can get the mol atom map of the sidechain
        #NOTE: since the sidechain oemols are NOT zero-indexed anymore, we need to match by name (since they are unique identifiers)
        local_atom_map_nonstereo_sidechain = SmallMoleculeSetProposalEngine._get_mol_atom_map(old_oemol_res,
                                                                                              new_oemol_res,
                                                                                              atom_expr = PROTEIN_ATOM_EXPRESSION,
                                                                                              bond_expr = PROTEIN_BOND_EXPRESSION,
                                                                                              allow_ring_breaking = True,
                                                                                              matching_criterion = 'name')

        #check the atom map thus far:
        _logger.debug(f"\t\t\tlocal atom map nonstereo sidechain: {local_atom_map_nonstereo_sidechain}")

        #preserve chirality of the sidechain
        # _logger.warning(f"\t\t\told oemols: {[atom.GetIdx() for atom in old_oemol_res.GetAtoms()]}")
        # _logger.warning(f"\t\t\tnew oemols: {[atom.GetIdx() for atom in new_oemol_res.GetAtoms()]}")
        local_atom_map_stereo_sidechain = SmallMoleculeSetProposalEngine.preserve_chirality(old_oemol_res, new_oemol_res, local_atom_map_nonstereo_sidechain)

        _logger.debug(f"\t\t\tlocal atom map stereo sidechain: {local_atom_map_stereo_sidechain}")
        sidechain_fixed_map = {}
        for new_oemol_idx, old_oemol_idx in local_atom_map_stereo_sidechain.items():
            sidechain_fixed_map[new_oemol_to_res_map[new_oemol_idx]] = old_oemol_to_res_map[old_oemol_idx]
        _logger.debug(f"\t\t\tsidechain fixed map: {sidechain_fixed_map}")


        #make sure that CB is mapped; otherwise the residue will not be contiguous
        found_CB = False
        new_atoms = {atom.index: atom.name for atom in new_res.atoms()}
        old_atoms = {atom.index: atom.name for atom in old_res.atoms()}

        for new_index, old_index in sidechain_fixed_map.items():
            new_name, old_name = new_atoms[new_index], old_atoms[old_index]
            if new_name == 'CB' and old_name == 'CB':
                found_CB = True
        if not found_CB:
            _logger.debug(f"\t\t\tno 'CB' found!!!.  removing local atom map stereo sidechain...")
            local_atom_map_stereo_sidechain = {}

        _logger.debug(f"\t\t\tthe local atom map (backbone) is {local_atom_map}")
        #update the local map
        local_atom_map.update(local_atom_map_stereo_sidechain)
        _logger.debug(f"\t\t\tthe local atom map (total) is {local_atom_map}")

        #correct the map
        #now we have to update the atom map indices
        _logger.debug(f"\t\t\tadjusting the atom map with topology indices...")
        fixed_map = {}
        for new_oemol_idx, old_oemol_idx in local_atom_map.items():
            fixed_map[new_oemol_to_res_map[new_oemol_idx]] = old_oemol_to_res_map[old_oemol_idx]

        adjusted_atom_map = fixed_map
        _logger.debug(f"\t\t\tadjusted_atom_map: {adjusted_atom_map}")

        index_to_name_new = {atom.index: atom.name for atom in new_res.atoms()}
        index_to_name_old = {atom.index: atom.name for atom in old_res.atoms()}
        map_atom_names = [(index_to_name_new[new_idx], index_to_name_old[old_idx]) for new_idx, old_idx in adjusted_atom_map.items()]
        _logger.debug(f"\t\t\tthe mapped atom names are: {map_atom_names}")

            #and all of the environment atoms should already be handled
        return adjusted_atom_map, old_res_to_oemol_map, new_res_to_oemol_map, local_atom_map_stereo_sidechain, old_oemol_res, new_oemol_res, old_oemol_res_copy, new_oemol_res_copy

    def _get_mol_atom_matches(self, current_molecule, proposed_molecule, first_atom_index_old, first_atom_index_new):
        """
        Given two molecules, returns the mapping of atoms between them.

        Arguments
        ---------
        current_molecule : openeye.oechem.oemol object
             The current molecule in the sampler
        proposed_molecule : openeye.oechem.oemol object
             The proposed new molecule
        first_atom_index_old : int
            The index of the first atom in the old resiude/current molecule
        first_atom_index_new : int
            The index of the first atom in the new residue/proposed molecule
        Note: Since FFAllAngleGeometryEngine.oemol_from_residue creates a new topology for the specified residue,
        the atom indices in the output oemol (i.e. current_molecule and proposed_molecule) are reset to start at 0.
        Therefore, first_atom_index_old and first_atom_index_new are used to correct the indices such that they match
        the atom indices of the original old and new residues.
        Returns
        -------
        new_to_old_atom_map : dict, key : index of atom in new residue, value : index of atom in old residue
        """
        # Load current and proposed residues as OEGraphMol objects
        oegraphmol_current = oechem.OEGraphMol(current_molecule)
        oegraphmol_proposed = oechem.OEGraphMol(proposed_molecule)

        # Instantiate Maximum Common Substructures (MCS) object and intialize properties
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

        # Forcibly match C and O atoms in proposed residue to atoms in current residue
        for matched_atom_name in ['C','O']:
            force_match = oechem.OEHasAtomName(matched_atom_name)
            old_atom, new_atom = forcibly_matched(mcs, oegraphmol_proposed, force_match)
            this_match = oechem.OEMatchPairAtom(old_atom, new_atom)
            assert mcs.AddConstraint(this_match)

        # Generate matches using MCS
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        unique = True
        matches = [m for m in mcs.Match(oegraphmol_proposed, unique)]

        # Handle case where there are no matches
        if len(matches) == 0:
            from perses.utils.openeye import describe_oemol
            msg = 'No matches found in _get_mol_atom_matches.\n'
            msg += '\n'
            msg += 'oegraphmol_current:\n'
            msg += describe_oemol(oegraphmol_current)
            msg += '\n'
            msg += 'oegraphmol_proposed:\n'
            msg += describe_oemol(oegraphmol_proposed)
            raise Exception(msg)

        # Select match and generate atom map
        match = np.random.choice(matches)
        new_to_old_atom_map = {}
        for match_pair in match.GetAtoms():
            if match_pair.pattern.GetAtomicNum() == 1 and match_pair.target.GetAtomicNum() == 1:  # Do not map hydrogens
                continue
            O2_index_current = current_molecule.NumAtoms() - 2
            O2_index_proposed = proposed_molecule.NumAtoms() -2
            if 'O2' in match_pair.pattern.GetName() and 'O2' in match_pair.target.GetName() and match_pair.pattern.GetIdx() == O2_index_current and match_pair.target.GetIdx() == O2_index_proposed:  # Do not map O2 if its second to last index in atom (this O2 was added to oemol to complete the residue)
                continue
            old_index = match_pair.pattern.GetData("topology_index")
            new_index = match_pair.target.GetData("topology_index")
            new_to_old_atom_map[new_index] = old_index

        return new_to_old_atom_map

    def compute_state_key(self, topology):
        """
        utility method to define a state key;
        state key takes the following form:
        """
        for chain in topology.chains():
            if chain.id == self._chain_id:
                break
        chemical_state_key = ''
        for index, res in enumerate(chain.residues()):
            if index > 0:
                chemical_state_key += '-'
            chemical_state_key += res.name

        return chemical_state_key

    @staticmethod
    def validate_core_atoms_with_system(topology_proposal):
        """
        Utility function to ensure that the valence terms and nonbonded exceptions do not change between alchemical and environment atoms.

        Arguments
        ---------
        topology_proposal : TopologyProposal
            topology proposal

        Returns
        -------
        validated : bool
            whether the assertion is validated
        """
        old_system, new_system = topology_proposal.old_system, topology_proposal.new_system

        #check if there are bonds between non alchemical atoms and core atoms in both systems
        old_alchemical_to_nonalchemical_bonds = []
        new_alchemical_to_nonalchemical_bonds = []

        #loop over old topology
        for bond in topology_proposal.old_topology.bonds():
            if bond[0].index in topology_proposal._old_alchemical_atoms and bond[1].index in topology_proposal.old_environment_atoms:
                old_alchemical_to_nonalchemical_bonds.append((bond[0], bond[1]))
            elif bond[1].index in topology_proposal._old_alchemical_atoms and bond[0].index in topology_proposal.old_environment_atoms:
                old_alchemical_to_nonalchemical_bonds.append((bond[1], bond[0]))
            else:
                pass

        #loop over new topology
        for bond in topology_proposal.new_topology.bonds():
            if bond[0].index in topology_proposal._new_alchemical_atoms and bond[1].index in topology_proposal.new_environment_atoms:
                new_alchemical_to_nonalchemical_bonds.append((bond[0], bond[1]))
            elif bond[1].index in topology_proposal._new_alchemical_atoms and bond[0].index in topology_proposal.new_environment_atoms:
                new_alchemical_to_nonalchemical_bonds.append((bond[1], bond[0]))
            else:
                pass

        assert len(old_alchemical_to_nonalchemical_bonds) == len(new_alchemical_to_nonalchemical_bonds), f"the number of alchemical to nonalchemical bonds in old and new topologies is not equal"

        #assert that all of the alchemical atoms (bonded to nonalchemical atoms) are 'core'
        new_to_old_pair = {}
        for alch_atom, nonalch_atom in old_alchemical_to_nonalchemical_bonds:
            assert alch_atom.index in list(topology_proposal.core_new_to_old_atom_map.values()), f"the old alchemical atom ({alch_atom.index}) is not a core atom!"
            assert nonalch_atom.index in topology_proposal.old_environment_atoms, f"the nonalchemical atom ({nonalch_atom.index}) is not an environment atom!"
            appropriate_new_bonds = [bond for bond in new_alchemical_to_nonalchemical_bonds if bond[0].index == topology_proposal.old_to_new_atom_map[alch_atom.index] and bond[1].index == topology_proposal.old_to_new_atom_map[nonalch_atom.index]]
            assert len(appropriate_new_bonds) == 1, f"there is no match between the old bond to the new bond"
            new_to_old_pair[appropriate_new_bonds[0]] = (alch_atom, nonalch_atom)


        #if there is at least one alchemical to nonalchemical bond, then we should check the system...
        old_forces = {force.__class__.__name__ : force for force in [old_system.getForce(index) for index in range(old_system.getNumForces())]}
        new_forces = {force.__class__.__name__ : force for force in [new_system.getForce(index) for index in range(new_system.getNumForces())]}
        assert set(list(old_forces.keys())) == set(list(new_forces.keys())), f"the old and new forces do not match: {old_forces, new_forces}"

        for new_bond, old_bond in new_to_old_pair.items():
            new_atom_idx1, new_atom_idx2 = new_bond[0].index, new_bond[1].index
            old_atom_idx1, old_atom_idx2 = old_bond[0].index, old_bond[1].index
            _logger.debug(f"\titerating through new atoms {new_atom_idx1, new_atom_idx2} with old atoms {old_atom_idx1, old_atom_idx2}")

            if 'HarmonicBondForce' in list(old_forces.keys()):
                old_bond_terms = [param for param in [old_forces['HarmonicBondForce'].getBondParameters(bond_index) for bond_index in range(old_forces['HarmonicBondForce'].getNumBonds())] if set((old_atom_idx1, old_atom_idx2)) == set(param[:2])]
                new_bond_terms = [param for param in [new_forces['HarmonicBondForce'].getBondParameters(bond_index) for bond_index in range(new_forces['HarmonicBondForce'].getNumBonds())] if set((new_atom_idx1, new_atom_idx2)) == set(param[:2])]
                assert all(old_bond_term[2:] == new_bond_term[2:] for old_bond_term, new_bond_term in zip(old_bond_terms, new_bond_terms)), f"the bond terms do not match. old terms: {old_bond_terms}; new terms: {new_bond_terms}"
            if 'HarmonicAngleForce' in list(old_forces.keys()):
                old_angle_terms = [param for param in [old_forces['HarmonicAngleForce'].getAngleParameters(angle_index) for angle_index in range(old_forces['HarmonicAngleForce'].getNumAngles())] if set((old_atom_idx1, old_atom_idx2)).issubset(set(param[:3]))]
                new_angle_terms = [param for param in [new_forces['HarmonicAngleForce'].getAngleParameters(angle_index) for angle_index in range(new_forces['HarmonicAngleForce'].getNumAngles())] if set((new_atom_idx1, new_atom_idx2)).issubset(set(param[:3]))]
                assert len(old_angle_terms) == len(new_angle_terms), f"the number of old angle and new angle terms do not match: \n{old_angle_terms}\n{new_angle_terms}"
                #assert set([tuple(old_term[3:]) for old_term in old_angle_terms]) == set([tuple(new_term[3:]) for new_term in new_angle_terms]), f"the old and new angle terms, respectively, do not match: {old_angle_terms, new_angle_terms}"

            if "PeriodicTorsionForce" in list(old_forces.keys()):
                old_torsion_terms = [param for param in [old_forces['PeriodicTorsionForce'].getTorsionParameters(torsion_index) for torsion_index in range(old_forces['PeriodicTorsionForce'].getNumTorsions())] if set((old_atom_idx1, old_atom_idx2)).issubset(set(param[:4]))]
                new_torsion_terms = [param for param in [new_forces['PeriodicTorsionForce'].getTorsionParameters(torsion_index) for torsion_index in range(new_forces['PeriodicTorsionForce'].getNumTorsions())] if set((new_atom_idx1, new_atom_idx2)).issubset(set(param[:4]))]
                assert len(old_torsion_terms) == len(new_torsion_terms), f"the number of old torsion and new torsion terms do not match: \n{old_torsion_terms}\n{new_torsion_terms}"
                #assert set([tuple(old_term[4:]) for old_term in old_torsion_terms]) == set([tuple(new_term[4:]) for new_term in new_torsion_terms]), f"the old and new torsion terms, respectively, do not match: {old_torsion_terms, new_torsion_terms}"

        if "NonbondedForce" in list(old_forces.keys()):
            #make sure that the environment atoms do not change params
            for new_idx, old_idx in topology_proposal.new_to_old_atom_map.items():
                if new_idx in topology_proposal._new_environment_atoms:
                    new_params = new_forces['NonbondedForce'].getParticleParameters(new_idx)[1:]
                    old_params = old_forces['NonbondedForce'].getParticleParameters(old_idx)[1:]
                    assert new_params == old_params, f"the environment new_to_old pair {new_idx, old_idx} do not have the same nonbonded params: {new_params, old_params}"

            #then we have to check the exceptions between env atoms and between env-core atoms
            old_exceptions = {set(old_forces['NonbondedForce'].getExceptionParameters(idx)[:2]):old_forces['NonbondedForce'].getExceptionParameters(idx)[2:] for idx in range(old_forces['NonbondedForce'].getNumExceptions())}
            for new_exception_idx in range(new_forces['NonbondedForce'].getNumExceptions()):
                new_exception_parms = new_forces['NonbondedForce'].getExceptionParameters(new_exception_idx)
                new_exception_pair = new_exception_parms[:2]
                if set(new_exception_pair).issubset(topology_proposal._new_environment_atoms) or len(set(new_exception_pair).intersection(topology_proposal._new_environment_atoms)) == 1:
                    old_exception_pair = (topology_proposal.new_to_old_atom_map[new_exception_pair[0]], topology_proposal.new_to_old_atom_map[new_exception_pair[1]])
                    old_exception_params = old_exceptions[set(old_exception_pair)]
                    assert new_exception_parms == old_exception_params, f"new exception params of new env atoms {new_exception_pair} is not equal to the exception params of old env pair {old_exception_pair}"

        return True



class PointMutationEngine(PolymerProposalEngine):
    """
    ProposalEngine for generating point mutation variants of a wild-type polymer

    Examples
    --------

    Mutations of a terminally-blocked peptide

    >>> from openmmtools.testsystems import AlanineDipeptideExplicit
    >>> testsystem = AlanineDipeptideExplicit()
    >>> system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

    >>> from topology_proposal import PointMutationEngine
    >>> engine = PointMutationEngine(topology, system_generator, chain_id='A', residues_allowed_to_mutate='max_point_mutants=1)

    """

    # TODO: Overhaul API to make it easier to specify mutations
    def __init__(self,
                 wildtype_topology,
                 system_generator,
                 chain_id,
                 proposal_metadata=None,
                 max_point_mutants=1,
                 residues_allowed_to_mutate=None,
                 allowed_mutations=None,
                 verbose=False,
                 always_change=True,
                 aggregate=False):
        """
        Create a PointMutationEngine for proposing point mutations of a biopolymer component of a system.

        Note: When instantiating this class, be sure to include cap residues ('ACE', 'NME') in the input topology.
        Otherwise, mutating the first residue will result in a residue template mismatch.
        See openmmtools/data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb for reference.

        Parameters
        ----------
        wildtype_topology : openmm.app.Topology
            The Topology object describing the system
        system_generator : SystemGenerator
            The SystemGenerator to generate new parameterized System objects
        chain_id : str
            id of the chain to mutate
            (using the first chain with the id, if there are multiple)
        proposal_metadata : dict, optional, default=None
            Contains information necessary to initialize proposal engine
        max_point_mutants : int, optional, default=1
            If not None, limit the number of point mutations that are allowed simultaneously
        residues_allowed_to_mutate : list(str), optional, default=None
            Contains residue ids
            If not specified, engine assumes all residues (except ACE and NME caps) may be mutated.
        allowed_mutations : list(tuple), optional, default=None
            ('residue id to mutate','desired mutant residue name (3-letter code)')
            Example:
                Desired systems are wild type T4 lysozyme, T4 lysozyme L99A, and T4 lysozyme L99A/M102Q
                allowed_mutations = [
                    ('99', 'ALA'),
                    ('102','GLN')
                ]
            If this is not specified, the engine will propose a random amino acid at a random location.
        always_change : bool, optional, default=True
            Have the proposal engine always propose a state different from the current state.
            If the current state is WT, always propose a mutation.
            If the current state is mutant, always propose a different mutant or WT.
        aggregate : bool, optional, default=False
            Have the proposal engine aggregate mutations.
            If aggregate is set to False, the engine will undo mutants in the current topology such that the next proposal
            if the WT state
            If aggregate is set to True, the engine will not undo the mutants from the current topology, thereby allowing
            each proposal to contain multiple mutations.

        """
        super(PointMutationEngine,self).__init__(system_generator, chain_id, proposal_metadata=proposal_metadata, verbose=verbose, always_change=always_change, aggregate=aggregate)

        assert isinstance(wildtype_topology, app.Topology)

        # Check that provided topology has specified chain.
        chain_ids_in_topology = [chain.id for chain in wildtype_topology.chains()]
        if chain_id not in chain_ids_in_topology:
            raise Exception(f"Specified chain id {chain_id} not found in the wildtype topology. choices are {chain_ids_in_topology}")

        if max_point_mutants != 1:
            raise ValueError('max_point_mutants != 1 not yet supported')

        self._max_point_mutants = max_point_mutants
        self._wildtype = wildtype_topology
        self._ff = system_generator.forcefield
        self._templates = self._ff._templates
        self._residues_allowed_to_mutate = residues_allowed_to_mutate
        self._allowed_mutations = allowed_mutations
        if proposal_metadata is None:
            proposal_metadata = dict()
        self._metadata = proposal_metadata

    def _choose_mutant(self, topology, metadata):
        """
        Method to choose a mutant.

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
            topology of the protein
        metadata : dict
            metadata associated with mutant choice

        Returns
        -------
        index_to_new_residues : dict
            dict of {index: new_residue}
        metadata : dict
            input metadata
        """
        chain_id = self._chain_id

        _logger.debug(f"\t\tcomputing old mutant key...")
        old_key = self._compute_mutant_key(topology, chain_id) #compute the key of the given topology and the appropriate chain
        _logger.debug(f"\t\t\told mutant key: {old_key}")

        _logger.debug(f"\t\tcomputing index_to_new_residues...")
        index_to_new_residues = self._undo_old_mutants(topology, chain_id, old_key) #pull the index: new residues that are different from WT
        _logger.debug(f"\t\t\tindex_to_new_residues: {index_to_new_residues}")

        if index_to_new_residues != {} and not self._aggregate:  # Starting state is mutant and mutations should not be aggregated
            _logger.debug(f"\t\tthe starting state is a mutant, but mutations are not being aggregated.")
            pass  # At this point, index_to_new_residues contains mutations that need to be undone. This iteration will result in WT,
        else:  # Starting state is WT or starting state is mutant and mutations should be aggregated
            _logger.debug(f"\t\tthe starting state is WT or mutant, and mutations can aggregate.")
            index_to_new_residues = dict()
            if self._allowed_mutations is not None:
                _logger.debug(f"\t\tthe allowed mutations are prespecified as {self._allowed_mutations}; generating index_to_new_residues...")
                allowed_mutations = self._allowed_mutations
                index_to_new_residues = self._choose_mutation_from_allowed(topology, chain_id, allowed_mutations, index_to_new_residues, old_key)
            else:
                # index_to_new_residues : dict, key : int (index) , value : str (three letter residue name)
                #TODO: check _propose_mutation
                index_to_new_residues = self._propose_mutation(topology, chain_id, index_to_new_residues)
            _logger.debug(f"\t\t\tindex_to_new_residues: {index_to_new_residues}")
        # metadata['mutations'] : list(str (three letter WT residue name - index - three letter MUT residue name) )
        metadata['mutations'] = self._save_mutations(topology, index_to_new_residues)
        return index_to_new_residues, metadata

    def _undo_old_mutants(self, topology, chain_id, old_key):
        """
        Function to find the residue indices in the chain_id with residues that are different from WT.  This is a dict of form {idx : res.name}.

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
            topology of the protein
        chain_id : str
            id of the chain of interest (this is a PointMutationEngine attribute)
        old_key : str
            str of the form generated by self._compute_mutant_key

        Returns
        -------
        index_to_new_residues : dict
            dict of {index: new_residue}
        """
        index_to_new_residues = dict()
        if old_key == 'WT': #there are no mutants, so return an empty dict
            return index_to_new_residues

        #search for appropriate chain
        found_chain = False
        for chain in topology.chains():
            if chain.id == chain_id:
                found_chain = True
                break
        if not found_chain:
            raise Exception(f"chain id {chain_id} was not found in the topology")

        residue_id_to_index = {residue.id : residue.index for residue in chain.residues()}

        for mutant in old_key.split('-'):
            old_res = mutant[:3] #3 letter old res
            residue_id = mutant[3:-3] #id of res
            index_to_new_residues[residue_id_to_index[residue_id]] = old_res #update the index to new res ONLY for mutated residues

        return index_to_new_residues

    def _choose_mutation_from_allowed(self, topology, chain_id, allowed_mutations, index_to_new_residues, old_key):
        """
        Used when allowed mutations have been specified
        Assume (for now) uniform probability of selecting each specified mutant

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        chain_id : str
        allowed_mutations : list(tuple)
            list of allowed mutations -- each mutation is a tuple of the residue id and three-letter amino acid code of desired mutant
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
            contains information to mutate back to WT as starting point for new mutants
        old_key : str
            chemical_state_key for old topology

        Returns
        -------
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        """

        # Set chain and create id-index mapping for residues in chain
        chain_found = False
        for anychain in topology.chains():
            if anychain.id == chain_id:
                # chain : simtk.openmm.app.topology.Chain
                chain = anychain
                chain_found = True
                break

        if not chain_found:
            chains = [chain.id for chain in topology.chains()]
            raise Exception("Chain '%s' not found in Topology. Chains present are: %s" % (chain_id, str(chains)))

        residue_id_to_index = {residue.id : residue.index for residue in chain.residues()}

        # Define location probabilities and propose a location/mutant state
        if self._always_change:
            # location_prob : np.array, probability value for each mutant state at their respective locations in allowed_mutations (uniform).
            if old_key == 'WT':
                location_prob = [1.0 / len(allowed_mutations)] * len(allowed_mutations)
                proposed_location = np.random.choice(range(len(allowed_mutations)), p=location_prob)
            else:
                current_mutations = []
                for mutant in old_key.split('-'):
                    residue_id = mutant[3:-3]
                    new_res = mutant[-3:]
                    current_mutations.append((residue_id, new_res))

                new_mutations = []
                for mutation in allowed_mutations:
                    if mutation not in current_mutations:
                        new_mutations.append(mutation)
                if not new_mutations:
                    raise Exception("The old topology state contains all allowed mutations (%s). Please specify additional mutations." % allowed_mutations[0])

                location_prob = [1.0 / (len(new_mutations))] * len(new_mutations)
                proposed_location = np.random.choice(range(len(new_mutations)), p=location_prob)

        else:
            if old_key == 'WT':
                location_prob = [1.0 / (len(allowed_mutations)+1.0)] * (len(allowed_mutations)+1)
                proposed_location = np.random.choice(range(len(allowed_mutations) + 1), p=location_prob)
            else:
                location_prob = [1.0 / len(allowed_mutations)] * len(allowed_mutations)
                proposed_location = np.random.choice(range(len(allowed_mutations)), p=location_prob)

        # If the proposed state is the same as the current state
        # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
        if old_key == 'WT' and proposed_location == len(allowed_mutations):
            # Choose WT and return empty index_to_new_residues
            return index_to_new_residues
        elif old_key != 'WT':
            for mutant in old_key.split('-'):
                residue_id = mutant[3:-3]
                new_res = mutant[-3:]
                if allowed_mutations.index((residue_id, new_res)) == proposed_location:
                    return index_to_new_residues #it is already mutated

        residue_id = allowed_mutations[proposed_location][0]
        residue_name = allowed_mutations[proposed_location][1]
        # Verify residue with mutation exists in old topology and is not the first or last residue
        # original_residue : simtk.openmm.app.topology.Residue
        original_residue = ''
        for res in chain.residues():
            if res.index == residue_id_to_index[residue_id]:
                original_residue = res
                break
        if not original_residue:
            raise Exception("User-specified an allowed mutation at residue %s , but that residue does not exist" % residue_id)
        if original_residue.index == 0 or original_residue.index == topology.getNumResidues() - 1:
            raise Exception("Residue not found. Be sure you are not trying to mutate the first or last residue."
                            " If you wish to modify one of these residues, make sure you have added cap residues to the input topology.")

        # Check if mutated residue's name is same as residue's name in old topology
        if original_residue.name in ['HID', 'HIE']:
            # original_residue.name = 'HIS'
            pass
        if original_residue.name == residue_name: #there is no mutation to be done
            return index_to_new_residues

        # Save proposed mutation to index_to_new_residues
        # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
        index_to_new_residues[residue_id_to_index[residue_id]] = residue_name

        # Randomly choose HIS template ('HIS' does not exist as a template)
        # if residue_name == 'HIS':
        #     his_state = ['HIE','HID']
        #     his_prob = [1/len(his_state)] * len(his_state)
        #     his_choice = np.random.choice(range(len(his_state)), p=his_prob)
        #     index_to_new_residues[residue_id_to_index[residue_id]] = his_state[his_choice]
        return index_to_new_residues

    def _propose_mutation(self, topology, chain_id, index_to_new_residues):
        """

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
        chain_id : str
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
            contains information to mutate back to WT as starting point for new mutants
        old_key : str
            chemical_state_key for old topology
        Returns
        -------
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        """

        # Set chain and create id-index mapping for residues in chain
        # chain : simtk.openmm.app.topology.Chain
        chain_found = False
        for anychain in topology.chains():
            if anychain.id == chain_id:
                chain = anychain
                chain_found = True
                residue_id_to_index = {residue.id: residue.index for residue in chain.residues()}
                if self._residues_allowed_to_mutate is None:
                    chain_residues = [res for res in chain.residues() if res.index != 0 and res.index != topology.getNumResidues()-1]
                    # num_residues : int
                    num_residues = len(chain_residues)
                else:
                    for res_id in self._residues_allowed_to_mutate:
                        if res_id not in residue_id_to_index.keys():
                            raise Exception(
                                "Residue id '%s' not found in Topology. Residue ids present are: %s. "
                                "\n\t Note: The type of the residue id must be 'str'" % (res_id, str(residue_id_to_index.keys())))
                    num_residues = len(self._residues_allowed_to_mutate)
                    chain_residues = self._mutable_residues(chain)
                break
        if not chain_found:
            chains = [chain.id for chain in topology.chains()]
            raise Exception("Chain '%s' not found in Topology. Chains present are: %s" % (chain_id, str(chains)))

        # Define location probabilities
        # location_prob : np.array, probability value for each residue location (uniform)
        location_prob = [1.0/num_residues] * num_residues

        # Propose a location at which to mutate the residue
        # proposed_location : int, index of chosen entry in location_prob
        proposed_location = np.random.choice(range(num_residues), p=location_prob)

        # Rename residue to HIS if it uses one of the HIS-derived templates
        # original_residue : simtk.openmm.app.topology.Residue
        original_residue = chain_residues[proposed_location]
        if original_residue.name in ['HIE', 'HID']:
            #original_residue.name = 'HIS'
            pass

        if self._residues_allowed_to_mutate is None:
            proposed_location = original_residue.index
        else:
            proposed_location = residue_id_to_index[self._residues_allowed_to_mutate[proposed_location]]

        # Define probabilities for amino acid options and choose one
        # amino_prob : np.array, probability value for each amino acid option (uniform)
        aminos = self._aminos
        if self._always_change:
            amino_prob = [1.0/(len(aminos)-1)] * len(aminos)
            amino_prob[aminos.index(original_residue.name)] = 0.0
        else:
            amino_prob = [1.0/len(aminos)] * len(aminos)
        # proposed_amino_index : int, index of three letter residue name in aminos list
        proposed_amino_index = np.random.choice(range(len(aminos)), p=amino_prob)

        # Save proposed mutation to index_to_new_residues
        # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
        index_to_new_residues[proposed_location] = aminos[proposed_amino_index]

        # Randomly choose HIS template ('HIS' does not exist as a template)
        if aminos[proposed_amino_index] == 'HIS':
            # his_state = ['HIE','HID']
            # his_prob = [1 / len(his_state)] * len(his_state)
            # his_choice = np.random.choice(range(len(his_state)), p=his_prob)
            # index_to_new_residues[proposed_location] = his_state[his_choice]
            pass
        return index_to_new_residues

    def _mutable_residues(self, chain):
        chain_residues = [residue for residue in chain.residues() if residue.id in self._residues_allowed_to_mutate]
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

    def _compute_mutant_key(self,
                            topology,
                            chain_id):
        """
        Compute the key of a mutant topology

        Arguments
        ---------
        topology : simtk.openmm.app.Topology
            topology of the protein

        chain_id : str
            chain id in the topology to be computed

        """
        mutant_key = ''
        chain = None
        wildtype = self._wildtype
        anychains = []
        anywt_chains = []

        #pull the appropriate chain from the (potentially) modified `topology`
        for anychain in topology.chains():
            anychains.append(anychain.id)
            if anychain.id == chain_id:
                chain = anychain
                break

        if chain is None:
            raise Exception(f"Chain {chain_id} not found.  Available chains are {anychains}")

        #pull the appropriate wt chain
        for anywt_chain in wildtype.chains():
            anywt_chains.append(anywt_chain)
            if anywt_chain.id == chain_id:
                wt_chain = anywt_chain
                break

        if not wt_chain:
            raise Exception(f"Chain {chain_id} not found.  Available chains are {anywt_chains}")

        assert len(list(wt_chain.residues())) == len(list(chain.residues())), f"the wt chain and the topology chain do not have the same number of residues."
        for wt_res, res in zip(wt_chain.residues(), chain.residues()):
            if wt_res.name != res.name:
                if mutant_key:
                    mutant_key += '-' # add a hyphen space for every residue that is different between the wt chain and the topology chain
                mutant_key += str(wt_res.name)+str(res.id)+str(res.name) #mutant key has the form for each residue: (wt_res_name, res.id, res_name)
        if not mutant_key:
            mutant_key = 'WT'
        return mutant_key

class PeptideLibraryEngine(PolymerProposalEngine):
    """
    Note: The PeptideLibraryEngine currently doesn't work because PolymerProposalEngine has been modified to handle
    only one mutation at a time (in accordance with the geometry engine).
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
                # his_state = ['HIE','HID']
                # his_prob = np.array([0.5 for i in range(len(his_state))])
                # his_choice = np.random.choice(range(len(his_state)),p=his_prob)
                # index_to_new_residues[residue_index] = his_state[his_choice]
                pass

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
    particle_charges : bool, optional, default=True
        If False, particle charges will be zeroed
    exception_charges : bool, optional, default=True
        If False, exception charges will be zeroed.
    particle_epsilon : bool, optional, default=True
        If False, particle LJ epsilon will be zeroed.
    exception_epsilon : bool, optional, default=True
        If False, exception LJ epsilon will be zeroed.
    torsions : bool, optional, default=True
        If False, torsions will be zeroed.
    """

    def __init__(self, forcefields_to_use, forcefield_kwargs=None, metadata=None, use_antechamber=True, barostat=None,
        particle_charge=True, exception_charge=True, particle_epsilon=True, exception_epsilon=True,
        torsions=True, angles=True):
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

        self._particle_charge = particle_charge
        self._exception_charge = exception_charge
        self._particle_epsilon = particle_epsilon
        self._exception_epsilon = exception_epsilon
        self._torsions = torsions

    def getForceField(self):
        """
        Return the associated ForceField object.

        Returns
        -------
        forcefield : simtk.openmm.app.ForceField
            The current ForceField object.
        """
        return self._forcefield

    def build_system(self, new_topology, check_system=False):
        """
        Build a system from the new_topology, adding templates
        for the molecules in oemol_list

        Parameters
        ----------
        new_topology : simtk.openmm.app.Topology object
            The topology of the system
        check_system : book, optional, default=False`
            If True, will check system for issues following creation

        Returns
        -------
        new_system : openmm.System
            A system object generated from the topology
        """
        # TODO: Write some debug info if exception is raised
        system = self._forcefield.createSystem(new_topology, **self._forcefield_kwargs)

        # Turn off various force classes for debugging if requested
        for force in system.getForces():
            if force.__class__.__name__ == 'NonbondedForce':
                for index in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(index)
                    if not self._particle_charge:
                        charge *= 0
                    if not self._particle_epsilon:
                        epsilon *= 0
                    force.setParticleParameters(index, charge, sigma, epsilon)
                for index in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(index)
                    if not self._exception_charge:
                        chargeProd *= 0
                    if not self._exception_epsilon:
                        epsilon *= 0
                    force.setExceptionParameters(index, p1, p2, chargeProd, sigma, epsilon)
            elif force.__class__.__name__ == 'PeriodicTorsionForce':
                for index in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, K = force.getTorsionParameters(index)
                    if not self._torsions:
                        K *= 0
                    force.setTorsionParameters(index, p1, p2, p3, p4, periodicity, phase, K)

        # Add barostat if requested.
        if self._barostat is not None:
            MAXINT = np.iinfo(np.int32).max
            barostat = openmm.MonteCarloBarostat(*self._barostat)
            seed = np.random.randint(MAXINT)
            barostat.setRandomNumberSeed(seed)
            system.addForce(barostat)

        # See if any torsions have duplicate atoms.
        if check_system:
            from perses.tests import utils
            utils.check_system(system)

        return system

    @property
    def ffxmls(self):
        return self._forcefield_xmls

    @property
    def forcefield(self):
        return self._forcefield

class DummyForceField(object):
    """
    Dummy force field that can add basic parameters to any system for testing purposes.
    """
    def createSystem(self, topology, **kwargs):
        """
        Create a System object with simple parameters from the provided Topology

        Any kwargs are ignored.

        Parameters
        ----------
        topology : simtk.openmm.app.Topology
            The Topology to be parameterized

        Returns
        -------
        system : simtk.openmm.System
            The System object
        """
        from openmmtools.constants import kB
        kT = kB * 300*unit.kelvin

        # Create a System
        system = openmm.System()

        # Add particles
        nonbonded = openmm.CustomNonbondedForce('100/(r/0.1)^4')
        nonbonded.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic);
        nonbonded.setCutoffDistance(1*unit.nanometer)
        system.addForce(nonbonded)
        mass = 12.0 * unit.amu
        for atom in topology.atoms():
            nonbonded.addParticle([])
            system.addParticle(mass)

        # Build a list of which atom indices are bonded to each atom
        bondedToAtom = []
        for atom in topology.atoms():
            bondedToAtom.append(set())
        for (atom1, atom2) in topology.bonds():
            bondedToAtom[atom1.index].add(atom2.index)
            bondedToAtom[atom2.index].add(atom1.index)
        return bondedToAtom

        # Add bonds
        bond_force = openmm.HarmonicBondForce()
        r0 = 1.0 * unit.angstroms
        sigma_r = 0.1 * unit.angstroms
        Kr = kT / sigma_r**2
        for atom1, atom2 in topology.bonds():
            bond_force.addBond(atom1.index, atom2.index, r0, Kr)
        system.addForce(bond_force)

        # Add angles
        uniqueAngles = set()
        for bond in topology.bonds():
            for atom in bondedToAtom[bond.atom1]:
                if atom != bond.atom2:
                    if atom < bond.atom2:
                        uniqueAngles.add((atom, bond.atom1, bond.atom2))
                    else:
                        uniqueAngles.add((bond.atom2, bond.atom1, atom))
            for atom in bondedToAtom[bond.atom2]:
                if atom != bond.atom1:
                    if atom > bond.atom1:
                        uniqueAngles.add((bond.atom1, bond.atom2, atom))
                    else:
                        uniqueAngles.add((atom, bond.atom2, bond.atom1))
        angles = sorted(list(uniqueAngles))
        theta0 = 109.5 * unit.degrees
        sigma_theta = 10 * unit.degrees
        Ktheta = kT / sigma_theta**2
        angle_force = openmm.HarmonicAngleForce()
        for (atom1, atom2, atom3) in angles:
            angles.addAngle(atom1.index, atom2.index, atom3.index, theta0, Ktheta)
        system.addForce(angle_force)

        # Make a list of all unique proper torsions
        uniquePropers = set()
        for angle in angles:
            for atom in bondedToAtom[angle[0]]:
                if atom not in angle:
                    if atom < angle[2]:
                        uniquePropers.add((atom, angle[0], angle[1], angle[2]))
                    else:
                        uniquePropers.add((angle[2], angle[1], angle[0], atom))
            for atom in bondedToAtom[angle[2]]:
                if atom not in angle:
                    if atom > angle[0]:
                        uniquePropers.add((angle[0], angle[1], angle[2], atom))
                    else:
                        uniquePropers.add((atom, angle[2], angle[1], angle[0]))
        propers = sorted(list(uniquePropers))
        torsion_force = openmm.PeriodicTorsionForce()
        periodicity = 3
        phase = 0.0 * unit.degrees
        Kphi = 0.0 * kT
        for (atom1, atom2, atom3, atom4) in propers:
            torsion_force.add_torsion(atom1.index, atom2.index, atom3.index, atom4.index, periodicity, phase, Kphi)
        system.addForce(torsion_force)

        return system

class DummySystemGenerator(SystemGenerator):
    """
    Dummy SystemGenerator that employs a universal simple force field.

    """
    def __init__(self, forcefields_to_use, barostat=None, **kwargs):
        """
        Create a DummySystemGenerator with universal simple force field.

        All parameters except 'barostat' are ignored.

        """
        self._forcefield = DummyForceField()
        self._forcefield_xmls = list()
        self._forcefield_kwargs = dict()
        self._barostat = None
        if barostat is not None:
            pressure = barostat.getDefaultPressure()
            if hasattr(barostat, 'getDefaultTemperature'):
                temperature = barostat.getDefaultTemperature()
            else:
                temperature = barostat.getTemperature()
            frequency = barostat.getFrequency()
            self._barostat = (pressure, temperature, frequency)

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
                 atom_expr=None, bond_expr=None, map_strength='default', proposal_metadata=None,
                 storage=None, always_change=True, atom_map=None):

        # Default atom and bond expressions for MCSS
        if atom_expr is None:
            _logger.info(f'Setting the atom expression to {map_strength}')
            _logger.info(type(map_strength))
            if map_strength == 'default':
                self.atom_expr = DEFAULT_ATOM_EXPRESSION
            elif map_strength == 'weak':
                self.atom_expr = WEAK_ATOM_EXPRESSION
            elif map_strength == 'strong':
                self.atom_expr = STRONG_ATOM_EXPRESSION
            else:
                _logger.warning(f"User defined map_strength: {map_strength} not recognised, setting to default")
                self.atom_expr = DEFAULT_ATOM_EXPRESSION
        else:
            self.atom_expr = atom_expr
            _logger.info(f'Setting the atom expression to user defined: {atom_expr}')
            _logger.info('If map_strength has been set, it will be ignored')
        if bond_expr is None:
            _logger.info(f'Setting the bond expression to {map_strength}')
            if map_strength == 'default':
                self.bond_expr = DEFAULT_BOND_EXPRESSION
            elif map_strength == 'weak':
                self.bond_expr = WEAK_BOND_EXPRESSION
            elif map_strength == 'strong':
                self.bond_expr = STRONG_BOND_EXPRESSION
            else:
                _logger.warning(f"User defined map_strength: {map_strength} not recognised, setting to default")
                self.bond_expr = DEFAULT_BOND_EXPRESSION
        else:
            self.bond_expr = bond_expr
            _logger.info(f'Setting the bond expression to user defined: {bond_expr}')
            _logger.info('If map_strength has been set, it will be ignored')
        self._allow_ring_breaking = True # allow ring breaking

        # Canonicalize all SMILES strings
        self._smiles_list = [SmallMoleculeSetProposalEngine.canonicalize_smiles(smiles) for smiles in set(list_of_smiles)]
        _logger.info(f"smiles list {list_of_smiles} has been canonicalized to {self._smiles_list}")

        self._n_molecules = len(self._smiles_list)

        self._residue_name = residue_name
        self._generated_systems = dict()
        self._generated_topologies = dict()
        self._matches = dict()

        self._storage = None
        if storage is not None:
            self._storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        _logger.info(f"creating probability matrix...")
        self._probability_matrix = self._calculate_probability_matrix(self._smiles_list)

        self._atom_map = atom_map

        super(SmallMoleculeSetProposalEngine, self).__init__(system_generator, proposal_metadata=proposal_metadata, always_change=always_change)

    def propose(self,
                current_system,
                current_topology,
                current_mol = None,
                proposed_mol = None,
                preserve_chirality = True,
                current_metadata = None):
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
        preserve_chirality : bool, default True
            whether to preserve the chirality of the small molecule

        Returns
        -------
        proposal : TopologyProposal object
           topology proposal object
        """
        _logger.info(f"conducting proposal from {self._smiles_list[0]} to {self._smiles_list[1]}...")
        from perses.utils.openeye import createSMILESfromOEMol
        # Determine SMILES string for current small molecule
        if current_mol is None:
            _logger.info(f"current mol was not specified (it is advisable to prespecify an oemol); creating smiles and oemol...")
            current_mol_smiles, current_mol = self._topology_to_smiles(current_topology)
        else:
            _logger.info(f"current mol specified; creating associated smiles...")
            current_mol_smiles = createSMILESfromOEMol(current_mol)
            _logger.info(f"generated current mol smiles: {current_mol_smiles}")

        # Remove the small molecule from the current Topology object
        _logger.info(f"creating current receptor topology by removing small molecule from current topology...")
        current_receptor_topology = self._remove_small_molecule(current_topology)

        # Find the initial atom index of the small molecule in the current topology
        old_mol_start_index, len_old_mol = self._find_mol_start_index(current_topology)
        self.old_mol_start_index = old_mol_start_index
        self.len_old_mol = len_old_mol
        _logger.info(f"small molecule start index: {old_mol_start_index}")
        _logger.info(f"small molecule has {len_old_mol} atoms.")

        # Determine atom indices of the small molecule in the current topology
        old_alchemical_atoms = range(old_mol_start_index, len_old_mol)
        _logger.info(f"old alchemical atom indices: {old_alchemical_atoms}")

        # Select the next molecule SMILES given proposal probabilities
        if proposed_mol is None:
            _logger.info(f"the proposed oemol is not specified; proposing a new molecule from proposal matrix P(M_new | M_old)...")
            proposed_mol_smiles, proposed_mol, logp_proposal = self._propose_molecule(current_system, current_topology, current_mol_smiles)
            _logger.info(f"proposed mol smiles: {proposed_mol_smiles}")
            _logger.info(f"logp proposal: {logp_proposal}")
        else:
            # TODO: Make sure we're using canonical mol to smiles conversion
            proposed_mol_smiles = oechem.OEMolToSmiles(proposed_mol)
            proposed_mol_smiles = SmallMoleculeSetProposalEngine.canonicalize_smiles(proposed_mol_smiles)
            _logger.info(f"proposed mol detected with smiles {proposed_mol_smiles} and logp_proposal of 0.0")
            logp_proposal = 0.0

        # Build the new Topology object, including the proposed molecule
        _logger.info(f"building new topology with proposed molecule and current receptor topology...")
        new_topology = self._build_new_topology(current_receptor_topology, proposed_mol)
        new_mol_start_index, len_new_mol = self._find_mol_start_index(new_topology)
        self.new_mol_start_index = new_mol_start_index
        self.len_new_mol = len_new_mol
        _logger.info(f"new molecule has a start index of {new_mol_start_index} and {len_new_mol} atoms.")

        # Generate an OpenMM System from the proposed Topology
        _logger.info(f"proceeding to build the new system from the new topology...")
        new_system = self._system_generator.build_system(new_topology)

        # Determine atom mapping between old and new molecules
        _logger.info(f"determining atom map between old and new molecules...")
        if not self._atom_map:
            _logger.info(f"the atom map is not specified; proceeding to generate an atom map...")
            mol_atom_map = self._get_mol_atom_map(current_mol, proposed_mol, atom_expr=self.atom_expr,
                                                  bond_expr=self.bond_expr, verbose=self.verbose,
                                                  allow_ring_breaking=self._allow_ring_breaking)

        else:
            _logger.info(f"atom map is pre-determined as {mol_atom_map}")
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

        # now to correct for possible constraint problems
        adjusted_atom_map = SmallMoleculeSetProposalEngine._constraint_repairs(adjusted_atom_map, current_system, new_system, current_topology, new_topology)
        non_offset_new_to_old_atom_map = copy.deepcopy(adjusted_atom_map)
        min_keys, min_values = min(non_offset_new_to_old_atom_map.keys()), min(non_offset_new_to_old_atom_map.values())
        self.non_offset_new_to_old_atom_map = mol_atom_map

        #create NetworkXMolecule for each molecule
        old_networkx_molecule = NetworkXMolecule(mol_oemol = current_mol, mol_residue = current_topology, residue_to_oemol_map = {i: j for i, j in zip(range(old_mol_start_index, len_old_mol), range(len_old_mol - old_mol_start_index))})
        new_networkx_molecule = NetworkXMolecule(mol_oemol = proposed_mol, mol_residue = new_topology, residue_to_oemol_map = {i: j for i, j in zip(range(new_mol_start_index, len_new_mol), range(len_new_mol - new_mol_start_index))})

        # Create the TopologyProposal object
        proposal = TopologyProposal(logp_proposal=logp_proposal, new_to_old_atom_map=adjusted_atom_map,
            old_topology=current_topology, new_topology=new_topology,
            old_system=current_system, new_system=new_system,
            old_alchemical_atoms=old_alchemical_atoms,
            old_chemical_state_key=current_mol_smiles, new_chemical_state_key=proposed_mol_smiles,
            old_residue_name=self._residue_name, new_residue_name=self._residue_name,
            old_networkx_residue = old_networkx_molecule,
            new_networkx_residue = new_networkx_molecule)

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
        _logger.info(f"\tmolecule name specified from residue: {self._residue_name}.")

        matching_molecules = [res for res in topology.residues() if res.name[:3] == molecule_name[:3]]  # Find residue in topology by searching for residues with name "MOL"
        if len(matching_molecules) > 1:
            raise ValueError("More than one residue with the same name!")
        if len(matching_molecules) == 0:
            raise ValueError(f"No residue found with the resname {molecule_name[:3]}")
        mol_res = matching_molecules[0]
        oemol = forcefield_generators.generateOEMolFromTopologyResidue(mol_res)
        _logger.info(f"\toemol generated!")
        smiles_string = oechem.OECreateSmiString(oemol, OESMILES_OPTIONS)
        final_smiles_string = smiles_string
        _logger.info(f"\tsmiles generated from oemol: {final_smiles_string}")
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
        _logger.info(f"\tsetting proposed oemol title to {self._residue_name}")
        oemol_proposed.SetTitle(self._residue_name)
        _logger.info(f"\tcreating mol topology from oemol...")
        mol_topology = forcefield_generators.generateTopologyFromOEMol(oemol_proposed)
        new_topology = app.Topology()
        _logger.info(f"\tappending current receptor topology to new mol topology...")
        append_topology(new_topology, current_receptor_topology)
        append_topology(new_topology, mol_topology)
        # Copy periodic box vectors.
        if current_receptor_topology._periodicBoxVectors != None:
            _logger.info(f"\tperiodic box vectors of the current receptor is specified; copying to new topology...")
            new_topology._periodicBoxVectors = copy.deepcopy(current_receptor_topology._periodicBoxVectors)

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

    @staticmethod
    def enumerate_ring_bonds(molecule, ring_membership, ring_index):
        """Enumerate OEBond objects in ring."""
        for bond in molecule.GetBonds():
            if (ring_membership[bond.GetBgnIdx()] == ring_index) and (ring_membership[bond.GetEndIdx()] == ring_index):
                yield bond

    @staticmethod
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
        for cycle in SmallMoleculeSetProposalEngine.enumerate_cycle_basis(molecule1):
            for bond in cycle:
                # All bonds in this cycle must also be present in molecule2
                if not ((bond.GetBgn() in atom_map) and (bond.GetEnd() in atom_map)):
                    return True # there are no corresponding atoms in molecule2
                if not atom_map[bond.GetBgn()].GetBond(atom_map[bond.GetEnd()]):
                    return True # corresponding atoms have no bond in molecule2
        return False # no rings in molecule1 are broken in molecule2

    @staticmethod
    def preserves_rings(match,
                        current_mol,
                        proposed_mol,
                        matching_criterion = 'index'):
        """
        Returns True if the transformation allows ring systems to be broken or created.

        Arguments
        ---------
        match : oechem.OEMCSSearch.Match iter
            entry in oechem.OEMCSSearch.Match object
        current_mol : openeye.oechem.oemol object
        proposed_mol : openeye.oechem.oemol object
        matching_criterion : str, default 'index'
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        Returns
        -------
        breaks_ring : bool
            whether the transformation breaks the ring

        """
        pattern_to_target_map = SmallMoleculeSetProposalEngine._create_pattern_to_target_map(current_mol, proposed_mol, match, matching_criterion)

        if SmallMoleculeSetProposalEngine.breaks_rings_in_transformation(current_mol, proposed_mol, pattern_to_target_map):
            return False

        target_to_pattern_map = { target_atoms[matchpair.target.GetIdx()] : pattern_atoms[matchpair.pattern.GetIdx()] for matchpair in match.GetAtoms() }
        if SmallMoleculeSetProposalEngine.breaks_rings_in_transformation(proposed_mol, current_mol, target_to_pattern_map):
            return False

        return True

    @staticmethod
    def _create_pattern_to_target_map(current_mol, proposed_mol, match, matching_criterion = 'index'):
        """
        Create a dict of {pattern_atom: target_atom}

        Arguments
        ---------
        current_mol : openeye.oechem.oemol object
        proposed_mol : openeye.oechem.oemol object
        match : oechem.OEMCSSearch.Match iter
            entry in oechem.OEMCSSearch.Match object
        matching_criterion : str, default 'index'
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        Returns
        -------
        pattern_to_target_map : dict
            {pattern_atom: target_atom}
        """
        if matching_criterion == 'index':
            pattern_atoms = { atom.GetIdx() : atom for atom in current_mol.GetAtoms() }
            target_atoms = { atom.GetIdx() : atom for atom in proposed_mol.GetAtoms() }
            pattern_to_target_map = { pattern_atoms[matchpair.pattern.GetIdx()] : target_atoms[matchpair.target.GetIdx()] for matchpair in match.GetAtoms() }
        elif matching_criterion == 'name':
            pattern_atoms = {atom.GetName(): atom for atom in current_mol.GetAtoms()}
            target_atoms = {atom.GetName(): atom for atom in proposed_mol.GetAtoms()}
            pattern_to_target_map = {pattern_atoms[matchpair.pattern.GetName()]: target_atoms[matchpair.target.GetName()] for matchpair in match.GetAtoms()}
        else:
            raise Exception(f"matching criterion {matching_criterion} is not currently supported")
        return pattern_to_target_map


    @staticmethod
    def check_molecule_name_uniqueness(molecule):
        """
        check that the oemol has unique name identifiers

        Arguments
        ---------
        molecule : openeye.oechem.oemol
            molecule to check

        Returns
        -------
        validated : bool
            whether the names are unique identifiers
        """
        if not len(set([atom.GetName() for atom in molecule.GetAtoms()])) == len([atom.GetName() for atom in molecule.GetAtoms()]):
            _logger.warning(f"\t\t\tthe molecule atoms are not uniquely named")
            return False
        else:
            return True


    @staticmethod
    def preserve_chirality(current_mol, proposed_mol, new_to_old_atom_map):
        """
        filters the new_to_old_atom_map for chirality preservation
        The current scheme is implemented as follows:
        for atom_new, atom_old in new_to_old.items():
            if atom_new is R/S and atom_old is undefined:
                # we presume that one of the atom neighbors is being changed, so map it accordingly
            elif atom_new is undefined and atom_old is R/S:
                # we presume that one of the atom neighbors is not being mapped, so map it accordingly
            elif atom_new is R/S and atom_old is R/S:
                # we presume nothing is changing
            elif atom_new is S/R and atom_old is R/S:
                # we presume that one of the neighbors is changing
                # check if all of the neighbors are being mapped:
                    if True, flip two
                    else: do nothing
        """
        pattern_atoms = { atom.GetIdx() : atom for atom in current_mol.GetAtoms() }
        target_atoms = { atom.GetIdx() : atom for atom in proposed_mol.GetAtoms() }
        # _logger.warning(f"\t\t\told oemols: {pattern_atoms}")
        # _logger.warning(f"\t\t\tnew oemols: {target_atoms}")
        copied_new_to_old_atom_map = copy.deepcopy(new_to_old_atom_map)

        for new_index, old_index in new_to_old_atom_map.items():

            if target_atoms[new_index].IsChiral() and not pattern_atoms[old_index].IsChiral():
                #make sure that not all the neighbors are being mapped
                #get neighbor indices:
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if all(nbr in set(list(new_to_old_atom_map.keys())) for nbr in neighbor_indices):
                    _logger.warning(f"the atom map cannot be reconciled with chirality preservation!  It is advisable to conduct a manual atom map.")
                    return {}
                else:
                    #try to remove a hydrogen
                    hydrogen_maps = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms() if atom.GetAtomicNum() == 1]
                    mapped_hydrogens = [_idx for _idx in hydrogen_maps if _idx in list(new_to_old_atom_map.keys())]
                    if mapped_hydrogens != []:
                        del copied_new_to_old_atom_map[mapped_hydrogens[0]]
                    else:
                        _logger.warning(f"there may be a geometry problem!  It is advisable to conduct a manual atom map.")
            elif not target_atoms[new_index].IsChiral() and pattern_atoms[old_index].IsChiral():
                #we have to assert that one of the neighbors is being deleted
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if any(nbr_idx not in list(new_to_old_atom_map.keys()) for nbr_idx in neighbor_indices):
                    pass
                else:
                    _logger.warning(f"the atom map cannot be reconciled with chirality preservation since no hydrogens can be deleted!  It is advisable to conduct a manual atom map.")
                    return {}
            elif target_atoms[new_index].IsChiral() and pattern_atoms[old_index].IsChiral() and oechem.OEPerceiveCIPStereo(current_mol, pattern_atoms[old_index]) == oechem.OEPerceiveCIPStereo(proposed_mol, target_atoms[new_index]):
                #check if all the atoms are mapped
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if all(nbr in set(list(new_to_old_atom_map.keys())) for nbr in neighbor_indices):
                    pass
                else:
                    _logger.warning(f"the atom map cannot be reconciled with chirality preservation since all atom neighbors are being mapped!  It is advisable to conduct a manual atom map.")
                    return {}
            elif target_atoms[new_index].IsChiral() and pattern_atoms[old_index].IsChiral() and oechem.OEPerceiveCIPStereo(current_mol, pattern_atoms[old_index]) != oechem.OEPerceiveCIPStereo(proposed_mol, target_atoms[new_index]):
                neighbor_indices = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms()]
                if all(nbr in set(list(new_to_old_atom_map.keys())) for nbr in neighbor_indices):
                    _logger.warning(f"the atom map cannot be reconciled with chirality preservation since all atom neighbors are being mapped!  It is advisable to conduct a manual atom map.")
                    return {}
                else:
                    #try to remove a hydrogen
                    hydrogen_maps = [atom.GetIdx() for atom in target_atoms[new_index].GetAtoms() if atom.GetAtomicNum() == 1]
                    mapped_hydrogens = [_idx for _idx in hydrogen_maps if _idx in list(new_to_old_atom_map.keys())]
                    if mapped_hydrogens != []:
                        del copied_new_to_old_atom_map[mapped_hydrogens[0]]
                    else:
                        _logger.warning(f"there may be a geometry problem.  It is advisable to conduct a manual atom map.")

        return copied_new_to_old_atom_map #was this really an indentation error?


    @staticmethod
    def rank_degenerate_maps(old_mol,
                             new_mol,
                             matches,
                             matching_criterion = 'index'):
        """
        If the atom/bond expressions for maximal substructure is relaxed, then the maps with the highest scores will likely be degenerate.
        Consequently, it is important to reduce the degeneracy with other tests.

        This test will give each match a score wherein every atom matching with the same atomic number (in aromatic rings) will
        receive a +1 score.

        Arguments
        ---------
        old_mol : openeye.oechem.oemol object
        new_mol : openeye.oechem.oemol object
        matches : oechem.OEMCSSearch.Match objects
            the MCS match oemol objects of the old_mol to new_mol map
        matching_criterion : str, default 'index'
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        Returns
        -------
        top_aliph_matches : dict of {int: oechem.OEMCSSearch.Match}
            {index: oechem.OEMCSSearch.Match}
        """
        score_list = {}
        for idx, match in enumerate(matches):
            counter_arom, counter_aliph = 0, 0
            pattern_to_target_map = SmallMoleculeSetProposalEngine._create_pattern_to_target_map(old_mol, new_mol, match, matching_criterion)
            for pattern_atom, target_atom in pattern_to_target_map.items():
                old_index, new_index = pattern_atom.GetIdx(), target_atom.GetIdx()
                old_atom, new_atom = pattern_atom, target_atom

                if old_atom.IsAromatic() and new_atom.IsAromatic(): #if both are aromatic
                    if old_atom.GetAtomicNum() == new_atom.GetAtomicNum():
                        counter_arom += 1
                else: # TODO: specify whether a single atom is aromatic/aliphatic (for ring form/break purposes)
                    old_atomic_num, new_atomic_num = old_atom.GetAtomicNum(), new_atom.GetAtomicNum()
                    if old_atomic_num != 1 and new_atomic_num == old_atomic_num:
                        counter_aliph += 1

            score_list[idx] = (counter_arom, counter_aliph)

        # return a list of matches with the most aromatic matches
        max_arom_score = max([tup[0] for tup in score_list.values()])
        top_arom_match_dict = {index: match for index, match in enumerate(matches) if score_list[index][0] == max_arom_score}

        #filter further for aliphatic matches...
        max_aliph_score = max([score_list[idx][1] for idx in top_arom_match_dict.keys()])
        top_aliph_matches = [top_arom_match_dict[idx] for idx in top_arom_match_dict.keys() if score_list[idx][1] == max_aliph_score]

        return top_aliph_matches

    @staticmethod
    def hydrogen_mapping_exceptions(old_mol,
                                    new_mol,
                                    match,
                                    matching_criterion):
        """
        Returns an atom map that omits hydrogen-to-nonhydrogen atom maps AND X-H to Y-H where element(X) != element(Y)
        or aromatic(X) != aromatic(Y)

        Arguments
        ---------
        old_mol : openeye.oechem.oemol object
        new_mol : openeye.oechem.oemol object
        match : oechem.OEMCSSearch.Match iter
        matching_criterion : str, default 'index'
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        """
        new_to_old_atom_map = {}
        pattern_to_target_map = SmallMoleculeSetProposalEngine._create_pattern_to_target_map(old_mol, new_mol, match, matching_criterion)
        for pattern_atom, target_atom in pattern_to_target_map.items():
            old_index, new_index = pattern_atom.GetIdx(), target_atom.GetIdx()
            old_atom, new_atom = pattern_atom, target_atom

            #Check if a hydrogen was mapped to a non-hydroden (basically the xor of is_h_a and is_h_b)
            if (old_atom.GetAtomicNum() == 1) != (new_atom.GetAtomicNum() == 1):
                continue

            new_to_old_atom_map[new_index] = old_index

        return new_to_old_atom_map


    @staticmethod
    def _constraint_repairs(atom_map, old_system, new_system, old_topology, new_topology):
        """
        Given an adjusted atom map (corresponding to the true indices of the new: old atoms in their respective systems), iterate through all of the
        atoms in the map that are hydrogen and check if the constraint length changes; if so, we do not map.
        """
        old_hydrogens = list(atom.index for atom in old_topology.atoms() if atom.element == app.Element.getByAtomicNumber(1))
        new_hydrogens = list(atom.index for atom in new_topology.atoms() if atom.element == app.Element.getByAtomicNumber(1))
        #wrapping constraints
        old_constraints, new_constraints = {}, {}
        for idx in range(old_system.getNumConstraints()):
            atom1, atom2, length = old_system.getConstraintParameters(idx)
            if atom1 in old_hydrogens:
                old_constraints[atom1] = length
            elif atom2 in old_hydrogens:
                old_constraints[atom2] = length

        for idx in range(new_system.getNumConstraints()):
            atom1, atom2, length = new_system.getConstraintParameters(idx)
            if atom1 in new_hydrogens:
                new_constraints[atom1] = length
            elif atom2 in new_hydrogens:
                new_constraints[atom2] = length

         #iterate through the atom indices in the new_to_old map, check bonds for pairs, and remove appropriate matches
        to_delete = []
        for new_index, old_index in atom_map.items():
            if new_index in new_constraints.keys() and old_index in old_constraints.keys(): # both atom indices are hydrogens
                old_length, new_length = old_constraints[old_index], new_constraints[new_index]
                if not old_length == new_length: #then we have to remove it from
                    to_delete.append(new_index)

        for idx in to_delete:
            del atom_map[idx]

        return atom_map

    @staticmethod
    def _get_mol_atom_map(current_molecule,
                          proposed_molecule,
                          atom_expr=None,
                          bond_expr=None,
                          verbose=False,
                          allow_ring_breaking=True,
                          matching_criterion = 'index'):
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
        matching_criterion : str, default 'index'
            the best atom map is pulled based on some ranking criterion;
            if 'index', the best atom map is chosen based on the map with the maximum number of atom index matches
            if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
            else, raise exception.
            NOTE: the matching criterion pulls pattern and target matches based on indices or names;
            if 'names' is chosen, it is first asserted that the current_molecule and proposed_molecule have atoms that are uniquely named.


        Returns
        -------
        top_match : dict
            new_to_old_atom_map
        """
        if atom_expr is None:
            _logger.warning('atom_expr not set. using DEFAULT')
            atom_expr = DEFAULT_ATOM_EXPRESSION
        if bond_expr is None:
            _logger.warning('atom_expr not set. using DEFAULT')
            bond_expr = DEFAULT_BOND_EXPRESSION

        if matching_criterion == 'name':
            #assert the names are unique:
            for _mol in [current_molecule, proposed_molecule]:
                if SmallMoleculeSetProposalEngine.check_molecule_name_uniqueness(current_molecule):
                    pass
                else:
                    _logger.warning(f"\tname uniqueness assertion failed. returning empty top_match dict")
                    return {}

        # this ensures that the hybridization of the oemols is done for correct atom mapping
        oechem.OEAssignHybridization(current_molecule)
        oechem.OEAssignHybridization(proposed_molecule)

        # retrieve graph mols
        oegraphmol_current = oechem.OEGraphMol(current_molecule) # pattern molecule
        oegraphmol_proposed = oechem.OEGraphMol(proposed_molecule) # target molecule

        #mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)
        mcs.Init(oegraphmol_current, atom_expr, bond_expr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        unique = False
        matches = [m for m in mcs.Match(oegraphmol_proposed, unique)]
        _logger.debug(f"\tnumber of matches: {len(matches)}")

        if allow_ring_breaking is False:
            # Filter the matches to remove any that allow ring breaking
            matches = [m for m in matches if SmallMoleculeSetProposalEngine.preserves_rings(m, oegraphmol_current, oegraphmol_proposed, matching_criterion = matching_criterion)]

        if not matches:
            #raise Exception(f"There are no atom map matches that preserve rings!  It is advisable to conduct a manual atom mapping.")
            _logger.warn(f"There are no atom map matches that preserve the ring!  It is advisable to conduct a manual atom map.")
            return {}

        try:
            top_matches = SmallMoleculeSetProposalEngine.rank_degenerate_maps(current_molecule, proposed_molecule, matches, matching_criterion) #remove the matches with the lower rank score (filter out bad degeneracies)
        except Exception as e:
            _logger.warning(f"\t rank_degenerate_maps: {e}")
            top_matches = matches

        _logger.debug(f"\tthere are {len(top_matches)} top matches")
        max_num_atoms = max([match.NumAtoms() for match in top_matches])
        _logger.debug(f"\tthe max number of atom matches is: {max_num_atoms}; there are {len([m for m in top_matches if m.NumAtoms() == max_num_atoms])} matches herein")
        new_top_matches = [m for m in top_matches if m.NumAtoms() == max_num_atoms]
        new_to_old_atom_maps = [SmallMoleculeSetProposalEngine.hydrogen_mapping_exceptions(current_molecule, proposed_molecule, match, matching_criterion) for match in new_top_matches]
        _logger.debug(f"\tnew to old atom maps with most atom hits: {new_to_old_atom_maps}")


        #now all else is equal; we will choose the map with the highest overlap of atom indices
        index_overlap_numbers = []
        if matching_criterion == 'index':
            for map in new_to_old_atom_maps:
                hit_number = 0
                for key, value in map.items():
                    if key == value:
                        hit_number += 1
                index_overlap_numbers.append(hit_number)
        elif matching_criterion == 'name':
            for map in new_to_old_atom_maps:
                hit_number = 0
                map_tuples = list(map.items())
                atom_map = {atom_new: atom_old for atom_new, atom_old in zip(list(proposed_molecule.GetAtoms()), list(current_molecule.GetAtoms())) if (atom_new.GetIdx(), atom_old.GetIdx()) in map_tuples}
                for key, value in atom_map.items():
                    if key.GetName() == value.GetName():
                        hit_number += 1
                index_overlap_numbers.append(hit_number)
        else:
            raise Exception(f"the ranking criteria {ranking_criteria} is not supported.")

        max_index_overlap_number = max(index_overlap_numbers)
        _logger.debug(f"\tmax index overlap num: {max_index_overlap_number}")
        max_index = index_overlap_numbers.index(max_index_overlap_number)
        _logger.debug(f"\tchose {new_to_old_atom_maps[max_index]}")

        top_match = new_to_old_atom_maps[max_index]

        return top_match

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
            _logger.info(f"\tcurrent smiles index: {current_smiles_idx}")
        except ValueError as e:
            msg = f"Current SMILES string {molecule_smiles} not found in canonical molecule set.\nMolecule set: {self._smiles_list}"
            raise Exception(msg)

        # Propose a new molecule
        molecule_probabilities = self._probability_matrix[current_smiles_idx, :]
        _logger.info(f"\tmolecule probabilities: {molecule_probabilities}")
        proposed_smiles_idx = np.random.choice(range(len(self._smiles_list)), p=molecule_probabilities)
        _logger.info(f"\tproposed smiles index chosen: {proposed_smiles_idx}")
        reverse_probability = self._probability_matrix[proposed_smiles_idx, current_smiles_idx]
        forward_probability = molecule_probabilities[proposed_smiles_idx]
        _logger.info(f"\tforward probability: {forward_probability}")
        _logger.info(f"\treverse probability: {reverse_probability}")
        proposed_smiles = self._smiles_list[proposed_smiles_idx]
        _logger.info(f"\tproposed smiles: {proposed_smiles}")
        logp = np.log(reverse_probability) - np.log(forward_probability)
        from perses.utils.openeye import smiles_to_oemol
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
        from perses.tests.utils import createSystemFromSMILES
        import itertools
        from perses.rjmc.geometry import ProposalOrderTools
        from perses.tests.test_geometry_engine import oemol_to_openmm_system
        safe_smiles = set()
        smiles_pairs = set()
        smiles_set = set(smiles_list)

        for mol1, mol2 in itertools.combinations(smiles_list, 2):
            smiles_pairs.add((mol1, mol2))

        for smiles_pair in smiles_pairs:
            mol1, sys1, pos1, top1 = createSystemFromSMILES(smiles_pair[0])
            mol2, sys2, pos2, top2 = createSystemFromSMILES(smiles_pair[1])

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

    def propose(self, current_system, current_topology, current_smiles=None, proposed_mol=None, map_index=None, current_metadata=None):
        """
        Propose the next state, given the current state

        Parameters
        ----------
        current_system : openmm.System object
            the system of the current state
        current_topology : app.Topology object
            the topology of the current state
        current_smiles : str, default None
            Specify the current SMILES string to avoid perceiving it from the topology. If None, perceive from topology.
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
        # Determine SMILES string for current small molecule if the SMILES isn't specified
        if current_smiles is None:
            current_mol_smiles, _ = self._topology_to_smiles(current_topology)
        else:
            current_mol_smiles = current_smiles

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

class NetworkXMolecule(object):
    """
    Creates a networkx representation of an atom set to allow for easy querying
    """
    def __init__(self, mol_oemol, mol_residue, residue_to_oemol_map):
        """
        mol_oemol : oechem.OEMol
            oemol object to interpret atom topologies in terms of oemol characteristics
        mol_residue : simtk.openmm.app.topology.Residue
            base from which to interpret a molecule
        residue_to_oemol_map : dict
            map of the residue indices to the oemol indices

        #NOTE: the atoms comprising the mol_residue must be a subset fo the mol_oemol atoms
        """
        #subset assertion
        assert set([atom.name for atom in mol_residue.atoms()]).issubset(set([atom.GetName() for atom in mol_oemol.GetAtoms()])), f"the mol_residue is not a subset of the mol_oemol"

        #the first thing to do is to create a simple undirected graph based on covalency
        self.mol_oemol = mol_oemol
        self.mol_residue = mol_residue
        self.residue_to_oemol_map = residue_to_oemol_map
        _logger.debug(f"\tresidue_to_oemol_map: {residue_to_oemol_map}")
        self.graph = nx.Graph()

        self.reverse_residue_to_oemol_map = {val : key for key, val in residue_to_oemol_map.items()}
        oemol_atom_dict = {atom.GetIdx() : atom for atom in self.mol_oemol.GetAtoms()}
        _logger.debug(f"\toemol_atom_dict: {oemol_atom_dict}")
        reverse_oemol_atom_dict = {val : key for key, val in oemol_atom_dict.items()}

        #try to perceive chirality
        for atom in self.mol_oemol.GetAtoms():
            nbrs = [] #we have to get the neighbors first
            for bond in atom.GetBonds():
                nbor = bond.GetNbr(atom)
                nbrs.append(nbor)

            match_found = False

            if atom.IsChiral() and len(nbrs) >= 4:
                stereo = oechem.OEPerceiveCIPStereo(self.mol_oemol, atom)
                oechem.OESetCIPStereo(self.mol_oemol, atom, stereo)
                match_found = True
                if not match_found:
                    raise Exception("Error: Stereochemistry was not assigned to all chiral atoms from the smiles string. (i.e. stereochemistry is undefined)")

        #add atoms
        _logger.debug(f"\tadding atoms to networkx graph")
        for atom in mol_residue.atoms():
            atom_index = atom.index
            _logger.debug(f"\t\tadding top atom index: {atom_index}")
            self.graph.add_node(atom_index)
            self.graph.nodes[atom_index]['openmm_atom'] = atom
            _logger.debug(f"\t\tcorresponding oemol index: {residue_to_oemol_map[atom_index]}")
            self.graph.nodes[atom_index]['oechem_atom'] = oemol_atom_dict[residue_to_oemol_map[atom_index]]

        #make a simple list of the nodes for bookkeeping purposes
        #if the res is bonded to another res, then we do not want to include that in the oemol...
        nodes_set = set(list(self.graph.nodes()))
        for bond in mol_residue.bonds():
            bond_atom0, bond_atom1 = bond[0].index, bond[1].index
            if set([bond_atom0, bond_atom1]).issubset(nodes_set):
                self.graph.add_edge(bond[0].index, bond[1].index)
                self.graph.edges[bond[0].index, bond[1].index]['openmm_bond'] = bond
            else:
                pass

        for bond in self.mol_oemol.GetBonds():
            index_a, index_b = bond.GetBgnIdx(), bond.GetEndIdx()
            try:
                index_rev_a = self.reverse_residue_to_oemol_map[index_a]
                index_rev_b = self.reverse_residue_to_oemol_map[index_b]

                if (index_rev_a, index_rev_b) in list(self.graph.edges()) or (index_rev_b, index_rev_a) in list(self.graph.edges()):
                    self.graph.edges[index_rev_a, index_rev_b]['oemol_bond'] = bond
            except Exception as e:
                _logger.debug(f"\tbond oemol loop exception: {e}")
                pass

        _logger.debug(f"\tgraph nodes: {self.graph.nodes()}")

    def remove_oemols_from_graph(self):
        """
        Remove oemol atoms and bonds from the graph
        """
        for atom in self.graph.nodes(data=True):
            atom[1]['oechem_atom'] = None
        for bond in self.graph.edges(data=True):
            bond[2]['oemol_bond'] = None

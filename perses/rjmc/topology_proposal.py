"""
This file contains the base classes for topology proposals
"""

from simtk.openmm import app

import copy
import logging
import itertools
import os
import numpy as np
import networkx as nx
import openmoltools.forcefield_generators as forcefield_generators
from perses.storage import NetCDFStorageView
from perses.rjmc.geometry import NoTorsionError
from functools import partial
from simtk import unit # needed for unit-bearing quantity defaults
try:
    from subprocess import getoutput  # If python 3
except ImportError:
    from commands import getoutput  # If python 2

################################################################################
# LOGGER
################################################################################

import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("proposal_generator")
_logger.setLevel(logging.INFO)

################################################################################
# UTILITIES
################################################################################

def add_method(object, function):
        """
        Bind a function to an object instance

        Parameters
        ----------
        object : class instance
            object to which function will be bound
        function : function
            function which will be bound to object
        """
        setattr(object, function.__name__, partial(function, object))

def set_residue_oemol_and_openmm_topology_attributes(object, residue_oemol, residue_topology, residue_to_oemol_map):
    """
    Add the following attributes to an openmm.Topology:
        current openmm topology with a residue oemol,
        openmm.Topology.residue,
        and the corresponding index map.

    Parameters
    ----------
    object : class instance
        object to which function will be bound
    residue_oemol : openeye.oechem.OEMol
        oemol of the residue of interest
    residue_topology : simtk.openmm.Topology.residue
        the residue of interest
    residue_to_oemol_map : dict
        dictionary of the residue_topology indices to the residue_oemol indices

    NOTE: the atoms comprising the residue_topology must be a subset fo the residue_oemol atoms
    """
    assert set([atom.name for atom in residue_topology.atoms()]).issubset(set([atom.GetName() for atom in residue_oemol.GetAtoms()])), f"the self.residue_topology is not a subset of the self.residue_oemol"
    for attribute, name in zip([residue_oemol, residue_topology, residue_to_oemol_map], ['residue_oemol', 'residue_topology', 'residue_to_oemol_map']):
        setattr(object, name, attribute)

    reverse_residue_to_oemol_map = {val : key for key, val in residue_to_oemol_map.items()}
    setattr(object, 'reverse_residue_to_oemol_map', reverse_residue_to_oemol_map)

def _get_networkx_molecule(self):
    """
    Returns
    -------
    graph : NetworkX.Graph
        networkx representation of the residue
    """
    import openeye.oechem as oechem
    graph = nx.Graph()

    oemol_atom_dict = {atom.GetIdx() : atom for atom in self.residue_oemol.GetAtoms()}
    _logger.debug(f"\toemol_atom_dict: {oemol_atom_dict}")
    reverse_oemol_atom_dict = {val : key for key, val in oemol_atom_dict.items()}

    #try to perceive chirality
    for atom in self.residue_oemol.GetAtoms():
        nbrs = [] #we have to get the neighbors first
        for bond in atom.GetBonds():
            nbor = bond.GetNbr(atom)
            nbrs.append(nbor)

        match_found = False

        if atom.IsChiral() and len(nbrs) >= 4:
            stereo = oechem.OEPerceiveCIPStereo(self.residue_oemol, atom)
            oechem.OESetCIPStereo(self.residue_oemol, atom, stereo)
            match_found = True
            if not match_found:
                raise Exception("Error: Stereochemistry was not assigned to all chiral atoms from the smiles string. (i.e. stereochemistry is undefined)")

    #add atoms
    _logger.debug(f"\tadding atoms to networkx graph")
    for atom in self.residue_topology.atoms():
        atom_index = atom.index
        _logger.debug(f"\t\tadding top atom index: {atom_index}")
        graph.add_node(atom_index)
        graph.nodes[atom_index]['openmm_atom'] = atom
        _logger.debug(f"\t\tcorresponding oemol index: {self.residue_to_oemol_map[atom_index]}")
        graph.nodes[atom_index]['oechem_atom'] = oemol_atom_dict[self.residue_to_oemol_map[atom_index]]

    #make a simple list of the nodes for bookkeeping purposes
    #if the res is bonded to another res, then we do not want to include that in the oemol...
    nodes_set = set(list(graph.nodes()))
    for bond in self.residue_topology.bonds():
        bond_atom0, bond_atom1 = bond[0].index, bond[1].index
        if set([bond_atom0, bond_atom1]).issubset(nodes_set):
            graph.add_edge(bond[0].index, bond[1].index)
            graph.edges[bond[0].index, bond[1].index]['openmm_bond'] = bond
        else:
            pass

    for bond in self.residue_oemol.GetBonds():
        index_a, index_b = bond.GetBgnIdx(), bond.GetEndIdx()
        try:
            index_rev_a = self.reverse_residue_to_oemol_map[index_a]
            index_rev_b = self.reverse_residue_to_oemol_map[index_b]

            if (index_rev_a, index_rev_b) in list(graph.edges()) or (index_rev_b, index_rev_a) in list(graph.edges()):
                graph.edges[index_rev_a, index_rev_b]['oemol_bond'] = bond
        except Exception as e:
            _logger.debug(f"\tbond oemol loop exception: {e}")

    _logger.debug(f"\tgraph nodes: {graph.nodes()}")
    return graph

def augment_openmm_topology(topology, residue_oemol, residue_topology, residue_to_oemol_map):
    """
    Add the networkx builder tools as attribute and methods to the specified topology

    Parameters
    ----------
    topology : simtk.openmm.topology.Topology
        topology that will be augmented
    residue_oemol : openeye.oechem.OEMol
        oemol of the residue of interest
    residue_topology : simtk.openmm.Topology.residue
        the residue of interest
    residue_to_oemol_map : dict
        dictionary of the residue_topology indices to the residue_oemol indices
    """
    set_residue_oemol_and_openmm_topology_attributes(topology, residue_oemol, residue_topology, residue_to_oemol_map)
    add_method(topology, _get_networkx_molecule)



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

def has_h_mapped(atommap, mola, molb):
    """
    Parameters
    mola : oechem.OEMol
    molb : oechem.OEMol
    """

    import openeye.oechem as oechem
    for a_atom, b_atom in atommap.items():
        if mola.GetAtom(oechem.OEHasAtomIdx(a_atom)).GetAtomicNum() == 1 or molb.GetAtom(oechem.OEHasAtomIdx(b_atom)).GetAtomicNum() == 1:
            return True

    return False

class InvalidMappingException(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class AtomMapper(object):
    """ Generates map between two molecules.
    As this doesn't generate a system, it will not be
    accurate for whether hydrogens are mapped or not.
    It also doesn't check that this is feasible to simulate,
    but is just helpful for testing options.

    Parameters
    ----------
    current_molecule : oechem.OEMol
        starting molecule
    proposed_molecule : oechem.OEMol
        molecule to move to
    atom_expr : oechem.OEExprOpts
        atom requirement to consider two bonds as the same
    bond_expr : oechem.OEExprOpts
        bond requirement to consider two bonds as the same
    allow_ring_breaking : bool, default=True
        Wether or not to allow ring breaking in map
    map_strategy : str, default = 'core'

    Attributes
    ----------
    map : type
        Description of attribute `map`.
    _get_mol_atom_map : type
        Description of attribute `_get_mol_atom_map`.
    current_molecule
    proposed_molecule
    atom_expr
    bond_expr
    allow_ring_breaking
    map_strategy

    """

    def _score_maps(mol_A, mol_B, maps):
        """ Gives a score for how well each map in a list
        recapitulates the geometry of ligand B.
        If the geometry of ligand B is known, it can identify the closest map,
        if it's not known, it can still be helpful as maps with the same score
        are redundant --- i.e. a flipped phenyl ring.


        Parameters
        ----------
        mol_A : oechem.oemol
            old molecule
        mol_B : oechem.oemol
            new molecule
        maps : list(dict)
            list of maps to score

        Returns
        -------
        list
            list of the distance scores

        """
        coords_A = np.zeros(shape=(mol_A.NumAtoms(), 3))
        for i in mol_A.GetCoords():
            coords_A[i] = mol_A.GetCoords()[i]
        coords_B = np.zeros(shape=(mol_B.NumAtoms(), 3))
        for i in mol_B.GetCoords():
            coords_B[i] = mol_B.GetCoords()[i]
        from scipy.spatial.distance import cdist

        all_to_all = cdist(coords_A, coords_B, 'euclidean')

        mol_B_H = {x.GetIdx(): x.IsHydrogen() for x in mol_B.GetAtoms()}
        all_scores = []
        for M in maps:
            map_score = 0
            for atom in M:
                if not mol_B_H[atom]:  # skip H's - only look at heavy atoms
                    map_score += all_to_all[M[atom], atom]
            all_scores.append(map_score)
        return all_scores

    @staticmethod
    def _get_mol_atom_map_by_positions(molA, molB, coordinate_tolerance=0.2*unit.angstroms):
        """
        Return an atom map whereby atoms are mapped if the positions of the atoms are overlapping via np.isclose

        Parameters
        ----------
        molA : openeye.oechem.OEMol
            First molecule
        molA : openeye.oechem.OEMol
            Second molecule
        coordinate_tolerance : simtk.unit.Quantity with units of length, optional, default=0.2*unit.angstroms
            The absolute coordinate tolerance for mapping atoms

        Returns
        -------
        mapping : dict of int : int
            mapping[molB_index] = molA_index is the mapping of atoms from molA to molB that are geometrically close

        .. TODO :: Add sanity checking and ensure that partial rings are de-mapped.

        .. TODO :: Use openff.toolkit.topology.Molecule instead of OEMol

        """
        molA_positions = molA.GetCoords() # coordinates (Angstroms)
        molB_positions = molB.GetCoords() # coordinates (Angstroms)
        molB_backward_positions = {val: key for key, val in molB_positions.items()}

        rtol = 0.0 # relative tolerane
        atol = coordinate_tolerance / unit.angstroms # absolute tolerance (Angstroms)

        returnable = {}
        for a_idx, a_pos_tup in molA_positions.items():
            match_pos = [idx for idx, b_pos in molB_positions.items() if np.allclose(np.array(a_pos_tup), np.array(b_pos), rtol=rtol, atol=atol)]
            if not len(match_pos) in [0,1]:
                raise Exception(f"there are multiple molB positions with the same coordinates as molA index {a_idx} (by OEMol indexing)")
            if len(match_pos) == 1:
                returnable[match_pos[0]] = a_idx
                #returnable[a_idx] = match_pos[0]

        return returnable

    @staticmethod
    def _get_mol_atom_map(molA,
                          molB,
                          atom_expr=None,
                          bond_expr=None,
                          map_strength='default',
                          allow_ring_breaking=True,
                          matching_criterion='index',
                          external_inttypes=False,
                          map_strategy='core'):
        """Find a suitable atom map between two molecules, according
        to the atom_expr, bond_expr or map_strength

        Parameters
        ----------
        molA : oechem.oemol
            old molecule
        molB : oechem.oemol
            new molecule
        atom_expr : int, default=None
            integer corresponding to atom matching, see `perses.openeye.generate_expression`
        bond_expr : int, default=None
            integer corresponding to bond matching, see `perses.openeye.generate_expression`
        map_strength : str, default 'default'
            pre-defined mapping strength that can be one of ['strong', 'default', 'weak']
            this will be ignored if either atom_expr or bond_expr have been defined.
        allow_ring_breaking : bool, optional, default=True
             If False, will check to make sure rings are not being broken or formed.
        matching_criterion : str, default 'index'
             The best atom map is pulled based on some ranking criteria;
             if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
             if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
             else: raise Exception.
             NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                    if 'names' is chosen, it is first asserted that the current_oemol and the proposed_oemol have atoms that are uniquely named
        external_inttypes : bool, default False
            If True, IntTypes already assigned to oemols will be used for mapping, if IntType is in the atom or bond expression.
            Otherwise, IntTypes will be overwritten such as to ensure rings of different sizes are not matched.
        map_strategy : str, default='core'
            determines which map is considered the best and returned
            can be one of ['geometry', 'matching_criterion', 'random', 'weighted-random', 'return-all']
            - core will return the map with the largest number of atoms in the core. If there are multiple maps with the same highest score, then `matching_criterion` is used to tie break
            - geometry uses the coordinates of the molB oemol to calculate the heavy atom distance between the proposed map and the actual geometry
            this can be vital for getting the orientation of ortho- and meta- substituents correct in constrained (i.e. protein-like) environments.
            this is ONLY useful if the positions of ligand B are known and/or correctly aligned.
            - matching_criterion uses the `matching_criterion` flag to pick which of the maps best satisfies a 2D requirement.
            - random will use a random map of those that are possible
            - weighted-random uses a map chosen at random, proportional to how close it is in geometry to ligand B. The same as for 'geometry', this requires the coordinates of ligand B to be meaninful
            - return-all BREAKS THE API as it returns a list of dicts, rather than list. This is intended for development code, not main pipeline.
        Returns
        -------
        dict
            dictionary of scores (keys) and maps (dict)

        """
        import openeye.oechem as oechem

        # weak requirements for mapping atoms == more atoms mapped, more in core
        # atoms need to match in aromaticity. Same with bonds.
        # maps ethane to ethene, CH3 to NH2, but not benzene to cyclohexane
        WEAK_ATOM_EXPRESSION = oechem.OEExprOpts_EqAromatic | oechem.OEExprOpts_EqNotAromatic #| oechem.OEExprOpts_IntType
        WEAK_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

        # default atom expression, requires same aromaticitiy and hybridization
        # bonds need to match in bond order
        # ethane to ethene wouldn't map, CH3 to NH2 would map but CH3 to HC=O wouldn't
        DEFAULT_ATOM_EXPRESSION = oechem.OEExprOpts_Hybridization #| oechem.OEExprOpts_IntType
        DEFAULT_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

        # strong requires same hybridization AND the same atom type
        # bonds are same as default, require them to match in bond order
        STRONG_ATOM_EXPRESSION = oechem.OEExprOpts_Hybridization | oechem.OEExprOpts_HvyDegree | oechem.OEExprOpts_DefaultAtoms
        STRONG_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

        allowed_map_strategy = ['core','geometry', 'matching_criterion', 'random', 'weighted-random', 'return-all']
        assert map_strategy in allowed_map_strategy, f'map_strategy cannot be {map_strategy}, it must be one of the allowed options {allowed_map_strategy}.'
        _logger.info(f'Using {map_strategy} to chose best atom map')


        map_strength_dict = {'default': [DEFAULT_ATOM_EXPRESSION, DEFAULT_BOND_EXPRESSION],
                             'weak': [WEAK_ATOM_EXPRESSION, WEAK_BOND_EXPRESSION],
                             'strong': [STRONG_ATOM_EXPRESSION, STRONG_BOND_EXPRESSION]}
        if map_strength is None:
            map_strength = 'default'

        if atom_expr is None:
            _logger.debug(f'No atom expression defined, using map strength : {map_strength}')
            atom_expr = map_strength_dict[map_strength][0]
        if bond_expr is None:
            _logger.debug(f'No bond expression defined, using map strength : {map_strength}')
            bond_expr = map_strength_dict[map_strength][1]


        if not external_inttypes or allow_ring_breaking:
            molA = AtomMapper._assign_ring_ids(molA)
            molB = AtomMapper._assign_ring_ids(molB)

        from perses.utils.openeye import get_scaffold
        scaffoldA = get_scaffold(molA)
        scaffoldB = get_scaffold(molB)

        for atom in scaffoldA.GetAtoms():
            atom.SetIntType(AtomMapper._assign_atom_ring_id(atom))
        for atom in scaffoldB.GetAtoms():
            atom.SetIntType(AtomMapper._assign_atom_ring_id(atom))


        scaffold_maps = AtomMapper._get_all_maps(scaffoldA, scaffoldB,
                                                 atom_expr=oechem.OEExprOpts_RingMember | oechem.OEExprOpts_IntType,
                                                 bond_expr=oechem.OEExprOpts_RingMember,
                                                 external_inttypes=True,
                                                 unique=False,
                                                 matching_criterion=matching_criterion)


        _logger.info(f'Scaffold has symmetry of {len(scaffold_maps)}')

        if len(scaffold_maps) == 0:
            _logger.warning('Two molecules are not similar to have a common scaffold')
            _logger.warning('Proceeding with direct mapping of molecules, but please check atom mapping and the geometry of the ligands.')

            # if no commonality with the scaffold, don't use it.
            # why weren't matching arguments carried to these mapping functions? is there an edge case that i am missing?
            # it still doesn't fix the protein sidechain mapping problem
            all_molecule_maps = AtomMapper._get_all_maps(molA, molB,
                                                     external_inttypes=external_inttypes,
                                                     atom_expr=atom_expr,
                                                     bond_expr=bond_expr,
                                                     matching_criterion=matching_criterion)
            _logger.info(f'len {all_molecule_maps}')
            for x in all_molecule_maps:
                _logger.info(x)

        else:

            max_mapped = max([len(m) for m in scaffold_maps])
            _logger.info(f'There are {len(scaffold_maps)} before filtering')
            scaffold_maps = [m for m in scaffold_maps if len(m) == max_mapped]
            _logger.info(f'There are {len(scaffold_maps)} after filtering to remove maps with fewer matches than {max_mapped} atoms')

            scaffold_A_maps = AtomMapper._get_all_maps(molA, scaffoldA,
                                     atom_expr=oechem.OEExprOpts_AtomicNumber,
                                     bond_expr=0,
                                     matching_criterion=matching_criterion)
            _logger.info(f'{len(scaffold_A_maps)} scaffold maps for A')
            scaffold_A_map = scaffold_A_maps[0]
            _logger.info(f'Scaffold to molA: {scaffold_A_map}')
            assert len(scaffold_A_map) == scaffoldA.NumAtoms(), f'Scaffold should be fully contained within the molecule it came from. {len(scaffold_A_map)} in map, and {scaffoldA.NumAtoms()} in scaffold'


            scaffold_B_maps = AtomMapper._get_all_maps(molB, scaffoldB,
                                     atom_expr=oechem.OEExprOpts_AtomicNumber,
                                     bond_expr=0,
                                     matching_criterion=matching_criterion)
            _logger.info(f'{len(scaffold_B_maps)} scaffold maps for B')
            scaffold_B_map = scaffold_B_maps[0]
            _logger.info(f'Scaffold to molB: {scaffold_B_map}')
            assert len(scaffold_B_map) == scaffoldB.NumAtoms(), f'Scaffold should be fully contained within the molecule it came from. {len(scaffold_B_map)} in map, and {scaffoldB.NumAtoms()} in scaffold'

            # now want to find all of the maps
            # for all of the possible scaffold  symmetries
            all_molecule_maps = []
            for scaffold_map in scaffold_maps:
                if external_inttypes is False and allow_ring_breaking is True:
                    # reset the IntTypes
                    for atom in molA.GetAtoms():
                        atom.SetIntType(0)
                    for atom in molB.GetAtoms():
                        atom.SetIntType(0)

                    index = 1
                    for scaff_b_id, scaff_a_id in scaffold_map.items():
                        for atom in molA.GetAtoms():
                            if atom.GetIdx() == scaffold_A_map[scaff_a_id]:
                                atom.SetIntType(index)
                        for atom in molB.GetAtoms():
                            if atom.GetIdx() == scaffold_B_map[scaff_b_id]:
                                atom.SetIntType(index)
                        index += 1
                    for atom in molA.GetAtoms():
                        if atom.GetIntType() == 0:
                            atom.SetIntType(AtomMapper._assign_atom_ring_id(atom))
                    for atom in molB.GetAtoms():
                        if atom.GetIntType() == 0:
                            atom.SetIntType(AtomMapper._assign_atom_ring_id(atom))

                molecule_maps = AtomMapper._get_all_maps(molA, molB,
                                                     external_inttypes=True,
                                                     atom_expr=atom_expr,
                                                     bond_expr=bond_expr,
                                                     matching_criterion=matching_criterion)
                all_molecule_maps.extend(molecule_maps)

        if not allow_ring_breaking:
            # Filter the matches to remove any that allow ring breaking
            all_molecule_maps = [m for m in all_molecule_maps if AtomMapper.preserves_rings(m, molA, molB)]
            _logger.info(f'Checking maps to see if they break rings')
        if len(all_molecule_maps) == 0:
            _logger.warning('No maps found. Try relaxing match criteria or setting allow_ring_breaking to True')
            return None

        if map_strategy == 'return-all':
            _logger.warning('Returning a list of all maps, rather than a dictionary.')
            return all_molecule_maps


        #  TODO - there will be other options that we might want in future here
        #  maybe _get_mol_atom_map  should return a list of maps and then we have
        # a pick_map() function elsewhere?
        # but this would break the API so I'm not doing it now
        if len(all_molecule_maps) == 1:
            _logger.info('Only one map so returning that one')
            return all_molecule_maps[0] #  can this be done in a less ugly way??
        if map_strategy == 'geometry':
            molecule_maps_scores = AtomMapper._remove_redundant_maps(molA, molB, all_molecule_maps)
            _logger.info(f'molecule_maps_scores: {molecule_maps_scores.keys()}')
            _logger.info('Returning map with closest geometry satisfaction')
            return molecule_maps_scores[min(molecule_maps_scores)]
        elif map_strategy == 'core':
            core_count = [len(m) for m in all_molecule_maps]
            maximum_core_atoms = max(core_count)
            if core_count.count(maximum_core_atoms) == 1:
                _logger.info('Returning map with most atoms in core')
                return all_molecule_maps[core_count.index(maximum_core_atoms)]
            else:
                best_maps = [m for c, m in zip(core_count, all_molecule_maps) if c == maximum_core_atoms]
                best_map = AtomMapper._score_nongeometric(molA, molB, best_maps, matching_criterion)
                _logger.info(f'{len(best_maps)} have {maximum_core_atoms} core atoms. Using matching_criterion {matching_criterion} to return the best of those')
                return best_map
        elif map_strategy == 'matching_criterion':
            _logger.info('Returning map that best satisfies matching_criterion')
            best_map = AtomMapper._score_nongeometric(molA, molB, all_molecule_maps, matching_criterion)
            return best_map
        elif map_strategy == 'random':
            _logger.info('Returning map at random')
            return np.random.choice(all_molecule_maps.values())
        elif map_strategy == 'weighted-random':
            molecule_maps_scores = AtomMapper._remove_redundant_maps(molA, molB, all_molecule_maps)
            _logger.info(f'molecule_maps_scores: {molecule_maps_scores.keys()}')
            _logger.info('Returning random map proportional to the geometic distance')
            return np.random.choice(molecule_maps_scores.values(),
                                    [x**-1 for x in molecule_maps_scores.keys()])

    @staticmethod
    def _remove_redundant_maps(molA, molB, all_molecule_maps):
        """For a set of maps, it will filter out those that result in
        the same geometries. From redundant maps, one is chosen randomly.

        Parameters
        ----------
        molA : oechem.oemol
            old molecule
        molB : oechem.oemol
            new molecule
        all_molecule_maps : list(dict)
            list of mappings to check for redundancies
            where the maps are molB to molA

        Returns
        -------
        dict
            dictionary of scores (keys) and maps (dict of new-to-old atom indices)
            where the score is the sum of the distances in cartesian space between atoms that have been mapped
            this helps identify when two core atoms that have been assigned to eachother are actually far away.
            maps are molB indices to molA indices.

        """
        scores = AtomMapper._score_maps(molA, molB, all_molecule_maps)
        _logger.info(f'{len(scores)} maps are reduced to {len(set(scores))}')
        clusters = {}
        for s, mapping in zip(scores, all_molecule_maps):
            if s not in clusters:
                #  doesn't matter which one is chosen as all is equal
                clusters[s] = mapping
        return clusters

    @staticmethod
    def _get_all_maps(current_oemol,
                      proposed_oemol,
                      atom_expr=None,
                      bond_expr=None,
                      map_strength='default',
                      allow_ring_breaking=True,
                      external_inttypes=False,
                      unique=True,
                      matching_criterion='index'):
        """Generate all  possible maps between two oemols

        Parameters
        ----------
        current_oemol : oechem.oemol
            old molecule
        proposed_oemol : oechem.oemol
            new molecule
        atom_expr : int, default=None
            integer corresponding to atom matching, see `perses.openeye.generate_expression`
        bond_expr : int, default=None
            integer corresponding to bond matching, see `perses.openeye.generate_expression`
        map_strength : str, default 'default'
            pre-defined mapping strength that can be one of ['strong', 'default', 'weak']
            this will be ignored if either atom_expr or bond_expr have been defined.
        allow_ring_breaking : bool, optional, default=True
             If False, will check to make sure rings are not being broken or formed.
        external_inttypes : bool, default False
            If True, IntTypes already assigned to oemols will be used for mapping, if IntType is in the atom or bond expression.
            Otherwise, IntTypes will be overwritten such as to ensure rings of different sizes are not matched.
        unique : bool, default True
            openeye kwarg which either returns all maps, or filters out redundant ones
        Returns
        -------
        dict
            dictionary of scores (keys) and maps (dict)

        """
        import openeye.oechem as oechem

        # weak requirements for mapping atoms == more atoms mapped, more in core
        # atoms need to match in aromaticity. Same with bonds.
        # maps ethane to ethene, CH3 to NH2, but not benzene to cyclohexane
        WEAK_ATOM_EXPRESSION = oechem.OEExprOpts_EqAromatic | oechem.OEExprOpts_EqNotAromatic #| oechem.OEExprOpts_IntType
        WEAK_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

        # default atom expression, requires same aromaticitiy and hybridization
        # bonds need to match in bond order
        # ethane to ethene wouldn't map, CH3 to NH2 would map but CH3 to HC=O wouldn't
        DEFAULT_ATOM_EXPRESSION = oechem.OEExprOpts_Hybridization #| oechem.OEExprOpts_IntType
        DEFAULT_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

        # strong requires same hybridization AND the same atom type
        # bonds are same as default, require them to match in bond order
        STRONG_ATOM_EXPRESSION = oechem.OEExprOpts_Hybridization | oechem.OEExprOpts_HvyDegree | oechem.OEExprOpts_DefaultAtoms
        STRONG_BOND_EXPRESSION = oechem.OEExprOpts_DefaultBonds

        map_strength_dict = {'default': [DEFAULT_ATOM_EXPRESSION, DEFAULT_BOND_EXPRESSION],
                             'weak': [WEAK_ATOM_EXPRESSION, WEAK_BOND_EXPRESSION],
                             'strong': [STRONG_ATOM_EXPRESSION, STRONG_BOND_EXPRESSION]}
        if map_strength is None:
            map_strength = 'default'

        if atom_expr is None:
            _logger.debug(f'No atom expression defined, using map strength : {map_strength}')
            atom_expr = map_strength_dict[map_strength][0]
        if bond_expr is None:
            _logger.debug(f'No bond expression defined, using map strength : {map_strength}')
            bond_expr = map_strength_dict[map_strength][1]

        # this ensures that the hybridization of the oemols is done for correct atom mapping
        oechem.OEAssignHybridization(current_oemol)
        oechem.OEAssignHybridization(proposed_oemol)
        oegraphmol_current = oechem.OEGraphMol(current_oemol)  # pattern molecule
        oegraphmol_proposed = oechem.OEGraphMol(proposed_oemol)  # target molecule

        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)
        mcs.Init(oegraphmol_current, atom_expr, bond_expr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        matches = [m for m in mcs.Match(oegraphmol_proposed, unique)]
        _logger.debug(f'all matches have atom counts of : {[m.NumAtoms() for m in matches]}')

        all_mappings = []
        for match in matches:
            all_mappings.append(AtomMapper.hydrogen_mapping_exceptions(current_oemol, proposed_oemol, match, matching_criterion))
        return all_mappings


    @staticmethod
    def _score_nongeometric(molA, molB, maps,
                            matching_criterion='index'):
        """
        Given two molecules, returns the mapping of atoms between them using the match with the greatest number of atoms

        Arguments
        ---------
        molA : openeye.oemol
            first molecule
        molB : openeye.oemol
            second molecule
        maps :  list(dict)
            list of maps with which to identify the best match given matching_criterion
        matching_criterion : str, default 'index'
             The best atom map is pulled based on some ranking criteria;
             if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
             if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
             else: raise Exception.
             NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                    if 'names' is chosen, it is first asserted that the current_oemol and the proposed_oemol have atoms that are uniquely named
        Returns
        -------
        matches : list of match
            list of the matches between the molecules, or None if no matches possible

        """

        _logger.info(f'Finding best map using matching_criterion {matching_criterion}')
        max_num_atoms = max([len(m) for m in maps])
        _logger.debug(f"\tthere are {len(maps)} top matches with at most {max_num_atoms} before hydrogen exceptions")

        # now all else is equal; we will choose the map with the highest overlap of atom indices
        index_overlap_numbers = []
        if matching_criterion == 'index':
            for map in maps:
                hit_number = 0
                for key, value in map.items():
                    if key == value:
                        hit_number += 1
                index_overlap_numbers.append(hit_number)
        elif matching_criterion == 'name':
            for map in maps:
                hit_number = 0
                map_tuples = list(map.items())
                atom_map = {atom_new: atom_old for atom_new, atom_old in zip(list(molB.GetAtoms()), list(molA.GetAtoms())) if (atom_new.GetIdx(), atom_old.GetIdx()) in map_tuples}
                for key, value in atom_map.items():
                    if key.GetName() == value.GetName():
                        hit_number += 1
                index_overlap_numbers.append(hit_number)
        else:
            raise Exception(f"the ranking criteria {matching_criterion} is not supported.")

        max_index_overlap_number = max(index_overlap_numbers)
        max_index = index_overlap_numbers.index(max_index_overlap_number)
        map = maps[max_index]

        return map

    @staticmethod
    def _find_closest_map(mol_A, mol_B, maps):
        """ from a list of maps, finds the one that is geometrically the closest match for molecule B

        Parameters
        ----------
        mol_A : oechem.oemol
            The first moleule in the mapping
        mol_B : oechem.oemol
            Second molecule in the mapping
        maps : list(dict)
            A list of maps to search through

        Returns
        -------
        dict
            the single best match from all maps

        """
        if len(maps) == 1:
            return maps[0]
        coords_A = np.zeros(shape=(mol_A.NumAtoms(), 3))
        for i in mol_A.GetCoords():
            coords_A[i] = mol_A.GetCoords()[i]
        coords_B = np.zeros(shape=(mol_B.NumAtoms(), 3))
        for i in mol_B.GetCoords():
            coords_B[i] = mol_B.GetCoords()[i]
        from scipy.spatial.distance import cdist

        all_to_all = cdist(coords_A, coords_B, 'euclidean')

        mol_B_H = {x.GetIdx(): x.IsHydrogen() for x in mol_B.GetAtoms()}
        all_scores = []
        for M in maps:
            map_score = 0
            for atom in M:
                if not mol_B_H[atom]:  # skip H's - only look at heavy atoms
                    map_score += all_to_all[M[atom], atom]
            all_scores.append(map_score/len(M))
        _logger.debug(f'Mapping scores: {all_scores}')

        # returning lowest score
        best_map_index = np.argmin(all_scores)
        _logger.debug(f'Returning map index: {best_map_index}')
        return maps[best_map_index]

    @staticmethod
    def _create_pattern_to_target_map(current_mol, proposed_mol, match, matching_criterion = 'index'):
        """
        Create a dict of {pattern_atom: target_atom}

        Parameters
        ----------
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
    def hydrogen_mapping_exceptions(old_mol,
                                    new_mol,
                                    match,
                                    matching_criterion):
        """
        Returns an atom map that omits hydrogen-to-nonhydrogen atom maps AND X-H to Y-H where element(X) != element(Y)
        or aromatic(X) != aromatic(Y)
        Parameters
        ----------
        old_mol : openeye.oechem.oemol object
        new_mol : openeye.oechem.oemol object
        match : openeye.oechem.OEMatchBase
        matching_criterion : str, default

        Returns
        -------
        new_to_old_atom_map : dict
            map of new to old atom indices
        """
        import openeye.oechem as oechem
        new_to_old_atom_map = {}
        pattern_to_target_map = AtomMapper._create_pattern_to_target_map(old_mol, new_mol, match, matching_criterion)
        for pattern_atom, target_atom in pattern_to_target_map.items():
            old_index, new_index = pattern_atom.GetIdx(), target_atom.GetIdx()
            old_atom, new_atom = pattern_atom, target_atom

            #Check if a hydrogen was mapped to a non-hydroden (basically the xor of is_h_a and is_h_b)
            if (old_atom.GetAtomicNum() == 1) != (new_atom.GetAtomicNum() == 1):
                continue

            new_to_old_atom_map[new_index] = old_index

        return new_to_old_atom_map

    @staticmethod
    def _assign_atom_ring_id(atom, max_ring_size=10):
        """ Returns the int type based on the ring occupancy
        of the atom

        Parameters
        ----------
        atom : oechem.OEAtomBase
            atom to compute integer of
        max_ring_size : int, default = 10
            Largest ring size that will be checked for

        Returns
        -------
        ring_as_base_two : int
            binary integer corresponding to the atoms ring membership
        """
        import openeye.oechem as oechem
        rings = ''
        for i in range(3, max_ring_size+1): #  smallest feasible ring size is 3
            rings += str(int(oechem.OEAtomIsInRingSize(atom, i)))
        ring_as_base_two = int(rings, 2)
        return ring_as_base_two

    @staticmethod
    def _assign_bond_ring_id(bond, max_ring_size=10):
        import openeye.oechem as oechem
        """ Returns the int type based on the ring occupancy
        of the bond

        Parameters
        ----------
        bond : oechem.OEBondBase
            atom to compute integer of
        max_ring_size : int, default = 10
            Largest ring size that will be checked for

        Returns
        -------
        ring_as_base_two : int
            binary integer corresponding to the bonds ring membership
        """
        rings = ''
        for i in range(3, max_ring_size+1): #  smallest feasible ring size is 3
            rings += str(int(oechem.OEBondIsInRingSize(bond, i)))
        ring_as_base_two = int(rings, 2)
        return ring_as_base_two


    @staticmethod
    def _assign_ring_ids(molecule, max_ring_size=10):
        """ Sets the Int of each atom in the oemol to a number
        corresponding to the ring membership of that atom

        Parameters
        ----------
        molecule : oechem.OEMol
            oemol to assign ring ID to
        max_ring_size : int, default = 10
            Largest ring size that will be checked for

        Returns
        -------
        """
        for atom in molecule.GetAtoms():
            atom.SetIntType(AtomMapper._assign_atom_ring_id(atom, max_ring_size=max_ring_size))
        for bond in molecule.GetBonds():
            bond.SetIntType(AtomMapper._assign_bond_ring_id(bond, max_ring_size=max_ring_size))
        return molecule

    @staticmethod
    def _assign_distance_ids(old_mol, new_mol, distance=0.3):
        """ Gives atoms in both molecules matching Int numbers if they are close
        to each other. This should ONLY be  used if the geometry (i.e. binding mode)
        of both molecules are known, and they are aligned to the same frame of reference.

        this function is invariant to which is passed in as {old|new}_mol
        Parameters
        ----------
        old_mol : oechem.OEMol
            first molecule to compare
        new_mol : oechem.OEMol
            second molecule to compare
        distance : float, default = 0.3
            Distance (in angstrom) that two atoms need to be closer than to be
            labelled as the same.

        Returns
        -------
        copies of old_mol and new_mol, with IntType set according to inter-molecular distances
        """
        _logger.info(f'Using a distance of {distance} to force the mapping of close atoms')
        from scipy.spatial.distance import cdist
        unique_integer = 1
        for atomA, coordsA in zip(old_mol.GetAtoms(), old_mol.GetCoords().values()):
            for atomB, coordsB in zip(new_mol.GetAtoms(), new_mol.GetCoords().values()):
                distances_ij = cdist([coordsA], [coordsB], 'euclidean')[0]
                if distances_ij < distance:
                    atomA.SetIntType(unique_integer)
                    atomB.SetIntType(unique_integer)
                    unique_integer += 1
        return old_mol, new_mol

    @staticmethod
    def preserves_rings(new_to_old_map, current, proposed):
        """Returns True if the transformation allows ring
        systems to be broken or created."""
        if AtomMapper.breaks_rings_in_transformation(new_to_old_map,
                                                     proposed):
            return False
        old_to_new_map = {i : j for j,i in new_to_old_map.items()}
        if AtomMapper.breaks_rings_in_transformation(old_to_new_map,
                                                     current):
            return False
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
        import openeye.oechem as oechem
        pattern_atoms = { atom.GetIdx() : atom for atom in current_mol.GetAtoms() }
        target_atoms = { atom.GetIdx() : atom for atom in proposed_mol.GetAtoms() }
        # _logger.warning(f"\t\t\told oemols: {pattern_atoms}")
        # _logger.warning(f"\t\t\tnew oemols: {target_atoms}")
        copied_new_to_old_atom_map = copy.deepcopy(new_to_old_atom_map)
        _logger.info(new_to_old_atom_map)

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
    def breaks_rings_in_transformation(atom_map, current):
            """Return True if the transformation from molecule1 to molecule2 breaks rings.

            Parameters
            ----------
            atom_map : dict of OEAtom : OEAtom
                atom_map[molecule1_atom] is the corresponding molecule2 atom
            current : oechem.OEMol
                Initial molecule whose rings are to be checked for not being broken

            Returns
            -------
            bool
            """
            for cycle in AtomMapper.enumerate_cycle_basis(current):
                cycle_size = len(cycle)
                # first check that ALL of the ring is in the map or out
                atoms_in_cycle = set([bond.GetBgn().GetIdx() for bond in cycle] + [bond.GetEnd().GetIdx() for bond in cycle])
                number_of_cycle_atoms_mapped = 0
                for atom in atoms_in_cycle:
                    if atom in atom_map:
                        number_of_cycle_atoms_mapped += 1
                _logger.info(number_of_cycle_atoms_mapped)
                if number_of_cycle_atoms_mapped == 0:
                    # none of the ring is mapped - ALL unique, so continue
                    continue
                if number_of_cycle_atoms_mapped != len(atoms_in_cycle):
                    return True # not all atoms in ring are mapped
            return False  # no rings in molecule1 are broken in molecule2

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
    def rank_degenerate_maps(matches, current, proposed):
        """If the atom/bond expressions for maximal substructure is relaxed,
         then the maps with the highest scores will likely be degenerate.
        Consequently, it is important to reduce the degeneracy with other tests

        This test will give each match a score wherein every atom matching
        with the same atomic number (in aromatic rings) will
        receive a +1 score.

        Parameters
        ----------
        matches : type
            Description of parameter `matches`.
        current : oechem.OEMol
            oemol of first molecule
        proposed : oechem.OEMol
            oemol of second molecule

        Returns
        -------
        list of matches
            Ordered list of the matches

        """
        """
        """
        import openeye.oechem as oechem
        score_list = {}
        for idx, match in enumerate(matches):
            counter_arom, counter_aliph = 0, 0
            for matchpair in match.GetAtoms():
                old_index, new_index = matchpair.pattern.GetIdx(), matchpair.target.GetIdx()
                old_atom, new_atom = current.GetAtom(oechem.OEHasAtomIdx(old_index)), proposed.GetAtom(oechem.OEHasAtomIdx(new_index))

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

    def save_atom_mapping(self,filename='atom_map.png'):
        from perses.utils.smallmolecules import render_atom_mapping
        render_atom_mapping(filename, self.current_mol, self.proposed_mol, self.atom_map)

class TopologyProposal(object):
    """
    This is a container class with convenience methods to access various objects needed
    for a topology proposal

    Parameters
    ----------
    new_topology : simtk.openmm.topology.Topology object (augmented)
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.topology.Topology object (augmented)
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
    new_topology : simtk.openmm.topology.Topology object (augmented)
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.topology.Topology object (augmented)
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
    metadata : dict
        additional information of interest about the state
    """

    def __init__(self,
                 new_topology, new_system,
                 old_topology, old_system,
                 logp_proposal=None,
                 new_to_old_atom_map=None, old_alchemical_atoms=None,
                 old_chemical_state_key=None, new_chemical_state_key=None,
                 old_residue_name='MOL', new_residue_name='MOL',
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
        self._old_environment_atoms = set(range(old_system.getNumParticles())) - self._old_alchemical_atoms
        self._new_environment_atoms = set(range(new_system.getNumParticles())) - self._new_alchemical_atoms
        self._metadata = metadata
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

    Parameters
    ----------
    system_generator : SystemGenerator
        The SystemGenerator to use to generate new System objects for proposed Topology objects
    proposal_metadata : dict
        Contains information necessary to initialize proposal engine

    Properties
    ----------
    chemical_state_list : list of str
         a list of all the chemical states that this proposal engine may visit.
    """

    def __init__(self, system_generator, proposal_metadata=None, always_change=True, **kwargs):
        self._system_generator = system_generator
        self._always_change = always_change
        #super(ProposalEngine, self).__init__(**kwargs)

    def propose(self, current_system, current_topology, current_metadata=None):
        """
        Base interface for proposal method.

        Parameters
        ----------
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

    @property
    def chemical_state_list(self):
        raise NotImplementedError("This ProposalEngine does not expose a list of possible chemical states.")

class PolymerProposalEngine(ProposalEngine):
    """
    Base class for ProposalEngine implementations that modify polymer components of systems.

    This base class is not meant to be invoked directly.
    """

    _aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL'] # common naturally-occurring amino acid names
                    # Note this does not include PRO since there's a problem with OpenMM's template DEBUG
    _positive_aminos = ['ARG', 'HIS', 'LYS']
    _negative_aminos = ['ASP', 'GLU']

    def _get_neutrals(aminos, positive, negative):
        excluded = positive + negative
        return [amino for amino in aminos if amino not in excluded]

    _neutral_aminos = _get_neutrals(_aminos, _positive_aminos, _negative_aminos)

    # TODO: Document meaning of 'aggregate'
    def __init__(self, system_generator, chain_id, proposal_metadata=None, always_change=True, aggregate=False):
        """
        Create a polymer proposal engine.

        This base class is not meant to be invoked directly.

        Parameters
        ----------
        system_generator : SystemGenerator
            The SystemGenerator to use to generate perturbed systems
        chain_id : str
            The chain identifier in the Topology object to be mutated
        proposal_metadata : dict, optional, default=None
            Any metadata to be maintained
        always_change : bool, optional, default=True
            If True, will not propose self transitions
        aggregate : bool, optional, default=False
            (FIXME: Description needed!!)


        """
        _logger.debug(f"Instantiating PolymerProposalEngine")
        super(PolymerProposalEngine,self).__init__(system_generator=system_generator, proposal_metadata=proposal_metadata, always_change=always_change)
        self._chain_id = chain_id # chain identifier defining polymer to be modified
        self._aminos_3letter_to_1letter_map = {'ALA' : 'A' ,
                                                'ARG' : 'R' ,
                                                'ASN' : 'N' ,
                                                'ASP' : 'D' ,
                                                'CYS' : 'C' ,
                                                'GLN' : 'Q' ,
                                                'GLU' : 'E' ,
                                                'GLY' : 'G' ,
                                                'HIS' : 'H' ,
                                                'ILE' : 'I' ,
                                                'LEU' : 'L' ,
                                                'LYS' : 'K' ,
                                                'MET' : 'M' ,
                                                'PHE' : 'F' ,
                                                'PRO' : 'P' ,
                                                'SER' : 'S' ,
                                                'THR' : 'T' ,
                                                'TRP' : 'W' ,
                                                'TYR' : 'Y' ,
                                                'VAL' : 'V' }
        self._aggregate = aggregate # ?????????

    @staticmethod
    def generate_oemol_from_pdb_template(pdbfile):
        from perses.utils.openeye import createOEMolFromSDF
        current_oemol = createOEMolFromSDF(pdbfile, add_hydrogens = True)
        if not len(set([atom.GetName() for atom in current_oemol.GetAtoms()])) == len([atom.GetName() for atom in current_oemol.GetAtoms()]):
            raise Exception(f"the atoms in the oemol are not uniquely named.")

        #formatting all canonical atom names from pdb
        for atom in current_oemol.GetAtoms():
            name_with_spaces = atom.GetName()
            name_without_spaces = name_with_spaces.replace(" ", "")
            if name_without_spaces[0].isdigit():
                name_without_spaces = name_without_spaces[1:] + name_without_spaces[0]
            atom.SetName(name_without_spaces)
        return current_oemol

    @staticmethod
    def _get_charge_difference(current_resname, new_resname):
        """
        return the charge of the old res - charge new res

        Parameters
        ----------
        current_resname : str
            three letter identifier for original residue
        new_resname : str
            three letter identifier for new residue

        Returns
        -------
        chargediff : int
            charge(new_res) - charge(old_res)
        """
        assert new_resname in PolymerProposalEngine._aminos
        assert current_resname in PolymerProposalEngine._aminos

        new_rescharge, current_rescharge = 0,0
        resname_to_charge = {current_resname: 0, new_resname: 0}
        for resname in [new_resname, current_resname]:
            if resname in PolymerProposalEngine._negative_aminos:
                resname_to_charge[resname] -= 1
            elif resname in PolymerProposalEngine._positive_aminos:
                resname_to_charge[resname] += 1

        return resname_to_charge[current_resname] - resname_to_charge[new_resname]

    @staticmethod
    def get_water_indices(charge_diff,
                               new_positions,
                               new_topology,
                               radius=0.8):
        """
        Choose random water(s) (at least `radius` nm away from the protein) to turn into ion(s). Returns the atom indices of the water(s) (index w.r.t. new_topology)

        Parameters
        ----------
        charge_diff : int
            the charge difference between the old_system - new_system
        new_positions : np.ndarray(N, 3)
            positions (nm) of atoms corresponding to new_topology
        new_topology : openmm.Topology
            topology of new system
        radius : float, default 0.8
            minimum distance (in nm) that all candidate waters must be from 'protein atoms'

        Returns
        -------
        ion_indices : np.array(abs(charge_diff)*3)
            indices of water atoms to be turned into ions
        """

        import mdtraj as md
        # Create trajectory
        traj = md.Trajectory(new_positions[np.newaxis, ...], md.Topology.from_openmm(new_topology))
        water_atoms = traj.topology.select(f"water")
        query_atoms = traj.top.select('protein')

        # Get water atoms within radius of protein
        neighboring_atoms = md.compute_neighbors(traj, radius, query_atoms, haystack_indices=water_atoms)[0]

        # Get water atoms outside of radius of protein
        nonneighboring_residues = set([atom.residue.index for atom in traj.topology.atoms if (atom.index in water_atoms) and (atom.index not in neighboring_atoms)])
        assert len(nonneighboring_residues) > 0, "there are no available nonneighboring waters"
        # Choose N random nonneighboring waters, where N is determined based on the charge_diff
        choice_residues = np.random.choice(list(nonneighboring_residues), size=abs(charge_diff), replace=False)

        # Get the atom indices in the water(s)
        choice_indices = np.array([[atom.index for atom in traj.topology.residue(res).atoms] for res in choice_residues])

        return np.ndarray.flatten(choice_indices)



    def propose(self,
                current_system,
                current_topology,
                current_metadata=None):
        """
        Generate a TopologyProposal

        Parameters
        ----------
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
            _logger.debug('PolymerProposalEngine: No changes to topology proposed, returning old system and topology')
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
        atom_map, old_res_to_oemol_map, new_res_to_oemol_map, local_atom_map_stereo_sidechain, current_oemol_sidechain, proposed_oemol_sidechain, old_oemol_res_copy, new_oemol_res_copy  = self._construct_atom_map(residue_map, old_topology, index_to_new_residues, new_topology)

        _logger.debug(f"\tadding indices of the 'C' backbone atom in the next residue and the 'N' atom in the previous")
        _logger.debug(f"\t{list(index_to_new_residues.keys())[0]}")
        extra_atom_map = self._find_adjacent_residue_atoms(old_topology, new_topology, list(index_to_new_residues.keys())[0])
        _logger.debug(f"\tfound extra atom map: {extra_atom_map}")

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
        # TODO: Remove build_system() branch once we convert entirely to new openmm-forcefields SystemBuilder
        if hasattr(self._system_generator, 'create_system'):
            new_system = self._system_generator.create_system(new_topology)
        else:
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

        # Create TopologyProposal.
        current_res = [res for res in current_topology.residues() if res.index == chosen_res_index][0]
        proposed_res = [res for res in new_topology.residues() if res.index == chosen_res_index][0]
        augment_openmm_topology(topology = old_topology, residue_oemol = old_oemol_res_copy, residue_topology = current_res, residue_to_oemol_map = old_res_to_oemol_map)
        augment_openmm_topology(topology = new_topology, residue_oemol = new_oemol_res_copy, residue_topology = proposed_res, residue_to_oemol_map = new_res_to_oemol_map)

        topology_proposal = TopologyProposal(logp_proposal = 0.,
                                             new_to_old_atom_map = atom_map,
                                             old_topology = old_topology,
                                             new_topology  = new_topology,
                                             old_system = old_system,
                                             new_system = new_system,
                                             old_alchemical_atoms = [atom.index for atom in current_res.atoms()] + list(extra_atom_map.values()),
                                             old_chemical_state_key = old_chemical_state_key,
                                             new_chemical_state_key = new_chemical_state_key,
                                             old_residue_name = old_res_name,
                                             new_residue_name = new_res_name)

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


        return topology_proposal

    def _find_adjacent_residue_atoms(self, old_topology, new_topology, mutated_residue_index):
        """
        return the maps of the adjacent residue atoms; here, we will ALWAYS consider the atoms of the residues adjacent to the mutation residue to be core

        Parameters
        ----------
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

        assert new_prev_res.name == old_prev_res.name, f"the new residue left adjacent to mutation res (name {new_prev_res.name}) is not the name of the old residue left adjacent to mutation res (name {old_prev_res.name})"
        assert new_next_res.name == old_next_res.name, f"the new residue right adjacent to mutation res (name {new_next_res.name}) is not the name of the old residue right adjacent to mutation res (name {old_next_res.name})"

        new_next_res_to_old_next_res_map = {new_atom.index : old_atom.index for new_atom, old_atom in zip(new_next_res.atoms(), old_next_res.atoms())}
        new_prev_res_to_old_prev_res_map = {new_atom.index : old_atom.index for new_atom, old_atom in zip(new_prev_res.atoms(), old_prev_res.atoms())}
        _logger.debug(f"\t\tnew_next_res_to_old_next_res_map : {new_next_res_to_old_next_res_map}")
        _logger.debug(f"\t\tnew_prev_res_to_old_prev_res_map : {new_prev_res_to_old_prev_res_map}")

        # new_next_res_N_index = [atom.index for atom in new_next_res.atoms() if atom.name.replace(" ", "") == 'N']
        # old_next_res_N_index = [atom.index for atom in old_next_res.atoms() if atom.name.replace(" ", "") == 'N']
        #
        # new_prev_res_C_index = [atom.index for atom in new_prev_res.atoms() if atom.name.replace(" ", "") == 'C']
        # old_prev_res_C_index = [atom.index for atom in old_prev_res.atoms() if atom.name.replace(" ", "") == 'C']
        #
        # for _list in [new_next_res_N_index, old_next_res_N_index, new_prev_res_C_index, old_prev_res_C_index]:
        #     assert len(_list) == 1, f"atoms in the next or prev residue are not uniquely named"
        #
        # new_to_old_map = {new_next_res_N_index[0]: old_next_res_N_index[0],
        #                   new_prev_res_C_index[0]: old_prev_res_C_index[0]}

        new_next_res_to_old_next_res_map.update(new_prev_res_to_old_prev_res_map)
        new_to_old_map = new_next_res_to_old_next_res_map
        return new_to_old_map




    def _choose_mutant(self, topology, metadata):
        """
        Dummy function in parent (PolymerProposalEngine) class to choose a mutant

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
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
        Parameters
        ----------
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
            Parameters
            ----------
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

        Parameters
        ----------
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
        current_oemol : openeye.oechem.oemol object
            copy of modified old oemol sidechain
        proposed_oemol : openeye.oechem.oemol object
            copy of modified new oemol sidechain
        old_oemol_res_copy : openeye.oechem.oemol object
            copy of modified old oemol
        new_oemol_res_copy : openeye.oechem.oemol object
            copy of modified new oemol
        """
        from pkg_resources import resource_filename
        import openeye.oechem as oechem #must this be explicit?

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

        new_res_index_to_name = {atom.index: atom.name for atom in new_res.atoms()}
        old_res_index_to_name = {atom.index: atom.name for atom in old_res.atoms()}

        _logger.debug(f"\t\t\told topology res names: {old_res_index_to_name}")
        _logger.debug(f"\t\t\tnew topology res names: {new_res_index_to_name}")

        old_res_name = old_res.name
        new_res_name = new_res.name

        #make correction for HIS
        his_templates = ['HIE', 'HID']
        if old_res_name in his_templates:
            old_res_name = 'HIS'
        if new_res_name in his_templates:
            new_res_name = 'HIS'
        else:
            pass

        current_residue_pdb_filename = resource_filename('perses', os.path.join('data', 'amino_acid_templates', f"{old_res_name}.pdb"))
        proposed_residue_pdb_filename = resource_filename('perses', os.path.join('data', 'amino_acid_templates', f"{new_res_name}.pdb"))

        current_oemol = PolymerProposalEngine.generate_oemol_from_pdb_template(current_residue_pdb_filename)
        proposed_oemol = PolymerProposalEngine.generate_oemol_from_pdb_template(proposed_residue_pdb_filename)

        old_oemol_res_copy = copy.deepcopy(current_oemol)
        new_oemol_res_copy = copy.deepcopy(proposed_oemol)


        _logger.debug(f"\t\t\told_oemol_res names: {[(atom.GetIdx(), atom.GetName()) for atom in current_oemol.GetAtoms()]}")
        _logger.debug(f"\t\t\tnew_oemol_res names: {[(atom.GetIdx(), atom.GetName()) for atom in proposed_oemol.GetAtoms()]}")

        #create bookkeeping dictionaries
        old_res_to_oemol_map = {atom.index: current_oemol.GetAtom(oechem.OEHasAtomName(atom.name)).GetIdx() for atom in old_res.atoms()}
        new_res_to_oemol_map = {atom.index: proposed_oemol.GetAtom(oechem.OEHasAtomName(atom.name)).GetIdx() for atom in new_res.atoms()}

        old_oemol_name_idx = {atom.GetName(): atom.GetIdx() for atom in current_oemol.GetAtoms()}
        new_oemol_name_idx = {atom.GetName(): atom.GetIdx() for atom in proposed_oemol.GetAtoms()}

        _logger.debug(f"\t\t\told_res_to_oemol_map: {old_res_to_oemol_map}")
        _logger.debug(f"\t\t\tnew_res_to_oemol_map: {new_res_to_oemol_map}")

        old_oemol_to_res_map = {val: key for key, val in old_res_to_oemol_map.items()}
        new_oemol_to_res_map = {val: key for key, val in new_res_to_oemol_map.items()}

        # HBM - these don't seem to be used anywhere
        #old_res_to_oemol_molecule_map = {atom.index: current_oemol.GetAtom(oechem.OEHasAtomName(atom.name)) for atom in old_res.atoms()}
        #new_res_to_oemol_molecule_map = {atom.index: proposed_oemol.GetAtom(oechem.OEHasAtomName(atom.name)) for atom in new_res.atoms()}



        #initialize_the atom map
        local_atom_map = {}

        #now remove backbones in both molecules and map them separately
        backbone_atoms = ['C', 'CA', 'N', 'O', 'H', 'HA', "H'"]
        # TODO dom make this a seperate function
        old_atoms_to_delete, new_atoms_to_delete = [], []
        for atom in proposed_oemol.GetAtoms():
            if atom.GetName() in backbone_atoms:
                try: #to get the backbone atom with the same naem in the old_oemol_res
                    old_corresponding_backbones = [_atom for _atom in current_oemol.GetAtoms() if _atom.GetName() == atom.GetName()]
                    if old_corresponding_backbones == []:
                        #this is an exception when the old oemol res is a glycine.  if this is the case, then we do not map HA2 or HA3
                        assert set(['HA2', 'HA3']).issubset([_atom.GetName() for _atom in current_oemol.GetAtoms()]), f"old oemol residue is not a GLY template"
                        #we have to map HA3 to HA (old, new)
                        old_corresponding_backbones = [_atom for _atom in current_oemol.GetAtoms() if _atom.GetName() == 'HA3' and atom.GetName() == 'HA']
                    assert len(old_corresponding_backbones) == 1, f"there can only be one corresponding backbone in the old molecule; corresponding backbones: {[atom.GetName() for atom in old_corresponding_backbones]}"
                    old_corresponding_backbone = old_corresponding_backbones[0]
                    if not atom.GetName() == "H'": #throw out the extra H
                        local_atom_map[atom.GetIdx()] = old_corresponding_backbone.GetIdx()
                    old_atoms_to_delete.append(old_corresponding_backbone)
                    new_atoms_to_delete.append(atom)
                    assert proposed_oemol.DeleteAtom(atom), f"failed to delete new_oemol atom {atom}"
                    assert current_oemol.DeleteAtom(old_corresponding_backbone), f"failed to delete old_oemol atom {old_corresponding_backbone}"
                except Exception as e:
                    raise Exception(f"failed to map the backbone separately: {e}")


        _logger.debug(f"\t\t\told_oemol_res names: {[(atom.GetIdx(), atom.GetName()) for atom in current_oemol.GetAtoms()]}")
        _logger.debug(f"\t\t\tnew_oemol_res names: {[(atom.GetIdx(), atom.GetName()) for atom in proposed_oemol.GetAtoms()]}")

        old_sidechain_oemol_indices_to_name = {atom.GetIdx(): atom.GetName() for atom in current_oemol.GetAtoms()}
        new_sidechain_oemol_indices_to_name = {atom.GetIdx(): atom.GetName() for atom in proposed_oemol.GetAtoms()}


        #now we can get the mol atom map of the sidechain
        #NOTE: since the sidechain oemols are NOT zero-indexed anymore, we need to match by name (since they are unique identifiers)
        break_bool = False if old_res_name == 'TRP' or new_res_name == 'TRP' else True # Set allow_ring_breaking to be False if the transformation involves TRP
        _logger.debug(f"\t\t\t allow ring breaking: {break_bool}")
        local_atom_map_nonstereo_sidechain = AtomMapper._get_mol_atom_map(current_oemol, proposed_oemol, map_strength='strong', matching_criterion='name', map_strategy='matching_criterion', allow_ring_breaking=break_bool)

        #check the atom map thus far:
        _logger.debug(f"\t\t\tlocal atom map nonstereo sidechain: {local_atom_map_nonstereo_sidechain}")

        #preserve chirality of the sidechain
        # _logger.warning(f"\t\t\told oemols: {[atom.GetIdx() for atom in self.current_molecule.GetAtoms()]}")
        # _logger.warning(f"\t\t\tnew oemols: {[atom.GetIdx() for atom in new_oemol_res.GetAtoms()]}")
        if local_atom_map_nonstereo_sidechain is not None:
            local_atom_map_stereo_sidechain = AtomMapper.preserve_chirality(current_oemol, proposed_oemol, local_atom_map_nonstereo_sidechain)
        else:
            local_atom_map_stereo_sidechain = {}

        _logger.debug(f"\t\t\tlocal atom map stereo sidechain: {local_atom_map_stereo_sidechain}")

        #fix the sidechain indices w.r.t. full oemol
        sidechain_fixed_map = {}
        mapped_names = []
        for new_sidechain_idx, old_sidechain_idx in local_atom_map_stereo_sidechain.items():
            new_name, old_name = new_sidechain_oemol_indices_to_name[new_sidechain_idx], old_sidechain_oemol_indices_to_name[old_sidechain_idx]
            mapped_names.append((new_name, old_name))
            new_full_oemol_idx, old_full_oemol_idx = new_oemol_name_idx[new_name], old_oemol_name_idx[old_name]
            sidechain_fixed_map[new_full_oemol_idx] = old_full_oemol_idx

        _logger.debug(f"\t\t\toemol sidechain fixed map: {sidechain_fixed_map}")


        #make sure that CB is mapped; otherwise the residue will not be contiguous
        found_CB = False
        if any(item[0] == 'CB' and item[1] == 'CB' for item in mapped_names):
            found_CB = True

        if not found_CB:
            _logger.debug(f"\t\t\tno 'CB' found!!!.  removing local atom map stereo sidechain...")
            sidechain_fixed_map = {}

        _logger.debug(f"\t\t\tthe local atom map (backbone) is {local_atom_map}")
        #update the local map
        local_atom_map.update(sidechain_fixed_map)
        _logger.debug(f"\t\t\tthe local atom map (total) is {local_atom_map}")

        #correct the map
        #now we have to update the atom map indices
        _logger.debug(f"\t\t\tadjusting the atom map with topology indices...")
        topology_index_map = {}
        for new_oemol_idx, old_oemol_idx in local_atom_map.items():
            topology_index_map[new_oemol_to_res_map[new_oemol_idx]] = old_oemol_to_res_map[old_oemol_idx]


        _logger.debug(f"\t\t\ttopology_atom_map: {topology_index_map}")

        mapped_atoms = [(new_res_index_to_name[new_idx], old_res_index_to_name[old_idx]) for new_idx, old_idx in topology_index_map.items()]
        _logger.debug(f"\t\t\tthe mapped atom names are: {mapped_atoms}")

            #and all of the environment atoms should already be handled
        return topology_index_map, old_res_to_oemol_map, new_res_to_oemol_map, local_atom_map, current_oemol, proposed_oemol, old_oemol_res_copy, new_oemol_res_copy

    def _get_mol_atom_matches(self, current_molecule, proposed_molecule, first_atom_index_old, first_atom_index_new):
        """
        Given two molecules, returns the mapping of atoms between them.

        Parameters
        ----------
        current_molecule : openeye.oechem.oemol object
             The current molecule in the sampler
        proposed_molecule : openeye.oechem.oemol object
             The proposed new molecule
        first_atom_index_old : int
            The index of the first atom in the old resiude/current molecule
        first_atom_index_new : int
            The index of the first atom in the new residue/proposed molecule
        Note: Since FFAllAngleGeometryEngine._oemol_from_residue creates a new topology for the specified residue,
        the atom indices in the output oemol (i.e. current_molecule and proposed_molecule) are reset to start at 0.
        Therefore, first_atom_index_old and first_atom_index_new are used to correct the indices such that they match
        the atom indices of the original old and new residues.
        Returns
        -------
        new_to_old_atom_map : dict, key : index of atom in new residue, value : index of atom in old residue
        """
        import openeye.oechem as oechem
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

        Parameters
        ----------
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
            For example, the desired systems are wild type T4 lysozyme, T4 lysozyme L99A, and T4 lysozyme L99A/M102Q:
            ``allowed_mutations = [ ('99', 'ALA'), ('102','GLN') ]``. If this is not specified, the engine will propose
            a random amino acid at a random location.
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
        super(PointMutationEngine, self).__init__(system_generator, chain_id, proposal_metadata=proposal_metadata, always_change=always_change, aggregate=aggregate)

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

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
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
                    chain_residues = [res for res in chain.residues() if res.index != 0 and res.index != topology.getNumResidues()-1 and res.name in self._aminos]
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
        Parameters
        ----------
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

        Parameters
        ----------
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

    Parameters
    ----------
    system_generator : SystemGenerator
    library : list of strings
        each string is a 1-letter-code list of amino acid sequence
    chain_id : str
        id of the chain to mutate
        (using the first chain with the id, if there are multiple)
    proposal_metadata : dict, optional
        Contains information necessary to initialize proposal engine
    """

    def __init__(self, system_generator, library, chain_id, proposal_metadata=None, always_change=True):
        super(PeptideLibraryEngine,self).__init__(system_generator, chain_id, proposal_metadata=proposal_metadata, always_change=always_change)
        self._library = library
        self._ff = system_generator.forcefield
        self._templates = self._ff._templates

    def _choose_mutant(self, topology, metadata):
        """
        Used when library of pepide sequences has been provided
        Assume (for now) uniform probability of selecting each peptide
        Parameters
        ----------
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


class SmallMoleculeSetProposalEngine(ProposalEngine):
    """
    This class proposes new small molecules from a prespecified set. It uses
    uniform proposal probabilities, but can be extended. The user is responsible
    for providing a list of smiles that can be interconverted! The class includes
    extra functionality to assist with that (it is slow).

    .. TODO :: Split out proposal engine for two-molecule relative free energy calculations.

    Attributes
    ----------
    system_generator : SystemGenerator
        The SystemGenerator initialized with appropriate force fields used to produce System objects from Topology objects
    atom_expr : ???
        TODO: What does this do?
    bond_expr : ???
        TODO: What does this do?
    map_strength : ???
        TODO: What does this do?
    preserve_chirality : bool, default=True
        whether to preserve the chirality of the small molecule
    map_strategy : str. default='matching_criterion'
        TODO: What does this do?
    external_inttypes : bool, default=False
        TODO: What is this?
    use_given_geometries : bool, default=False
        if True, the oemol atom map is generated by atoms that overlap
    given_geometries_tolerance : simtk.unit.Quantity with units of length, default=0.2*angstrom
        If use_given_geometries=True, use this tolerance for identifying mapped atoms

    """

    system_generator = None
    atom_expr = None
    bond_expr = None
    map_strength = 'default'
    preserve_chirality = True,
    external_inttypes = False,
    map_strategy = 'matching_criterion'
    use_given_geometries = False
    given_geometries_tolerance = 0.2 * unit.angstroms

    def __init__(self, list_of_oemols, system_generator, residue_name='MOL', storage=None, **kwargs):
        """
        Create a SmallMoleculeSetProposalEngine

        Parameters
        ----------
        list_of_smiles : list of string
            list of smiles that will be sampled
        system_generator : SystemGenerator
            The SystemGenerator initialized with appropriate force fields used to produce System objects from Topology objects
        residue_name : str
            The name that will be used for small molecule residues in the topology
        proposal_metadata : dict
            metadata for the proposal engine
        storage : NetCDFStorageView, optional, default=None
            If specified, write statistics to this storage
        current_metadata : dict
            dict containing current smiles as a key

        kwargs can be used to set class attributes

        """
        super(SmallMoleculeSetProposalEngine, self).__init__(system_generator=system_generator, **kwargs)
        # This needs to be exposed, and only set in one place
        self.system_generator = system_generator
        self.list_of_oemols = list_of_oemols
        self._n_molecules = len(self.list_of_oemols)

        self._residue_name = residue_name
        self._generated_systems = dict()
        self._generated_topologies = dict()
        self._matches = dict()

        self._list_of_smiles = []
        from perses.utils.openeye import createSMILESfromOEMol
        for mol in self.list_of_oemols:
            smiles = createSMILESfromOEMol(mol)
            self._list_of_smiles.append(SmallMoleculeSetProposalEngine.canonicalize_smiles(smiles))

        self._storage = storage
        if self._storage is not None:
            self._storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # no point in doing this if there are only two molecules
        if self._n_molecules != 2:
            _logger.info(f"creating probability matrix...")
            self._probability_matrix = self._calculate_probability_matrix()

        # Set modifiable class attributes if specified
        # TODO: This can be dangerous if a parameter is misspelled
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def propose(self,
                current_system,
                current_topology,
                atom_map = None, # TODO: Can this be moved to class attribute?
                current_mol_id = 0, # TODO: Can this be moved to class attribute?
                proposed_mol_id = None, # TODO: Can this be moved to class attribute?
                current_metadata = None):
        """
        Propose the next state, given the current state

        Parameters
        ----------
        current_system : openmm.System object
            the system of the current state
        current_topology : app.Topology object
            the topology of the current state
        atom_map : dict, default None
            dict of atom indices
        current_mol_id : int, optional, default=0
            Index of starting oemol, default is first in list
        proposed_mol_id : int, optional, default=None
            If specified, index of oemol to propose, if None, an oemol from the list is chosen

        Returns
        -------
        proposal : TopologyProposal object
           topology proposal object
        """
        self.current_mol_id = current_mol_id
        if len(self.list_of_oemols) == 2:
            # only two molecules so...
            if self.current_mol_id == 0:
                self.proposed_mol_id = 1
            elif self.current_mol_id == 1:
                self.proposed_mol_id = 0
        else:
            self.proposed_mol_id = proposed_mol_id
        self.current_molecule = self.list_of_oemols[self.current_mol_id]

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
        old_alchemical_atoms = range(old_mol_start_index, old_mol_start_index+len_old_mol)
        _logger.info(f"old alchemical atom indices: {old_alchemical_atoms}")

        # Select the next molecule SMILES given proposal probabilities
        if self.proposed_mol_id is None:
            _logger.info(f"the proposed oemol is not specified; proposing a new molecule from proposal matrix P(M_new | M_old)...")
            self.proposed_mol_id, self.proposed_molecule, logp_proposal = self._propose_molecule(current_system, current_topology, self.current_mol_id)
        else:
            self.proposed_molecule = self.list_of_oemols[self.proposed_mol_id]
            proposed_mol_smiles = self._list_of_smiles[self.proposed_mol_id]
            _logger.info(f"proposed mol detected with smiles {proposed_mol_smiles} and logp_proposal of 0.0")
            logp_proposal = 0.0

        _logger.info(f"conducting proposal from {self._list_of_smiles[self.current_mol_id]} to {self._list_of_smiles[self.proposed_mol_id]}...")

        # Build the new Topology object, including the proposed molecule
        _logger.info(f"building new topology with proposed molecule and current receptor topology...")
        new_topology = self._build_new_topology(current_receptor_topology, self.proposed_molecule)
        new_mol_start_index, len_new_mol = self._find_mol_start_index(new_topology)
        self.new_mol_start_index = new_mol_start_index
        self.len_new_mol = len_new_mol
        _logger.info(f"new molecule has a start index of {new_mol_start_index} and {len_new_mol} atoms.")

        # Generate an OpenMM System from the proposed Topology
        _logger.info(f"proceeding to build the new system from the new topology...")
        # TODO: Remove build_system() branch once we convert entirely to new openmm-forcefields SystemBuilder
        if hasattr(self._system_generator, 'create_system'):
            new_system = self._system_generator.create_system(new_topology)
        else:
            new_system = self._system_generator.build_system(new_topology)

        # Determine atom mapping between old and new molecules
        _logger.info(f"determining atom map between old and new molecules...")
        if atom_map is None:
            _logger.info(f"the atom map is not specified; proceeding to generate an atom map...")
            if self.use_given_geometries:
                mol_atom_map = AtomMapper._get_mol_atom_map_by_positions(self.current_molecule, self.proposed_molecule, tolerance=self.given_geometries_tolerance)
            else:
                mol_atom_map = AtomMapper._get_mol_atom_map(self.current_molecule, self.proposed_molecule, atom_expr=self.atom_expr, bond_expr=self.bond_expr, map_strength=self.map_strength, external_inttypes=self.external_inttypes, map_strategy=self.map_strategy)
        else:
            _logger.info(f"atom map is pre-determined as {atom_map}")
            mol_atom_map = atom_map

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
        _logger.debug(f"mapped {len(adjusted_atom_map)} before constraint repairs")
        adjusted_atom_map = SmallMoleculeSetProposalEngine._constraint_repairs(adjusted_atom_map, current_system, new_system, current_topology, new_topology)
        _logger.debug(f"mapped {len(adjusted_atom_map)} after constraint repairs")
        non_offset_new_to_old_atom_map = copy.deepcopy(adjusted_atom_map)
        # TODO is the following line needed? It doesn't seem to be used
        min_keys, min_values = min(non_offset_new_to_old_atom_map.keys()), min(non_offset_new_to_old_atom_map.values())
        self.non_offset_new_to_old_atom_map = mol_atom_map

        #create NetworkXMolecule for each molecule
        current_residue = [res for res in current_topology.residues() if res.name == self._residue_name][0]
        proposed_residue = [res for res in new_topology.residues() if res.name == self._residue_name][0]

        augment_openmm_topology(topology = current_topology, residue_oemol = self.current_molecule, residue_topology = current_residue, residue_to_oemol_map = {i: j for i, j in zip(range(old_mol_start_index, old_mol_start_index + len_old_mol), range(len_old_mol))})
        augment_openmm_topology(topology = new_topology, residue_oemol = self.proposed_molecule, residue_topology = proposed_residue, residue_to_oemol_map = {i: j for i, j in zip(range(new_mol_start_index, new_mol_start_index + len_new_mol), range(len_new_mol))})

        # Create the TopologyProposal object
        proposal = TopologyProposal(logp_proposal=logp_proposal,
                                    new_to_old_atom_map=adjusted_atom_map,
                                    old_topology=current_topology,
                                    new_topology=new_topology,
                                    old_system=current_system,
                                    new_system=new_system,
                                    old_alchemical_atoms=old_alchemical_atoms,
                                    old_chemical_state_key=self._list_of_smiles[self.current_mol_id],
                                    new_chemical_state_key=self._list_of_smiles[self.proposed_mol_id],
                                    old_residue_name=self._residue_name,
                                    new_residue_name=self._residue_name)

        ndelete = proposal.old_system.getNumParticles() - len(proposal.old_to_new_atom_map.keys())
        ncreate = proposal.new_system.getNumParticles() - len(proposal.old_to_new_atom_map.keys())
        _logger.info(f'Proposed transformation would delete {ndelete} atoms and create {ncreate} atoms.')

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
        # TODO can this go in perses/utils/openeye?
        import openeye.oechem as oechem
        OESMILES_OPTIONS = oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_ISOMERIC | oechem.OESMILESFlag_Hydrogens

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
        # TODO can this go in perses/utils/openeye?
        import openeye.oechem as oechem
        OESMILES_OPTIONS = oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_ISOMERIC | oechem.OESMILESFlag_Hydrogens

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

    def _propose_molecule(self, system, topology, current_mol_id, exclude_self=False):
        """
        Propose a new molecule given the current molecule.

        The current scheme uses a probability matrix computed via _calculate_probability_matrix.

        Parameters
        ----------
        system : simtk.openmm.System object
            The current system
        topology : simtk.openmm.app.Topology object
            The current topology
        positions : [n, 3] np.ndarray of floats (Quantity nm)
            The current positions of the system
        current_mol_id : int
            The index of the current molecule
        exclude_self : bool, optional, default=True
            If True, exclude self-transitions

        Returns
        -------
        proposed_mol_id : int
             The index of the proposed molecule
        mol : oechem.OEMol
            The next molecule to simulate
        logp_proposal : float
            contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
            log [P(Mold | Mnew) / P(Mnew | Mold)]
        """
        # Compute contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
        # log [P(Mold | Mnew) / P(Mnew | Mold)]

        # Propose a new molecule
        molecule_probabilities = self._probability_matrix[current_mol_id, :]
        _logger.info(f"\tmolecule probabilities: {molecule_probabilities}")
        proposed_mol_id = np.random.choice(range(len(self._list_of_smiles)), p=molecule_probabilities)
        _logger.info(f"\tproposed molecule index: {proposed_mol_id}")
        reverse_probability = self._probability_matrix[proposed_mol_id, current_mol_id]
        forward_probability = molecule_probabilities[proposed_mol_id]
        _logger.info(f"\tforward probability: {forward_probability}")
        _logger.info(f"\treverse probability: {reverse_probability}")
        proposed_smiles = self._list_of_smiles[proposed_mol_id]
        _logger.info(f"\tproposed molecule smiles: {proposed_smiles}")
        proposed_mol = self.list_of_oemols[proposed_mol_id]
        logp = np.log(reverse_probability) - np.log(forward_probability)
        _logger.info(f"\tlogP proposal: {logp}")
        return proposed_mol_id, proposed_mol, logp

    def _calculate_probability_matrix(self):
        """
        Calculate the matrix of probabilities of choosing A | B
        based on normalized MCSS overlap. Does not check for torsions!
        Parameters
        ----------
        oemol_list : list of oechem.OEMol
            list of oemols to be potentially selected

        Returns
        -------
        probability_matrix : [n, n] np.ndarray
            probability_matrix[Mold, Mnew] is the probability of choosing molecule Mnew given the current molecule is Mold

        """
        n_mols = len(self.list_of_oemols)
        probability_matrix = np.zeros([n_mols, n_mols])
        for i in range(n_mols):
            for j in range(i):
                oemol_i = self.list_of_oemols[i]
                oemol_j = self.list_of_oemols[j]
                atom_map = AtomMapper._get_mol_atom_map(oemol_i, oemol_j)
                if not atom_map:
                    n_atoms_matching = 0
                    continue
                n_atoms_matching = len(atom_map.keys())
                probability_matrix[i, j] = n_atoms_matching
                probability_matrix[j, i] = n_atoms_matching
        #normalize the rows:
        for i in range(n_mols):
            row_sum = np.sum(probability_matrix[i, :])
            try:
                probability_matrix[i, :] /= row_sum
            except ZeroDivisionError:
                print("One molecule is completely disconnected!")
                raise

        if self._storage:
            self._storage.write_object('molecule_smiles_list', self._list_of_smiles)
            self._storage.write_array('probability_matrix', probability_matrix)

        return probability_matrix

    @property
    def chemical_state_list(self):
        return self._list_of_smiles

    # TODO remove? It's not used anywhere?
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
        from perses.rjmc.geometry import ProposalOrderTools
        safe_smiles = set()
        smiles_pairs = set()
        smiles_set = set(smiles_list)

        for mol1, mol2 in itertools.combinations(smiles_list, 2):
            smiles_pairs.add((mol1, mol2))

        for smiles_pair in smiles_pairs:
            mol1, sys1, pos1, top1 = createSystemFromSMILES(smiles_pair[0])
            mol2, sys2, pos2, top2 = createSystemFromSMILES(smiles_pair[1])

            new_to_old_atom_map = AtomMapper._get_mol_atom_map()
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

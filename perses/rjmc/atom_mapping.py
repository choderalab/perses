"""
Tools for computing atom mappings for relative free energy transformations.
These tools operate exclusively on OpenEye Toolkit openeye.oechem.OEMol objects.

"""

from simtk import unit

import copy
import os
import numpy as np

################################################################################
# LOGGER
################################################################################

import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("atom_mapping")
_logger.setLevel(logging.INFO)

################################################################################
# EXCEPTIONS
################################################################################

class InvalidMappingException(Exception):
    """
    Invalid atom mapping for relative free energy transformation.

    """
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

################################################################################
# ATOM MAPPING
################################################################################

class AtomMapping(object):
    """
    A container representing an atom mapping between two small molecules.

    This object is mutable, but only valid atom mappings can be stored.
    The validation check occurs whenever a new atom mapping is set.

    .. todo :: Figure out how this should work for biopolymers.

    .. todo :: Migrate to openff.toolkit.topology.Molecule when able

    Attributes
    ----------
    old_mol : openff.toolkit.topology.Molecule
        Copy of the first molecule to be mapped
    new_mol : openff.toolkit.topology.Molecule
        Copy of the second molecule to be mapped
    n_mapped_atoms : int
        The number of mapped atoms.
        Read-only property.
    new_to_old_atom_map : dict of int : int
        new_to_old_atom_map[new_atom_index] is the atom index in old_oemol corresponding to new_atom_index in new_oemol
        A copy is returned, but this attribute can be set.
        Zero-based indexing within the atoms in old_mol and new_mol is used.
    old_to_new_atom_map : dict of int : int
        old_to_new_atom_map[old_atom_index] is the atom index in new_oemol corresponding to old_atom_index in old_oemol
        A copy is returned, but this attribute can be set.
        Zero-based indexing within the atoms in old_mol and new_mol is used.

    Examples
    --------

    Create an atom mapping for ethane -> ethanol
    >>> from openff.toolkit.topology import Molecule
    >>> ethane = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])')
    >>> ethanol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]')
    >>> atom_mapping = AtomMapping(ethane, ethanol, old_to_new_atom_map={0:0, 4:4})

    """
    def __init__(self, old_mol, new_mol, new_to_old_atom_map=None, old_to_new_atom_map=None):
        """
        Construct an AtomMapping object.

        Once constructed, either new_to_old_atom_map or old_to_new_atom_map can be accessed or set.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            First molecule to be mapped
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            Second molecule to be mapped
        new_to_old_atom_map : dict of int : int
            new_to_old_atom_map[new_atom_index] is the atom index in old_oemol corresponding to new_atom_index in new_oemol

        """
        # Store molecules
        # TODO: Should we be allowing unspecified stereochemistry?
        from openff.toolkit.topology import Molecule
        self.old_mol = Molecule(old_mol, allow_undefined_stereo=True)
        self.new_mol = Molecule(new_mol, allow_undefined_stereo=True)

        # Store atom maps
        if (old_to_new_atom_map is not None) and (new_to_old_atom_map is not None):
            raise ValueError('Only one of old_to_new_atom_map or new_to_old_atom_map can be specified')
        if (old_to_new_atom_map is None) and (new_to_old_atom_map is None):
            raise ValueError('One of old_to_new_atom_map or new_to_old_atom_map must be specified')
        if old_to_new_atom_map is not None:
            self.old_to_new_atom_map = old_to_new_atom_map
        if new_to_old_atom_map is not None:
            self.new_to_old_atom_map = new_to_old_atom_map

    def __repr__(self):
        return f"AtomMapping(Molecule.from_smiles('{self.old_mol.to_smiles(mapped=True)}'), Molecule.from_smiles('{self.new_mol.to_smiles(mapped=True)}'), old_to_new_atom_map={self.old_to_new_atom_map})"

    def __str__(self):
        return f'AtomMapping : {self.old_mol.to_smiles(mapped=True)} -> {self.new_mol.to_smiles(mapped=True)} : mapped atoms {self.old_to_new_atom_map}'

    def __hash__(self):
        """Compute unique hash that accounts for atom ordering in molecules and permutation invariance of dictionary items"""
        return hash( ( self.old_mol.to_smiles(mapped=True), self.new_mol.to_smiles(mapped=True), frozenset(self.old_to_new_atom_map.items()) ) )

    def _validate(self):
        """
        Validate the atom mapping is consistent with stored molecules.
        """
        # Ensure mapping is not empty
        if len(self.new_to_old_atom_map) == 0:
            raise InvalidMappingException(f'Atom mapping contains no mappped atoms')
        # Ensure all keys and values are integers
        if not (     all(isinstance(x, int) for x in self.new_to_old_atom_map.keys())
                 and all(isinstance(x, int) for x in self.new_to_old_atom_map.values())
               ):
            raise InvalidMappingException(f'Atom mapping contains non-integers:\n{self.new_to_old_atom_map}')

        # Check to make sure atom indices are within valid range
        if (   not set(self.new_to_old_atom_map.keys()).issubset(range(self.new_mol.n_atoms))
            or not set(self.new_to_old_atom_map.values()).issubset(range(self.old_mol.n_atoms))
           ):
            raise InvalidMappingException('Atom mapping contains invalid atom indices:\n{self.new_to_old_atom_map}\nold_mol: {self.old_mol}\nnew_mol: {self.new_mol}')

        # Make sure mapping is one-to-one
        if len(set(self.new_to_old_atom_map.keys())) != len(set(self.new_to_old_atom_map.values())):
            raise InvalidMappingException('Atom mapping is not one-to-one:\n{self.new_to_old_atom_map}')

    @property
    def new_to_old_atom_map(self):
        import copy
        return copy.deepcopy(self._new_to_old_atom_map)

    @new_to_old_atom_map.setter
    def new_to_old_atom_map(self, value):
        self._new_to_old_atom_map = dict(value)
        self._validate()

    @property
    def old_to_new_atom_map(self):
        # Construct reversed atom map on the fly
        return dict(map(reversed, self._new_to_old_atom_map.items()))

    @old_to_new_atom_map.setter
    def old_to_new_atom_map(self, value):
        self._new_to_old_atom_map = dict(map(reversed, value.items()))
        self._validate()

    @property
    def n_mapped_atoms(self):
        """The number of mapped atoms"""
        return len(self._new_to_old_atom_map)

    def render_image(self, filename, width=1200, height=600):
        """
        Render the atom mapping to an image or PDF.

        .. note :: This currently requires the OpenEye toolkit.

        .. todo ::

           * Add support for biopolymer mapping rendering
           * Add support for non-OpenEye rendering?

        Parameters
        ----------
        filename : str
            The image filename to write to.
            Format automatically detected from file suffix.
        width : int, optional, default=1200
            Width in pixels
        height : int, optional, default=600
            Height in pixels

        """
        from openeye import oechem, oedepict

        molecule1 = self.old_mol.to_openeye()
        molecule2 = self.new_mol.to_openeye()
        new_to_old_atom_map = self.new_to_old_atom_map

        oechem.OEGenerate2DCoordinates(molecule1)
        oechem.OEGenerate2DCoordinates(molecule2)

        # Add both to an OEGraphMol reaction
        rmol = oechem.OEGraphMol()
        rmol.SetRxn(True)
        def add_molecule(mol):
            # Add atoms
            new_atoms = list()
            old_to_new_atoms = dict()
            for old_atom in mol.GetAtoms():
                new_atom = rmol.NewAtom(old_atom.GetAtomicNum())
                new_atom.SetFormalCharge(old_atom.GetFormalCharge())
                new_atoms.append(new_atom)
                old_to_new_atoms[old_atom] = new_atom
            # Add bonds
            for old_bond in mol.GetBonds():
                rmol.NewBond(old_to_new_atoms[old_bond.GetBgn()], old_to_new_atoms[old_bond.GetEnd()], old_bond.GetOrder())
            return new_atoms, old_to_new_atoms

        [new_atoms_1, old_to_new_atoms_1] = add_molecule(molecule1)
        [new_atoms_2, old_to_new_atoms_2] = add_molecule(molecule2)

        # Label reactant and product
        for atom in new_atoms_1:
            atom.SetRxnRole(oechem.OERxnRole_Reactant)
        for atom in new_atoms_2:
            atom.SetRxnRole(oechem.OERxnRole_Product)

        core1 = oechem.OEAtomBondSet()
        core2 = oechem.OEAtomBondSet()
        # add all atoms to the set
        core1.AddAtoms(new_atoms_1)
        core2.AddAtoms(new_atoms_2)
        # Label mapped atoms
        core_change = oechem.OEAtomBondSet()
        index =1
        for (index2, index1) in new_to_old_atom_map.items():
            new_atoms_1[index1].SetMapIdx(index)
            new_atoms_2[index2].SetMapIdx(index)
            # now remove the atoms that are core, so only uniques are highlighted
            core1.RemoveAtom(new_atoms_1[index1])
            core2.RemoveAtom(new_atoms_2[index2])
            if new_atoms_1[index1].GetAtomicNum() != new_atoms_2[index2].GetAtomicNum():
                # this means the element type is changing
                core_change.AddAtom(new_atoms_1[index1])
                core_change.AddAtom(new_atoms_2[index2])
            index += 1
        # Set up image options
        itf = oechem.OEInterface()
        oedepict.OEConfigureImageOptions(itf)
        ext = oechem.OEGetFileExtension(filename)
        if not oedepict.OEIsRegisteredImageFile(ext):
            raise ValueError(f'Unknown image type for filename {filename}')
        ofs = oechem.oeofstream()
        if not ofs.open(filename):
            raise IOError(f'Cannot open output file {filename}')

        # Setup depiction options
        oedepict.OEConfigure2DMolDisplayOptions(itf, oedepict.OE2DMolDisplaySetup_AromaticStyle)
        opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
        oedepict.OESetup2DMolDisplayOptions(opts, itf)
        opts.SetBondWidthScaling(True)
        opts.SetAtomPropertyFunctor(oedepict.OEDisplayAtomMapIdx())
        opts.SetAtomColorStyle(oedepict.OEAtomColorStyle_WhiteMonochrome)

        # Depict reaction with component highlights
        oechem.OEGenerate2DCoordinates(rmol)
        rdisp = oedepict.OE2DMolDisplay(rmol, opts)

        if core1.NumAtoms() != 0:
            oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEPink),oedepict.OEHighlightStyle_Stick, core1)
        if core2.NumAtoms() != 0:
            oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEPurple),oedepict.OEHighlightStyle_Stick, core2)
        if core_change.NumAtoms() != 0:
            oedepict.OEAddHighlighting(rdisp, oechem.OEColor(oechem.OEGreen),oedepict.OEHighlightStyle_Stick, core_change)
        oedepict.OERenderMolecule(ofs, ext, rdisp)
        ofs.close()

    def creates_or_breaks_rings(self):
        """Determine whether the mapping causes rings to be created or broken in transformation.

        Returns
        -------
        breaks_rings : bool
            Returns True if the atom mapping would cause rings to be created or broken
        """
        import networkx as nx
        # Check that any bond between mapped atoms in one molecule is also bonded in the other molecule
        for mol1, mol2, atom_mapping in [
            (self.old_mol, self.new_mol, self.old_to_new_atom_map),
            (self.new_mol, self.old_mol, self.new_to_old_atom_map),
            ]:
            mol1_graph = mol1.to_networkx()
            mol2_graph = mol2.to_networkx()
            # Check that all bonds between mapped atoms in mol1 are mapped in mol2
            for mol1_edge in mol1_graph.edges:
                if set(mol1_edge).issubset(atom_mapping.keys()):
                    mol2_edge = (atom_mapping[mol1_edge[0]], atom_mapping[mol1_edge[1]])
                    if not mol2_graph.has_edge(*mol2_edge):
                        return True
        # For every cycle in the molecule, check that ALL atoms in the cycle are mapped or not mapped
        for molecule, mapped_atoms in [
            (self.old_mol, self.old_to_new_atom_map.keys()),
            (self.new_mol, self.old_to_new_atom_map.values())
           ]:
            graph = molecule.to_networkx()
            for cycle in nx.simple_cycles(graph.to_directed()):
                n_atoms_in_cycle = len(cycle)
                if n_atoms_in_cycle < 3:
                    # Cycle must contain at least three atoms to be useful
                    continue
                n_atoms_mapped = len( set(cycle).intersection(mapped_atoms) )
                if not ((n_atoms_mapped==0) or (n_atoms_in_cycle==n_atoms_mapped)):
                    return True
        return False

    def unmap_partially_mapped_cycles(self):
        """De-map atoms to ensure the partition function will be factorizable.

        This algorithm builds a graph for old and new molecules where edges connect bonded atoms where both atoms are mapped.
        We then find the largest connected subgraph and de-map all other atoms

        .. todo :: Change this algorithm to operate on the hybrid graph

        .. todo :: Check to make sure that we don't end up with problematic mappings.

        """
        import networkx as nx

        # Save initial mapping
        import copy
        initial_mapping = copy.deepcopy(self)

        # Traverse all cycles and de-map any atoms that are in partially mapped cycles
        # making sure to check that bonds in mapped atoms are concordant.
        atoms_to_demap = dict()
        for mol1, mol2, atom_mapping, selection in [
            (self.old_mol, self.new_mol, self.old_to_new_atom_map, 'old'),
            (self.new_mol, self.old_mol, self.new_to_old_atom_map, 'new')
           ]:
            atoms_to_demap[selection] = set()
            mol1_graph = mol1.to_networkx()
            mol2_graph = mol2.to_networkx()
            for cycle in nx.simple_cycles(mol1_graph.to_directed()):
                n_atoms_in_cycle = len(cycle)
                if n_atoms_in_cycle < 3:
                    # Need at least three atoms in a cycle
                    continue

                # Check that all atoms and bonds are mapped
                def is_cycle_mapped(mol1_graph, mol2_graph, cycle, atom_mapping):
                    n_atoms_in_cycle = len(cycle)
                    for index in range(n_atoms_in_cycle):
                        mol1_atom1, mol1_atom2 = cycle[index], cycle[(index+1)%n_atoms_in_cycle]
                        if not ((mol1_atom1 in atom_mapping) and (mol1_atom2 in atom_mapping)):
                            return False
                        mol2_atom1, mol2_atom2 = atom_mapping[mol1_atom1], atom_mapping[mol1_atom2]
                        if not mol2_graph.has_edge(mol2_atom1, mol2_atom2):
                            return False

                    # All atoms and bonds in cycle are mapped correctly
                    return True

                if not is_cycle_mapped(mol1_graph, mol2_graph, cycle, atom_mapping):
                    # De-map any atoms in this map
                    for atom_index in cycle:
                        atoms_to_demap[selection].add(atom_index)

        # Update mapping to eliminate any atoms in partially mapped cycles
        if len(atoms_to_demap['old'])>0 or len(atoms_to_demap['new'])>0:
            _logger.info(f"AtomMapping.unmap_partially_mapped_cycles(): Demapping atoms that were in partially mapped cycles: {atoms_to_demap}")
        self.old_to_new_atom_map = { old_atom : new_atom for old_atom, new_atom in self.old_to_new_atom_map.items() if (old_atom not in atoms_to_demap['old']) and (new_atom not in atoms_to_demap['new']) }

        # Construct old_mol graph pruning any edges that do not share bonds
        # correctly mapped in both molecules
        old_mol_graph = self.old_mol.to_networkx()
        new_mol_graph = self.new_mol.to_networkx()
        for edge in old_mol_graph.edges:
            if not set(edge).issubset(self.old_to_new_atom_map.keys()):
                # Remove the edge because bond is not between mapped atoms
                old_mol_graph.remove_edge(*edge)
                _logger.info(f'Demapping old_mol edge {edge} because atoms are not mapped')
            else:
                # Both atoms are mapped
                # Ensure atoms are also bonded in new_mol
                if not new_mol_graph.has_edge(self.old_to_new_atom_map[edge[0]], self.old_to_new_atom_map[edge[1]]):
                    old_mol_graph.remove_edge(*edge)
                    _logger.info(f'Demapping old_mol edge {edge} because atoms are not bonded in new_mol')

        # Find the largest connected component of the graph
        connected_components = [component for component in nx.connected_components(old_mol_graph)]
        connected_components.sort(reverse=True, key=lambda subgraph : len(subgraph))
        largest_connected_component = connected_components[0]
        _logger.info(f"AtomMapping.unmap_partially_mapped_cycles(): Connected component sizes: {[len(component) for component in connected_components]}")

        # Check to make sure we haven't screwed something up
        if len(largest_connected_component) == 0:
            msg = f'AtomMapping.unmap_partially_mapped_cycles(): Largest connected component has too few atoms ({len(largest_connected_component)} atoms)\n'
            msg += f'  Initial mapping (initial-mapping.png): {self}\n'
            msg += f'  largest_connected_component: {largest_connected_component}\n'
            initial_mapping.render_image('initial-mapping.png')
            raise AssertionError(msg)

        # Update mapping to include only largest connected component atoms
        self.old_to_new_atom_map = { old_atom : new_atom for old_atom, new_atom in self.old_to_new_atom_map.items() if (old_atom in largest_connected_component) }


        _logger.info(f"AtomMapping.unmap_partially_mapped_cycles(): Number of mapped atoms changed from {len(initial_mapping.old_to_new_atom_map)} -> {len(self.old_to_new_atom_map)}")

        # Check to make sure we haven't screwed something up
        if self.creates_or_breaks_rings() == True:
            msg = f'AtomMapping.unmap_partially_mapped_cycles() failed to eliminate all ring creation/breaking. This indicates a programming logic error.\n'
            msg += f'  Initial mapping (initial-mapping.png): {initial_mapping}\n'
            msg += f'  After demapping (final-mapping.png)  : {self}\n'
            msg += f'  largest_connected_component: {largest_connected_component}\n'
            initial_mapping.render_image('initial-mapping.png')
            self.render_image('final-mapping.png')
            raise InvalidMappingException(msg)

    def preserve_chirality(self):
        """
        Alter the atom mapping to preserve chirality

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

        .. todo :: Check that chirality is correctly handled.

        """
        # TODO: Simplify this.

        import openeye.oechem as oechem
        pattern_atoms = { atom.GetIdx() : atom for atom in self.old_mol.to_openeye().GetAtoms() }
        target_atoms  = { atom.GetIdx() : atom for atom in self.new_mol.to_openeye().GetAtoms() }
        # _logger.warning(f"\t\t\told oemols: {pattern_atoms}")
        # _logger.warning(f"\t\t\tnew oemols: {target_atoms}")
        copied_new_to_old_atom_map = copy.deepcopy(self.new_to_old_atom_map)
        _logger.info(self.new_to_old_atom_map)

        for new_index, old_index in self.new_to_old_atom_map.items():

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

        # Update atom map
        self.new_to_old_atom_map = copied_new_to_old_atom_map

################################################################################
# ATOM MAPPERS
################################################################################

class AtomMapper(object):
    """
    Generate atom mappings between two molecules for relative free energy transformations.

    .. note ::

      As this doesn't generate a system, it will not be
      accurate for whether hydrogens are mapped or not.
      It also doesn't check that this is feasible to simulate,
      but is just helpful for testing options.

    .. todo ::

       * Expose options for whether bonds to hydrogen are constrained or not (and hence should be mapped or not)

       * Find a better way to express which mappings are valid for the hybrid topology factory

    Attributes
    ----------
    use_positions : bool, optional, default=True
        If True, will attempt to use positions of molecules to determine optimal mappings.
        If False, will only use maximum common substructure (MCSS).
    atom_expr : openeye.oechem.OEExprOpts
        Override for atom matching expression; None if default is to be used.
    bond_expr : openeye.oechem.OEExprOpts
        Override for bond matching expression; None if default is to be used.
    allow_ring_breaking : bool
        Wether or not to allow ring breaking in map
    coordinate_tolerance : simtk.unit.Quantity, optional, default=0.25*simtk.unit.angstroms
        Coordinate tolerance for geometry-derived mappings.

    Examples
    --------

    Create an AtomMapper factory:

    >>> atom_mapper = AtomMapper()

    You can also configure it after it has been created:

    >>> atom_mapper.use_positions = True # use positions in scoring mappings if available
    >>> atom_mapper.allow_ring_breaking = False # don't allow rings to be broken
    >>> from openeye import oechem
    >>> atom_mapper.atom_expr = oechem.OEExprOpts_Hybridization # override default atom_expr

    Specify two molecules without positions

    >>> from openff.toolkit.topology import Molecule
    >>> ethane = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])')
    >>> ethanol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]')

    Retrieve all mappings between two molecules

    >>> atom_mappings = atom_mapper.get_all_mappings(ethane, ethanol)

    Retrieve optimal mapping between molecules

    >>> atom_mapping = atom_mapper.get_best_mapping(ethane, ethanol)

    Stochastically sample a mapping between molecules

    >>> atom_mapping = atom_mapper.get_sampled_mapping(ethane, ethanol)

    We can access (or modify) either the old-to-new atom mapping or new-to-old atom mapping,
    and both will be kept in a self-consistent state:

    >>> atom_mapping.old_to_new_atom_map
    >>> atom_mapping.new_to_old_atom_map

    Copies of the initial and final molecules are also available as OpenFF Molecule objects:

    >>> atom_mapping.old_mol
    >>> atom_mapping.new_mol

    The AtomMapper can also utilize positions in generating atom mappings.
    If positions are available, they will be used to derive mappings by default.

    >>> atom_mapper.use_positions = True # use positions in scoring mappings if available
    >>> old_mol = Molecule.from_file('old_mol.sdf')
    >>> new_mol = Molecule.from_file('new_mol.sdf')
    >>> atom_mapping = atom_mapper.get_best_mapping(old_mol, new_mol)

    The mapping can also be generated only from positions,
    rather than scoring mappings from MCSS:

    >>> atom_mapping = atom_mapper.generate_atom_mapping_from_positions(old_mol, new_mol)

    The tolerance for position scoring or position-derived mappings can be adjusted in the AtomMapper factory:

    >>> from simtk import unit
    >>> atom_mapper.coordinate_tolerance = 0.3*unit.angstroms

    """
    def __init__(self,
        map_strength='default',
        atom_expr=None, bond_expr=None,
        use_positions=True,
        allow_ring_breaking=False,
        external_inttypes=False,
        matching_criterion='index',
        coordinate_tolerance=0.25*unit.angstroms,
        ):
        """
        Create an AtomMapper factory.

        Parameters
        ----------
        map_strength : str, optional, default='default'
            Select atom mapping atom and bond expression defaults: ['strong', 'default', 'weak'].
            These can be overridden by specifying atom_expr or bond_expr.
        atom_expr : openeye.oechem.OEExprOpts
            Override for atom matching expression; None if default is to be used.
        bond_expr : openeye.oechem.OEExprOpts
            Override for bond matching expression; None if default is to be used.
        use_positions : bool, optional, default=True
            If True, will attempt to use positions of molecules to determine optimal mappings.
            If False, will use maximum common substructure (MCSS).
        allow_ring_breaking : bool, default=False
            Wether or not to allow ring breaking in map
        external_inttypes : bool, optional, default=False
            If True, IntTypes already assigned to oemols will be used for mapping, if IntType is in the atom or bond expression.
            Otherwise, IntTypes will be overwritten such as to ensure rings of different sizes are not matched.
        matching_criterion : str, optional, default='index'
            The best atom map is pulled based on some ranking criteria;
            if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
            if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
            else: raise Exception.
            NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                   if 'names' is chosen, it is first asserted that old and new molecules have atoms that are uniquely named
        coordinate_tolerance : simtk.unit.Quantity, optional, default=0.25*simtk.unit.angstroms
            Coordinate tolerance for geometry-derived mappings.

        """
        # Configure default object attributes
        self.use_positions = use_positions
        self.allow_ring_breaking = allow_ring_breaking
        self.external_inttypes = external_inttypes
        self.matching_criterion = matching_criterion
        self.coordinate_tolerance = coordinate_tolerance

        # Determine atom and bond expressions
        import openeye.oechem as oechem
        DEFAULT_EXPRESSIONS = {
            # weak requirements for mapping atoms == more atoms mapped, more in core
            # atoms need to match in aromaticity. Same with bonds.
            # maps ethane to ethene, CH3 to NH2, but not benzene to cyclohexane
            'weak' : {
                'atom' : oechem.OEExprOpts_EqAromatic | oechem.OEExprOpts_EqNotAromatic, #| oechem.OEExprOpts_IntType
                'bond' : oechem.OEExprOpts_DefaultBonds
            },
            # default atom expression, requires same aromaticitiy and hybridization
            # bonds need to match in bond order
            # ethane to ethene wouldn't map, CH3 to NH2 would map but CH3 to HC=O wouldn't
            'default' : {
                'atom' : oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember,
                #'atom' : oechem.OEExprOpts_Hybridization, #| oechem.OEExprOpts_IntType
                'bond' : oechem.OEExprOpts_DefaultBonds
            },
            # strong requires same hybridization AND the same atom type
            # bonds are same as default, require them to match in bond order
            'strong' : {
                'atom' : oechem.OEExprOpts_Hybridization | oechem.OEExprOpts_HvyDegree | oechem.OEExprOpts_DefaultAtoms,
                'bond' : oechem.OEExprOpts_DefaultBonds
            }
        }

        if map_strength is None:
            map_strength = 'default'
        if atom_expr is None:
            _logger.debug(f'No atom expression defined, using map strength : {map_strength}')
            atom_expr = DEFAULT_EXPRESSIONS[map_strength]['atom']
        if bond_expr is None:
            _logger.debug(f'No bond expression defined, using map strength : {map_strength}')
            bond_expr = DEFAULT_EXPRESSIONS[map_strength]['bond']

        self.atom_expr = atom_expr
        self.bond_expr = bond_expr

    def get_all_mappings(self, old_mol, new_mol):
        """Retrieve all valid atom mappings and their scores for the proposed transformation.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mappings : list of AtomMapping
            All valid atom mappings

        Examples:
        ---------
        Specify two molecules without positions

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])')
        >>> ethanol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]')

        Retrieve all mappings between two molecules

        >>> atom_mappings = atom_mapper.get_all_mappings(ethane, ethanol)

        You can also specify OEMols for one or more of the molecules:

        >>> atom_mappings = atom_mapper.get_all_mappings(ethane.to_openeye(), ethanol.to_openeye())
        >>> atom_mappings = atom_mapper.get_all_mappings(ethane.to_openeye(), ethanol)

        """
        atom_mappings = set() # all unique atom mappings found

        # Create OpenFF and OEMol copies of input molecules, retaining properties if provided
        # TODO: Do we really need to retain OEMol atom/bond inttypes?
        def copy_molecule(mol):
            """Generate OpenFF and OEMol copies of a given molecule,
            retaining OEMol properties (such as atom/bond inttypes) if present.

            Parameters
            ----------
            mol : openeye.oechem.OEMol or openeye.oechem.OEGraphMol or openff.toolkit.topology.Molecule
                Molecule to be copied

            Returns
            -------
            offmol : openff.toolkit.topology.Molecule
                The OpenFF Molecule
            oemol : openeye.oechem.OEMol
                A copy of the OEMol, preserving data if provided
            """
            from openff.toolkit.topology import Molecule
            offmol = Molecule(mol)
            try:
                # Retain OEMol if provided, since it may have int atom and bond types
                # TODO: Revisit whether we actually need this
                from openeye import oechem
                oemol = oechem.OEMol(mol)
            except (TypeError, NotImplementedError) as e:
                oemol = offmol.to_openeye()

            return offmol, oemol

        old_offmol, old_oemol = copy_molecule(old_mol)
        new_offmol, new_oemol = copy_molecule(new_mol)

        # Annotate OEMol representations with ring IDs
        # TODO: What is all this doing
        if (not self.external_inttypes) or self.allow_ring_breaking:
            self._assign_ring_ids(old_oemol)
            self._assign_ring_ids(new_oemol)

        from perses.utils.openeye import get_scaffold
        old_oescaffold = get_scaffold(old_oemol)
        new_oescaffold = get_scaffold(new_oemol)

        self._assign_ring_ids(old_oescaffold, assign_atoms=True, assign_bonds=False)
        self._assign_ring_ids(new_oescaffold, assign_atoms=True, assign_bonds=False)

        # Generate scaffold maps
        # TODO: Why are these hard-coded?
        from openeye import oechem
        scaffold_maps = AtomMapper._get_all_maps(old_oescaffold, new_oescaffold,
                                                 atom_expr=oechem.OEExprOpts_RingMember | oechem.OEExprOpts_IntType,
                                                 bond_expr=oechem.OEExprOpts_RingMember,
                                                 external_inttypes=True,
                                                 unique=False,
                                                 matching_criterion=self.matching_criterion)

        _logger.info(f'Scaffold has symmetry of {len(scaffold_maps)}')

        if len(scaffold_maps) == 0:
            # There are no scaffold maps, so attempt to generate maps between molecules using the factory parameters
            _logger.warning('Two molecules are not similar to have a common scaffold')
            _logger.warning('Proceeding with direct mapping of molecules, but please check atom mapping and the geometry of the ligands.')

            # if no commonality with the scaffold, don't use it.
            # why weren't matching arguments carried to these mapping functions? is there an edge case that i am missing?
            # it still doesn't fix the protein sidechain mapping problem
            generated_atom_mappings = AtomMapper._get_all_maps(old_oemol, new_oemol,
                                                        external_inttypes=self.external_inttypes,
                                                        atom_expr=self.atom_expr,
                                                        bond_expr=self.bond_expr,
                                                        matching_criterion=self.matching_criterion)
            _logger.info(f'{len(generated_atom_mappings)} mappings were generated by AtomMapper._get_all_maps()')
            for x in all_molecule_maps:
                _logger.info(x)

            atom_mappings.update(generated_atom_mappings)

            # TODO: Package maps as AtomMapping objects

        else:
            # Some scaffold mappings have been found, so do something fancy
            # TODO: What exactly is it we're doing?

            # Keep only those scaffold match(es) with maximum score
            # TODO: Will this cause difficulties when trying to stochastically propose maps in both directions,
            # or when we want to retain all maps?
            _logger.info(f'There are {len(scaffold_maps)} scaffold mappings before filtering by score')
            scores = [ self.score_mapping(atom_mapping) for atom_mapping in scaffold_maps ]
            scaffold_maps = [ atom_mapping for index, atom_mapping in enumerate(scaffold_maps) if scores[index]==max(scores) ]
            _logger.info(f'There are {len(scaffold_maps)} after filtering to remove lower-scoring scaffold maps')

            # Determine mappings from scaffold to original molecule
            # TODO: Rework this logic to use openff Molecule
            def determine_scaffold_to_molecule_mapping(oescaffold, oemol):
                """Determine mapping of scaffold to full molecule.

                Parameters
                ----------
                oescaffold : openeye.oechem.OEMol
                    The scaffold within the complete molecule
                oemol : openeye.oechem.OEMol
                    The complete molecule

                Returns
                -------
                scaffold_to_molecule_map : dict of int : int
                    scaffold_to_molecule_map[scaffold_atom_index] is the atom index in oemol corresponding to scaffold_atom_index in oescaffold

                """
                scaffold_to_molecule_maps = AtomMapper._get_all_maps(oescaffold, oemol,
                                                atom_expr=oechem.OEExprOpts_AtomicNumber,
                                                bond_expr=0,
                                                matching_criterion=self.matching_criterion)
                _logger.info(f'{len(scaffold_to_molecule_maps)} scaffold maps found')
                scaffold_to_molecule_map = scaffold_to_molecule_maps[0]
                _logger.info(f'Scaffold to molecule map: {scaffold_to_molecule_map}')
                assert len(scaffold_to_molecule_map.old_to_new_atom_map) == oescaffold.NumAtoms(), f'Scaffold should be fully contained within the molecule it came from: map: {scaffold_to_molecule_map}\n{oescaffold.NumAtoms()} atoms in scaffold'
                return scaffold_to_molecule_map

            old_scaffold_to_molecule_map = determine_scaffold_to_molecule_mapping(old_oescaffold, old_oemol)
            new_scaffold_to_molecule_map = determine_scaffold_to_molecule_mapping(new_oescaffold, new_oemol)

            # now want to find all of the maps
            # for all of the possible scaffold symmetries
            # TODO: Re-work this algorithm
            for scaffold_map in scaffold_maps:
                if (self.external_inttypes is False) and (self.allow_ring_breaking is True):
                    # reset the IntTypes
                    for oeatom in old_oemol.GetAtoms():
                        oeatom.SetIntType(0)
                    for oeatom in new_oemol.GetAtoms():
                        oeatom.SetIntType(0)

                    # Assign scaffold-mapped atoms in the real molecule an IntType equal to their mapping index
                    old_oeatoms = [ atom for atom in old_oemol.GetAtoms() ]
                    new_oeatoms = [ atom for atom in new_oemol.GetAtoms() ]
                    index = 1
                    for old_scaffold_atom_index, new_scaffold_atom_index in scaffold_map.old_to_new_atom_map.items():
                        old_oeatoms[old_scaffold_to_molecule_map.old_to_new_atom_map[old_scaffold_atom_index]].SetIntType(index)
                        new_oeatoms[new_scaffold_to_molecule_map.old_to_new_atom_map[new_scaffold_atom_index]].SetIntType(index)
                        index += 1
                    # Assign remaining unmapped atoms in the real molecules an IntType determined by their ring classes
                    self._assign_ring_ids(old_oemol, only_assign_if_zero=True)
                    self._assign_ring_ids(new_oemol, only_assign_if_zero=True)

                atom_mappings_for_this_scaffold_map = AtomMapper._get_all_maps(old_oemol, new_oemol,
                                                        external_inttypes=True,
                                                        atom_expr=self.atom_expr,
                                                        bond_expr=self.bond_expr,
                                                        matching_criterion=self.matching_criterion)
                atom_mappings.update(atom_mappings_for_this_scaffold_map)

        if not self.allow_ring_breaking:
            # Filter the matches to remove any that allow ring breaking
            _logger.info(f'Fixing mappings to not create or break rings')
            for atom_mapping in atom_mappings:
                atom_mapping.unmap_partially_mapped_cycles()

        # TODO: Should we attempt to preserve chirality here for all atom mappings?
        # Or is this just for biopolymer residues?

        if len(atom_mappings) == 0:
            _logger.warning('No maps found. Try relaxing match criteria or setting allow_ring_breaking to True')
            return None

        # Render set of AtomMapping to a list to return
        return list(atom_mappings)

    def get_best_mapping(self, old_mol, new_mol):
        """Retrieve the best mapping between old and new molecules.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            Atom mapping with the best score

        Examples
        --------
        Retrieve best-scoring mapping between ethane and ethanol

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])')
        >>> ethanol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]')
        >>> atom_mapping = atom_mapper.get_best_mapping(ethane, ethanol)

        """
        import time
        initial_time = time.time()

        import numpy as np
        atom_mappings = self.get_all_mappings(old_mol, new_mol)
        if (atom_mappings is None) or len(atom_mappings)==0:
            return None

        scores = np.array([ self.score_mapping(atom_mapping) for atom_mapping in atom_mappings ])
        best_map_index = np.argmax(scores)

        elapsed_time = time.time() - initial_time
        _logger.info(f'get_best_mapping took {elapsed_time:.3f} s')

        return atom_mappings[best_map_index]

    def get_sampled_mapping(self, old_mol, new_mol):
        """Stochastically generate a mapping between old and new molecules selected proportional to its score.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            Atom mapping with the best score

        Examples
        --------
        Sample a mapping stochasticaly between ethane and ethanol

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])')
        >>> ethanol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]')
        >>> atom_mapping = atom_mapper.get_sampled_mapping(ethane, ethanol)

        """
        import numpy as np
        atom_mappings = self.get_all_maps(old_mol, new_mol)
        scores = np.array([ self.score_mapping(atom_mapping) for atom_mapping in atom_mappings ])
        # Compute normalized probability for sampling from mappings
        p = scores/np.sum(scores)
        # Select mapping with associated this probability
        selected_map_index = np.random.choice(np.arange(len(scores)), p=p)
        # Return the sampled mapping
        return atom_mappings[selected_map_index]

    def propose_mapping(old_mol, new_mol):
        """Propose new mapping stochastically and compute associated forward and reverse probabilities.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            Atom mapping with the best score
        logP_forward : float
            log probability of selecting atom_mapping in forward direction
        logP_reverse : float
            log probability of selecting atom_mapping in reverse direction

        Examples
        --------
        Propose a stochastic mapping between ethane and ethanol, computing the probability
        of both the forward mapping and reverse mapping choices:

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])')
        >>> ethanol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]')
        >>> atom_mapping, logP_forward, logP_reverse = atom_mapper.propose_mapping(ethane, ethanol)


        """
        # TODO: Stochastically select mapping, then compute forward and reverse log probabilities that same mapping
        #       would be used in forward and reverse directions (for new_mol -> old_mol)

        raise NotImplementedError('This feature has not been implemented yet')

    def score_mapping(self, atom_mapping):
        """Gives a score to each map.

        If molecule positions are available, the inverse of the total Euclidean deviation between heavy atoms is returned.
        If no positions are available, the number of mapped atoms is returned.

        This method can be overridden by subclasses to experiment with different schemes for prioritizing atom maps.

        Parameters
        ----------
        atom_mapping : AtomMapping
            The atom mapping to score

        Returns
        -------
        score : float
            A score for the atom mapping, where larger scores indicate better maps.

        Examples
        --------
        Compute scores for all mappings between ethane and ethanol:

        >>> from openff.toolkit.topology import Molecule
        >>> ethane = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])')
        >>> ethanol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]')
        >>> atom_mappings = atom_mapper.propose_mapping(ethane, ethanol)
        >>> scores = [ atom_mapper.score_mapping(atom_mapping) for atom_mapping in atom_mappings ]

        """

        # Score by positions
        if self.use_positions and (atom_mapping.old_mol.conformers is not None) and (atom_mapping.new_mol.conformers is not None):
            # Get all-to-all atom distance matrix
            # TODO: Only compute heavy atom distances
            from simtk import unit
            old_positions = atom_mapping.old_mol.conformers[0] / unit.angstroms
            new_positions = atom_mapping.new_mol.conformers[0] / unit.angstroms

            def dist(a, b):
                """Compute distance between numpy d-vectors a and b.

                Parameters
                ----------
                a, b : numpy.array (d,) arrays
                    Vectors to compute distance between

                Returns
                -------
                distance : float
                    The distance
                """
                import numpy as np
                return np.linalg.norm(b-a)

            # Score mapped heavy atoms using a Gaussian overlap function
            # TODO: Perhaps we should just weight hydrogens different from heavy atoms?
            map_score = 0.0
            for old_atom_index, new_atom_index in atom_mapping.old_to_new_atom_map.items():
                weight = 1.0
                if (atom_mapping.old_mol.atoms[old_atom_index].atomic_number==1) and (atom_mapping.new_mol.atoms[new_atom_index].atomic_number==1):
                    weight = 0.0 # hydrogen weight

                nsigma = dist(old_positions[old_atom_index,:], new_positions[new_atom_index,:]) / self.coordinate_tolerance
                map_score += weight * np.exp(-0.5 * nsigma**2)

        else:
            # There are no positions, so compute score derived from mapping
            # This is inspired by the former rank_degenerate_maps code
            # https://github.com/choderalab/perses/blob/412750c457712da1875c7beabfe88b2838f7f197/perses/rjmc/topology_proposal.py#L1123

            old_oeatoms = { oeatom.GetIdx() : oeatom for oeatom in atom_mapping.old_mol.to_openeye().GetAtoms() }
            new_oeatoms = { oeatom.GetIdx() : oeatom for oeatom in atom_mapping.new_mol.to_openeye().GetAtoms() }

            # Generate filtered mappings
            old_to_new_atom_map = atom_mapping.old_to_new_atom_map

            mapped_atoms = {
                old_index : new_index for old_index, new_index in old_to_new_atom_map.items()
                }

            mapped_aromatic_atoms = {
                old_index : new_index for old_index, new_index in old_to_new_atom_map.items()
                    if old_oeatoms[old_index].IsAromatic() and new_oeatoms[new_index].IsAromatic()
                }

            mapped_heavy_atoms = {
                old_index : new_index for old_index, new_index in old_to_new_atom_map.items()
                    if (old_oeatoms[old_index].GetAtomicNum()>1) and (new_oeatoms[new_index].GetAtomicNum()>1)
                }

            mapped_ring_atoms = {
                old_index : new_index for old_index, new_index in old_to_new_atom_map.items()
                    if old_oeatoms[old_index].IsInRing() and new_oeatoms[new_index].IsInRing()
                }

            # These weights are totally arbitrary
            map_score = 1.0 * len(mapped_atoms) \
                      + 0.8 * len(mapped_aromatic_atoms) \
                      + 0.5 * len(mapped_heavy_atoms) \
                      + 0.4 * len(mapped_ring_atoms)

        return map_score

    def generate_atom_mapping_from_positions(self, old_mol, new_mol):
        """Generate an atom mapping derived entirely from atom position proximity.

        The resulting map will be cleaned up by de-mapping hydrogens and rings as needed.

        Parameters
        ----------
        old_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The initial molecule for the transformation.
        new_mol : openff.toolkit.topology.Molecule or openeye.oechem.OEMol
            The final molecule for the transformation.

        Returns
        -------
        atom_mapping : AtomMapping
            The atom mapping determined from positions.
            mapping[molB_index] = molA_index is the mapping of atoms from molA to molB that are geometrically close

        Examples
        --------
        Derive atom mapping from positions:

        >>> atom_mapper = AtomMapper()
        >>> old_mol = Molecule.from_file('old_mol.sdf')
        >>> new_mol = Molecule.from_file('new_mol.sdf')
        >>> atom_mapping = atom_mapper.generate_atom_mapping_from_positions(old_mol, new_mol)

        """
        from openff.toolkit.topology import Molecule
        distance_unit = unit.angstroms # distance unit for comparisons

        # Coerce to openff Molecule
        old_mol = Molecule(old_mol)
        new_mol = Molecule(new_mol)

        # Check to ensure conformers are defined
        if (old_mol.conformers is None) or (new_mol.conformers is None):
            raise InvalidMappingException(f'Both old and new molecules must have at least one conformer defined.')

        # Get conformers in common distance unit as numpy arrays
        old_mol_positions = old_mol.conformers[0] / distance_unit
        new_mol_positions = new_mol.conformers[0] / distance_unit

        # TODO: Refactor
        molA_positions = old_mol.to_openeye().GetCoords() # coordinates (Angstroms)
        molB_positions = new_mol.to_openeye().GetCoords() # coordinates (Angstroms)
        molB_backward_positions = {val: key for key, val in molB_positions.items()}

        # Define closeness criteria for np.allclose
        rtol = 0.0 # relative tolerane
        atol = self.coordinate_tolerance / distance_unit # absolute tolerance (Angstroms)

        old_to_new_atom_map = dict()
        for old_atom_index in range(old_mol.n_atoms):
            # Determine which new atom indices match the old atom
            new_atom_matches = [
                new_atom_index
                for new_atom_index in range(new_mol.n_atoms)
                if np.allclose(old_mol_positions[old_atom_index,:], new_mol_positions[new_atom_index,:], rtol=rtol, atol=atol)
                ]
            if not len(new_atom_matches) in [0,1]:
                raise InvalidMappingException(f"there are multiple new positions with the same coordinates as old atom {old_atom_index} for coordinate tolerance {self.coordinate_tolerance}")
            if len(new_atom_matches) == 1:
                new_atom_index = new_atom_matches[0]
                old_to_new_atom_map[old_atom_index] = new_atom_index

        atom_mapping = AtomMapping(old_mol, new_mol, old_to_new_atom_map=old_to_new_atom_map)

        # De-map rings if needed
        if not self.allow_ring_breaking:
            atom_mapping.unmap_partially_mapped_cycles()

        return atom_mapping

    @staticmethod
    def _get_all_maps(old_oemol, new_oemol,
        external_inttypes=False,
        atom_expr=None,
        bond_expr=None,
        # TODO: Should 'unique' be False by default?
        # See https://docs.eyesopen.com/toolkits/python/oechemtk/patternmatch.html#section-patternmatch-mcss
        unique=True,
        matching_criterion='index',
        ):
        """Generate all possible maps between two oemols

        Parameters
        ----------
        old_oemol : openeye.oechem.OEMol
            old molecule
        new_oemol : openeye.oechem.OEMol
            new molecule
        external_inttypes : bool, optional, default=False
            If True, IntTypes already assigned to oemols will be used for mapping, if IntType is in the atom or bond expression.
            Otherwise, IntTypes will be overwritten such as to ensure rings of different sizes are not matched.
        atom_expr : openeye.oechem.OEExprOpts
            Override for atom matching expression; None if default is to be used.
        bond_expr : openeye.oechem.OEExprOpts
            Override for bond matching expression; None if default is to be used.
        unique : bool, optional, default=True
            Passed to MCSS Match
        matching_criterion : str, optional, default='index'
            The best atom map is pulled based on some ranking criteria;
            if 'index', the best atom map is chosen based on the map with the maximum number of atomic index matches;
            if 'name', the best atom map is chosen based on the map with the maximum number of atom name matches
            else: raise Exception.
            NOTE : the matching criterion pulls patterns and target matches based on indices or names;
                   if 'names' is chosen, it is first asserted that the old and new molecules have atoms that are uniquely named

        Returns
        -------
        atom_mappings : list of AtomMappings
            All unique atom mappings

        .. todo :: Do we need the 'unique' argument here?

        .. todo :: Can we refactor to get rid of this function?

        """
        import openeye.oechem as oechem

        if atom_expr is None:
            atom_expr = self.atom_expr
        if bond_expr is None:
            bond_expr = self.bond_expr

        # this ensures that the hybridization of the oemols is done for correct atom mapping
        oechem.OEAssignHybridization(old_oemol)
        oechem.OEAssignHybridization(new_oemol)
        old_oegraphmol = oechem.OEGraphMol(old_oemol)  # pattern molecule
        new_oegraphmol = oechem.OEGraphMol(new_oemol)  # target molecule

        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)
        mcs.Init(old_oegraphmol, atom_expr, bond_expr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        matches = [m for m in mcs.Match(new_oegraphmol, unique)]
        _logger.debug(f'all matches have atom counts of : {[m.NumAtoms() for m in matches]}')

        atom_mappings = set()
        for match in matches:
            atom_mapping = AtomMapper._create_atom_mapping(old_oemol, new_oemol, match, matching_criterion)
            atom_mappings.add(atom_mapping)

        # Render to a list to return mappings
        return list(atom_mappings)

    @staticmethod
    def _create_pattern_to_target_map(old_oemol, new_oemol, match, matching_criterion='index'):
        """
        Create a dict of {pattern_atom: target_atom}

        Parameters
        ----------
        old_oemol : openeye.oechem.OEMol
            old molecule
        new_oemol : openeye.oechem.OEMol
            new molecule
        match : oechem.OEMCSSearch.Match iterable
            entry in oechem.OEMCSSearch.Match object
        matching_criterion : str, optional, default='index'
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        Returns
        -------
        pattern_to_target_map : dict of OEAtom : OEAtom
            {pattern_atom: target_atom}

        """
        if matching_criterion == 'index':
            pattern_atoms = { atom.GetIdx() : atom for atom in old_oemol.GetAtoms() }
            target_atoms  = { atom.GetIdx() : atom for atom in new_oemol.GetAtoms() }
            pattern_to_target_map = { pattern_atoms[matchpair.pattern.GetIdx()] : target_atoms[matchpair.target.GetIdx()] for matchpair in match.GetAtoms() }
        elif matching_criterion == 'name':
            pattern_atoms = { atom.GetName(): atom for atom in old_oemol.GetAtoms() }
            target_atoms  = { atom.GetName(): atom for atom in new_oemol.GetAtoms() }
            pattern_to_target_map = { pattern_atoms[matchpair.pattern.GetName()]: target_atoms[matchpair.target.GetName()] for matchpair in match.GetAtoms() }
        else:
            raise Exception(f"matching criterion {matching_criterion} is not currently supported")

        return pattern_to_target_map

    def _create_atom_mapping(old_oemol, new_oemol, match, matching_criterion):
        """
        Returns an AtomMapping that omits hydrogen-to-nonhydrogen atom maps
        as well as any X-H to Y-H where element(X) != element(Y) or aromatic(X) != aromatic(Y)

        Parameters
        ----------
        old_oemol : openeye.oechem.OEMol object
            The old molecules
        new_oemol : openeye.oechem.OEMol object
            The new molecule
        match : openeye.oechem.OEMatchBase iterable
            entry in oechem.OEMCSSearch.Match object
        matching_criterion : str
            Matching criterion for _create_pattern_to_target_map.
            whether the pattern to target map is chosen based on atom indices or names (which should be uniquely defined)
            allowables: ['index', 'name']

        Returns
        -------
        atom_mapping : AtomMapping
            The atom mapping

        """
        # TODO : Overhaul this to use OpenFF Molecule
        import openeye.oechem as oechem
        new_to_old_atom_map = dict()
        pattern_to_target_map = AtomMapper._create_pattern_to_target_map(old_oemol, new_oemol, match, matching_criterion)
        for pattern_oeatom, target_oeatom in pattern_to_target_map.items():
            old_index, new_index = pattern_oeatom.GetIdx(), target_oeatom.GetIdx()
            old_oeatom, new_oeatom = pattern_oeatom, target_oeatom

            # Check if a hydrogen was mapped to a non-hydroden (basically the xor of is_h_a and is_h_b)
            if (old_oeatom.GetAtomicNum() == 1) != (new_oeatom.GetAtomicNum() == 1):
                continue

            # Check if X-H to Y-H changes where element(X) != element(Y) or aromatic(X) != aromatic(Y)
            if (old_oeatom.GetAtomicNum() == 1) and (new_oeatom.GetAtomicNum() == 1):
                X = [ bond.GetNbr(old_oeatom) for bond in old_oeatom.GetBonds() ][0]
                Y = [ bond.GetNbr(new_oeatom) for bond in new_oeatom.GetBonds() ][0]
                if ( X.GetAtomicNum() != Y.GetAtomicNum() ) or ( X.IsAromatic() != Y.IsAromatic() ):
                    continue

            new_to_old_atom_map[new_index] = old_index

        return AtomMapping(old_oemol, new_oemol, new_to_old_atom_map=new_to_old_atom_map)

    @staticmethod
    def _assign_ring_ids(oemol, max_ring_size=10, assign_atoms=True, assign_bonds=True, only_assign_if_zero=False):
        """ Sets the Int of each atom in the oemol to a number
        corresponding to the ring membership of that atom

        Parameters
        ----------
        oemol : openeye.oechem.OEMol
            oemol to assign ring ID to
        assign_atoms : bool, optional, default=True
            If True, assign atoms
        assign_bonds : bool, optional, default=True
            If True, assign bonds
        max_ring_size : int, optional, default=10
            Largest ring size that will be checked for
        only_assign_if_zero : bool, optional, default=False
            If True, will only assign atom IntTypes to atoms and bonds with non-zero IntType;
            bond IntTypes will not be assigned

        """
        def _assign_ring_id(oeobj, max_ring_size=10):
            import openeye.oechem as oechem
            """Returns the int type based on the ring occupancy of the atom or bond

            Parameters
            ----------
            oeobj : openeye.oechem.OEAtomBase or openeye.oechem.OEBondBase
                atom or bond to compute ring membership integer for
            max_ring_size : int, optional, default=10
                Largest ring size that will be checked for

            Returns
            -------
            ring_as_base_two : int
                Integer encoding binary ring membership for atom or bond
            """
            import openeye.oechem as oechem

            if hasattr(oeobj, 'GetAtomicNum'):
                fun = oechem.OEAtomIsInRingSize
            elif hasattr(oeobj, 'GetOrder'):
                fun = oechem.OEBondIsInRingSize
            else:
                raise ValueError(f'Argument {oeobj} is not an OEAtom or OEBond')

            rings = ''
            for i in range(3, max_ring_size+1): #  smallest feasible ring size is 3
                rings += str(int(fun(oeobj, i)))
            ring_as_base_two = int(rings, 2)
            return ring_as_base_two

        if assign_atoms:
            for oeatom in oemol.GetAtoms():
                if only_assign_if_zero and oeatom.GetIntType() != 0:
                    continue
                oeatom.SetIntType(_assign_ring_id(oeatom, max_ring_size=max_ring_size))

        if assign_bonds:
            for oebond in oemol.GetBonds():
                oebond.SetIntType(_assign_ring_id(oebond, max_ring_size=max_ring_size))

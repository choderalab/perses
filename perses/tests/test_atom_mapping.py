from pkg_resources import resource_filename
import os
import pytest
import unittest

from perses.utils.smallmolecules import render_atom_mapping
from openff.toolkit.topology import Molecule
from perses.rjmc.atom_mapping import AtomMapper, AtomMapping, InvalidMappingException

class AtomMappingTest(unittest.TestCase):
    """Test AtomMapping object."""
    def setUp(self):
        """Create useful common objects for testing."""
        from openff.toolkit.topology import Molecule
        self.old_mol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])([H:7])') # ethane
        self.new_mol = Molecule.from_smiles('[C:0]([H:1])([H:2])([H:3])[C:4]([H:5])([H:6])[O:7][H:8]') # ethanol
        self.old_to_new_atom_map = { 0:0, 4:4 }
        self.new_to_old_atom_map = dict(map(reversed, self.old_to_new_atom_map.items()))

    def test_create(self):
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        assert atom_mapping.old_to_new_atom_map == self.old_to_new_atom_map
        assert atom_mapping.new_to_old_atom_map == self.new_to_old_atom_map
        assert atom_mapping.n_mapped_atoms == 2
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, new_to_old_atom_map=self.new_to_old_atom_map)
        assert atom_mapping.old_to_new_atom_map == self.old_to_new_atom_map
        assert atom_mapping.new_to_old_atom_map == self.new_to_old_atom_map
        assert atom_mapping.n_mapped_atoms == 2

    def test_validation_fail(self):
        # Empty mapping
        with pytest.raises(InvalidMappingException) as excinfo:
            atom_mapping = AtomMapping(self.old_mol, self.new_mol, { })
        # Non-integers
        with pytest.raises(InvalidMappingException) as excinfo:
            atom_mapping = AtomMapping(self.old_mol, self.new_mol, { 0:0, 4:4, 5:'a' })
        # Invalid atom indices
        with pytest.raises(InvalidMappingException) as excinfo:
            atom_mapping = AtomMapping(self.old_mol, self.new_mol, { 0:0, 4:4, 9:9 })
        # Duplicated atom indices
        with pytest.raises(InvalidMappingException) as excinfo:
            atom_mapping = AtomMapping(self.old_mol, self.new_mol, { 0:0, 4:4, 3:4 })

    def test_set_and_get_mapping(self):
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        # Set old-to-new map
        atom_mapping.old_to_new_atom_map = self.old_to_new_atom_map
        assert atom_mapping.old_to_new_atom_map == self.old_to_new_atom_map
        assert atom_mapping.new_to_old_atom_map == self.new_to_old_atom_map
        # Set new-to-old map
        atom_mapping.new_to_old_atom_map = self.new_to_old_atom_map
        assert atom_mapping.old_to_new_atom_map == self.old_to_new_atom_map
        assert atom_mapping.new_to_old_atom_map == self.new_to_old_atom_map

    def test_repr(self):
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        repr(atom_mapping)

    def test_str(self):
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        str(atom_mapping)

    def test_render_image(self):
        import tempfile
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        for suffix in ['.pdf', '.png', '.svg']:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmpfile:
                atom_mapping.render_image(tmpfile.name)

    def test_ring_breaking_detection(self):
        # Test simple ethane -> ethanol transformation
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        assert atom_mapping.creates_or_breaks_rings() == False

        # Define benzene -> napthalene transformation
        from openff.toolkit.topology import Molecule
        old_mol = Molecule.from_smiles('[c:0]1[c:1][c:2][c:3][c:4][c:5]1') # benzene
        new_mol = Molecule.from_smiles('[c:0]12[c:1][c:2][c:3][c:4][c:5]2[c:6][c:7][c:8][c:9]1') # napthalene
        old_to_new_atom_map = { 0:0, 1:1, 2:2, 3:3, 4:4, 5:5 }
        new_to_old_atom_map = dict(map(reversed, self.old_to_new_atom_map.items()))
        atom_mapping = AtomMapping(old_mol, new_mol, old_to_new_atom_map=old_to_new_atom_map)
        assert atom_mapping.creates_or_breaks_rings() == True

    def test_unmap_partially_mapped_cycles(self):
        # Test simple ethane -> ethanol transformation
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        n_mapped_atoms_old = atom_mapping.n_mapped_atoms
        atom_mapping.unmap_partially_mapped_cycles()
        assert atom_mapping.n_mapped_atoms == n_mapped_atoms_old

        # Test methyl-cyclohexane -> methyl-cyclopentane, demapping the ring transformation
        from openff.toolkit.topology import Molecule
        old_mol = Molecule.from_smiles('[C:0][C:1]1[C:2][C:3][C:4][C:5][C:6]1') # methyl-cyclohexane
        new_mol = Molecule.from_smiles('[C:0][C:1]1[C:2][C:3][C:4][C:5]1') # methyl-cyclopentane
        old_to_new_atom_map = { 0:0, 1:1, 2:2, 3:3, 5:4, 6:5 }
        new_to_old_atom_map = dict(map(reversed, self.old_to_new_atom_map.items()))
        atom_mapping = AtomMapping(old_mol, new_mol, old_to_new_atom_map=old_to_new_atom_map)
        assert atom_mapping.creates_or_breaks_rings() == True
        atom_mapping.unmap_partially_mapped_cycles()
        assert atom_mapping.old_to_new_atom_map == {0:0} # only methyl group should remain mapped

    def test_preserve_chirality(self):
        # Test simple ethane -> ethanol transformation
        atom_mapping = AtomMapping(self.old_mol, self.old_mol, old_to_new_atom_map=self.old_to_new_atom_map)
        n_mapped_atoms_old = atom_mapping.n_mapped_atoms
        atom_mapping.preserve_chirality()
        assert atom_mapping.n_mapped_atoms == n_mapped_atoms_old

        # Test resolution of incorrect stereochemistry
        from openff.toolkit.topology import Molecule
        old_mol = Molecule.from_smiles('[C@H:0]([Cl:1])([Br:2])([F:3])')
        new_mol = Molecule.from_smiles('[C@@H:0]([Cl:1])([Br:2])([F:3])')
        atom_mapping = AtomMapping(old_mol, new_mol, old_to_new_atom_map={0:0, 1:1, 2:2, 3:3})
        atom_mapping.preserve_chirality()
        assert atom_mapping.old_to_new_atom_map == {0:0, 1:1, 2:2, 3:3} # TODO: Check this

class TestAtomMapper(unittest.TestCase):
    def setUp(self):
        self.molecules = dict()
        for dataset_name in ['CDK2', 'p38', 'Tyk2', 'Thrombin', 'PTP1B', 'MCL1', 'Jnk1', 'Bace']:
            # Read molecules
            dataset_path = 'data/schrodinger-jacs-datasets/%s_ligands.sdf' % dataset_name
            sdf_filename = resource_filename('perses', dataset_path)
            self.molecules[dataset_name] = Molecule.from_file(sdf_filename, allow_undefined_stereo=True)

    def test_molecular_atom_mapping(self):
        """Test the creation of atom maps between pairs of molecules from the JACS benchmark set.
        """
        # Create and configure an AtomMapper
        atom_mapper = AtomMapper()

        # Test mappings for JACS dataset ligands
        # TODO: Uncomment other test datasets
        for dataset_name in ['CDK2']: #, 'p38', 'Tyk2', 'Thrombin', 'PTP1B', 'MCL1', 'Jnk1', 'Bace']:
            molecules = self.molecules[dataset_name]

            # Build atom map for some transformations.
            #for (molecule1, molecule2) in combinations(molecules, 2): # too slow
            from itertools import combinations
            for old_index, new_index in combinations(range(len(molecules)), 2):
                atom_mapping = atom_mapper.get_best_mapping(molecules[old_index], molecules[new_index])
                # TODO: Perform quality checks
                # Render mapping for visual inspection
                filename = f'mapping-{dataset_name}-{old_index}-to-{new_index}.png'
                atom_mapping.render_image(filename)

    def test_map_strategy(self):
        """
        Test the creation of atom maps between pairs of molecules from the JACS benchmark set.
        """
        # Create and configure an AtomMapper
        from openeye import oechem
        atom_expr = oechem.OEExprOpts_IntType
        bond_expr = oechem.OEExprOpts_RingMember
        atom_mapper = AtomMapper(atom_expr=atom_expr, bond_expr=bond_expr)

        # Test mappings for JACS dataset ligands
        for dataset_name in ['Jnk1']:
            molecules = self.molecules[dataset_name]

            # Jnk1 ligands 0 and 2 have meta substituents that face opposite each other in the active site.
            # When ignoring position information, the mapper should align these groups, and put them both in the core.
            # When using position information, the mapper should see that the orientations differ and chose
            # to unmap (i.e. put both these groups in core) such as to get the geometry right at the expense of
            # mapping fewer atoms

            # Ignore positional information when scoring mappings
            atom_mapper.use_positions = False
            atom_mapping = atom_mapper.get_best_mapping(molecules[0], molecules[2])
            assert len(atom_mapping.new_to_old_atom_map) == 36, f'Expected meta groups methyl C to map onto ethyl O\n{atom_mapping}'

            # Use positional information to score mappings
            atom_mapper.use_positions = True
            atom_mapping = atom_mapper.get_best_mapping(molecules[0], molecules[2])
            assert len(atom_mapping.new_to_old_atom_map) == 35,  f'Expected meta groups methyl C to NOT map onto ethyl O as they are distal in cartesian space\n{atom_mapping}'

            # Explicitly construct mapping from positional information alone
            atom_mapping = atom_mapper.generate_atom_mapping_from_positions(molecules[0], molecules[2])
            assert len(atom_mapping.new_to_old_atom_map) == 35,  f'Expected meta groups methyl C to NOT map onto ethyl O as they are distal in cartesian space\n{atom_mapping}'

    def test_simple_heterocycle_mapping(self):
        """
        Test the ability to map conjugated heterocycles (that preserves all rings).  Will assert that the number of ring members in both molecules is the same.
        """
        # TODO: generalize this to test for ring breakage and closure.

        iupac_pairs = [
            ('benzene', 'pyridine')
            ]

        # Create and configure an AtomMapper
        atom_mapper = AtomMapper(allow_ring_breaking=False)


        for old_iupac, new_iupac in iupac_pairs:
            from openff.toolkit.topology import Molecule
            old_mol = Molecule.from_iupac(old_iupac)
            new_mol = Molecule.from_iupac(new_iupac)
            atom_mapping = atom_mapper.get_best_mapping(old_mol, new_mol)

            assert len(atom_mapping.old_to_new_atom_map) > 0

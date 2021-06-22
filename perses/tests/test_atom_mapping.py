from pkg_resources import resource_filename
import os

from perses.utils.smallmolecules import render_atom_mapping
from openff.toolkit.topology import Molecule
from perses.rjmc.atom_mapping import AtomMapper

def test_ring_breaking_detection():
    """
    Test the detection of ring-breaking transformations.

    """
    from openmoltools.openeye import iupac_to_oemol, generate_conformers
    molecule1 = iupac_to_oemol("naphthalene")
    molecule2 = iupac_to_oemol("benzene")
    molecule1 = generate_conformers(molecule1,max_confs=1)
    molecule2 = generate_conformers(molecule2,max_confs=1)

    # Allow ring breaking
    new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecule1, molecule2, allow_ring_breaking=True)
    if not len(new_to_old_atom_map) > 0:
        filename = 'mapping-error.png'
        #render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
        msg = 'Napthalene -> benzene transformation with allow_ring_breaking=True is not returning a valid mapping\n'
        msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
        msg += str(new_to_old_atom_map)
        raise Exception(msg)

    new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecule1, molecule2, allow_ring_breaking=False)
    if new_to_old_atom_map is not None: # atom mapper should not retain _any_ atoms in default mode
        filename = 'mapping-error.png'
        #render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
        msg = 'Napthalene -> benzene transformation with allow_ring_breaking=False is erroneously allowing ring breaking\n'
        msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
        msg += str(new_to_old_atom_map)
        raise Exception(msg)

def test_molecular_atom_mapping():
    """
    Test the creation of atom maps between pairs of molecules from the JACS benchmark set.

    """
    from openeye import oechem

    # Test mappings for JACS dataset ligands
    for dataset_name in ['CDK2']: #, 'p38', 'Tyk2', 'Thrombin', 'PTP1B', 'MCL1', 'Jnk1', 'Bace']:
        # Read molecules
        dataset_path = 'data/schrodinger-jacs-datasets/%s_ligands.sdf' % dataset_name
        mol2_filename = resource_filename('perses', dataset_path)
        ifs = oechem.oemolistream(mol2_filename)
        molecules = list()
        for mol in ifs.GetOEGraphMols():
            molecules.append(oechem.OEGraphMol(mol))

        # Build atom map for some transformations.
        #for (molecule1, molecule2) in combinations(molecules, 2): # too slow
        molecule1 = molecules[0]
        for i, molecule2 in enumerate(molecules[1:]):
            new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecule1, molecule2)
            # Make sure we aren't mapping hydrogens onto anything else
            atoms1 = [atom for atom in molecule1.GetAtoms()]
            atoms2 = [atom for atom in molecule2.GetAtoms()]
            #for (index2, index1) in new_to_old_atom_map.items():
            #    atom1, atom2 = atoms1[index1], atoms2[index2]
            #    if (atom1.GetAtomicNum()==1) != (atom2.GetAtomicNum()==1):
            filename = 'mapping-error-%d.png' % i
            render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
            #msg = 'Atom atomic number %d is being mapped to atomic number %d\n' % (atom1.GetAtomicNum(), atom2.GetAtomicNum())
            msg = 'molecule 1 : %s\n' % oechem.OECreateIsoSmiString(molecule1)
            msg += 'molecule 2 : %s\n' % oechem.OECreateIsoSmiString(molecule2)
            msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
            msg += str(new_to_old_atom_map)
            print(msg)
            #        raise Exception(msg)

def test_map_strategy():
    """
    Test the creation of atom maps between pairs of molecules from the JACS benchmark set.

    """
    from openeye import oechem

    # Test mappings for JACS dataset ligands
    for dataset_name in ['Jnk1']:
        # Read molecules
        dataset_path = 'data/schrodinger-jacs-datasets/%s_ligands.sdf' % dataset_name
        mol2_filename = resource_filename('perses', dataset_path)
        ifs = oechem.oemolistream(mol2_filename)
        molecules = list()
        for mol in ifs.GetOEGraphMols():
            molecules.append(oechem.OEGraphMol(mol))

        atom_expr = oechem.OEExprOpts_IntType
        bond_expr = oechem.OEExprOpts_RingMember

        # the 0th and 1st Jnk1 ligand have meta substituents that face opposite eachother
        # in the active site. Using `map_strategy=matching_criterion` should align these groups, and put them
        # both in the core. Using `map_strategy=geometry` should see that the orientations differ and chose
        # to unmap (i.e. put both these groups in core) such as to get the geometry right at the expense of
        # mapping fewer atoms
        new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecules[0], molecules[1],atom_expr=atom_expr,bond_expr=bond_expr)
        assert len(new_to_old_atom_map) == 37, 'Expected meta groups methyl C to map onto ethyl O'

        new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecules[0], molecules[1],atom_expr=atom_expr,bond_expr=bond_expr,map_strategy='geometry')
        assert len(new_to_old_atom_map) == 35,  'Expected meta groups methyl C to NOT map onto ethyl O as they are distal in cartesian space'


def test_simple_heterocycle_mapping(iupac_pairs = [('benzene', 'pyridine')]):
    """
    Test the ability to map conjugated heterocycles (that preserves all rings).  Will assert that the number of ring members in both molecules is the same.
    """
    # TODO: generalize this to test for ring breakage and closure.
    from openmoltools.openeye import iupac_to_oemol
    from openeye import oechem

    for iupac_pair in iupac_pairs:
        old_oemol, new_oemol = iupac_to_oemol(iupac_pair[0]), iupac_to_oemol(iupac_pair[1])
        new_to_old_map = AtomMapper._get_mol_atom_map(old_oemol, new_oemol, allow_ring_breaking=False)

        # Assert that the number of ring members is consistent in the mapping...
        num_hetero_maps = 0
        for new_index, old_index in new_to_old_map.items():
            old_atom, new_atom = old_oemol.GetAtom(oechem.OEHasAtomIdx(old_index)), new_oemol.GetAtom(oechem.OEHasAtomIdx(new_index))
            if old_atom.IsInRing() and new_atom.IsInRing():
                if old_atom.GetAtomicNum() != new_atom.GetAtomicNum():
                    num_hetero_maps += 1

        assert num_hetero_maps > 0, f"there are no differences in atomic number mappings in {iupac_pair}"

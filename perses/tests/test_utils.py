"""
Unit tests for test utilities.

"""

__author__ = 'John D. Chodera'

################################################################################
# GLOBAL IMPORTS
################################################################################

from nose.plugins.attrib import attr

################################################################################
# Suppress matplotlib logging
################################################################################

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

################################################################################
# CONSTANTS
################################################################################

################################################################################
# TESTS
################################################################################

@attr('travis')
def test_sanitizeSMILES():
    """
    Test SMILES sanitization.
    """
    from perses.tests.utils import sanitizeSMILES

    smiles_list = ['CC', 'CCC', '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]']

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='drop')
    if len(sanitized_smiles_list) != 2:
        raise Exception("Molecules with undefined stereochemistry are not being properly dropped (size=%d)." % len(sanitized_smiles_list))

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='expand')
    if len(sanitized_smiles_list) != 4:
        raise Exception("Molecules with undefined stereochemistry are not being properly expanded (size=%d)." % len(sanitized_smiles_list))

    # Check that all molecules can be round-tripped
    from perses.rjmc.topology_proposal import OESMILES_OPTIONS
    from openeye import oechem
    for smiles in sanitized_smiles_list:
        molecule = oechem.OEGraphMol()
        oechem.OESmilesToMol(molecule, smiles)
        isosmiles = oechem.OECreateSmiString(molecule, OESMILES_OPTIONS)
        if (smiles != isosmiles):
            raise Exception("Molecule '%s' was not properly round-tripped (result was '%s')" % (smiles, isosmiles))

@attr('travis')
def test_generate_test_topology_proposal():
    """Test generate_vacuum_topology_proposal"""
    from perses.tests.utils import generate_test_topology_proposal

    # Create the topology proposal
    topology_proposal, old_positions, new_positions = generate_test_topology_proposal()

    # Test solvated system
    topology_proposal, old_positions, new_positions = generate_test_topology_proposal(solvent=True)

    # Test writing of atom mapping
    topology_proposal, old_positions, new_positions = generate_test_topology_proposal(write_atom_mapping=True)

    # Test other molecule names
    topology_proposal, old_positions, new_positions = generate_test_topology_proposal(old_iupac_name='benzene', new_iupac_name='toluene')
    topology_proposal, old_positions, new_positions = generate_test_topology_proposal(old_iupac_name='toluene', new_iupac_name='catechol')

@attr('travis')
def test_generate_vacuum_hostguest_proposal():
    """Test generate_vacuum_hostguest_proposal"""
    from perses.tests.utils import generate_vacuum_hostguest_proposal
    # Create the topology proposal
    topology_proposal, old_positions, new_positions = generate_vacuum_hostguest_proposal()

@attr('travis')
def test_createOEMolFromIUPAC():
    """Test createOEMolFromIUPAC"""
    from perses.tests.utils import createOEMolFromIUPAC

    # Create a few molecules
    iupac_list = [
        'ethane',
        'phenol',
        'aspirin',
    ]
    for iupac in iupac_list:
        oemol = createOEMolFromIUPAC(iupac)

    # Test setting the title
    oemol = createOEMolFromIUPAC('ethane', title='XYZ')
    assert oemol.GetTitle() == 'XYZ'

@attr('travis')
def test_createOEMolFromSMILES():
    """Test createOEMolFromSMILES"""
    from perses.tests.utils import createOEMolFromSMILES

    # Create a few molecules
    smiles_list = [
        'CC', # ethane
        'c1ccc(cc1)O', # phenol
        'O=C(C)Oc1ccccc1C(=O)O', # aspirin
    ]
    for smiles in smiles_list:
        oemol = createOEMolFromSMILES(smiles)

    # Test setting the title
    oemol = createOEMolFromSMILES('CC', title='XYZ')
    assert oemol.GetTitle() == 'XYZ'

@attr('travis')
def test_createSystemFromIUPAC():
    """Test createSystemFromIUPAC"""
    from perses.tests.utils import createSystemFromIUPAC

    # Create a few molecules
    iupac_list = [
        'ethane',
        'phenol',
        'aspirin',
    ]
    for iupac in iupac_list:
        oemol, system, positions, topology = createSystemFromIUPAC(iupac)
        resnames = [ residue.name for residue in topology.residues() ]
        assert resnames[0] == iupac

    # Test setting the residue name
    oemol, system, positions, topology = createSystemFromIUPAC('ethane', resname='XYZ')
    resnames = [ residue.name for residue in topology.residues() ]
    assert resnames[0] == 'XYZ'

"""
Unit tests for the hybrid relative alchemical factory.

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

def create_vacuum_hybrid_system(old_iupac_name="styrene", new_iupac_name="2-phenylethanol"):
    """
    Generate hybrid alchemical System for a transformation between two molecules in vacuum.

    Parameters
    ----------
    old_iupac_name : str
        The IUPAC name of the initial molecule.
    new_iupac_name : str
        The IUPAC name of the final molecule.

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        The TopologyProposal for the transformation
    hybrid_topology_factory : perses.annihilation.new_relative.HybridTopologyFactory
        The HybridTopologyFactory

    """
    # Create old system
    from perses.tests.utils import createSystemFromIUPAC
    old_oemol, old_system, old_positions, old_topology = createSystemFromIUPAC(old_iupac_name)
    # TODO: Check to make sure that the residue name is 'MOL'
    print([residue.name for residue in old_topology.residues()])
    assert old_oemol.GetTitle() == 'MOL', "old_oemol.GetTitle() != 'MOL' (instead is '{}')".format(old_oemol.GetTitle())

    # Create new molecule
    from perses.tests.utils import createOEMolFromIUPAC
    new_oemol = createOEMolFromIUPAC(new_iupac_name)
    assert new_oemol.GetTitle() == 'MOL', "new_oemol.GetTitle() != 'MOL' (instead is '{}')".format(new_oemol.GetTitle())

    # Create a SystemGenerator for systems in vacuum
    from simtk.openmm import app
    from perses.rjmc.topology_proposal import SystemGenerator
    from perses.tests.utils import get_data_filename
    forcefield_files = [get_data_filename("data/gaff.xml")]
    system_generator = SystemGenerator(forcefield_files, forcefield_kwargs={'constraints': app.HBonds})

    # Create a proposal engine to transform from one molecule to the other in vacuum
    from perses.rjmc.topology_proposal import TwoMoleculeSetProposalEngine
    proposal_engine = TwoMoleculeSetProposalEngine(old_oemol, new_oemol, system_generator, residue_name="MOL")

    # Create a topology proposal
    topology_proposal = proposal_engine.propose(old_system, old_topology)

    # Create a hybrid topology factory
    from perses.annihilation.new_relative import HybridTopologyFactory
    hybrid_factory = HybridTopologyFactory(topology_proposal, positions, positions)

    # Return the topology proposal and hybrid factory
    return topology_proposal, hybrid_factory

@attr('travis')
def test_vacuum_hybrid_system():
    """Test the creation of vacuum hybrid systems."""
    iupac_name_pairs = [
        ('propane', '1-chloropropane'),
        ('benzene', 'phenol'),
        ('styrene', '2-phenylethanol'),
    ]
    for old_iupac_name, new_iupac_name in iupac_name_pairs:
        topology_factory, hybrid_factory = create_vacuum_hybrid_system(old_iupac_name=old_iupac_name, new_iupac_name=new_iupac_name)

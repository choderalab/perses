################################################################################
# IMPORTS
################################################################################
import copy
from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from functools import partial
from pkg_resources import resource_filename
from perses.rjmc import geometry
from perses.rjmc.topology_proposal import SystemGenerator, TopologyProposal, SmallMoleculeSetProposalEngine
from openeye import oechem
if sys.version_info >= (3, 0):
    from io import StringIO
    from subprocess import getstatusoutput
else:
    from cStringIO import StringIO
    from commands import getstatusoutput
from openmmtools.constants import kB
from openmmtools import alchemy, states
from perses.tests.utils import render_atom_mapping

################################################################################
# CONSTANTS
################################################################################

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

LOGP_FORWARD_THRESHOLD = 1e3
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("Reference")

def test_logp_forward_check_for_vacuum_topology_proposal(current_mol_name = 'propane', proposed_mol_name = 'octane', num_iterations = 10):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.  Assert that the logp_forward < 1e3.
    This assertion will fail if the proposal order tool proposed the placement of the a carbon before a previously defined carbon in the alkane.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    current_mol, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(current_mol_name)
    proposed_mol = createOEMolFromIUPAC(proposed_mol_name)

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=current_mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old, current_mol=current_mol, proposed_mol=proposed_mol)

    # show atom mapping
    filename = str(current_mol_name)+str(proposed_mol_name)+'.pdf'
    render_atom_mapping(filename,current_mol,proposed_mol,topology_proposal.new_to_old_atom_map)

    for _ in range(num_iterations):
        #generate new positions with geometry engine
        new_positions, logp_forward, rjmc_info, atoms_with_positions_reduced_potential, final_context_reduced_potential = geometry_engine.propose(topology_proposal, pos_old, beta)
        assert logp_forward < 1e3, "A heavy atom was proposed in an improper order"


#test_logp_forward_check_for_vacuum_topology_proposal()

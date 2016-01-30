"""
Unit tests for NCMC switching engine.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import sys, math
import numpy as np
from functools import partial
from openeye import oechem
if sys.version_info >= (3, 0):
    from io import StringIO
    from subprocess import getstatusoutput
else:
    from cStringIO import StringIO
    from commands import getstatusoutput

################################################################################
# CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

################################################################################
# TESTS
################################################################################

# TODO: Move some of these utility routines to openmoltools.

def extractPositionsFromOEMOL(molecule):
    positions = unit.Quantity(np.zeros([molecule.NumAtoms(), 3], np.float32), unit.angstroms)
    coords = molecule.GetCoords()
    for index in range(molecule.NumAtoms()):
        positions[index,:] = unit.Quantity(coords[index], unit.angstroms)
    return positions

def createOEMolFromIUPAC(iupac_name='bosutinib'):
    from openeye import oechem, oeiupac, oeomega

    # Create molecule.
    mol = oechem.OEMol()
    oeiupac.OEParseIUPACName(mol, iupac_name)
    mol.SetTitle(iupac_name)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(mol, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(mol)

    # Create atom names.
    oechem.OETriposAtomNames(mol)

    # Assign geometry
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)
    omega(mol)

    return mol

def createSystemFromIUPAC(iupac_name):
    """
    Create an openmm system out of an oemol

    Parameters
    ----------
    iupac_name : str
        IUPAC name

    Returns
    -------
    molecule : openeye.OEMol
        OEMol molecule
    system : openmm.System object
        OpenMM system
    positions : [n,3] np.array of floats
        Positions
    topology : openmm.app.Topology object
        Topology
    """

    # Create OEMol
    molecule = createOEMolFromIUPAC(iupac_name)

    # Generate a topology.
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    topology = generateTopologyFromOEMol(molecule)

    # Initialize a forcefield with GAFF.
    # TODO: Fix path for `gaff.xml` since it is not yet distributed with OpenMM
    from simtk.openmm.app import ForceField
    forcefield = ForceField('gaff.xml')

    # Generate template and parameters.
    from openmoltools.forcefield_generators import generateResidueTemplate
    [template, ffxml] = generateResidueTemplate(molecule)

    # Register the template.
    forcefield.registerResidueTemplate(template)

    # Add the parameters.
    forcefield.loadFile(StringIO(ffxml))

    # Create the system.
    system = forcefield.createSystem(topology)

    # Extract positions
    positions = extractPositionsFromOEMOL(molecule)

    return (molecule, system, positions, topology)

def align_molecules(mol1, mol2):
    """
    MCSS two OEmols. Return the mapping of new : old atoms
    """
    mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
    atomexpr = oechem.OEExprOpts_AtomicNumber
    bondexpr = 0
    mcs.Init(mol1, atomexpr, bondexpr)
    mcs.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())
    unique = True
    match = [m for m in mcs.Match(mol2, unique)][0]
    new_to_old_atom_mapping = {}
    for matchpair in match.GetAtoms():
        old_index = matchpair.pattern.GetIdx()
        new_index = matchpair.target.GetIdx()
        new_to_old_atom_mapping[new_index] = old_index
    return new_to_old_atom_mapping

def simulate(system, positions, nsteps=500, timestep=1.0*unit.femtoseconds, temperature=300.0*unit.kelvin, collision_rate=20.0/unit.picoseconds):
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(nsteps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return positions

def check_alchemical_elimination(topology_proposal, ncmc_nsteps=50, NSIGMA_MAX=6.0):
    """
    Test alchemical elimination engine on alanine dipeptide null transformation.

    Parameters
    ----------
    topology_proposal : TopologyProposal
        The topology proposal to test.
    ncmc_steps : int, optional, default=50
        Number of NCMC switching steps, or 0 for instantaneous switching.
    NSIGMA_MAX : float, optional, default=6.0
        Number of standard errors away from analytical solution tolerated before Exception is thrown
    """
    # Initialize engine
    from perses.annihilation.ncmc_switching import NCMCEngine
    ncmc_engine = NCMCEngine(nsteps=ncmc_nsteps)

    #from perses.rjmc import geometry
    #geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})

    print("Atom mapping:")
    print(topology_proposal.new_to_old_atom_map)

    positions = topology_proposal.old_positions
    niterations = 20 # number of round-trip switching trials
    logP_insert_n = np.zeros([niterations], np.float64)
    logP_delete_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        # Equilibrate
        positions = simulate(topology_proposal.old_system, positions)

        # Delete atoms
        [ncmc_positions, logP_delete] = ncmc_engine.integrate(topology_proposal, positions, direction='delete')

        # Check that positions are not NaN
        if(np.any(np.isnan(ncmc_positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC annihilation")

        # Propose geometry change.
        #topology_proposal.old_positions = old_positions
        #new_positions, logp_proposal = geometry_engine.propose(topology_proposal)

        # Insert atoms
        #[new_positions, logP_insert] = ncmc_engine.integrate(topology_proposal, new_positions, direction='insert')

        # Compute total probability
        logP_delete_n[iteration] = logP_delete
        #logP_insert_n[iteration] = logP_insert
        print("Iteration %5d : %16.8f" % (iteration, logP_delete))

def test_alchemical_elimination_molecule():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, and 50 switching steps.

    """
    molecule_name_1 = 'ethane'
    molecule_name_2 = 'pentane'
    [molecule1, sys1, pos1, top1] = createSystemFromIUPAC(molecule_name_1)
    [molecule2, sys2, pos2, top2] = createSystemFromIUPAC(molecule_name_2)
    new_to_old_atom_mapping = align_molecules(molecule1, molecule2)
    from perses.rjmc.topology_proposal import SmallMoleculeTopologyProposal
    topology_proposal = SmallMoleculeTopologyProposal(
        new_topology=top2, new_system=sys2, old_topology=top1, old_system=sys1,
        old_positions=pos1, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_mapping, metadata={'test':0.0})

    for ncmc_nsteps in [0, 1, 50]:
        f = partial(check_alchemical_elimination, topology_proposal, ncmc_nsteps=ncmc_nsteps)
        f.description = "Testing alchemical elimination for '%s' -> '%s' with %d NCMC steps" % (molecule_name_1, molecule_name_2, ncmc_nsteps)
        yield f

def test_alchemical_elimination_peptide():
    # Create an alanine dipeptide null transformation, where N-methyl group is deleted and then inserted.
    from openmmtools import testsystems
    testsystem = testsystems.AlanineDipeptideVacuum()
    from perses.rjmc.topology_proposal import TopologyProposal
    new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    from perses.rjmc.topology_proposal import TopologyProposal
    topology_proposal = TopologyProposal(
        old_system=testsystem.system, old_topology=testsystem.topology, old_positions=testsystem.positions,
        new_system=testsystem.system, new_topology=testsystem.topology,
        logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())

    for ncmc_nsteps in [0, 1, 50]:
        f = partial(check_alchemical_elimination, topology_proposal, ncmc_nsteps=ncmc_nsteps)
        f.description = "Testing alchemical elimination using alanine dipeptide with %d NCMC steps" % ncmc_nsteps
        yield f

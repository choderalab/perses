"""
Unit tests for NCMC switching engine.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from functools import partial
from pkg_resources import resource_filename
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

def generate_gaff_xml():
    """
    Return a file-like object for `gaff.xml`
    """
    from openmoltools import amber
    gaff_dat_filename = amber.find_gaff_dat()

    # Generate ffxml file contents for parmchk-generated frcmod output.
    leaprc = StringIO("parm = loadamberparams %s" % gaff_dat_filename)
    import parmed
    params = parmed.amber.AmberParameterSet.from_leaprc(leaprc)
    params = parmed.openmm.OpenMMParameterSet.from_parameterset(params)
    citations = """\
Wang, J., Wang, W., Kollman P. A.; Case, D. A. "Automatic atom type and bond type perception in molecular mechanical calculations". Journal of Molecular Graphics and Modelling , 25, 2006, 247260.
Wang, J., Wolf, R. M.; Caldwell, J. W.;Kollman, P. A.; Case, D. A. "Development and testing of a general AMBER force field". Journal of Computational Chemistry, 25, 2004, 1157-1174.
"""
    ffxml = str()
    gaff_xml = StringIO(ffxml)
    provenance=dict(OriginalFile='gaff.dat', Reference=citations)
    params.write(gaff_xml, provenance=provenance)

    return gaff_xml

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    In the source distribution, these files are in ``perses/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the openmoltools folder).

    """

    fn = resource_filename('perses', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

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
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    forcefield = ForceField(gaff_xml_filename)

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

def check_alchemical_null_elimination(topology_proposal, ncmc_nsteps=50, NSIGMA_MAX=6.0):
    """
    Test alchemical elimination engine on null transformations, where some atoms are deleted and then reinserted in a cycle.

    Parameters
    ----------
    topology_proposal : TopologyProposal
        The topology proposal to test.
        This must be a null transformation, where topology_proposal.old_system == topology_proposal.new_system
    ncmc_steps : int, optional, default=50
        Number of NCMC switching steps, or 0 for instantaneous switching.
    NSIGMA_MAX : float, optional, default=6.0
        Number of standard errors away from analytical solution tolerated before Exception is thrown
    """
    # Initialize engine
    from perses.annihilation.ncmc_switching import NCMCEngine
    ncmc_engine = NCMCEngine(nsteps=ncmc_nsteps)

    # Make sure that old system and new system are identical.
    if not (topology_proposal.old_system == topology_proposal.new_system):
        raise Exception("topology_proposal must be a null transformation for this test (old_system == new_system)")
    for (k,v) in topology_proposal.new_to_old_atom_map.items():
        if k != v:
            raise Exception("topology_proposal must be a null transformation for this test (retailed atoms must map onto themselves)")

    nequil = 5 # number of equilibration iterations
    niterations = 30 # number of round-trip switching trials
    logP_insert_n = np.zeros([niterations], np.float64)
    logP_delete_n = np.zeros([niterations], np.float64)
    positions = topology_proposal.old_positions
    print("")
    for iteration in range(nequil):
        positions = simulate(topology_proposal.old_system, positions)
    for iteration in range(niterations):
        # Equilibrate
        positions = simulate(topology_proposal.old_system, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Delete atoms
        [positions, logP_delete] = ncmc_engine.integrate(topology_proposal, positions, direction='delete')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC deletion")

        # Insert atoms
        [positions, logP_insert] = ncmc_engine.integrate(topology_proposal, positions, direction='insert')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC insertion")

        # Compute total probability
        logP_delete_n[iteration] = logP_delete
        logP_insert_n[iteration] = logP_insert
        #print("Iteration %5d : delete %16.8f kT | insert %16.8f kT" % (iteration, logP_delete, logP_insert))

    # Check free energy difference is withing NSIGMA_MAX standard errors of zero.
    logP_n = logP_delete_n + logP_insert_n
    work_n = - logP_n
    from pymbar import EXP
    [df, ddf] = EXP(work_n)
    #print("df = %12.6f +- %12.5f kT" % (df, ddf))
    if (abs(df) > NSIGMA_MAX * ddf):
        msg = 'Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0\n' % (ncmc_nsteps, df, ddf, NSIGMA_MAX)
        msg += 'delete logP:\n'
        msg += str(logP_delete_n) + '\n'
        msg += 'insert logP:\n'
        msg += str(logP_insert_n) + '\n'
        msg += 'logP:\n'
        msg += str(logP_n) + '\n'
        raise Exception(msg)

def test_alchemical_elimination_molecule():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, and 50 switching steps.

    """
    molecule_names = ['ethane', 'pentane', 'benzene', 'phenol', 'biphenyl', 'imatinib']
    for molecule_name in molecule_names:
        [molecule, system, positions, topology] = createSystemFromIUPAC(molecule_name)
        natoms = system.getNumParticles()
        # Eliminate half of the molecule
        # TODO: Use a more rigorous scheme to make sure we are really cutting the molecule in half and not just eliminating hydrogens or something.
        new_to_old_atom_map = { index : index for index in range(int(natoms/2)) }

        from perses.rjmc.topology_proposal import SmallMoleculeTopologyProposal
        topology_proposal = SmallMoleculeTopologyProposal(
            new_topology=topology, new_system=system, old_topology=topology, old_system=system,
            old_positions=positions, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata={'test':0.0})
        for ncmc_nsteps in [0, 1, 50]:
            f = partial(check_alchemical_null_elimination, topology_proposal, ncmc_nsteps=ncmc_nsteps)
            f.description = "Testing alchemical null elimination for '%s' with %d NCMC steps" % (molecule_name, ncmc_nsteps)
            yield f

def disable_alchemical_elimination_peptide():
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
        f = partial(check_alchemical_null_elimination, topology_proposal, ncmc_nsteps=ncmc_nsteps)
        f.description = "Testing alchemical elimination using alanine dipeptide with %d NCMC steps" % ncmc_nsteps
        yield f

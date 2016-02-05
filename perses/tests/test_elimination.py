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
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

################################################################################
# TESTS
################################################################################

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

def simulate(system, positions, nsteps=500, timestep=1.0*unit.femtoseconds, temperature=temperature, collision_rate=5.0/unit.picoseconds, platform=None):
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    if platform == None:
        context = openmm.Context(system, integrator)
    else:
        context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(nsteps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)
    return [positions, velocities]

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
    platform = openmm.Platform.getPlatformByName('Reference')
    ncmc_engine = NCMCEngine(temperature=temperature, nsteps=ncmc_nsteps, platform=platform)

    # Make sure that old system and new system are identical.
    if not (topology_proposal.old_system == topology_proposal.new_system):
        raise Exception("topology_proposal must be a null transformation for this test (old_system == new_system)")
    for (k,v) in topology_proposal.new_to_old_atom_map.items():
        if k != v:
            raise Exception("topology_proposal must be a null transformation for this test (retailed atoms must map onto themselves)")

    nequil = 5 # number of equilibration iterations
    niterations = 50 # number of round-trip switching trials
    logP_insert_n = np.zeros([niterations], np.float64)
    logP_delete_n = np.zeros([niterations], np.float64)
    logP_switch_n = np.zeros([niterations], np.float64)
    positions = topology_proposal.old_positions
    print("")
    for iteration in range(nequil):
        [positions, velocities] = simulate(topology_proposal.old_system, positions, platform=platform)
    for iteration in range(niterations):
        # Equilibrate
        [positions, velocities] = simulate(topology_proposal.old_system, positions, platform=platform)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Delete atoms
        [positions, logP_delete, potential_delete] = ncmc_engine.integrate(topology_proposal, positions, direction='delete')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC deletion")

        # Insert atoms
        [positions, logP_insert, potential_insert] = ncmc_engine.integrate(topology_proposal, positions, direction='insert')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC insertion")

        # Compute probability of switching geometries.
        logP_switch = - (potential_insert - potential_delete) / kT

        # Compute total probability
        logP_delete_n[iteration] = logP_delete
        logP_insert_n[iteration] = logP_insert
        logP_switch_n[iteration] = logP_switch
        #print("Iteration %5d : delete %16.8f kT | insert %16.8f kT | geometry switch %16.8f" % (iteration, logP_delete, logP_insert, logP_switch))

    # Check free energy difference is withing NSIGMA_MAX standard errors of zero.
    logP_n = logP_delete_n + logP_insert_n + logP_switch_n
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

def test_ncmc_alchemical_integrator_stability():
    """
    Test NCMCAlchemicalIntegrator

    """
    molecule_names = ['ethane', 'pentane', 'benzene', 'phenol', 'biphenyl', 'imatinib']
    for molecule_name in molecule_names:
        from perses.tests.utils import createSystemFromIUPAC
        [molecule, system, positions, topology] = createSystemFromIUPAC(molecule_name)

        # Eliminate half of the molecule
        # TODO: Use a more rigorous scheme to make sure we are really cutting the molecule in half and not just eliminating hydrogens or something.
        alchemical_atoms = [ index for index in range(int(system.getNumParticles()/2)) ]

        # Create an alchemically-modified system.
        from alchemy import AbsoluteAlchemicalFactory
        alchemical_factory = AbsoluteAlchemicalFactory(system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=True)

        # Return the alchemically-modified system in fully-interacting form.
        alchemical_system = alchemical_factory.createPerturbedSystem()

        # Create an NCMC switching integrator.
        from perses.annihilation.ncmc_switching import NCMCAlchemicalIntegrator
        temperature = 300.0 * unit.kelvin
        functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : 'lambda^2' }
        ncmc_integrator = NCMCAlchemicalIntegrator(temperature, alchemical_system, functions, direction='delete', nsteps=10, timestep=1.0*unit.femtoseconds)

        # Create a Context
        context = openmm.Context(alchemical_system, ncmc_integrator)
        context.setPositions(positions)

        # Run the integrator
        ncmc_integrator.step(1)

        # Check positions are finite
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        if np.isnan(np.any(positions / positions.unit)):
            raise Exception('NCMCAlchemicalIntegrator gave NaN positions')
        if np.isnan(ncmc_integrator.getLogAcceptanceProbability()):
            raise Exception('NCMCAlchemicalIntegrator gave NaN logAcceptanceProbability')

        del context, ncmc_integrator

def test_ncmc_engine_molecule():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, 2, and 50 switching steps.
    """
    molecule_names = ['ethane', 'pentane', 'benzene', 'phenol', 'biphenyl', 'imatinib']
    for molecule_name in molecule_names:
        from perses.tests.utils import createSystemFromIUPAC
        [molecule, system, positions, topology] = createSystemFromIUPAC(molecule_name)
        natoms = system.getNumParticles()
        # Eliminate half of the molecule
        # TODO: Use a more rigorous scheme to make sure we are really cutting the molecule in half and not just eliminating hydrogens or something.
        new_to_old_atom_map = { index : index for index in range(int(natoms/2)) }

        from perses.rjmc.topology_proposal import SmallMoleculeTopologyProposal
        topology_proposal = SmallMoleculeTopologyProposal(
            new_topology=topology, new_system=system, old_topology=topology, old_system=system,
            old_positions=positions, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata={'test':0.0})
        for ncmc_nsteps in [0, 1, 2, 50]:
            f = partial(check_alchemical_null_elimination, topology_proposal, ncmc_nsteps=ncmc_nsteps)
            f.description = "Testing alchemical null elimination for '%s' with %d NCMC steps" % (molecule_name, ncmc_nsteps)
            yield f

def test_alchemical_elimination_peptide():
    """
    Test alchemical elimination for the alanine dipeptide.
    """
    # Create an alanine dipeptide null transformation, where N-methyl group is deleted and then inserted.
    from openmmtools import testsystems
    testsystem = testsystems.AlanineDipeptideVacuum()
    from perses.rjmc.topology_proposal import TopologyProposal
    new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    topology_proposal = TopologyProposal(
        old_system=testsystem.system, old_topology=testsystem.topology, old_positions=testsystem.positions,
        new_system=testsystem.system, new_topology=testsystem.topology,
        logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())

    for ncmc_nsteps in [0, 1, 2, 50]:
        f = partial(check_alchemical_null_elimination, topology_proposal, ncmc_nsteps=ncmc_nsteps)
        f.description = "Testing alchemical elimination using alanine dipeptide with %d NCMC steps" % ncmc_nsteps
        yield f

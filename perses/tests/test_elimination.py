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
from unittest import skipIf
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

def check_alchemical_null_elimination(topology_proposal, positions, ncmc_nsteps=50, NSIGMA_MAX=6.0, geometry=False):
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
    geometry : bool, optional, default=None
        If True, will also use geometry engine in the middle of the null transformation.
    """
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : '1.0', # don't soften bonds
        'lambda_angles' : '1.0', # don't soften angles
        'lambda_torsions' : 'lambda'
    }

    # Initialize engine
    from perses.annihilation.ncmc_switching import NCMCEngine
    ncmc_engine = NCMCEngine(temperature=temperature, functions=functions, nsteps=ncmc_nsteps)

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
    for iteration in range(nequil):
        [positions, velocities] = simulate(topology_proposal.old_system, positions)
    for iteration in range(niterations):
        # Equilibrate
        [positions, velocities] = simulate(topology_proposal.old_system, positions)

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
        #print("Iteration %5d : delete %16.8f kT | insert %16.8f kT | work %16.8f kT" % (iteration, logP_delete, logP_insert, -(logP_delete + logP_insert)))

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

def check_hybrid_null_elimination(topology_proposal, positions, ncmc_nsteps=50, NSIGMA_MAX=6.0, geometry=False):
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
    geometry : bool, optional, default=None
        If True, will also use geometry engine in the middle of the null transformation.
    """
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : 'lambda', # don't soften bonds
        'lambda_angles' : 'lambda', # don't soften angles
        'lambda_torsions' : 'lambda'
    }
    # Initialize engine
    from perses.annihilation.ncmc_switching import NCMCHybridEngine
    ncmc_engine = NCMCHybridEngine(temperature=temperature, functions=functions, nsteps=ncmc_nsteps)

    # Make sure that old system and new system are identical.
    if not (topology_proposal.old_system == topology_proposal.new_system):
        raise Exception("topology_proposal must be a null transformation for this test (old_system == new_system)")
    for (k,v) in topology_proposal.new_to_old_atom_map.items():
        if k != v:
            raise Exception("topology_proposal must be a null transformation for this test (retailed atoms must map onto themselves)")

    nequil = 5 # number of equilibration iterations
    niterations = 50 # number of round-trip switching trials
    logP_n = np.zeros([niterations], np.float64)
    for iteration in range(nequil):
        [positions, velocities] = simulate(topology_proposal.old_system, positions)
    for iteration in range(niterations):
        # Equilibrate
        [positions, velocities] = simulate(topology_proposal.old_system, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Hybrid NCMC from old to new
        [positions, new_old_positions, logP] = ncmc_engine.integrate(topology_proposal, positions, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on Hybrid NCMC switch")

        # Compute total probability
        logP_n[iteration] = logP

    # Check free energy difference is withing NSIGMA_MAX standard errors of zero.
    work_n = - logP_n
    from pymbar import EXP
    [df, ddf] = EXP(work_n)
    #print("df = %12.6f +- %12.5f kT" % (df, ddf))
    if (abs(df) > NSIGMA_MAX * ddf):
        msg = 'Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0\n' % (ncmc_nsteps, df, ddf, NSIGMA_MAX)
        msg += 'logP:\n'
        msg += str(logP_n) + '\n'
        raise Exception(msg)

@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
def test_alchemical_elimination_mutation():
    """
    Test alchemical elimination for mutations.

    """

    ff_filename = "amber99sbildn.xml"
    proposal_metadata = {'ffxmls':[ff_filename]}

    # Create peptide.
    from openmmtools import testsystems
    testsystem = testsystems.AlanineDipeptideVacuum(constraints=None)
    [topology, system, positions] = [testsystem.topology, testsystem.system, testsystem.positions]

    # Create forcefield.
    ff = app.ForceField(ff_filename)
    chain_id = '1'
    allowed_mutations = [[('2','GLY')]]

    from perses.rjmc.topology_proposal import SystemGenerator
    system_generator = SystemGenerator([ff_filename],
        forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None },
        use_antechamber=False, barostat=None)

    # Create a topology proposal fro mutating ALA -> GLY
    from perses.rjmc.topology_proposal import PointMutationEngine
    proposal_engine = PointMutationEngine(topology, system_generator, chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)
    topology_proposal = proposal_engine.propose(system, topology)

    # Modify atom mapping to get a null transformation.
    from perses.rjmc.topology_proposal import TopologyProposal
    new_to_old_atom_map = { atom1 : atom1 for atom1 in topology_proposal.new_to_old_atom_map }
    topology_proposal = TopologyProposal(
                new_topology=topology_proposal.old_topology, new_system=topology_proposal.old_system, old_topology=topology_proposal.old_topology, old_system=topology_proposal.old_system,
                old_chemical_state_key='AA', new_chemical_state_key='AG', logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=topology_proposal.metadata)

    for ncmc_nsteps in [0, 1, 2, 50]:
        f = partial(check_alchemical_null_elimination, topology_proposal, positions, ncmc_nsteps=ncmc_nsteps)
        f.description = "Testing alchemical null transformation of ALA sidechain in alanine dipeptide with %d NCMC steps" % ncmc_nsteps
        yield f

def test_ncmc_alchemical_integrator_stability_molecules():
    """
    Test NCMCAlchemicalIntegrator with various molecules

    """
    molecule_names = ['pentane', 'biphenyl', 'imatinib']
    if os.environ.get("TRAVIS", None) == 'true':
        molecule_names = ['pentane']

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
        from perses.annihilation.ncmc_switching import NCMCVVAlchemicalIntegrator
        temperature = 300.0 * unit.kelvin
        functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : 'lambda^2' }
        ncmc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, direction='delete', nsteps=10, timestep=1.0*unit.femtoseconds)

        # Create a Context
        context = openmm.Context(alchemical_system, ncmc_integrator)
        context.setPositions(positions)

        # Run the integrator
        ncmc_integrator.step(1)

        # Check positions are finite
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        if np.isnan(np.any(positions / positions.unit)):
            raise Exception('NCMCAlchemicalIntegrator gave NaN positions')
        if np.isnan(ncmc_integrator.getLogAcceptanceProbability(context)):
            raise Exception('NCMCAlchemicalIntegrator gave NaN logAcceptanceProbability')

        del context, ncmc_integrator

def test_ncmc_engine_molecule():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, 2, and 50 switching steps.
    """
    molecule_names = ['imatinib', 'pentane', 'biphenyl']
    if os.environ.get("TRAVIS", None) == 'true':
        molecule_names = ['pentane']

    for molecule_name in molecule_names:
        from perses.tests.utils import createSystemFromIUPAC
        [molecule, system, positions, topology] = createSystemFromIUPAC(molecule_name)
        natoms = system.getNumParticles()

        # DEBUG
        print(molecule_name)
        from openeye import oechem
        ofs = oechem.oemolostream('%s.mol2' % molecule_name)
        oechem.OEWriteMol2File(ofs, molecule)
        ofs.close()

        # Eliminate half of the molecule
        # TODO: Use a more rigorous scheme to make sure we are really cutting the molecule in half and not just eliminating hydrogens or something.
        new_to_old_atom_map = { atom.index : atom.index for atom in topology.atoms() if str(atom.element.name) in ['carbon','nitrogen'] }

        # DEBUG
        print(new_to_old_atom_map)

        from perses.rjmc.topology_proposal import TopologyProposal
        topology_proposal = TopologyProposal(
            new_topology=topology, new_system=system, old_topology=topology, old_system=system,
            old_chemical_state_key='', new_chemical_state_key='', logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata={'test':0.0})
        for ncmc_nsteps in [0, 1, 50]:
            f = partial(check_alchemical_null_elimination, topology_proposal, positions, ncmc_nsteps=ncmc_nsteps)
            f.description = "Testing two-stage null elimination for '%s' with %d NCMC steps" % (molecule_name, ncmc_nsteps)
            yield f

#@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
def test_ncmc_hybrid_engine_molecule():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, 2, and 50 switching steps.
    """
    molecule_names = ['imatinib', 'pentane', 'biphenyl']
    if os.environ.get("TRAVIS", None) == 'true':
        molecule_names = ['pentane']

    for molecule_name in molecule_names:
        from perses.tests.utils import createSystemFromIUPAC
        [molecule, system, positions, topology] = createSystemFromIUPAC(molecule_name)
        natoms = system.getNumParticles()
        # Eliminate half of the molecule
        # TODO: Use a more rigorous scheme to make sure we are really cutting the molecule in half and not just eliminating hydrogens or something.
        new_to_old_atom_map = { atom.index : atom.index for atom in topology.atoms() if str(atom.element.name) in ['carbon','nitrogen'] }

        from perses.rjmc.topology_proposal import TopologyProposal
        topology_proposal = TopologyProposal(
            new_topology=topology, new_system=system, old_topology=topology, old_system=system,
            old_chemical_state_key='', new_chemical_state_key='', logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata={'test':0.0})
        for ncmc_nsteps in [0, 1, 50]:
            f = partial(check_hybrid_null_elimination, topology_proposal, positions, ncmc_nsteps=ncmc_nsteps)
            f.description = "Testing hybrid null elimination for '%s' with %d NCMC steps" % (molecule_name, ncmc_nsteps)
            yield f

#@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
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
        old_system=testsystem.system, old_topology=testsystem.topology,
        old_chemical_state_key='AA', new_chemical_state_key='AA',
        new_system=testsystem.system, new_topology=testsystem.topology,
        logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())

    for ncmc_nsteps in [0, 1, 50]:
        f = partial(check_alchemical_null_elimination, topology_proposal, testsystem.positions, ncmc_nsteps=ncmc_nsteps)
        f.description = "Testing alchemical elimination using alanine dipeptide with %d NCMC steps" % ncmc_nsteps
        yield f

if __name__ == "__main__":
    for x in test_ncmc_engine_molecule():
        print(x.description)
        x()
    for x in test_ncmc_hybrid_engine_molecule():
        print(x.description)
        x()
#    test_ncmc_alchemical_integrator_stability_molecules()
#    for x in test_alchemical_elimination_mutation():
#        x()
#    for x in test_alchemical_elimination_peptide():
#        x()

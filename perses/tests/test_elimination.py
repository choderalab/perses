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

def simulate_hybrid(hybrid_system,functions, lambda_value, positions, nsteps=100, timestep=1.0*unit.femtoseconds, temperature=temperature, collision_rate=5.0/unit.picoseconds):
    platform = openmm.Platform.getPlatformByName("Reference")
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(hybrid_system, integrator, platform)
    for parameter in functions.keys():
        context.setParameter(parameter, lambda_value)
    context.setPositions(positions)
    integrator.step(nsteps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return positions

def generate_hybrid_test_topology(mol_name="naphthalene", ref_mol_name="benzene"):
    """
    Generate a test topology proposal and positions for the hybrid test.
    """
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC

    mol = createOEMolFromIUPAC(mol_name)
    m, system, positions, topology = createSystemFromIUPAC(mol_name)

    refmol = createOEMolFromIUPAC(ref_mol_name)

    #map one of the rings
    atom_map = SmallMoleculeSetProposalEngine._get_mol_atom_map(mol, refmol)

    #now use the mapped atoms to generate a new and old system with identical atoms mapped. This will result in the
    #same molecule with the same positions for lambda=0 and 1, and ensures a contiguous atom map
    effective_atom_map = {value : value for value in atom_map.values()}

    #make a topology proposal with the appropriate data:
    top_proposal = TopologyProposal(new_topology=topology, new_system=system, old_topology=topology, old_system=system, new_to_old_atom_map=effective_atom_map, new_chemical_state_key="n1", old_chemical_state_key='n2')

    return top_proposal, positions

def generate_solvated_hybrid_test_topology(mol_name="naphthalene", ref_mol_name="benzene"):

    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    mol = createOEMolFromIUPAC(mol_name)
    m, unsolv_system, pos, top = createSystemFromIUPAC(mol_name)

    refmol = createOEMolFromIUPAC(ref_mol_name)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)
    #map one of the rings
    atom_map = SmallMoleculeSetProposalEngine._get_mol_atom_map(mol, refmol)

    #now use the mapped atoms to generate a new and old system with identical atoms mapped. This will result in the
    #same molecule with the same positions for lambda=0 and 1, and ensures a contiguous atom map
    effective_atom_map = {value : value for value in atom_map.values()}

    modeller = app.Modeller(top, pos)
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    topology = modeller.getTopology()
    positions = modeller.getPositions()
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME)

    n_atoms_old_system = unsolv_system.getNumParticles()
    n_atoms_after_solvation = system.getNumParticles()

    for i in range(n_atoms_old_system, n_atoms_after_solvation):
        effective_atom_map[i] = i

    top_proposal = TopologyProposal(new_topology=topology, new_system=system, old_topology=topology, old_system=system, new_to_old_atom_map=effective_atom_map, new_chemical_state_key="n1", old_chemical_state_key='n2')

    return top_proposal, positions


def generate_solvated_hybrid_topology(mol_name="naphthalene", ref_mol_name="benzene"):

    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    mol = createOEMolFromIUPAC(mol_name)
    m, unsolv_system, pos, top = createSystemFromIUPAC(mol_name)

    refmol = createOEMolFromIUPAC(ref_mol_name)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)
    #map one of the rings
    atom_map = SmallMoleculeSetProposalEngine._get_mol_atom_map(mol, refmol)

    effective_atom_map = atom_map

    modeller = app.Modeller(top, pos)
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    topology = modeller.getTopology()
    positions = modeller.getPositions()
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME)

    n_atoms_old_system = unsolv_system.getNumParticles()
    n_atoms_after_solvation = system.getNumParticles()

    for i in range(n_atoms_old_system, n_atoms_after_solvation):
        effective_atom_map[i] = i

    top_proposal = TopologyProposal(new_topology=topology, new_system=system, old_topology=topology, old_system=system, new_to_old_atom_map=effective_atom_map, new_chemical_state_key="n1", old_chemical_state_key='n2')

    return top_proposal, positions

def check_alchemical_hybrid_elimination_bar(topology_proposal, positions, ncmc_nsteps=50, NSIGMA_MAX=6.0, geometry=False):
    """
    Check that the hybrid topology, where both endpoints are identical, returns a free energy within NSIGMA_MAX of 0.
    Parameters
    ----------
    topology_proposal
    positions
    ncmc_nsteps
    NSIGMA_MAX

    Returns
    -------

    """
    from perses.annihilation import NCMCGHMCAlchemicalIntegrator
    from perses.annihilation.new_relative import HybridTopologyFactory

    #make the hybrid topology factory:
    factory = HybridTopologyFactory(topology_proposal, positions, positions)

    platform = openmm.Platform.getPlatformByName("Reference")

    hybrid_system = factory.hybrid_system
    hybrid_topology = factory.hybrid_topology
    initial_hybrid_positions = factory.hybrid_positions

    n_iterations = 50 #number of times to do NCMC protocol

    #alchemical functions
    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }

    w_f = np.zeros(n_iterations)
    w_r = np.zeros(n_iterations)

    #make the alchemical integrators:
    #forward_integrator = NCMCGHMCAlchemicalIntegrator(temperature, hybrid_system, functions, nsteps=ncmc_nsteps, direction='insert')
    #reverse_integrator = NCMCGHMCAlchemicalIntegrator(temperature, hybrid_system, functions, nsteps=ncmc_nsteps, direction='delete')

    #first, do forward protocol (lambda=0 -> 1)
    for i in range(n_iterations):
        forward_integrator = NCMCGHMCAlchemicalIntegrator(temperature, hybrid_system, functions, nsteps=ncmc_nsteps, direction='insert')
        equil_positions = simulate_hybrid(hybrid_system,functions, 0.0, initial_hybrid_positions)
        context = openmm.Context(hybrid_system, forward_integrator, platform)
        context.setPositions(equil_positions)
        forward_integrator.step(ncmc_nsteps)
        w_f[i] = -1.0 * forward_integrator.getLogAcceptanceProbability(context)
        print(i)
        del context, forward_integrator

    #now, reverse protocol
    for i in range(n_iterations):
        reverse_integrator = NCMCGHMCAlchemicalIntegrator(temperature, hybrid_system, functions, nsteps=ncmc_nsteps, direction='delete')
        equil_positions = simulate_hybrid(hybrid_system,functions, 1.0, initial_hybrid_positions)
        context = openmm.Context(hybrid_system, reverse_integrator, platform)
        context.setPositions(equil_positions)
        reverse_integrator.step(ncmc_nsteps)
        w_r[i] = -1.0 * reverse_integrator.getLogAcceptanceProbability(context)
        print(i)
        del context, reverse_integrator

    from pymbar import BAR
    [df, ddf] = BAR(w_f, w_r)
    print("df = %12.6f +- %12.5f kT" % (df, ddf))
    if (abs(df) > NSIGMA_MAX * ddf):
        msg = 'Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0\n' % (ncmc_nsteps, df, ddf, NSIGMA_MAX)
        msg += 'logP_work_n:\n'
        msg += str(w_f) + '\n'
        msg += str(w_r) + '\n'
        raise Exception(msg)

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
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
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
    logP_work_n = np.zeros([niterations], np.float64)
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
        [positions, logP_delete_work, logP_delete_energy] = ncmc_engine.integrate(topology_proposal, positions, direction='delete')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC deletion")

        # Insert atoms
        [positions, logP_insert_work, logP_insert_energy] = ncmc_engine.integrate(topology_proposal, positions, direction='insert')

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on NCMC insertion")

        # Store log probability associated with work
        logP_work_n[iteration] = logP_delete_work + logP_insert_work
        logP_insert_n[iteration] = logP_insert_work
        logP_delete_n[iteration] = logP_delete_work
        #print("Iteration %5d : work %16.8f kT (delete %16.8f kT | insert %16.8f kT) | energy %16.8f kT (delete %16.8f kT | insert %16.8f kT)"
        #    % (iteration, logP_delete_work + logP_insert_work, logP_delete_work, logP_insert_work, logP_delete_energy + logP_insert_energy, logP_delete_energy, logP_insert_energy))

    # Check free energy difference is withing NSIGMA_MAX standard errors of zero.
    work_n = - logP_work_n
    from pymbar import EXP
    [df, ddf] = EXP(work_n)
    #print("df = %12.6f +- %12.5f kT" % (df, ddf))
    if (abs(df) > NSIGMA_MAX * ddf):
        msg = 'Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0\n' % (ncmc_nsteps, df, ddf, NSIGMA_MAX)
        msg += 'logP_delete_n:\n'
        msg += str(logP_delete_n) + '\n'
        msg += 'logP_insert_n:\n'
        msg += str(logP_insert_n) + '\n'
        msg += 'logP_work_n:\n'
        msg += str(logP_work_n) + '\n'
        raise Exception(msg)

def check_hybrid_round_trip_elimination(topology_proposal, positions, ncmc_nsteps=50, NSIGMA_MAX=6.0):
    """
    Test the hybrid system by switching between lambda = 1 and lambda = 0, then using BAR to compute the free energy
    difference. As the test is designed so that both endpoints are the same, the free energy difference should be zero.

    Parameters
    ----------
    topology_proposal : TopologyProposal
        The topology proposal to test.
        This must be a null transformation, where topology_proposal.old_system == topology_proposal.new_system
    ncmc_steps : int, optional, default=50
        Number of NCMC switching steps, or 0 for instantaneous switching.
    NSIGMA_MAX : float, optional, default=6.0
    """
    functions = {
        'lambda_sterics' : 'lambda',
        'lambda_electrostatics' : 'lambda',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }
    # Initialize engine
    from perses.annihilation import NCMCGHMCAlchemicalIntegrator
    from perses.annihilation.new_relative import HybridTopologyFactory

    #The current and "proposed" positions are the same, since the molecule is not changed.
    factory = HybridTopologyFactory(topology_proposal, positions, positions)

    forward_integrator = NCMCGHMCAlchemicalIntegrator(temperature, factory.hybrid_system, functions, nsteps=ncmc_nsteps, direction='insert')
    reverse_integrator = NCMCGHMCAlchemicalIntegrator(temperature, factory.hybrid_system, functions, nsteps=ncmc_nsteps, direction='delete')

    platform = openmm.Platform.getPlatformByName("Reference")

    forward_context = openmm.Context(factory.hybrid_system, forward_integrator, platform)
    reverse_context = openmm.Context(factory.hybrid_system, reverse_integrator, platform)

    # Make sure that old system and new system are identical.
    if not (topology_proposal.old_system == topology_proposal.new_system):
        raise Exception("topology_proposal must be a null transformation for this test (old_system == new_system)")
    for (k,v) in topology_proposal.new_to_old_atom_map.items():
        if k != v:
            raise Exception("topology_proposal must be a null transformation for this test (retailed atoms must map onto themselves)")

    nequil = 5 # number of equilibration iterations
    niterations = 50 # number of round-trip switching trials
    logP_work_n_f = np.zeros([niterations], np.float64)
    for iteration in range(nequil):
        positions = simulate_hybrid(factory.hybrid_system,functions, 0.0, factory.hybrid_positions)

    #do forward switching:
    for iteration in range(niterations):
        # Equilibrate
        positions = simulate_hybrid(factory.hybrid_system,functions, 0.0, factory.hybrid_positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Hybrid NCMC
        forward_integrator.reset()
        forward_context.setPositions(positions)
        forward_integrator.step(ncmc_nsteps)
        logP_work = forward_integrator.getTotalWork(forward_context)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on Hybrid NCMC switch")

        # Store log probability associated with work
        logP_work_n_f[iteration] = logP_work

    logP_work_n_r = np.zeros([niterations], np.float64)

    for iteration in range(nequil):
        positions = simulate_hybrid(factory.hybrid_system,functions, 1.0, factory.hybrid_positions)

    #do forward switching:
    for iteration in range(niterations):
        # Equilibrate
        positions = simulate_hybrid(factory.hybrid_system,functions, 1.0, factory.hybrid_positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Hybrid NCMC
        reverse_integrator.reset()
        reverse_context.setPositions(positions)
        reverse_integrator.step(ncmc_nsteps)
        logP_work = reverse_integrator.getTotalWork(forward_context)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on Hybrid NCMC switch")

        # Store log probability associated with work
        logP_work_n_r[iteration] = logP_work

    work_f = - logP_work_n_f
    work_r = - logP_work_n_r
    from pymbar import BAR
    [df, ddf] = BAR(work_f, work_r)
    print("df = %12.6f +- %12.5f kT" % (df, ddf))
    if (abs(df) > NSIGMA_MAX * ddf):
        msg = 'Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0\n' % (ncmc_nsteps, df, ddf, NSIGMA_MAX)
        msg += 'logP_work_n:\n'
        msg += str(work_f) + '\n'
        msg += str(work_r) + '\n'
        raise Exception(msg)

def check_hybrid_null_elimination(topology_proposal, positions, new_positions, ncmc_nsteps=50, NSIGMA_MAX=6.0, geometry=False):
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
        'lambda_sterics' : 'lambda',
        'lambda_electrostatics' : 'lambda',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }
    # Initialize engine
    from perses.annihilation.ncmc_switching import NCMCHybridEngine
    ncmc_engine = NCMCHybridEngine(temperature=temperature, functions=functions, nsteps=ncmc_nsteps)

    # Make sure that old system and new system are identical.
   # if not (topology_proposal.old_system == topology_proposal.new_system):
   #     raise Exception("topology_proposal must be a null transformation for this test (old_system == new_system)")
   # for (k,v) in topology_proposal.new_to_old_atom_map.items():
   #     if k != v:
   #         raise Exception("topology_proposal must be a null transformation for this test (retailed atoms must map onto themselves)")

    nequil = 5 # number of equilibration iterations
    niterations = 50 # number of round-trip switching trials
    logP_work_n = np.zeros([niterations], np.float64)
    for iteration in range(nequil):
        [positions, velocities] = simulate(topology_proposal.old_system, positions)
    for iteration in range(niterations):
        # Equilibrate
        [positions, velocities] = simulate(topology_proposal.old_system, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN during equilibration")

        # Hybrid NCMC from old to new
        [_, new_old_positions, logP_work, logP_energy] = ncmc_engine.integrate(topology_proposal, positions, positions)

        # Check that positions are not NaN
        if(np.any(np.isnan(positions / unit.angstroms))):
            raise Exception("Positions became NaN on Hybrid NCMC switch")

        # Store log probability associated with work
        logP_work_n[iteration] = logP_work
        #print("Iteration %5d : NCMC work %16.8f kT | NCMC energy %16.8f kT" % (iteration, logP_work, logP_energy))

    # Check free energy difference is withing NSIGMA_MAX standard errors of zero.
    work_n = - logP_work_n
    from pymbar import EXP
    [df, ddf] = EXP(work_n)
    print("df = %12.6f +- %12.5f kT" % (df, ddf))
    if (abs(df) > NSIGMA_MAX * ddf):
        msg = 'Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0\n' % (ncmc_nsteps, df, ddf, NSIGMA_MAX)
        msg += 'logP_work_n:\n'
        msg += str(logP_work_n) + '\n'
        raise Exception(msg)

# TODO: Re-enable this test once PointMutationEngine can return size of chemical space
@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
def test_alchemical_elimination_mutation():
    """
    Test alchemical elimination for mutations.
    """

    ff_filename = "amber99sbildn.xml"
    proposal_metadata = {'ffxmls':[ff_filename]}

    # Create peptide.
    from openmmtools import testsystems
    testsystem = testsystems.AlanineDipeptideVacuum()
    [topology, system, positions] = [testsystem.topology, testsystem.system, testsystem.positions]

    # Create forcefield.
    ff = app.ForceField(ff_filename)
    chain_id = '1'
    allowed_mutations = [[('2','GLY')]]

    from perses.rjmc.topology_proposal import SystemGenerator
    system_generator = SystemGenerator([ff_filename])

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
    Test NCMCAlchemicalIntegrator

    """
    molecule_names = ['pentane', 'biphenyl', 'imatinib']
    #if os.environ.get("TRAVIS", None) == 'true':
    #    molecule_names = ['pentane']

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
        nsteps = 10 # number of steps to run integration for
        functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : 'lambda^2' }
        ncmc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, direction='delete', nsteps=nsteps, timestep=1.0*unit.femtoseconds)

        # Create a Context
        context = openmm.Context(alchemical_system, ncmc_integrator)
        context.setPositions(positions)

        # Run the integrator
        ncmc_integrator.step(nsteps)

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
    molecule_names = ['pentane', 'biphenyl', 'imatinib']
    #if os.environ.get("TRAVIS", None) == 'true':
    #    molecule_names = ['pentane']

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
            f.description = "Testing alchemical null elimination for '%s' with %d NCMC steps" % (molecule_name, ncmc_nsteps)
            yield f

@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
def test_ncmc_hybrid_engine_molecule():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, 2, and 50 switching steps.
    """
    mols_and_refs = [['naphthalene', 'benzene'], ['pentane', 'propane'], ['biphenyl', 'benzene'], ['octane','propane']]
    #mols_and_refs=[['pentane', 'propane']]
    if os.environ.get("TRAVIS", None) == 'true':
        mols_and_refs = [['naphthalene', 'benzene']]

    for mol_ref in mols_and_refs:
        from perses.tests.utils import createSystemFromIUPAC
        [molecule, system, positions, topology] = createSystemFromIUPAC(mol_ref[0])

        topology_proposal, new_positions = generate_hybrid_test_topology(mol_name=mol_ref[0], ref_mol_name=mol_ref[1])
        #topology_proposal, positions = generate_solvated_hybrid_test_topology(mol_name=mol_ref[0], ref_mol_name=mol_ref[1])
        for ncmc_nsteps in [0, 1, 50]:
            f = partial(check_alchemical_hybrid_elimination_bar, topology_proposal, positions, ncmc_nsteps=ncmc_nsteps)
            f.description = "Testing alchemical null elimination for '%s' with %d NCMC steps" % (mol_ref[0], ncmc_nsteps)
            yield f

@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
def test_ncmc_hybrid_explicit_engine_molecule():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, 2, and 50 switching steps.
    """
    #mols_and_refs = [['naphthalene', 'benzene'], ['pentane', 'propane'], ['biphenyl', 'benzene'], ['octane','propane']]
    mols_and_refs=[['pentane', 'propane']]
    if os.environ.get("TRAVIS", None) == 'true':
        mols_and_refs = [['naphthalene', 'benzene']]

    for mol_ref in mols_and_refs:
        from perses.tests.utils import createSystemFromIUPAC
        [molecule, system, positions, topology] = createSystemFromIUPAC(mol_ref[0])

        #topology_proposal, new_positions = generate_hybrid_test_topology(mol_name=mol_ref[0], ref_mol_name=mol_ref[1])
        topology_proposal, positions = generate_solvated_hybrid_test_topology(mol_name=mol_ref[0], ref_mol_name=mol_ref[1])
        for ncmc_nsteps in [0, 1, 50]:
            f = partial(check_alchemical_hybrid_elimination_bar, topology_proposal, positions, ncmc_nsteps=ncmc_nsteps)
            f.description = "Testing alchemical null elimination for '%s' with %d NCMC steps" % (mol_ref[0], ncmc_nsteps)
            yield f

@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
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
    #for x in test_ncmc_engine_molecule():
    #    print(x.description)
    #    x()
    #generate_solvated_hybrid_test_topology()
    for x in test_ncmc_hybrid_explicit_engine_molecule():
        print(x.description)
        x()
#    test_ncmc_alchemical_integrator_stability_molecules()
#    for x in test_alchemical_elimination_mutation():
#        x()
#    for x in test_alchemical_elimination_peptide():
#        x()

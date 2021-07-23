###########################################
# IMPORTS
###########################################
from simtk.openmm import app
from simtk import unit, openmm
import numpy as np
import os
import random
from nose.tools import nottest
from pkg_resources import resource_filename


from perses.annihilation.relative import HybridTopologyFactory
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.tests import utils
from openmmtools.states import SamplerState
import openmmtools.mcmc as mcmc
import openmmtools.cache as cache
from unittest import skipIf

import pymbar.timeseries as timeseries

import pymbar

running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

try:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")
except Exception:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")

#############################################
# CONSTANTS
#############################################
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01
ENERGY_THRESHOLD = 1e-1
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("CPU")
aminos = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

def run_hybrid_endpoint_overlap(topology_proposal, current_positions, new_positions):
    """
    Test that the variance of the perturbation from lambda={0,1} to the corresponding nonalchemical endpoint is not
    too large.

    Parameters
    ----------
    topology_proposal : perses.rjmc.TopologyProposal
         TopologyProposal object describing the transformation
    current_positions : np.array, unit-bearing
         Positions of the initial system
    new_positions : np.array, unit-bearing
         Positions of the new system

    Returns
    -------
    hybrid_endpoint_results : list
       list of [df, ddf, N_eff] for 1 and 0
    """
    # Create the hybrid system:
    #hybrid_factory = HybridTopologyFactory(topology_proposal, current_positions, new_positions, use_dispersion_correction=True)
    hybrid_factory = HybridTopologyFactory(topology_proposal, current_positions, new_positions, use_dispersion_correction=False) # DEBUG

    # Get the relevant thermodynamic states:
    nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state = utils.generate_endpoint_thermodynamic_states(
        hybrid_factory.hybrid_system, topology_proposal)

    nonalchemical_thermodynamic_states = [nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state]

    alchemical_thermodynamic_states = [lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state]

    # Create an MCMCMove, BAOAB with default parameters (but don't restart if we encounter a NaN)
    mc_move = mcmc.LangevinDynamicsMove(n_restart_attempts=0, n_steps=100)

    initial_sampler_state = SamplerState(hybrid_factory.hybrid_positions, box_vectors=hybrid_factory.hybrid_system.getDefaultPeriodicBoxVectors())

    hybrid_endpoint_results = []
    all_results = []
    for lambda_state in (0, 1):
        result, non, hybrid = run_endpoint_perturbation(alchemical_thermodynamic_states[lambda_state],
                                        nonalchemical_thermodynamic_states[lambda_state], initial_sampler_state,
                                        mc_move, 100, hybrid_factory, lambda_index=lambda_state)
        all_results.append(non)
        all_results.append(hybrid)
        print('lambda {} : {}'.format(lambda_state,result))

        hybrid_endpoint_results.append(result)
    calculate_cross_variance(all_results)
    return hybrid_endpoint_results

def calculate_cross_variance(all_results):
    """
    Calculates the overlap (df and ddf) between the non-alchemical state at lambda=0 to the hybrid state at lambda=1 and visa versa
    These ensembles are not expected to have good overlap, as they are of explicitly different system, but provides a benchmark of appropriate dissimilarity
    """
    if len(all_results) != 4:
        return
    else:
        non_a = all_results[0]
        hybrid_a = all_results[1]
        non_b = all_results[2]
        hybrid_b = all_results[3]
    print('CROSS VALIDATION')
    [df, ddf] = pymbar.EXP(non_a - hybrid_b)
    print('df: {}, ddf: {}'.format(df, ddf))
    [df, ddf] = pymbar.EXP(non_b - hybrid_a)
    print('df: {}, ddf: {}'.format(df, ddf))
    return

def check_result(results, threshold=3.0, neffmin=10):
    """
    Ensure results are within threshold standard deviations and Neff_max > neffmin

    Parameters
    ----------
    results : list
        list of [df, ddf, Neff_max]
    threshold : float, default 3.0
        the standard deviation threshold
    neff_min : float, default 10
        the minimum number of acceptable samples
    """
    [df, ddf, t0, N_eff] = results

    if N_eff < neffmin:
        raise Exception("Number of effective samples %f was below minimum of %f" % (N_eff, neffmin))

    if ddf > threshold:
        raise Exception("Standard deviation of %f exceeds threshold of %f" % (ddf, threshold))

def test_networkx_proposal_order():
    """
    This test fails with a 'no topical torsions found' error with the old ProposalOrderTools
    """
    pairs=[('pentane','propane')]
    for pair in pairs:
        print('{} -> {}'.format(pair[0],pair[1]))
        test_simple_overlap(pair[0],pair[1])
        print('{} -> {}'.format(pair[1],pair[0]))
        test_simple_overlap(pair[1],pair[0])

def test_explosion():
    """
    This test fails with ridiculous DeltaF if the alchemical factory is misbehaving
    """
    pairs = [['2-phenyl ethanol', 'benzene']]
    for pair in pairs:
        print('{} -> {}'.format(pair[0],pair[1]))
        test_simple_overlap(pair[0],pair[1])
        print('{} -> {}'.format(pair[1],pair[0]))
        test_simple_overlap(pair[1],pair[0])

def test_vacuum_overlap_with_constraints():
    """
    Test that constraints do not cause problems for the hybrid factory in vacuum
    """
    test_simple_overlap('2-phenyl ethanol', 'benzene', forcefield_kwargs={'constraints' : app.HBonds})

def test_valence_overlap():
    """
    Test hybrid factory vacuum overlap with valence terms only
    """
    system_generator_kwargs = {
        'particle_charge' : False, 'exception_charge' : False, 'particle_epsilon' : False, 'exception_epsilon' : False, 'torsions' : True,
        }
    test_simple_overlap('2-phenyl ethanol', 'benzene', system_generator_kwargs=system_generator_kwargs)

def test_bonds_angles_overlap():
    """
    Test hybrid factory vacuum overlap with bonds and angles
    """
    system_generator_kwargs = {
        'particle_charge' : False, 'exception_charge' : False, 'particle_epsilon' : False, 'exception_epsilon' : False, 'torsions' : False,
        }
    test_simple_overlap('2-phenyl ethanol', 'benzene', system_generator_kwargs=system_generator_kwargs)

def test_sterics_overlap():
    """
    Test hybrid factory vacuum overlap with valence terms and sterics only
    """
    system_generator_kwargs = {
        'particle_charge' : False, 'exception_charge' : False, 'particle_epsilon' : True, 'exception_epsilon' : True, 'torsions' : True,
        }
    test_simple_overlap('2-phenyl ethanol', 'benzene', system_generator_kwargs=system_generator_kwargs)

def test_simple_overlap_pairs(pairs=None):
    """
    Test to run pairs of small molecule perturbations in vacuum, using test_simple_overlap, both forward and backward.

    Parameters
    ----------
    pairs : list of lists of str, optional, default=None
        Pairs of IUPAC names to test.
        If None, will test a default set:
        [['pentane','butane'],['fluorobenzene', 'chlorobenzene'],['benzene', 'catechol'],['benzene','2-phenyl ethanol'],['imatinib','nilotinib']]

        pentane <-> butane is adding a methyl group
        fluorobenzene <-> chlorobenzene perturbs one halogen to another, with no adding or removing of atoms
        benzene <-> catechol perturbing molecule in two positions simultaneously
        benzene <-> 2-phenyl ethanol addition of 3 heavy atom group
    """
    if pairs is None:
        pairs = [['pentane','butane'],['fluorobenzene', 'chlorobenzene'],['benzene', 'catechol'],['benzene','2-phenyl ethanol']] #'imatinib' --> 'nilotinib' atom mapping is bad

    for pair in pairs:
        print('{} -> {}'.format(pair[0],pair[1]))
        test_simple_overlap(pair[0],pair[1])
        # Now running the reverse
        print('{} -> {}'.format(pair[1],pair[0]))
        test_simple_overlap(pair[1],pair[0])

@nottest # This is, in fact, a helper function that is called in other working tests
@skipIf(running_on_github_actions, "Skip helper function on GH Actions")
def test_simple_overlap(name1='pentane', name2='butane', forcefield_kwargs=None, system_generator_kwargs=None):
    """Test that the variance of the hybrid -> real perturbation in vacuum is sufficiently small.

    Parameters
    ----------
    name1 : str
        IUPAC name of initial molecule
    name2 : str
        IUPAC name of final molecule
    forcefield_kwargs : dict, optional, default=None
        If None, these parameters are fed to the SystemGenerator
        Setting { 'constraints' : app.HBonds } will enable constraints to hydrogen
    system_generator_kwargs : dict, optional, default=None
        If None, these parameters are fed to the SystemGenerator
        Setting { 'particle_charge' : False } will turn off particle charges in parameterized systems
        Can also disable 'exception_charge', 'particle_epsilon', 'exception_epsilon', and 'torsions' by setting to False

    """
    topology_proposal, current_positions, new_positions = utils.generate_solvated_hybrid_test_topology(current_mol_name=name1, proposed_mol_name=name2, vacuum = True)
    results = run_hybrid_endpoint_overlap(topology_proposal, current_positions, new_positions)
    for idx, lambda_result in enumerate(results):
        try:
            check_result(lambda_result)
        except Exception as e:
            message = "pentane->butane failed at lambda %d \n" % idx
            message += str(e)
            raise Exception(message)

@skipIf(running_on_github_actions, "Skip expensive test on GH Actions")
def test_hostguest_overlap():
    """Test that the variance of the endpoint->nonalchemical perturbation is sufficiently small for host-guest system in vacuum"""
    topology_proposal, current_positions, new_positions = utils.generate_vacuum_hostguest_proposal()
    results = run_hybrid_endpoint_overlap(topology_proposal, current_positions, new_positions)

    for idx, lambda_result in enumerate(results):
        try:
            check_result(lambda_result)
        except Exception as e:
            message = "pentane->butane failed at lambda %d \n" % idx
            message += str(e)
            raise Exception(message)

@skipIf(running_on_github_actions, "Skip broken test on GH Actions")
@nottest # At the moment, the mapping between imatinib and nilotinib is faulty
def test_difficult_overlap():
    """Test that the variance of the endpoint->nonalchemical perturbation is sufficiently small for imatinib->nilotinib in solvent"""
    name1 = 'imatinib'
    name2 = 'nilotinib'

    print(name1, name2)
    topology_proposal, solvated_positions, new_positions = utils.generate_solvated_hybrid_test_topology(current_mol_name=name1, proposed_mol_name=name2)
    results = run_hybrid_endpoint_overlap(topology_proposal, solvated_positions, new_positions)

    for idx, lambda_result in enumerate(results):
        try:
            check_result(lambda_result)
        except Exception as e:
            message = "solvated imatinib->nilotinib failed at lambda %d \n" % idx
            message += str(e)
            raise Exception(message)

    print(name2, name1)
    topology_proposal, solvated_positions, new_positions = utils.generate_solvated_hybrid_test_topology(current_mol_name=name2, proposed_mol_name=name1)
    results = run_hybrid_endpoint_overlap(topology_proposal, solvated_positions, new_positions)

    for idx, lambda_result in enumerate(results):
        try:
            check_result(lambda_result)
        except Exception as e:
            message = "solvated imatinib->nilotinib failed at lambda %d \n" % idx
            message += str(e)
            raise Exception(message)

def run_endpoint_perturbation(lambda_thermodynamic_state, nonalchemical_thermodynamic_state, initial_hybrid_sampler_state, mc_move, n_iterations, factory,
    lambda_index=0, print_work=False, write_system=False, write_state=False, write_trajectories=False):
    """

    Parameters
    ----------
    lambda_thermodynamic_state : ThermodynamicState
        The thermodynamic state corresponding to the hybrid system at a lambda endpoint
    nonalchemical_thermodynamic_state : ThermodynamicState
        The nonalchemical thermodynamic state for the relevant endpoint
    initial_hybrid_sampler_state : SamplerState
        Starting positions for the sampler. Must be compatible with lambda_thermodynamic_state
    mc_move : MCMCMove
        The MCMove that will be used for sampling at the lambda endpoint
    n_iterations : int
        The number of iterations
    factory : HybridTopologyFactory
        The hybrid topology factory
    lambda_index : int, optional, default=0
        The index, 0 or 1, at which to retrieve nonalchemical positions
    print_work : bool, optional, default=False
        If True, will print work values
    write_system : bool, optional, default=False
        If True, will write alchemical and nonalchemical System XML files
    write_state : bool, optional, default=False
        If True, write alchemical (hybrid) State XML files each iteration
    write_trajectories : bool, optional, default=False
        If True, will write trajectories

    Returns
    -------
    df : float
        Free energy difference between alchemical and nonalchemical systems, estimated with EXP
    ddf : float
        Standard deviation of estimate, corrected for correlation, from EXP estimator.
    """
    import mdtraj as md

    # Run an initial minimization:
    mcmc_sampler = mcmc.MCMCSampler(lambda_thermodynamic_state, initial_hybrid_sampler_state, mc_move)
    mcmc_sampler.minimize(max_iterations=20)
    new_sampler_state = mcmc_sampler.sampler_state

    if write_system:
        with open(f'hybrid{lambda_index}-system.xml', 'w') as outfile:
            outfile.write(openmm.XmlSerializer.serialize(lambda_thermodynamic_state.system))
        with open(f'nonalchemical{lambda_index}-system.xml', 'w') as outfile:
            outfile.write(openmm.XmlSerializer.serialize(nonalchemical_thermodynamic_state.system))

    # Initialize work array
    w = np.zeros([n_iterations])
    non_potential = np.zeros([n_iterations])
    hybrid_potential = np.zeros([n_iterations])

    # Run n_iterations of the endpoint perturbation:
    hybrid_trajectory = unit.Quantity(np.zeros([n_iterations, lambda_thermodynamic_state.system.getNumParticles(), 3]), unit.nanometers) # DEBUG
    nonalchemical_trajectory = unit.Quantity(np.zeros([n_iterations, nonalchemical_thermodynamic_state.system.getNumParticles(), 3]), unit.nanometers) # DEBUG
    for iteration in range(n_iterations):
        # Generate a new sampler state for the hybrid system
        mc_move.apply(lambda_thermodynamic_state, new_sampler_state)

        # Compute the hybrid reduced potential at the new sampler state
        hybrid_context, integrator = cache.global_context_cache.get_context(lambda_thermodynamic_state)
        new_sampler_state.apply_to_context(hybrid_context, ignore_velocities=True)
        hybrid_reduced_potential = lambda_thermodynamic_state.reduced_potential(hybrid_context)

        if write_state:
            state = hybrid_context.getState(getPositions=True, getParameters=True)
            state_xml = openmm.XmlSerializer.serialize(state)
            with open(f'state{iteration}_l{lambda_index}.xml', 'w') as outfile:
                outfile.write(state_xml)

        # Construct a sampler state for the nonalchemical system
        if lambda_index == 0:
            nonalchemical_positions = factory.old_positions(new_sampler_state.positions)
        elif lambda_index == 1:
            nonalchemical_positions = factory.new_positions(new_sampler_state.positions)
        else:
            raise ValueError("The lambda index needs to be either one or zero for this to be meaningful")
        nonalchemical_sampler_state = SamplerState(nonalchemical_positions, box_vectors=new_sampler_state.box_vectors)

        if write_trajectories:
            state = hybrid_context.getState(getPositions=True)
            hybrid_trajectory[iteration,:,:] = state.getPositions(asNumpy=True)
            nonalchemical_trajectory[iteration,:,:] = nonalchemical_positions

        # Compute the nonalchemical reduced potential
        nonalchemical_context, integrator = cache.global_context_cache.get_context(nonalchemical_thermodynamic_state)
        nonalchemical_sampler_state.apply_to_context(nonalchemical_context, ignore_velocities=True)
        nonalchemical_reduced_potential = nonalchemical_thermodynamic_state.reduced_potential(nonalchemical_context)

        # Compute and store the work
        w[iteration] = nonalchemical_reduced_potential - hybrid_reduced_potential
        non_potential[iteration] = nonalchemical_reduced_potential
        hybrid_potential[iteration] = hybrid_reduced_potential

        if print_work:
            print(f'{iteration:8d} {hybrid_reduced_potential:8.3f} {nonalchemical_reduced_potential:8.3f} => {w[iteration]:8.3f}')

    if write_trajectories:
        if lambda_index == 0:
            nonalchemical_mdtraj_topology = md.Topology.from_openmm(factory._topology_proposal.old_topology)
        elif lambda_index == 1:
            nonalchemical_mdtraj_topology = md.Topology.from_openmm(factory._topology_proposal.new_topology)
        md.Trajectory(hybrid_trajectory / unit.nanometers, factory.hybrid_topology).save(f'hybrid{lambda_index}.pdb')
        md.Trajectory(nonalchemical_trajectory / unit.nanometers, nonalchemical_mdtraj_topology).save(f'nonalchemical{lambda_index}.pdb')

    # Analyze data and return results
    [t0, g, Neff_max] = timeseries.detectEquilibration(w)
    w_burned_in = w[t0:]
    [df, ddf] = pymbar.EXP(w_burned_in)
    ddf_corrected = ddf * np.sqrt(g)
    results = [df, ddf_corrected, t0, Neff_max]

    return results, non_potential, hybrid_potential

def compare_energies(mol_name="naphthalene", ref_mol_name="benzene",atom_expression=['Hybridization'],bond_expression=['Hybridization']):
    """
    Make an atom map where the molecule at either lambda endpoint is identical, and check that the energies are also the same.
    """
    from openmoltools.openeye import generate_conformers
    from openmmtools.constants import kB
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
    from perses.annihilation.relative import HybridTopologyFactory
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import simtk.openmm as openmm
    from perses.utils.openeye import iupac_to_oemol, extractPositionsFromOEMol
    from perses.utils.openeye import generate_expression
    from openmmforcefields.generators import SystemGenerator
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    from perses.tests.utils import validate_endstate_energies
    temperature = 300*unit.kelvin
    # Compute kT and inverse temperature.
    kT = kB * temperature
    beta = 1.0 / kT
    ENERGY_THRESHOLD = 1e-6

    atom_expr, bond_expr = generate_expression(atom_expression), generate_expression(bond_expression)

    mol = iupac_to_oemol(mol_name)
    mol = generate_conformers(mol, max_confs=1)

    refmol = iupac_to_oemol(ref_mol_name)
    refmol = generate_conformers(refmol,max_confs=1)

    from openff.toolkit.topology import Molecule
    molecules = [Molecule.from_openeye(oemol) for oemol in [refmol, mol]]
    barostat = None
    forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus}
    nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff}

    system_generator = SystemGenerator(forcefields = forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs, nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                                         small_molecule_forcefield = 'gaff-2.11', molecules=molecules, cache=None)

    # Make a topology proposal with the appropriate data:
    topology = generateTopologyFromOEMol(refmol)
    system = system_generator.create_system(topology)
    positions = extractPositionsFromOEMol(refmol)

    proposal_engine = SmallMoleculeSetProposalEngine([refmol, mol], system_generator, atom_expr=atom_expr, bond_expr=bond_expr, allow_ring_breaking=True)
    proposal = proposal_engine.propose(system, topology)
    geometry_engine = FFAllAngleGeometryEngine()
    new_positions, _ = geometry_engine.propose(proposal, positions, beta = beta, validate_energy_bookkeeping = False)
    _ = geometry_engine.logp_reverse(proposal, new_positions, positions, beta)

    factory = HybridTopologyFactory(proposal, positions, new_positions)
    if not proposal.unique_new_atoms:
        assert geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
        assert geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
        vacuum_added_valence_energy = 0.0
    else:
        added_valence_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential

    if not proposal.unique_old_atoms:
        assert geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
        assert geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
        subtracted_valence_energy = 0.0
    else:
        subtracted_valence_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential

    zero_state_error, one_state_error = validate_endstate_energies(factory._topology_proposal, factory, added_valence_energy, subtracted_valence_energy, beta = 1.0/(kB*temperature), ENERGY_THRESHOLD = ENERGY_THRESHOLD, platform = openmm.Platform.getPlatformByName('Reference'))
    return factory

def test_compare_energies():
    mols_and_refs = [['naphthalene', 'benzene'], ['pentane', 'propane'], ['biphenyl', 'benzene']]

    for mol_ref_pair in mols_and_refs:
        _ = compare_energies(mol_name=mol_ref_pair[0], ref_mol_name=mol_ref_pair[1])

def test_position_output():
    """
    Test that the hybrid returns the correct positions for the new and old systems after construction
    """
    from perses.annihilation.relative import HybridTopologyFactory
    import numpy as np

    # Generate topology proposal
    topology_proposal, old_positions, new_positions = utils.generate_solvated_hybrid_test_topology()

    factory = HybridTopologyFactory(topology_proposal, old_positions, new_positions)

    old_positions_factory = factory.old_positions(factory.hybrid_positions)
    new_positions_factory = factory.new_positions(factory.hybrid_positions)

    assert np.all(np.isclose(old_positions.in_units_of(unit.nanometers), old_positions_factory.in_units_of(unit.nanometers)))
    assert np.all(np.isclose(new_positions.in_units_of(unit.nanometers), new_positions_factory.in_units_of(unit.nanometers)))

def test_generate_endpoint_thermodynamic_states():
    """
    test whether the hybrid system zero and one thermodynamic states have the appropriate lambda values
    """
    topology_proposal, current_positions, new_positions = utils.generate_solvated_hybrid_test_topology(current_mol_name='propane', proposed_mol_name='pentane', vacuum = False)
    hybrid_factory = HybridTopologyFactory(topology_proposal, current_positions, new_positions, use_dispersion_correction=True)

    # Get the relevant thermodynamic states:
    _, _, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state = utils.generate_endpoint_thermodynamic_states(hybrid_factory.hybrid_system, topology_proposal)
    # Check the parameters for each state
    lambda_protocol = ['lambda_sterics_core','lambda_electrostatics_core','lambda_sterics_insert','lambda_electrostatics_insert','lambda_sterics_delete','lambda_electrostatics_delete']
    for value in lambda_protocol:
        if getattr(lambda_zero_thermodynamic_state, value) != 0.:
            raise Exception('Interaction {} not set to 0. at lambda = 0. {} set to {}'.format(value,value, getattr(lambda_one_thermodynamic_state, value)))
        if getattr(lambda_one_thermodynamic_state, value) != 1.:
            raise Exception('Interaction {} not set to 1. at lambda = 1. {} set to {}'.format(value,value, getattr(lambda_one_thermodynamic_state, value)))


def HybridTopologyFactory_energies(current_mol = 'toluene', proposed_mol = '1,2-bis(trifluoromethyl) benzene', validate_geometry_energy_bookkeeping = True):
    """
    Test whether the difference in the nonalchemical zero and alchemical zero states is the forward valence energy.  Also test for the one states.
    """
    from perses.tests.utils import generate_solvated_hybrid_test_topology, generate_endpoint_thermodynamic_states
    import openmmtools.cache as cache

    # Just test the solvated system
    top_proposal, old_positions, _ = generate_solvated_hybrid_test_topology(current_mol_name = current_mol, proposed_mol_name = proposed_mol)

    # Remove the dispersion correction
    top_proposal._old_system.getForce(3).setUseDispersionCorrection(False)
    top_proposal._new_system.getForce(3).setUseDispersionCorrection(False)

    # Run geometry engine to generate old and new positions
    _geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=100, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)
    _new_positions, _lp = _geometry_engine.propose(top_proposal, old_positions, beta, validate_geometry_energy_bookkeeping)
    _lp_rev = _geometry_engine.logp_reverse(top_proposal, _new_positions, old_positions, beta, validate_geometry_energy_bookkeeping)

    # Make the hybrid system, reset the CustomNonbondedForce cutoff
    HTF = HybridTopologyFactory(top_proposal, old_positions, _new_positions)
    hybrid_system = HTF.hybrid_system

    nonalch_zero, nonalch_one, alch_zero, alch_one = generate_endpoint_thermodynamic_states(hybrid_system, top_proposal)

    # Compute reduced energies for the nonalchemical systems...
    attrib_list = [(nonalch_zero, old_positions, top_proposal._old_system.getDefaultPeriodicBoxVectors()),
                    (alch_zero, HTF._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    (alch_one, HTF._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    (nonalch_one, _new_positions, top_proposal._new_system.getDefaultPeriodicBoxVectors())]

    rp_list = []
    for (state, pos, box_vectors) in attrib_list:
        context, integrator = cache.global_context_cache.get_context(state)
        samplerstate = SamplerState(positions = pos, box_vectors = box_vectors)
        samplerstate.apply_to_context(context)
        rp = state.reduced_potential(context)
        rp_list.append(rp)

    # Valence energy definitions
    forward_added_valence_energy = _geometry_engine.forward_final_context_reduced_potential - _geometry_engine.forward_atoms_with_positions_reduced_potential
    reverse_subtracted_valence_energy = _geometry_engine.reverse_final_context_reduced_potential - _geometry_engine.reverse_atoms_with_positions_reduced_potential

    nonalch_zero_rp, alch_zero_rp, alch_one_rp, nonalch_one_rp = rp_list[0], rp_list[1], rp_list[2], rp_list[3]
    # print(f"Difference between zeros: {nonalch_zero_rp - alch_zero_rp}; forward added: {forward_added_valence_energy}")
    # print(f"Difference between ones: {nonalch_zero_rp - alch_zero_rp}; forward added: {forward_added_valence_energy}")

    assert abs(nonalch_zero_rp - alch_zero_rp + forward_added_valence_energy) < ENERGY_THRESHOLD, f"The zero state alchemical and nonalchemical energy absolute difference {abs(nonalch_zero_rp - alch_zero_rp + forward_added_valence_energy)} is greater than the threshold of {ENERGY_THRESHOLD}."
    assert abs(nonalch_one_rp - alch_one_rp + reverse_subtracted_valence_energy) < ENERGY_THRESHOLD, f"The one state alchemical and nonalchemical energy absolute difference {abs(nonalch_one_rp - alch_one_rp + reverse_subtracted_valence_energy)} is greater than the threshold of {ENERGY_THRESHOLD}."

    print(f"Abs difference in zero alchemical vs nonalchemical systems: {abs(nonalch_zero_rp - alch_zero_rp + forward_added_valence_energy)}")
    print(f"Abs difference in one alchemical vs nonalchemical systems: {abs(nonalch_one_rp - alch_one_rp + reverse_subtracted_valence_energy)}")

def test_HybridTopologyFactory_energies(molecule_perturbation_list = [['naphthalene', 'benzene'], ['pentane', 'propane'], ['biphenyl', 'benzene']], validations = [False, True, False]):
    """
    Test whether the difference in the nonalchemical zero and alchemical zero states is the forward valence energy.  Also test for the one states.
    """
    for molecule_pair, validate in zip(molecule_perturbation_list, validations):
        print(f"\tconduct energy comparison for {molecule_pair[0]} --> {molecule_pair[1]}")
        HybridTopologyFactory_energies(current_mol = molecule_pair[0], proposed_mol = molecule_pair[1], validate_geometry_energy_bookkeeping = validate)

def test_RMSD_restraint():
    """
    test the creation of an RMSD restraint between core heavy atoms and protein CA atoms on a hostguest transformation in a periodic solvent.
    will assert the existence of an RMSD force, minimizes at lambda=0, and runs 500 steps of MD.

    """
    from pkg_resources import resource_filename
    from perses.app.relative_setup import RelativeFEPSetup
    from openmmtools.states import ThermodynamicState, SamplerState
    from openmmtools.integrators import LangevinIntegrator
    from perses.dispersed.utils import minimize

    # Setup directory
    ligand_sdf = resource_filename("perses", "data/given-geometries/ligands.sdf")
    host_pdb = resource_filename("perses", "data/given-geometries/receptor.pdb")

    setup = RelativeFEPSetup(
             ligand_input = ligand_sdf,
             old_ligand_index=0,
             new_ligand_index=1,
             forcefield_files = ['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml'],
             phases = ['complex', 'solvent', 'vacuum'],
             protein_pdb_filename=host_pdb,
             receptor_mol2_filename=None,
             pressure=1.0 * unit.atmosphere,
             temperature=300.0 * unit.kelvin,
             solvent_padding=9.0 * unit.angstroms,
             ionic_strength=0.15 * unit.molar,
             hmass=4*unit.amus,
             neglect_angles=False,
             map_strength='default',
             atom_expr=None,
             bond_expr=None,
             anneal_14s=False,
             small_molecule_forcefield='gaff-2.11',
             small_molecule_parameters_cache=None,
             trajectory_directory=None,
             trajectory_prefix=None,
             spectator_filenames=None,
             nonbonded_method = 'PME',
             complex_box_dimensions=None,
             solvent_box_dimensions=None,
             remove_constraints=False,
             use_given_geometries = False
             )
    phase = 'complex'
    top_prop = setup._complex_topology_proposal
    htf = HybridTopologyFactory(setup._complex_topology_proposal,
                                   setup.complex_old_positions,
                                   setup.complex_new_positions,
                                   rmsd_restraint=True
                                   )
    #assert there is at least a CV force
    force_names = {htf._hybrid_system.getForce(i).__class__.__name__: htf._hybrid_system.getForce(i) for i in range(htf._hybrid_system.getNumForces())}
    assert 'CustomCVForce' in list(force_names.keys())
    coll_var_name = force_names['CustomCVForce'].getCollectiveVariableName(0)
    assert coll_var_name == 'RMSD'
    coll_var = force_names['CustomCVForce'].getCollectiveVariable(0)
    coll_var_particles = coll_var.getParticles()
    assert len(coll_var_particles) > 0 #the number of particles is nonzero. this will cause problems otherwise
    #assert coll_var.usesPeriodicBoundaryConditions() #should this be the case?

    #make thermo and sampler state
    thermostate = ThermodynamicState(system = htf._hybrid_system, temperature = 300*unit.kelvin, pressure = 1.0*unit.atmosphere)
    ss = SamplerState(positions=htf._hybrid_positions, box_vectors = htf._hybrid_system.getDefaultPeriodicBoxVectors())

    #attempt to minimize
    minimize(thermostate, ss)

    #run simulation to validate no nans
    integrator = LangevinIntegrator(300*unit.kelvin, 5.0/unit.picosecond, 2.0*unit.femtosecond)
    context = thermostate.create_context(integrator)
    ss.apply_to_context(context)
    context.setVelocitiesToTemperature(300*unit.kelvin)

    integrator.step(500)

def RepartitionedHybridTopologyFactory_energies(topology, chain, system, positions, system_generator):
    """
    Test whether the difference in the nonalchemical zero and alchemical zero states is the forward valence energy.  Also test for the one states.
    Note that two RepartitionedHybridTopologyFactorys need to be generated (one for each endstate) because the energies need to be validated separately for each endstate.
    """

    from perses.rjmc.topology_proposal import PointMutationEngine
    from perses.annihilation.relative import RepartitionedHybridTopologyFactory
    from perses.tests.utils import validate_endstate_energies

    ENERGY_THRESHOLD = 1e-6

    for res in topology.residues():
        if res.id == '2':
            wt_res = res.name
    aminos_updated = [amino for amino in aminos if amino not in [wt_res, 'PRO', 'HIS', 'TRP', 'PHE', 'TYR']]
    mutant = random.choice(aminos_updated)
    print(f'Making mutation {wt_res}->{mutant}')

    # Create point mutation engine to mutate residue at id 2 to random amino acid
    point_mutation_engine = PointMutationEngine(wildtype_topology=topology,
                                                system_generator=system_generator,
                                                chain_id=chain,
                                                max_point_mutants=1,
                                                residues_allowed_to_mutate=['2'],  # the residue ids allowed to mutate
                                                allowed_mutations=[('2', mutant)],
                                                aggregate=True)  # always allow aggregation

    # Create topology proposal
    topology_proposal = point_mutation_engine.propose(current_system=system, current_topology=topology)

    # Create geometry engine
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine(metadata=None,
                                               use_sterics=False,
                                               n_bond_divisions=100,
                                               n_angle_divisions=180,
                                               n_torsion_divisions=360,
                                               verbose=True,
                                               storage=None,
                                               bond_softening_constant=1.0,
                                               angle_softening_constant=1.0,
                                               neglect_angles=False,
                                               use_14_nonbondeds=True)

    # Create geometry proposal
    new_positions, logp_proposal = geometry_engine.propose(topology_proposal, positions, beta,
                                                                   validate_energy_bookkeeping=True)
    logp_reverse = geometry_engine.logp_reverse(topology_proposal, new_positions, positions, beta,
                                                validate_energy_bookkeeping=True)

    if not topology_proposal.unique_new_atoms:
        assert geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
        assert geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
        vacuum_added_valence_energy = 0.0
    else:
        added_valence_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential

    if not topology_proposal.unique_old_atoms:
        assert geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
        assert geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
        subtracted_valence_energy = 0.0
    else:
        subtracted_valence_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential

    # Generate repartitioned htf at lambda = 0
    zero_htf = RepartitionedHybridTopologyFactory(topology_proposal=topology_proposal,
                          current_positions=positions,
                          new_positions=new_positions,
                          endstate=0)

    # Compute error at lambda = 0 endstate
    zero_state_error, _ = validate_endstate_energies(zero_htf._topology_proposal,
                                                     zero_htf,
                                                     added_valence_energy,
                                                     subtracted_valence_energy,
                                                     ENERGY_THRESHOLD=ENERGY_THRESHOLD,
                                                     platform=openmm.Platform.getPlatformByName('Reference'),
                                                     repartitioned_endstate=0)
    # Generate repartitioned htf at lambda = 1
    one_htf = RepartitionedHybridTopologyFactory(topology_proposal=topology_proposal,
                                                  current_positions=positions,
                                                  new_positions=new_positions,
                                                  endstate=1)

    # Compute error at lambda = 1 endstate
    _, one_state_error = validate_endstate_energies(one_htf._topology_proposal,
                                                     one_htf,
                                                     added_valence_energy,
                                                     subtracted_valence_energy,
                                                     ENERGY_THRESHOLD=ENERGY_THRESHOLD,
                                                     platform=openmm.Platform.getPlatformByName('Reference'),
                                                     repartitioned_endstate=1)

    # Check that endstate errors are below threshold
    assert abs(zero_state_error) < ENERGY_THRESHOLD, f"The zero state alchemical and nonalchemical energy absolute difference {abs(zero_state_error)} is greater than the threshold of {ENERGY_THRESHOLD}."
    assert abs(one_state_error) < ENERGY_THRESHOLD, f"The one state alchemical and nonalchemical energy absolute difference {abs(one_state_error)} is greater than the threshold of {ENERGY_THRESHOLD}."

    print(f"Abs difference in zero alchemical vs nonalchemical systems: {abs(zero_state_error)}")
    print(f"Abs difference in one alchemical vs nonalchemical systems: {abs(one_state_error)}")


def test_RepartitionedHybridTopologyFactory_energies():
    """
    Test whether the difference in the nonalchemical zero and alchemical zero states is the forward valence energy.  Also test for the one states.
    """

    from perses.tests.test_topology_proposal import generate_atp
    from openmmforcefields.generators import SystemGenerator

    # Test alanine dipeptide in vacuum
    atp, system_generator = generate_atp('vacuum')
    RepartitionedHybridTopologyFactory_energies(atp.topology, '1', atp.system, atp.positions, system_generator)

    # Test alanine dipeptide in solvent
    atp, system_generator = generate_atp('solvent')
    RepartitionedHybridTopologyFactory_energies(atp.topology, '1', atp.system, atp.positions, system_generator)

    # Test 8-mer peptide in vacuum
    peptide_filename = resource_filename('perses', 'data/8mer-example/4zuh_peptide_capped.pdb')
    pdb = app.PDBFile(peptide_filename)
    forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    barostat = None
    system_generator = SystemGenerator(forcefields=forcefield_files,
                                       barostat=barostat,
                                       forcefield_kwargs={'removeCMMotion': False,
                                                          'ewaldErrorTolerance': 0.00025,
                                                          'constraints': app.HBonds,
                                                          'hydrogenMass': 4 * unit.amus},
                                       periodic_forcefield_kwargs=None,
                                       small_molecule_forcefield='gaff-2.11',
                                       nonperiodic_forcefield_kwargs={'nonbondedMethod': app.NoCutoff},
                                       molecules=None,
                                       cache=None)
    system = system_generator.create_system(pdb.topology)
    positions = unit.quantity.Quantity(
                    value=np.array([list(atom_pos) for atom_pos in pdb.positions.value_in_unit_system(unit.md_unit_system)]),
                    unit=unit.nanometers)
    RepartitionedHybridTopologyFactory_energies(pdb.topology, 'C', system, positions, system_generator)

def flattenedHybridTopologyFactory_energies(topology, chain, system, positions, system_generator, repartitioned=False):
    """
    Test whether the difference in the nonalchemical zero and alchemical zero states is the forward valence energy.  Also test for the one states.
    Note that the torsions/1,4 exception terms of the off atoms are manually zeroed for the lambda = 0 endstate and the endstate error is computed. Then, this is repeated for the lambda = 1 endstate.
    """

    from perses.rjmc.topology_proposal import PointMutationEngine
    from perses.annihilation.relative import RepartitionedHybridTopologyFactory
    from perses.tests.utils import validate_endstate_energies

    ENERGY_THRESHOLD = 1e-6

    # Create point mutation engine to mutate residue at id 2 to a random amino acid
    aminos_updated = [amino for amino in aminos if amino not in ['ALA', 'PRO', 'HIS', 'TRP', 'PHE', 'TYR']]
    mutant = random.choice(aminos_updated)
    print(f'Making mutation ALA->{mutant}')
    point_mutation_engine = PointMutationEngine(wildtype_topology=topology,
                                                system_generator=system_generator,
                                                chain_id=chain,
                                                max_point_mutants=1,
                                                residues_allowed_to_mutate=['2'],  # the residue ids allowed to mutate
                                                allowed_mutations=[('2', mutant)],
                                                aggregate=True)  # always allow aggregation

    for endstate in range(2):
        # Create topology proposal
        topology_proposal = point_mutation_engine.propose(current_system=system, current_topology=topology)

        # Make list of off atoms that should have flattened torsions/exceptions
        off_atoms = topology_proposal.unique_new_atoms if endstate == 0 else topology_proposal.unique_old_atoms
        system = topology_proposal.old_system if endstate == 0 else topology_proposal.new_system

        # Flatten torsions involving off atoms
        periodic_torsion = system.getForce(2)
        for i in range(periodic_torsion.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = periodic_torsion.getTorsionParameters(i)
            if p1 in off_atoms or p2 in off_atoms or p3 in off_atoms or p4 in off_atoms:
                periodic_torsion.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0. * k)

        # Flatten exceptions involving off atoms
        nb_force = system.getForce(3)
        for i in range(nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
            if p1 in off_atoms or p2 in off_atoms:
                nb_force.setExceptionParameters(i, p1, p2, 0, sigma, 0)

        # Create geometry engine
        from perses.rjmc.geometry import FFAllAngleGeometryEngine
        geometry_engine = FFAllAngleGeometryEngine(metadata=None,
                                                   use_sterics=False,
                                                   n_bond_divisions=100,
                                                   n_angle_divisions=180,
                                                   n_torsion_divisions=360,
                                                   verbose=True,
                                                   storage=None,
                                                   bond_softening_constant=1.0,
                                                   angle_softening_constant=1.0,
                                                   neglect_angles=False,
                                                   use_14_nonbondeds=True)

        # Create geometry proposals
        new_positions, logp_proposal = geometry_engine.propose(topology_proposal, positions, beta,
                                                                       validate_energy_bookkeeping=True)
        logp_reverse = geometry_engine.logp_reverse(topology_proposal, new_positions, positions, beta,
                                                    validate_energy_bookkeeping=True)

        if not topology_proposal.unique_new_atoms:
            assert geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
            assert geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
            vacuum_added_valence_energy = 0.0
        else:
            added_valence_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential

        if not topology_proposal.unique_old_atoms:
            assert geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
            assert geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
            subtracted_valence_energy = 0.0
        else:
            subtracted_valence_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential

        if repartitioned:
            # Generate repartitioned htf
            htf = RepartitionedHybridTopologyFactory(topology_proposal=topology_proposal,
                                                    current_positions=positions,
                                                    new_positions=new_positions,
                                                    endstate=endstate)
            # Compute error at endstate
            zero_state_error, one_state_error = validate_endstate_energies(htf._topology_proposal,
                                                        htf,
                                                        added_valence_energy,
                                                        subtracted_valence_energy,
                                                        ENERGY_THRESHOLD=ENERGY_THRESHOLD,
                                                        platform=openmm.Platform.getPlatformByName('Reference'),
                                                        repartitioned_endstate=endstate)
        else:
            # Generate vanilla htf
            htf = HybridTopologyFactory(topology_proposal=topology_proposal,
                                        current_positions=positions,
                                        new_positions=new_positions)

            # Compute error at endstate for vanilla htf
            zero_state_error, one_state_error = validate_endstate_energies(htf._topology_proposal,
                                                             htf,
                                                             added_valence_energy,
                                                             subtracted_valence_energy,
                                                             ENERGY_THRESHOLD=ENERGY_THRESHOLD,
                                                             platform=openmm.Platform.getPlatformByName('Reference'))

        if endstate == 0:
            # Check that endstate errors are below threshold
            assert abs(zero_state_error) < ENERGY_THRESHOLD, f"The zero state alchemical and nonalchemical energy absolute difference {abs(zero_state_error)} is greater than the threshold of {ENERGY_THRESHOLD}."
            print(f"Abs difference in zero state alchemical vs nonalchemical systems: {abs(zero_state_error)}")
        else:
            # Check that endstate errors are below threshold
            assert abs(one_state_error) < ENERGY_THRESHOLD, f"The one state alchemical and nonalchemical energy absolute difference {abs(one_state_error)} is greater than the threshold of {ENERGY_THRESHOLD}."
            print(f"Abs difference in one state alchemical vs nonalchemical systems: {abs(one_state_error)}")

def test_flattenedHybridTopologyFactory_energies():
    """
        Test whether the difference in the nonalchemical zero and alchemical zero states is the forward valence energy.  Also test for the one states.
    """

    from perses.tests.test_topology_proposal import generate_atp

    # Test alanine dipeptide vanilla htf with flattened torsions and exceptions in vacuum
    atp, system_generator = generate_atp()
    flattenedHybridTopologyFactory_energies(atp.topology, '1', atp.system, atp.positions, system_generator)

    # Test alanine dipeptide repartitioned htf with flattened torsions and exceptions in vacuum
    atp, system_generator = generate_atp()
    flattenedHybridTopologyFactory_energies(atp.topology, '1', atp.system, atp.positions, system_generator, repartitioned=True)

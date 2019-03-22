from simtk.openmm import app
from simtk import unit, openmm
import numpy as np
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from perses.annihilation.new_relative import HybridTopologyFactory
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, SystemGenerator, TopologyProposal
from perses.tests import utils
import openeye.oechem as oechem
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
import openmmtools.mcmc as mcmc
import openmmtools.cache as cache
from unittest import skipIf
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

import pymbar.timeseries as timeseries

import copy
import pymbar

istravis = os.environ.get('TRAVIS', None) == 'true'

try:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")
except Exception:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")

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
    #create the hybrid system:
    hybrid_factory = HybridTopologyFactory(topology_proposal, current_positions, new_positions, use_dispersion_correction=True)

    #get the relevant thermodynamic states:
    nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state = utils.generate_endpoint_thermodynamic_states(
        hybrid_factory.hybrid_system, topology_proposal)

    nonalchemical_thermodynamic_states = [nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state]

    alchemical_thermodynamic_states = [lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state]

    #create an MCMCMove, BAOAB with default parameters
    mc_move = mcmc.LangevinDynamicsMove()

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
    df = results[0]
    ddf = results[1]
    N_eff = results[2]

    if N_eff < neffmin:
        raise Exception("Number of effective samples %f was below minimum of %f" % (N_eff, neffmin))

    if ddf > threshold:
        raise Exception("Standard deviation of %f exceeds threshold of %f" % (ddf, threshold))

def test_simple_overlap_pairs(pairs=[['pentane','butane'],['fluorobenzene', 'chlorobenzene'],['benzene', 'catechol'],['benzene','2-phenyl ethanol'],['imatinib','nilotinib']]):
    """
    Test to run pairs of small molecule perturbations in vacuum, using test_simple_overlap, both forward and backward.
    pentane <-> butane is adding a methyl group
    fluorobenzene <-> chlorobenzene perturbs one halogen to another, with no adding or removing of atoms
    benzene <-> catechol perturbing molecule in two positions simultaneously
    benzene <-> 2-phenyl ethanol addition of 3 heavy atom group 
    """
    # TODO remove line below
    pairs = [['2-phenyl ethanol','benzene']]
    for pair in pairs:
        print('{} -> {}'.format(pair[0],pair[1]))
        test_simple_overlap(pair[0],pair[1])
        # now running the reverse
#        print('{} -> {}'.format(pair[1],pair[0]))
#        test_simple_overlap(pair[1],pair[0])

def test_simple_overlap(name1='pentane',name2='butane'):
    """Test that the variance of the endpoint->nonalchemical perturbation is sufficiently small for pentane->butane in vacuum"""
    topology_proposal, current_positions, new_positions = utils.generate_vacuum_topology_proposal(current_mol_name=name1, proposed_mol_name=name2)
    results = run_hybrid_endpoint_overlap(topology_proposal, current_positions, new_positions)
    for idx, lambda_result in enumerate(results):
        try:
            check_result(lambda_result)
        except Exception as e:
            message = "pentane->butane failed at lambda %d \n" % idx
            message += str(e)
            raise Exception(message)

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

@skipIf(istravis, "Skip expensive test on travis")
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

def run_endpoint_perturbation(lambda_thermodynamic_state, nonalchemical_thermodynamic_state, initial_hybrid_sampler_state, mc_move, n_iterations, factory, lambda_index=0):
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
    lambda_index : int, optional default 0
        The index, 0 or 1, at which to retrieve nonalchemical positions

    Returns
    -------
    df : float
        Free energy difference between alchemical and nonalchemical systems, estimated with EXP
    ddf : float
        Standard deviation of estimate, corrected for correlation, from EXP estimator.
    """
    from simtk.openmm.app import PDBFile
    #run an initial minimization:
    mcmc_sampler = mcmc.MCMCSampler(lambda_thermodynamic_state, initial_hybrid_sampler_state, mc_move)
    mcmc_sampler.minimize(max_iterations=20)
    new_sampler_state = mcmc_sampler.sampler_state

    #initialize work array
    w = np.zeros([n_iterations])
    non_potential = np.zeros([n_iterations])
    hybrid_potential = np.zeros([n_iterations])

    #run n_iterations of the endpoint perturbation:
    for iteration in range(n_iterations):
        mc_move.apply(lambda_thermodynamic_state, new_sampler_state)
        #compute the reduced potential at the new state
        hybrid_context, integrator = cache.global_context_cache.get_context(lambda_thermodynamic_state)
        state = hybrid_context.getState(getPositions=True, getVelocities=True, getForces=True)
#        forces = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system) # unitless, in MD unit system
#        force_magnitudes = np.sqrt(np.sum(forces**2, 1))
#        for x in force_magnitudes:
#            print(x)
        state_xml = openmm.XmlSerializer.serialize(state)
        with open('state{}_l{}.xml'.format(iteration,lambda_index), 'w') as outfile:
            outfile.write(state_xml)
        new_sampler_state.apply_to_context(hybrid_context, ignore_velocities=True)
        hybrid_reduced_potential = lambda_thermodynamic_state.reduced_potential(hybrid_context)

        #generate a sampler state for the nonalchemical system
        if lambda_index == 0:
            nonalchemical_positions = factory.old_positions(new_sampler_state.positions)
        elif lambda_index == 1:
            nonalchemical_positions = factory.new_positions(new_sampler_state.positions)
        else:
            raise ValueError("The lambda index needs to be either one or zero for this to be meaningful")
        pdbfile = open('state{}_l{}.pdb'.format(iteration,lambda_index),'w')
        PDBFile.writeModel(factory.hybrid_topology, new_sampler_state.positions, file=pdbfile)
        nonalchemical_sampler_state = SamplerState(nonalchemical_positions, box_vectors=new_sampler_state.box_vectors)

        #compute the reduced potential at the nonalchemical system as well:
        nonalchemical_context, integrator = cache.global_context_cache.get_context(nonalchemical_thermodynamic_state)
        nonalchemical_sampler_state.apply_to_context(nonalchemical_context, ignore_velocities=True)
        nonalchemical_reduced_potential = nonalchemical_thermodynamic_state.reduced_potential(nonalchemical_context)
        w[iteration] = nonalchemical_reduced_potential - hybrid_reduced_potential
        non_potential[iteration] = nonalchemical_reduced_potential
        hybrid_potential[iteration] = hybrid_reduced_potential
    [t0, g, Neff_max] = timeseries.detectEquilibration(w)
    w_burned_in = w[t0:]

    [df, ddf] = pymbar.EXP(w_burned_in)
    ddf_corrected = ddf * np.sqrt(g)

    return [df, ddf_corrected, Neff_max], non_potential, hybrid_potential 

def compare_energies(mol_name="naphthalene", ref_mol_name="benzene"):
    """
    Make an atom map where the molecule at either lambda endpoint is identical, and check that the energies are also the same.
    """
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
    from perses.annihilation.new_relative import HybridTopologyFactory
    import simtk.openmm as openmm

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC

    mol_name = "naphthalene"
    ref_mol_name = "benzene"

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

    factory = HybridTopologyFactory(top_proposal, positions, positions)

    alchemical_system = factory.hybrid_system
    alchemical_positions = factory.hybrid_positions

    integrator = openmm.VerletIntegrator(1)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(alchemical_system, integrator, platform)

    context.setPositions(alchemical_positions)

    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }

    #set all to zero
    for parm in functions.keys():
        context.setParameter(parm, 0.0)

    initial_energy = context.getState(getEnergy=True).getPotentialEnergy()

    #set all to one
    for parm in functions.keys():
        context.setParameter(parm, 1.0)

    final_energy = context.getState(getEnergy=True).getPotentialEnergy()

    if np.abs(final_energy - initial_energy) > 1.0e-6*unit.kilojoule_per_mole:
        raise Exception("The energy at the endpoints was not equal for molecule %s" % mol_name)

def test_compare_energies():
    mols_and_refs = [['naphthalene', 'benzene'], ['pentane', 'propane'], ['biphenyl', 'benzene']]

    for mol_ref_pair in mols_and_refs:
        compare_energies(mol_name=mol_ref_pair[0], ref_mol_name=mol_ref_pair[1])

def test_position_output():
    """
    Test that the hybrid returns the correct positions for the new and old systems after construction
    """
    from perses.annihilation.new_relative import HybridTopologyFactory
    import numpy as np

    #generate topology proposal
    topology_proposal, old_positions, new_positions = utils.generate_vacuum_topology_proposal()

    factory = HybridTopologyFactory(topology_proposal, old_positions, new_positions)

    old_positions_factory = factory.old_positions(factory.hybrid_positions)
    new_positions_factory = factory.new_positions(factory.hybrid_positions)

    assert np.all(np.isclose(old_positions.in_units_of(unit.nanometers), old_positions_factory.in_units_of(unit.nanometers)))
    assert np.all(np.isclose(new_positions.in_units_of(unit.nanometers), new_positions_factory.in_units_of(unit.nanometers)))

def test_generate_endpoint_thermodynamic_states():
    topology_proposal, current_positions, new_positions = utils.generate_vacuum_topology_proposal(current_mol_name='propane', proposed_mol_name='pentane')
    hybrid_factory = HybridTopologyFactory(topology_proposal, current_positions, new_positions, use_dispersion_correction=True)

    #get the relevant thermodynamic states:
    _, _, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state = utils.generate_endpoint_thermodynamic_states(hybrid_factory.hybrid_system, topology_proposal)
    # check the parameters for each state
    lambda_protocol = ['lambda_sterics_core','lambda_electrostatics_core','lambda_sterics_insert','lambda_electrostatics_insert','lambda_sterics_delete','lambda_electrostatics_delete']
    for value in lambda_protocol:
        if getattr(lambda_zero_thermodynamic_state, value) != 0.: 
            raise Exception('Interaction {} not set to 0. at lambda = 0. {} set to {}'.format(value,value, getattr(lambda_one_thermodynamic_state, value)))
        if getattr(lambda_one_thermodynamic_state, value) != 1.: 
            raise Exception('Interaction {} not set to 1. at lambda = 1. {} set to {}'.format(value,value, getattr(lambda_one_thermodynamic_state, value)))

if __name__ == '__main__':
    #test_compare_energies()
    #test_position_output()
    test_difficult_overlap()

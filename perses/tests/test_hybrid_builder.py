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
    for lambda_state in (0, 1):
        result = run_endpoint_perturbation(alchemical_thermodynamic_states[lambda_state],
                                        nonalchemical_thermodynamic_states[lambda_state], initial_sampler_state,
                                        mc_move, 100, hybrid_factory, lambda_index=lambda_state)
        print(result)

        hybrid_endpoint_results.append(result)

    return hybrid_endpoint_results

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

def test_simple_overlap():
    """Test that the variance of the endpoint->nonalchemical perturbation is sufficiently small for pentane->butane in vacuum"""
    topology_proposal, current_positions, new_positions = utils.generate_vacuum_topology_proposal(current_mol_name='propane', proposed_mol_name='butane')
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
    #run an initial minimization:
    mcmc_sampler = mcmc.MCMCSampler(lambda_thermodynamic_state, initial_hybrid_sampler_state, mc_move)
    mcmc_sampler.minimize(max_iterations=20)
    new_sampler_state = mcmc_sampler.sampler_state

    #initialize work array
    w = np.zeros([n_iterations])

    #run n_iterations of the endpoint perturbation:
    for iteration in range(n_iterations):
        mc_move.apply(lambda_thermodynamic_state, new_sampler_state)
        print(iteration)
        #compute the reduced potential at the new state
        hybrid_context, integrator = cache.global_context_cache.get_context(lambda_thermodynamic_state)
        new_sampler_state.apply_to_context(hybrid_context, ignore_velocities=True)
        hybrid_reduced_potential = lambda_thermodynamic_state.reduced_potential(hybrid_context)

        #generate a sampler state for the nonalchemical system
        if lambda_index == 0:
            nonalchemical_positions = factory.old_positions(new_sampler_state.positions)
        elif lambda_index == 1:
            nonalchemical_positions = factory.new_positions(new_sampler_state.positions)
        else:
            raise ValueError("The lambda index needs to be either one or zero for this to be meaningful")

        nonalchemical_sampler_state = SamplerState(nonalchemical_positions, box_vectors=new_sampler_state.box_vectors)

        #compute the reduced potential at the nonalchemical system as well:
        nonalchemical_context, integrator = cache.global_context_cache.get_context(nonalchemical_thermodynamic_state)
        nonalchemical_sampler_state.apply_to_context(nonalchemical_context, ignore_velocities=True)
        nonalchemical_reduced_potential = nonalchemical_thermodynamic_state.reduced_potential(nonalchemical_context)
        w[iteration] = nonalchemical_reduced_potential - hybrid_reduced_potential
    [t0, g, Neff_max] = timeseries.detectEquilibration(w)
    print(Neff_max)
    w_burned_in = w[t0:]

    [df, ddf] = pymbar.EXP(w_burned_in)
    ddf_corrected = ddf * np.sqrt(g)

    return [df, ddf_corrected, Neff_max]

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

if __name__ == '__main__':
    #test_compare_energies()
    #test_position_output()
    test_difficult_overlap()

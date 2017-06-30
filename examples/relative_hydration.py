import simtk.openmm as openmm
import simtk.unit as unit
from perses.annihilation.ncmc_switching import NCMCGHMCAlchemicalIntegrator
import progressbar
from perses.tests.utils import createSystemFromIUPAC, get_data_filename
from perses.annihilation.new_relative import HybridTopologyFactory
from perses.rjmc.topology_proposal import TopologyProposal, SmallMoleculeSetProposalEngine, SystemGenerator
from perses.rjmc.geometry import FFAllAngleGeometryEngine
import openeye.oechem as oechem
import numpy as np


kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

def simulate_hybrid(hybrid_system,functions, lambda_value, positions, nsteps=500, timestep=1.0*unit.femtoseconds, temperature=temperature, collision_rate=5.0/unit.picoseconds):
    platform = openmm.Platform.getPlatformByName("CUDA")
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(hybrid_system, integrator, platform)
    for parameter in functions.keys():
        context.setParameter(parameter, lambda_value)
    context.setPositions(positions)
    integrator.step(nsteps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return positions


def generate_solvated_hybrid_test_topology(mol_name="naphthalene", ref_mol_name="benzene"):

    from topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    m, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(mol_name)
    refmol = createOEMolFromIUPAC(ref_mol_name)

    initial_smiles = oechem.OEMolToSmiles(m)
    final_smiles = oechem.OEMolToSmiles(refmol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    modeller = app.Modeller(top_old, pos_old)
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()
    solvated_system = forcefield.createSystem(solvated_topology, nonbondedMethod=app.PME)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'])
    geometry_engine = FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, solvated_topology)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, solvated_positions, beta)

    return topology_proposal, solvated_positions, new_positions

def generate_vacuum_hybrid_topology(mol_name="naphthalene", ref_mol_name="benzene"):
    from topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    m, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(mol_name)
    refmol = createOEMolFromIUPAC(ref_mol_name)

    initial_smiles = oechem.OEMolToSmiles(m)
    final_smiles = oechem.OEMolToSmiles(refmol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'])
    geometry_engine = FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, pos_old, beta)

    return topology_proposal, pos_old, new_positions


def check_alchemical_hybrid_elimination_bar(topology_proposal, old_positions, new_positions, ncmc_nsteps=50, n_iterations=50, NSIGMA_MAX=6.0, geometry=False):
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

    #make the hybrid topology factory:
    factory = HybridTopologyFactory(topology_proposal, old_positions, new_positions)

    platform = openmm.Platform.getPlatformByName("CUDA")

    hybrid_system = factory.hybrid_system
    hybrid_topology = factory.hybrid_topology
    initial_hybrid_positions = factory.hybrid_positions


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
    forward_integrator = NCMCGHMCAlchemicalIntegrator(temperature, hybrid_system, functions, nsteps=ncmc_nsteps, direction='insert')
    forward_context = openmm.Context(hybrid_system, forward_integrator, platform)
    print("Minimizing for forward protocol...")
    forward_context.setPositions(initial_hybrid_positions)
    for parm in functions.keys():
        forward_context.setParameter(parm, 0.0)

    openmm.LocalEnergyMinimizer.minimize(forward_context, maxIterations=10)

    initial_state = forward_context.getState(getPositions=True, getEnergy=True)
    print("The initial energy after minimization is %s" % str(initial_state.getPotentialEnergy()))
    initial_forward_positions = initial_state.getPositions(asNumpy=True)
    equil_positions = simulate_hybrid(hybrid_system,functions, 0.0, initial_forward_positions)

    print("Beginning forward protocols")
    #first, do forward protocol (lambda=0 -> 1)
    with progressbar.ProgressBar(max_value=n_iterations) as bar:
        for i in range(n_iterations):
            equil_positions = simulate_hybrid(hybrid_system, functions, 0.0, equil_positions)
            forward_context.setPositions(equil_positions)
            forward_integrator.step(ncmc_nsteps)
            w_f[i] = -1.0 * forward_integrator.getLogAcceptanceProbability(forward_context)
            bar.update(i)

    del forward_context, forward_integrator

    reverse_integrator = NCMCGHMCAlchemicalIntegrator(temperature, hybrid_system, functions, nsteps=ncmc_nsteps, direction='delete')

    print("Minimizing for reverse protocol...")
    reverse_context = openmm.Context(hybrid_system, reverse_integrator, platform)
    reverse_context.setPositions(initial_hybrid_positions)
    for parm in functions.keys():
        reverse_context.setParameter(parm, 1.0)
    openmm.LocalEnergyMinimizer.minimize(reverse_context, maxIterations=10)
    initial_state = reverse_context.getState(getPositions=True, getEnergy=True)
    print("The initial energy after minimization is %s" % str(initial_state.getPotentialEnergy()))
    initial_reverse_positions = initial_state.getPositions(asNumpy=True)
    equil_positions = simulate_hybrid(hybrid_system,functions, 1.0, initial_reverse_positions, nsteps=1000)

    #now, reverse protocol
    print("Beginning reverse protocols...")
    with progressbar.ProgressBar(max_value=n_iterations) as bar:
        for i in range(n_iterations):
            equil_positions = simulate_hybrid(hybrid_system,functions, 1.0, equil_positions)
            reverse_context.setPositions(equil_positions)
            reverse_integrator.step(ncmc_nsteps)
            w_r[i] = -1.0 * reverse_integrator.getLogAcceptanceProbability(reverse_context)
            bar.update(i)
    del reverse_context, reverse_integrator

    from pymbar import BAR
    [df, ddf] = BAR(w_f, w_r)
    print("df = %12.6f +- %12.5f kT" % (df, ddf))


if __name__=="__main__":
    topology_proposal, old_positions, new_positions = generate_solvated_hybrid_test_topology()
    check_alchemical_hybrid_elimination_bar(topology_proposal, old_positions, new_positions, ncmc_nsteps=100, n_iterations=500)
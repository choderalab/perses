import numpy as np
import tqdm
from openmmtools import integrators, states, mcmc, constants
from perses.rjmc.topology_proposal import TopologyProposal
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.annihilation.ncmc_switching import NCMCEngine
from simtk import openmm, unit
from io import StringIO
from simtk.openmm import app
from perses.dispersed.feptasks import compute_reduced_potential
from dask import distributed
import mdtraj as md
temperature = 300.0*unit.kelvin
beta = 1.0 / (temperature*constants.kB)

def createOEMolFromIUPAC(iupac_name='bosutinib'):
    from openeye import oechem, oeiupac, oeomega

    # Create molecule.
    mol = oechem.OEMol()
    oeiupac.OEParseIUPACName(mol, iupac_name)
    mol.SetTitle("MOL")

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(mol, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(mol)

    # Create atom names.
    oechem.OETriposAtomNames(mol)

    # Create bond types
    oechem.OETriposBondTypeNames(mol)

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
    from perses.tests.utils import get_data_filename, extractPositionsFromOEMOL
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
    system = forcefield.createSystem(topology, removeCMMotion=False)

    # Extract positions
    positions = extractPositionsFromOEMOL(molecule)

    return (molecule, system, positions, topology)

def generate_solvated_hybrid_test_topology(current_mol_name="naphthalene", proposed_mol_name="benzene"):
    """
    Generate a test solvated topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

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
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators
    from openeye import oechem
    from perses.rjmc.topology_proposal import SystemGenerator, SmallMoleculeSetProposalEngine
    from perses.rjmc import geometry

    from perses.tests.utils import get_data_filename

    current_mol, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(current_mol_name)

    proposed_mol = createOEMolFromIUPAC(proposed_mol_name)
    proposed_mol.SetTitle("MOL")

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    modeller = app.Modeller(top_old, pos_old)
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()
    solvated_system = forcefield.createSystem(solvated_topology, nonbondedMethod=app.PME, removeCMMotion=False)
    barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, temperature, 50)

    solvated_system.addForce(barostat)

    gaff_filename = get_data_filename('data/gaff.xml')

    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], barostat=barostat, forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.PME})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name="MOL")

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, solvated_topology)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, solvated_positions, beta)

    return topology_proposal, solvated_positions, new_positions


def run_equilibrium(system, topology, configuration, n_steps, report_interval, filename):
    from mdtraj.reporters import HDF5Reporter
    integrator = integrators.LangevinIntegrator()
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(configuration)

    #equilibrate a little bit:
    simulation.step(10000)

    reporter = HDF5Reporter(filename, report_interval)
    simulation.reporters.append(reporter)
    simulation.step(n_steps)

def generate_solvated_topology_proposals(mol_a, mol_b):
    top_prop, cpos, npos = generate_solvated_hybrid_test_topology(current_mol_name=mol_a, proposed_mol_name=mol_b)
    reverse_top_prop = TopologyProposal(new_topology=top_prop.old_topology, new_system=top_prop.old_system,
                                        old_topology=top_prop.new_topology, old_system=top_prop.new_system,
                                        logp_proposal=0, new_to_old_atom_map=top_prop.old_to_new_atom_map, old_chemical_state_key=mol_b, new_chemical_state_key=mol_a)

    return top_prop, reverse_top_prop, cpos, npos

def traj_frame_to_sampler_state(traj: md.Trajectory, frame_number: int):
    xyz = traj.xyz[frame_number, :, :]
    box_vectors = traj.openmm_boxes(frame_number)
    sampler_state = states.SamplerState(unit.Quantity(xyz, unit=unit.nanometers), box_vectors=box_vectors)
    return sampler_state

def run_rj_proposals(top_prop, configuration_traj, use_sterics, ncmc_nsteps, n_replicates):
    ncmc_engine = NCMCEngine(nsteps=ncmc_nsteps, pressure=1.0*unit.atmosphere)
    geometry_engine = FFAllAngleGeometryEngine(use_sterics=use_sterics)
    initial_thermodynamic_state = states.ThermodynamicState(top_prop.old_system, temperature=temperature, pressure=1.0*unit.atmosphere)
    final_thermodynamic_state = states.ThermodynamicState(top_prop.new_system, temperature=temperature, pressure=1.0*unit.atmosphere)
    traj_indices = np.arange(0, configuration_traj.n_frames)
    results = np.zeros([n_replicates, 4])

    for i in tqdm.trange(n_replicates):
        frame_index = np.random.choice(traj_indices)
        initial_sampler_state = traj_frame_to_sampler_state(configuration_traj, frame_index)

        initial_logP = - compute_reduced_potential(initial_thermodynamic_state, initial_sampler_state)

        proposed_geometry, logP_geometry_forward = geometry_engine.propose(top_prop, initial_sampler_state.positions, beta)

        proposed_sampler_state = states.SamplerState(proposed_geometry, box_vectors=initial_sampler_state.box_vectors)

        final_old_sampler_state, final_sampler_state, logP_work, initial_hybrid_logP, final_hybrid_logP = ncmc_engine.integrate(top_prop, initial_sampler_state, proposed_sampler_state)

        final_logP = - compute_reduced_potential(final_thermodynamic_state, final_sampler_state)

        logP_reverse = geometry_engine.logp_reverse(top_prop, final_sampler_state.positions, final_old_sampler_state.positions, beta)

        results[i, 0] = initial_hybrid_logP - initial_logP
        results[i, 1] = logP_reverse - logP_geometry_forward
        results[i, 2] = final_logP - final_hybrid_logP
        results[i, 3] = logP_work

    return results

if __name__=="__main__":
    import yaml
    import sys
    import itertools
    import os

    input_filename = sys.argv[1]
    equilibrium = False if sys.argv[2] == '0' else True

    if not equilibrium:
        index = int(sys.argv[3]) - 1 # Correct for being off by one

    with open(input_filename, 'r') as yamlfile:
        options_dict = yaml.load(yamlfile)

    equilibrium_filename_a = "{}_{}.h5".format(options_dict['traj_prefix'], options_dict['molecules'][0])
    equilibrium_filename_b = "{}_{}.h5".format(options_dict['traj_prefix'], options_dict['molecules'][1])
    top_prop_forward_filename = "{}_{}.h5".format(options_dict['traj_prefix'], "top_prop_forward.npy")
    top_prop_reverse_filename = "{}_{}.h5".format(options_dict['traj_prefix'], "top_prop_reverse.npy")

    #if we need to set up equilibrium, then generate the topology proposals and
    if equilibrium:

         #now generate the topology proposals:
         fwd_top_prop, reverse_top_prop, cpos, npos = generate_solvated_topology_proposals(options_dict['molecules'][0], options_dict['molecules'][1])

         #write out the topology proposals:
         np.save(top_prop_forward_filename, fwd_top_prop)
         np.save(top_prop_reverse_filename, reverse_top_prop)

         n_steps = options_dict['eq_time'] * 1000 # timestep is 1fs, but we express time in ps
         report_interval = options_dict['eq_write_interval'] * 1000 # convert from ps -> fs again

         #run the equilibrium
         run_equilibrium(fwd_top_prop.old_system, fwd_top_prop.old_topology, cpos, n_steps, report_interval, equilibrium_filename_a)
         run_equilibrium(fwd_top_prop.new_system, fwd_top_prop.new_topology, npos, n_steps, report_interval, equilibrium_filename_b)

    # Otherwise, we want to run nonequilibrium from the equilibrium samples
    else:
        configuration_traj_a = md.load(equilibrium_filename_a)
        configuration_traj_b = md.load(equilibrium_filename_b)

        fwd_top_prop = np.load(top_prop_forward_filename).item()
        reverse_top_prop = np.load(top_prop_reverse_filename).item()

        n_replicates_neq = options_dict['n_replicates_neq']
        lengths = options_dict['lengths']
        use_sterics = options_dict['use_sterics']
        top_props = [fwd_top_prop, reverse_top_prop]
        config_trajs = [configuration_traj_a, configuration_traj_b]

        import itertools
        parameters = list(itertools.product([top_props[0]], [config_trajs[0]], use_sterics, lengths, [n_replicates_neq]))
        parameters.extend(list(itertools.product([top_props[1]], [config_trajs[1]], use_sterics, lengths, [n_replicates_neq])))

        parms_to_run = parameters[index]

        results = run_rj_proposals(parms_to_run[0], parms_to_run[1], parms_to_run[2], parms_to_run[3], parms_to_run[4])

        np.save("ncmc_{}_{}_{}_{}.npy".format(parms_to_run[0].old_chemical_state_key, parms_to_run[0].new_chemical_state_key, parms_to_run[2], parms_to_run[3]), results)







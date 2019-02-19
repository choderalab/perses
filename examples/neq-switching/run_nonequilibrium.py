from perses.annihilation.ncmc_switching import NCMCEngine
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, SystemGenerator, SmallMoleculeAtomMapper, PremappedSmallMoleculeSetProposalEngine
from openmmtools import states, constants
import tqdm
from simtk import openmm, unit
from simtk.openmm import app
import mdtraj as md
from perses.dispersed.feptasks import compute_reduced_potential
import numpy as np

def traj_frame_to_sampler_state(traj: md.Trajectory, frame_number: int,box_vectors):
    xyz = traj.xyz[frame_number, :, :]
    box_vectors = traj.openmm_boxes(frame_number)
    sampler_state = states.SamplerState(unit.Quantity(xyz, unit=unit.nanometers))
    return sampler_state

def run_rj_proposals(top_prop, configuration_traj, use_sterics, ncmc_nsteps, n_replicates, box_vectors, temperature=300.0*unit.kelvin):
    ncmc_engine = NCMCEngine(nsteps=ncmc_nsteps, pressure=1.0*unit.atmosphere)
    geometry_engine = FFAllAngleGeometryEngine(use_sterics=use_sterics)
    initial_thermodynamic_state = states.ThermodynamicState(top_prop.old_system, temperature=temperature, pressure=1.0*unit.atmosphere)
    final_thermodynamic_state = states.ThermodynamicState(top_prop.new_system, temperature=temperature, pressure=1.0*unit.atmosphere)
    traj_indices = np.arange(0, configuration_traj.n_frames)
    results = np.zeros([n_replicates, 7])
    beta = 1.0 / (temperature * constants.kB)

    for i in tqdm.trange(n_replicates):
        frame_index = np.random.choice(traj_indices)

        initial_sampler_state = traj_frame_to_sampler_state(configuration_traj, frame_index,box_vectors)
        initial_logP = - compute_reduced_potential(initial_thermodynamic_state, initial_sampler_state)

        proposed_geometry, logP_geometry_forward = geometry_engine.propose(top_prop, initial_sampler_state.positions, beta)

        proposed_sampler_state = states.SamplerState(proposed_geometry, box_vectors=initial_sampler_state.box_vectors)

        final_old_sampler_state, final_sampler_state, logP_work, initial_hybrid_logP, final_hybrid_logP = ncmc_engine.integrate(top_prop, initial_sampler_state, proposed_sampler_state)

        final_logP = - compute_reduced_potential(final_thermodynamic_state, final_sampler_state)

        logP_reverse = geometry_engine.logp_reverse(top_prop, final_sampler_state.positions, final_old_sampler_state.positions, beta)

        results[i, 0] = initial_logP
        results[i, 1] = logP_reverse
        results[i, 2] = final_logP
        results[i, 3] = logP_work
        results[i, 4] = initial_hybrid_logP
        results[i, 5] = final_hybrid_logP
        results[i, 6] = logP_geometry_forward

    return results


if __name__=="__main__":
    import sys
    import yaml
    import itertools
    import os
    from openeye import oechem

    yaml_filename = sys.argv[1]
    job_index = int(sys.argv[2])
    n_maps = int(sys.argv[3])
    map_index = job_index % n_maps
    molecule_index = job_index // n_maps

    with open(yaml_filename, "r") as yaml_file:
        options = yaml.load(yaml_file)

    setup_options = options['setup']
    equilibrium_options = options['equilibrium']
    nonequilibrium_options = options['nonequilibrium']

    n_ligands = nonequilibrium_options['n_ligands']
    equilibrium_output_directory = equilibrium_options['output_directory']
    project_prefix = setup_options['project_prefix']
    ncmc_nsteps = nonequilibrium_options['ncmc_length']
    n_replicates = nonequilibrium_options['n_attempts']
    nonequilibrium_output_directory = nonequilibrium_options['output_directory']
    setup_directory = setup_options['output_directory']
    ligand_filename = setup_options['ligand_filename']

    n_ligand_range = list(range(n_ligands))
    ligand_permutations = list(itertools.permutations(n_ligand_range, 2))

    ligand_pair_to_compute = ligand_permutations[molecule_index]

    initial_ligand = ligand_pair_to_compute[0]
    proposal_ligand = ligand_pair_to_compute[1]
    use_sterics = False
    temperature = 300.0*unit.kelvin


    equilibrium_snapshots_filename = os.path.join(equilibrium_output_directory, "{}_{}.h5".format(project_prefix, initial_ligand))
    configuration_traj = md.load(equilibrium_snapshots_filename)

    file_to_read = os.path.join(setup_directory, "{}_{}_initial.npy".format(project_prefix, initial_ligand))

    positions, topology, system, initial_smiles = np.load(file_to_read)
    topology = topology.to_openmm()
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    ifs = oechem.oemolistream()
    ifs.open(ligand_filename)

    # get the list of molecules
    mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]

    for idx, mol in enumerate(mol_list):
        mol.SetTitle("MOL{}".format(idx))
        oechem.OETriposAtomNames(mol)

    initial_mol = mol_list[initial_ligand]
    proposal_mol = mol_list[proposal_ligand]
    proposal_smiles = SmallMoleculeSetProposalEngine.canonicalize_smiles(oechem.OECreateCanSmiString(proposal_mol))
    current_smiles = SmallMoleculeSetProposalEngine.canonicalize_smiles(oechem.OECreateCanSmiString(initial_mol))

    barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, temperature, 50)

    system_generator = SystemGenerator(['amber14/protein.ff14SB.xml', 'gaff.xml', 'amber14/tip3p.xml', 'MCL1_ligands.xml'], barostat=barostat, forcefield_kwargs={'nonbondedMethod': app.PME,
                                                                        'constraints': app.HBonds,
                                                                        'hydrogenMass': 4 * unit.amus}, use_antechamber=False)

    atom_mapper_filename = os.path.join(setup_directory, "{}_atom_mapper.json".format(project_prefix))
    with open(atom_mapper_filename, 'r') as infile:
        atom_mapper = SmallMoleculeAtomMapper.from_json(infile.read())

    proposal_engine = PremappedSmallMoleculeSetProposalEngine(atom_mapper, system_generator)

    topology_proposal = proposal_engine.propose(system, topology, current_smiles=current_smiles, proposed_mol=proposal_mol, map_index=map_index)

    results = run_rj_proposals(topology_proposal, configuration_traj, use_sterics, ncmc_nsteps, n_replicates, system.getDefaultPeriodicBoxVectors() , temperature=temperature)

    if not os.path.exists(nonequilibrium_output_directory):
        os.mkdir(nonequilibrium_output_directory)

    np.save(os.path.join(nonequilibrium_output_directory, "{}_{}_{}.npy".format(project_prefix, initial_ligand, proposal_ligand)), results)


import numpy as np
import os
import tqdm
from openeye import oechem, oeiupac
from openmmtools import integrators, states, mcmc, constants
from openmoltools import forcefield_generators
from perses.rjmc.topology_proposal import TopologyProposal, SystemGenerator
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.annihilation.ncmc_switching import NCMCEngine
from perses.tests.utils import extractPositionsFromOEMOL
from simtk import openmm, unit
from io import StringIO
from simtk.openmm import app
import copy
from perses.dispersed.feptasks import compute_reduced_potential
import mdtraj as md

temperature = 300.0*unit.kelvin
beta = 1.0 / (temperature*constants.kB)

def generate_complex_topologies_and_positions(ligand_filename, protein_pdb_filename):

    ifs = oechem.oemolistream()
    ifs.open(ligand_filename)

    # get the list of molecules
    mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]

    mol_dict = {oechem.OEMolToSmiles(mol) : mol for mol in mol_list}

    ligand_topology_dict = {smiles : forcefield_generators.generateTopologyFromOEMol(mol) for smiles, mol in mol_dict}


    protein_pdbfile = open(protein_pdb_filename, 'r')
    pdb_file = app.PDBFile(protein_pdbfile)
    protein_pdbfile.close()
    receptor_positions = pdb_file.positions
    receptor_topology = pdb_file.topology
    receptor_md_topology = md.Topology.from_openmm(receptor_topology)

    n_receptor_atoms = receptor_md_topology.n_atoms

    complex_topologies = {}
    complex_positions = {}

    for smiles, ligand_topology in ligand_topology_dict.items():
        ligand_md_topology = md.Topology.from_openmm(ligand_topology)

        n_complex_atoms = ligand_md_topology.n_atoms + n_receptor_atoms
        copy_receptor_md_topology = copy.deepcopy(receptor_md_topology)

        complex_positions = unit.Quantity(np.array([n_complex_atoms, 3]), unit=unit.nanometers)

        complex_topology = copy_receptor_md_topology.join(ligand_md_topology)

        complex_topologies[smiles] = complex_topology

        ligand_positions = extractPositionsFromOEMOL(mol_dict[smiles])

        complex_positions[:n_receptor_atoms, :] = receptor_positions
        complex_positions[n_receptor_atoms:, :] = ligand_positions

        complex_positions[smiles] = complex_positions

    return complex_topologies, complex_positions

def solvate_system(topology, positions, system_generator, padding=9.0 * unit.angstrom, num_added=None, water_model='tip3p'):

    modeller = app.Modeller(topology, positions)
    modeller.addSolvent(system_generator._forcefield, model=water_model, padding=padding, numAdded=num_added)

    solvated_topology = modeller.topology
    solvated_positions = modeller.positions

    solvated_system = system_generator.build_system(solvated_topology)

    return solvated_positions, solvated_topology, solvated_system

def create_solvated_complex_systems(protein_pdb_filename, ligand_filename, output_directory, project_prefix):

    barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, temperature, 50)

    system_generator = SystemGenerator(['amber14/protein.ff14SB.xml', 'gaff.xml', 'amber14/tip3p.xml'], barostat=barostat, forcefield_kwargs={'nonbondedMethod': app.PME,
                                                                        'constraints': app.HBonds,
                                                                        'hydrogenMass': 4 * unit.amus})

    complex_topologies, complex_positions = generate_complex_topologies_and_positions(ligand_filename, protein_pdb_filename)

    list_of_smiles = list(complex_topologies.keys())

    initial_smiles = list_of_smiles[0]

    initial_topology = complex_topologies[initial_smiles]
    initial_positions = complex_positions[initial_smiles]

    solvated_initial_positions, solvated_topology, solvated_system = solvate_system(initial_topology, initial_positions, system_generator)

    md_topology = md.Topology.from_openmm(solvated_topology)

    num_added = md_topology.n_residues - initial_topology.n_residues

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    np.save("{}_{}_initial.npy".format(project_prefix, 0), (solvated_initial_positions, md_topology, solvated_system))

    for i in tqdm.trange(1, len(list_of_smiles)):

        smiles = list_of_smiles[i]

        topology = complex_topologies[smiles]
        positions = complex_positions[smiles]

        solvated_positions, solvated_topology, solvated_system = solvate_system(topology, positions, system_generator, padding=None, num_added=num_added)

        np.save("{}_{}_initial.npy".format(project_prefix, i),
                (solvated_positions, md.Topology.from_openmm(solvated_topology), solvated_system))

if __name__=="__main__":
    import sys
    import yaml

    yaml_filename = sys.argv[1]

    with open(yaml_filename, "r") as yaml_file:
        options = yaml.load(yaml_file)

    setup_options = options['setup']

    ligand_filename = setup_options['ligand_filename']
    protein_pdb_filename = setup_options['protein_pdb_filename']
    project_prefix = setup_options['project_prefix']
    output_directory = setup_options['output_directory']

    create_solvated_complex_systems(protein_pdb_filename, ligand_filename, output_directory, project_prefix)
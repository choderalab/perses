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
    """
    Generate the topologies and positions for complex phase simulations, given an input ligand file (in supported openeye
    format) and protein pdb file. Note that the input ligand file should have coordinates placing the ligand in the binding
    site.

    Parameters
    ----------
    ligand_filename : str
        Name of the file containing ligands
    protein_pdb_filename : str
        Name of the protein pdb file

    Returns
    -------
    complex_topologies_dict : dict of smiles: md.topology
        Dictionary of topologies for various complex systems
    complex_positions_dict : dict of smiles:  [n, 3] array of Quantity
        Positions for corresponding complexes
    """
    ifs = oechem.oemolistream()
    ifs.open(ligand_filename)

    # get the list of molecules
    mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]

    for idx, mol in enumerate(mol_list):
        mol.SetTitle("MOL{}".format(idx))
        oechem.OETriposAtomNames(mol)

    mol_dict = {oechem.OEMolToSmiles(mol) : mol for mol in mol_list}

    ligand_topology_dict = {smiles : forcefield_generators.generateTopologyFromOEMol(mol) for smiles, mol in mol_dict.items()}


    protein_pdbfile = open(protein_pdb_filename, 'r')
    pdb_file = app.PDBFile(protein_pdbfile)
    protein_pdbfile.close()
    receptor_positions = pdb_file.positions
    receptor_topology = pdb_file.topology
    receptor_md_topology = md.Topology.from_openmm(receptor_topology)

    n_receptor_atoms = receptor_md_topology.n_atoms

    complex_topologies = {}
    complex_positions_dict = {}

    for smiles, ligand_topology in ligand_topology_dict.items():
        ligand_md_topology = md.Topology.from_openmm(ligand_topology)

        n_complex_atoms = ligand_md_topology.n_atoms + n_receptor_atoms
        copy_receptor_md_topology = copy.deepcopy(receptor_md_topology)

        complex_positions = unit.Quantity(np.zeros([n_complex_atoms, 3]), unit=unit.nanometers)

        complex_topology = copy_receptor_md_topology.join(ligand_md_topology)

        complex_topologies[smiles] = complex_topology

        ligand_positions = extractPositionsFromOEMOL(mol_dict[smiles])

        complex_positions[:n_receptor_atoms, :] = receptor_positions
        complex_positions[n_receptor_atoms:, :] = ligand_positions

        complex_positions_dict[smiles] = complex_positions

    return complex_topologies, complex_positions_dict

def generate_ligand_topologies_and_positions(ligand_filename):
    """

    :param ligand_filename:
    :return:
    """
    ifs = oechem.oemolistream()
    ifs.open(ligand_filename)

    # get the list of molecules
    mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]

    for idx, mol in enumerate(mol_list):
        mol.SetTitle("MOL{}".format(idx))
        oechem.OETriposAtomNames(mol)

    mol_dict = {oechem.OEMolToSmiles(mol) : mol for mol in mol_list}

    ligand_topology_dict = {smiles : forcefield_generators.generateTopologyFromOEMol(mol) for smiles, mol in mol_dict.items()}

    ligand_topologies = {}
    ligand_positions_dict = {}

    for smiles, ligand_topology in ligand_topology_dict.items():
        ligand_md_topology = md.Topology.from_openmm(ligand_topology)

        ligand_topologies[smiles] = ligand_md_topology

        ligand_positions = extractPositionsFromOEMOL(mol_dict[smiles])

        ligand_positions_dict[smiles] = ligand_positions

    return ligand_topologies, ligand_positions_dict


def solvate_system(topology, positions, system_generator, padding=9.0 * unit.angstrom, num_added=None, water_model='tip3p'):

    modeller = app.Modeller(topology, positions)

    hs = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H'] and atom.residue.name[:3] != "MOL"]
    modeller.delete(hs)
    modeller.addHydrogens(forcefield=system_generator._forcefield)

    modeller.addSolvent(system_generator._forcefield, model=water_model, padding=padding, numAdded=num_added)

    solvated_topology = modeller.topology
    solvated_positions = modeller.positions

    solvated_system = system_generator.build_system(solvated_topology)

    return solvated_positions, solvated_topology, solvated_system

def create_systems(topologies_dict, positions_dict, output_directory, project_prefix, solvate=True):

    barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, temperature, 50)

    system_generator = SystemGenerator(['amber14/protein.ff14SB.xml', 'gaff.xml', 'amber14/tip3p.xml', 'MCL1_ligands.xml'], barostat=barostat, forcefield_kwargs={'nonbondedMethod': app.PME,
                                                                        'constraints': app.HBonds,
                                                                        'hydrogenMass': 4 * unit.amus}, use_antechamber=False)

    list_of_smiles = list(topologies_dict.keys())

    initial_smiles = list_of_smiles[0]

    initial_topology = topologies_dict[initial_smiles]
    initial_positions = positions_dict[initial_smiles]

    if solvate:
        solvated_initial_positions, solvated_topology, solvated_system = solvate_system(initial_topology.to_openmm(), initial_positions, system_generator)
    else:
        solvated_initial_positions = initial_positions
        solvated_topology = initial_topology
        solvated_system = system_generator.build_system(solvated_topology)

    md_topology = md.Topology.from_openmm(solvated_topology)

    if solvate:
        num_added = md_topology.n_residues - initial_topology.n_residues

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    np.save("{}_{}_initial.npy".format(project_prefix, 0), (solvated_initial_positions, md_topology, solvated_system))

    for i in tqdm.trange(1, len(list_of_smiles)):

        smiles = list_of_smiles[i]

        topology = topologies_dict[smiles]
        positions = positions_dict[smiles]

        if solvate:
            solvated_positions, solvated_topology, solvated_system = solvate_system(topology.to_openmm(), positions, system_generator, padding=None, num_added=num_added)
        else:
            solvated_positions = initial_positions
            solvated_topology = initial_topology
            solvated_system = system_generator.build_system(solvated_topology)

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

    if setup_options['phase'] == 'complex':
        topologies, positions = generate_complex_topologies_and_positions(ligand_filename,protein_pdb_filename)

    elif setup_options['phase'] == 'solvent':
        topologies, positions = generate_complex_topologies_and_positions(ligand_filename, protein_pdb_filename)

    else:
        raise ValueError("Phase must be either complex or solvent.")

    create_systems(topologies, positions, output_directory, project_prefix)
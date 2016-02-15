import simtk.openmm.app as app
import simtk.openmm as openmm
import simtk.unit as unit
import copy
from pkg_resources import resource_filename
import numpy as np
import os
try:
    from urllib.request import urlopen
    from io import StringIO
except:
    from urllib2 import urlopen
    from cStringIO import StringIO

temperature = 300*unit.kelvin
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
# Compute kT and inverse temperature.
kT = kB * temperature
beta = 1.0 / kT

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing
    In the source distribution, these files are in ``perses/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the openmoltools folder).
    """

    fn = resource_filename('perses', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

def extractPositionsFromOEMOL(molecule):
    positions = unit.Quantity(np.zeros([molecule.NumAtoms(), 3], np.float32), unit.angstroms)
    coords = molecule.GetCoords()
    for index in range(molecule.NumAtoms()):
        positions[index,:] = unit.Quantity(coords[index], unit.angstroms)
    return positions

def generate_initial_molecule(mol_smiles):
    """
    Generate an oemol with a geometry
    """
    import openeye.oechem as oechem
    import openeye.oeomega as oeomega
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, mol_smiles)
    mol.SetTitle("MOL")
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega(mol)
    return mol

def oemol_to_omm_ff(oemol, molecule_name):
    from perses.rjmc import topology_proposal
    from openmoltools import forcefield_generators
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])
    topology = forcefield_generators.generateTopologyFromOEMol(oemol)
    system = system_generator.build_system(topology)
    positions = extractPositionsFromOEMOL(oemol)
    return system, positions, topology

def test_small_molecule_proposals():
    """
    Make sure the small molecule proposal engine generates molecules
    """
    from perses.rjmc import topology_proposal
    from openmoltools import forcefield_generators
    import openeye.oechem as oechem
    list_of_smiles = ['CCC','CCCC','CCCCC']
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    stats_dict = {smiles : 0 for smiles in list_of_smiles}
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])
    proposal_engine = topology_proposal.SmallMoleculeSetProposalEngine(list_of_smiles, app.Topology(), system_generator)
    initial_molecule = generate_initial_molecule('CCC')
    initial_system, initial_positions, initial_topology = oemol_to_omm_ff(initial_molecule, "MOL")
    proposal = proposal_engine.propose(initial_system, initial_topology)
    for i in range(50):
        #positions are ignored here, and we don't want to run the geometry engine
        new_proposal = proposal_engine.propose(proposal.old_system, proposal.old_topology)
        stats_dict[new_proposal.new_chemical_state_key] += 1
        #check that the molecule it generated is actually the smiles we expect
        matching_molecules = [res for res in proposal.new_topology.residues() if res.name=='MOL']
        if len(matching_molecules) != 1:
            raise ValueError("More than one residue with the same name!")
        mol_res = matching_molecules[0]
        oemol = forcefield_generators.generateOEMolFromTopologyResidue(mol_res)
        assert oechem.OEMolToSmiles(oemol) == proposal.new_chemical_state_key
        proposal = new_proposal

def load_pdbid_to_openmm(pdbid):
    """
    create openmm topology without pdb file
    lifted from pandegroup/pdbfixer
    """
    url = 'http://www.rcsb.org/pdb/files/%s.pdb' % pdbid
    file = urlopen(url)
    contents = file.read().decode('utf-8')
    file.close()
    file = StringIO(contents)

    if _guessFileFormat(file, url) == 'pdbx':
        pdbx = app.PDBxFile(contents)
        topology = pdbx.topology
        positions = pdbx.positions
    else:
        pdb = app.PDBFile(file)
        topology = pdb.topology
        positions = pdb.positions

    return topology, positions

def _guessFileFormat(file, filename):
    """
    Guess whether a file is PDB or PDBx/mmCIF based on its filename and contents.
    authored by pandegroup
    """
    filename = filename.lower()
    if '.pdbx' in filename or '.cif' in filename:
        return 'pdbx'
    if '.pdb' in filename:
        return 'pdb'
    for line in file:
        if line.startswith('data_') or line.startswith('loop_'):
            file.seek(0)
            return 'pdbx'
        if line.startswith('HEADER') or line.startswith('REMARK') or line.startswith('TITLE '):
            file.seek(0)
            return 'pdb'
    file.seek(0)
    return 'pdb'

def test_specify_allowed_mutants():
    """
    Make sure proposals can be made using optional argument allowed_mutations
    """
    import perses.rjmc.topology_proposal as topology_proposal

    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"
    max_point_mutants = 1

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    chain_id = 'A'
    allowed_mutations = [[('5','GLU')],[('5','ASN'),('14','PHE')]]

    system_generator = topology_proposal.SystemGenerator([ff_filename])

    pm_top_engine = topology_proposal.PointMutationEngine(system_generator, max_point_mutants, chain_id, allowed_mutations=allowed_mutations)
    ntrials = 10
    for trian in range(ntrials):
        pm_top_proposal = pm_top_engine.propose(system, modeller.topology)
        # Check to make sure no out-of-bounds atoms are present in new_to_old_atom_map
        natoms_old = pm_top_proposal.old_system.getNumParticles()
        natoms_new = pm_top_proposal.new_system.getNumParticles()
        if not set(pm_top_proposal.new_to_old_atom_map.values()).issubset(range(natoms_old)):
            msg = "Some old atoms in TopologyProposal.new_to_old_atom_map are not in span of old atoms (1..%d):\n" % natoms_old
            msg += str(pm_top_proposal.new_to_old_atom_map)
            raise Exception(msg)
        if not set(pm_top_proposal.new_to_old_atom_map.keys()).issubset(range(natoms_new)):
            msg = "Some new atoms in TopologyProposal.new_to_old_atom_map are not in span of old atoms (1..%d):\n" % natoms_new
            msg += str(pm_top_proposal.new_to_old_atom_map)
            raise Exception(msg)

def test_run_point_mutation_propose():
    """
    Propose a random mutation in insulin
    """
    import perses.rjmc.topology_proposal as topology_proposal

    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"
    max_point_mutants = 1

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    chain_id = 'A'

    system_generator = topology_proposal.SystemGenerator([ff_filename])

    pm_top_engine = topology_proposal.PointMutationEngine(system_generator, max_point_mutants, chain_id)
    pm_top_proposal = pm_top_engine.propose(system, modeller.topology)


def test_mutate_from_every_amino_to_every_other():
    """
    Make sure mutations are successfuly between every possible pair of before-and-after residues
    Mutate Ecoli F-ATPase alpha subunit to all 20 amino acids (test going FROM all possibilities)
    Mutate each residue to all 19 alternatives
    """
    import perses.rjmc.topology_proposal as topology_proposal

    aminos = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

    failed_mutants = 0

    pdbid = "2A7U"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"
    max_point_mutants = 1

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    chain_id = 'A'

    system_generator = topology_proposal.SystemGenerator([ff_filename])

    pm_top_engine = topology_proposal.PointMutationEngine(system_generator, max_point_mutants, chain_id)

    current_system = system
    current_topology = modeller.topology
    current_positions = modeller.positions

    old_topology = copy.deepcopy(current_topology)

    metadata = dict()
    for atom in modeller.topology.atoms():
        atom.old_index = atom.index

    for chain in modeller.topology.chains():
        if chain.id == chain_id:
            # num_residues : int
            num_residues = len(chain._residues)
            break
    for k, proposed_amino in enumerate(aminos):
        proposed_location = k+1
        index_to_new_residues = dict()
        atom_map = dict()
        original_residue = chain._residues[proposed_location]
        if original_residue.name == proposed_amino:
            continue
        index_to_new_residues[proposed_location] = proposed_amino
        if proposed_amino == 'HIS':
            his_state = ['HIE','HID']
            his_prob = np.array([0.5 for i in range(len(his_state))])
            his_choice = np.random.choice(range(len(his_state)),p=his_prob)
            index_to_new_residues[proposed_location] = his_state[his_choice]
        metadata['mutations'] = pm_top_engine._save_mutations(modeller, index_to_new_residues)
        residue_map = pm_top_engine._generate_residue_map(modeller, index_to_new_residues)
        modeller, missing_atoms = pm_top_engine._delete_excess_atoms(modeller, residue_map)
        modeller = pm_top_engine._add_new_atoms(modeller, missing_atoms, residue_map)
        for k, atom in enumerate(modeller.topology.atoms()):
            atom.index=k
            try:
                atom_map[atom.index] = atom.old_index
            except AttributeError:
                pass

    new_sequence = list()
    for residue in modeller.topology.residues():
        if residue.index == 0:
            continue
        if residue.index == (num_residues -1):
            continue
        new_sequence.append(residue.name)
    assert [new_sequence[i] == aminos[i] for i in range(len(aminos))]

    new_topology = modeller.topology
    new_system = pm_top_engine._ff.createSystem(new_topology)

    current_system = new_system
    modeller = app.Modeller(new_topology, modeller.positions)
    current_topology = modeller.topology
    current_positions = modeller.positions

    old_topology = copy.deepcopy(current_topology)

    old_chemical_state_key = pm_top_engine.compute_state_key(old_topology)

    for chain in modeller.topology.chains():
        if chain.id == chain_id:
            # num_residues : int
            num_residues = len(chain._residues)
            break
    for proposed_location in range(num_residues):
        if proposed_location == 0 or proposed_location == num_residues-1:
            continue
        matching_amino_found = 0
        for proposed_amino in aminos:
            index_to_new_residues = dict()
            atom_map = dict()
            original_residue = chain._residues[proposed_location]
            if original_residue.name == proposed_amino or ((original_residue.name == 'HIE' or original_residue.name == 'HID') and proposed_amino == 'HIS'):
                matching_amino_found+=1
                continue
            index_to_new_residues[proposed_location] = proposed_amino
            if proposed_amino == 'HIS':
                his_state = ['HIE','HID']
                his_prob = np.array([0.5 for i in range(len(his_state))])
                his_choice = np.random.choice(range(len(his_state)),p=his_prob)
                index_to_new_residues[proposed_location] = his_state[his_choice]

            current_modeller = copy.deepcopy(modeller)

            metadata['mutations'] = pm_top_engine._save_mutations(current_modeller, index_to_new_residues)
            residue_map = pm_top_engine._generate_residue_map(current_modeller, index_to_new_residues)
            for res_pair in residue_map:
                residue = res_pair[0]
                name = res_pair[1]
                assert residue.index in index_to_new_residues.keys()
                assert index_to_new_residues[residue.index] == name
                assert residue.name+'-'+str(residue.id)+'-'+name in metadata['mutations']

            current_modeller, missing_atoms = pm_top_engine._delete_excess_atoms(current_modeller, residue_map)
            current_modeller = pm_top_engine._add_new_atoms(current_modeller, missing_atoms, residue_map)
            for res_pair in residue_map:
                residue = res_pair[0]
                name = res_pair[1]
                assert residue.name == name

            for k, atom in enumerate(current_modeller.topology.atoms()):
                atom.index=k
                try:
                    atom_map[atom.index] = atom.old_index
                except AttributeError:
                    pass
            new_topology = current_modeller.topology

            assert len(metadata['mutations']) <= max_point_mutants

            templates = pm_top_engine._ff.getMatchingTemplates(new_topology)
            assert [templates[index].name == residue.name for index, (residue, name) in enumerate(residue_map)]

            new_chemical_state_key = pm_top_engine.compute_state_key(new_topology)
            new_system = pm_top_engine._system_generator.build_system(new_topology)
            pm_top_proposal = topology_proposal.TopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=current_system, old_chemical_state_key=old_chemical_state_key, new_chemical_state_key=new_chemical_state_key, logp_proposal=0.0, new_to_old_atom_map=atom_map, metadata=metadata)
        assert matching_amino_found == 1

def test_run_peptide_library_engine():
    """
    Test example system with peptide and library
    """
    import perses.rjmc.topology_proposal as topology_proposal

    failed_mutants = 0

    pdbid = "1G3F"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)

    chain_id = 'B'
    to_delete = list()
    for chain in modeller.topology.chains():
        if chain.id != chain_id:
            to_delete.append(chain)
    modeller.delete(to_delete)
    modeller.deleteWater()
    modeller.addHydrogens()

    ff_filename = "amber99sbildn.xml"

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)

    system_generator = topology_proposal.SystemGenerator([ff_filename])
    library = ['AVILMFYQP','RHKDESTNQ','STNQCFGPZ']

    pl_top_library = topology_proposal.PeptideLibraryEngine(system_generator, library, chain_id)
    pl_top_proposal = pl_top_library.propose(system, modeller.topology)

if __name__ == "__main__":
    test_run_point_mutation_propose()
    test_mutate_from_every_amino_to_every_other()
    test_specify_allowed_mutants()
    test_run_peptide_library_engine()
    test_small_molecule_proposals()

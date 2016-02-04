import simtk.openmm.app as app
import simtk.openmm as openmm
import copy
import numpy as np
try:
    from urllib.request import urlopen
    from io import StringIO
except:
    from urllib2 import urlopen
    from cStringIO import StringIO

def test_small_molecule_proposals():
    """
    Make sure the small molecule proposal engine generates molecules uniformly
    """
    from perses.rjmc import topology_proposal
    list_of_smiles = ['CCC','CCCC','CCCCC']
    stats_dict = {smiles : 0 for smiles in list_of_smiles}
    system_generator = topology_proposal.SystemGenerator(['gaff.xml'])
    proposal_engine = topology_proposal.SmallMoleculeSetProposalEngine(list_of_smiles, app.Topology(), system_generator)
    initial_system =
    for i in range(50):




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
    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"
    max_point_mutants = 1
    proposal_metadata = {'ffxmls':[ff_filename]}

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    metadata = {'chain_id' : 'A'}
    allowed_mutations = [[('5','GLU')],[('5','ASN'),('14','PHE')]]

    import perses.rjmc.topology_proposal as topology_proposal

    pm_top_engine = topology_proposal.PointMutationEngine(max_point_mutants,proposal_metadata, allowed_mutations=allowed_mutations)
    pm_top_proposal = pm_top_engine.propose(system, modeller.topology, modeller.positions, metadata)


def test_run_point_mutation_propose():
    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"
    max_point_mutants = 1
    proposal_metadata = {'ffxmls':[ff_filename]}

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    metadata = {'chain_id' : 'A'}

    import perses.rjmc.topology_proposal as topology_proposal

    pm_top_engine = topology_proposal.PointMutationEngine(max_point_mutants,proposal_metadata)
    pm_top_proposal = pm_top_engine.propose(system, modeller.topology, modeller.positions, metadata)

def test_run_point_mutation_engine():

    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"
    max_point_mutants = 1
    proposal_metadata = {'ffxmls':[ff_filename]}

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    metadata = {'chain_id' : 'A'}

    import perses.rjmc.topology_proposal as topology_proposal

    pm_top_engine = topology_proposal.PointMutationEngine(max_point_mutants,proposal_metadata)

    current_system = system
    current_topology = modeller.topology
    current_positions = modeller.positions

    old_topology = copy.deepcopy(current_topology)
    #atom_map = dict()

    chain_id = metadata['chain_id']
    for atom in modeller.topology.atoms():
        atom.old_index = atom.index


    #index_to_new_residues = dict()
    aminos = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
    # chain : simtk.openmm.app.topology.Chain
    for chain in modeller.topology.chains():
        if chain.id == chain_id:
            # num_residues : int
            num_residues = len(chain._residues)
            break
    for proposed_location in range(num_residues):
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
                # how to count bonds


            for k, atom in enumerate(current_modeller.topology.atoms()):
                try:
                    atom.index=k
                    atom_map[atom.index] = atom.old_index
                except AttributeError:
                    pass
            new_topology = current_modeller.topology

            assert len(metadata['mutations']) <= max_point_mutants

            new_system = pm_top_engine._ff.createSystem(new_topology)
            pm_top_proposal = topology_proposal.PolymerTopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=current_system, old_positions=current_positions, logp_proposal=0.0, new_to_old_atom_map=atom_map, metadata=metadata)
        assert matching_amino_found == 1

   # return pm_top_proposal

def test_mutate_from_every_amino_to_every_other():
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
    proposal_metadata = {'ffxmls':[ff_filename]}

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    metadata = {'chain_id' : 'A'}

    import perses.rjmc.topology_proposal as topology_proposal

    pm_top_engine = topology_proposal.PointMutationEngine(max_point_mutants,proposal_metadata)

    current_system = system
    current_topology = modeller.topology
    current_positions = modeller.positions

    old_topology = copy.deepcopy(current_topology)
    #atom_map = dict()

    chain_id = metadata['chain_id']
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
                # how to count bonds

            for k, atom in enumerate(current_modeller.topology.atoms()):
                atom.index=k
                try:
                    atom_map[atom.index] = atom.old_index
                except AttributeError:
                    pass
            new_topology = current_modeller.topology

            templates = pm_top_engine._ff.getMatchingTemplates(new_topology)
            assert [templates[index].name == residue.name for index, (residue, name) in enumerate(residue_map)]

            new_system = pm_top_engine._ff.createSystem(new_topology)
            pm_top_proposal = topology_proposal.PolymerTopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=current_system, old_positions=current_positions, logp_proposal=0.0, new_to_old_atom_map=atom_map, metadata=metadata)
        assert matching_amino_found == 1



if __name__ == "__main__":
    test_run_point_mutation_propose()
    test_run_point_mutation_engine()
    test_mutate_from_every_amino_to_every_other()
    test_specify_allowed_mutants()



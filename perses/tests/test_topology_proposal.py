import simtk.openmm.app as app
from simtk.openmm.app.internal.pdbstructure import PdbStructure
import copy
import numpy as np
try:
    from urllib.request import urlopen
    from io import StringIO
except:
    from urllib2 import urlopen
    from cStringIO import StringIO


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
        structure = PdbStructure(file)
        pdb = app.PDBFile(structure)
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
    # It's certainly not a valid PDBx/mmCIF.  Guess that it's a PDB.
    file.seek(0)
    return 'pdb'

def _matchResidue(res, template, bondedToAtom):
    atoms = list(res.atoms())
    if len(atoms) != len(template.atoms):
        return None
    matches = len(atoms)*[0]
    hasMatch = len(atoms)*[False]

    renumberAtoms = {}
    for i in range(len(atoms)):
        renumberAtoms[atoms[i].index] = i
    bondedTo = []
    externalBonds = []
    for atom in atoms:
        bonds = [renumberAtoms[x] for x in bondedToAtom[atom.index] if x in renumberAtoms]
        bondedTo.append(bonds)
        externalBonds.append(len([x for x in bondedToAtom[atom.index] if x not in renumberAtoms]))

    residueTypeCount = {}
    for i, atom in enumerate(atoms):
        key = (atom.element, len(bondedTo[i]), externalBonds[i])
        if key not in residueTypeCount:
            residueTypeCount[key] = 1
        residueTypeCount[key] += 1
    templateTypeCount = {}
    for i, atom in enumerate(template.atoms):
        key = (atom.element, len(atom.bondedTo), atom.externalBonds)
        if key not in templateTypeCount:
            templateTypeCount[key] = 1
        templateTypeCount[key] += 1
    if residueTypeCount != templateTypeCount:
        return None

    if _findAtomMatches(atoms, template, bondedTo, externalBonds, matches, hasMatch, 0):
        return matches
    return None


def _findAtomMatches(atoms, template, bondedTo, externalBonds, matches, hasMatch, position):
    if position == len(atoms):
        return True
    elem = atoms[position].element
    name = atoms[position].name
    for i in range(len(atoms)):
        atom = template.atoms[i]
        if ((atom.element is not None and atom.element == elem) or (atom.element is None and atom.name == name)) and not hasMatch[i] and len(atom.bondedTo) == len(bondedTo[position]) and atom.externalBonds == externalBonds[position]:
            # See if the bonds for this identification are consistent

            allBondsMatch = all((bonded > position or matches[bonded] in atom.bondedTo for bonded in bondedTo[position]))
            if allBondsMatch:
                # This is a possible match, so trying matching the rest of the residue.

                matches[position] = i
                hasMatch[i] = True
                if _findAtomMatches(atoms, template, bondedTo, externalBonds, matches, hasMatch, position+1):
                    return True
                hasMatch[i] = False
    return False

def _match_template_and_res_name(topology, residue_map, forcefield):
    """
    Using code from forcefield.py createSystem(), match a residue to the template that will be found when creating system
    assert that template name matches intended residue name

    Verify correct functioning of _delete_excess_atoms(), _to_delete(), _to_delete_bonds(), and _add_new_atoms() by 
    asserting a template is found and the template matches the intended residue
    
    *** make sure consistent for terminal residues (residue name = N/C + three letter code)
    """
    atom_count = 0
    for atom in  topology.atoms():
        atom_count += 1
    assert atom_count == topology._numAtoms
    bonded_to_atom = list()
    atom_list = list(topology.atoms())
    for atom in topology.atoms():
        bonded_to_atom.append(set())
    assert len(bonded_to_atom) == len(atom_list)
    for bond in topology.bonds():
        assert bond[0].index in range(topology._numAtoms)
        assert bond[1].index in range(topology._numAtoms)
        bonded_to_atom[bond[0].index].add(bond[1].index)
        bonded_to_atom[bond[1].index].add(bond[0].index)

    for residue, new_name in residue_map:
        assert residue in topology.residues()
        assert residue.name == new_name
        template = None
        matches = None
        signature = app.forcefield._createResidueSignature([atom.element for atom in residue.atoms()])
        assert signature in forcefield._templateSignatures.keys()
        if forcefield._templateSignatures.has_key(signature):
            for t in forcefield._templateSignatures[signature]:
                matches = _matchResidue(residue, t, bonded_to_atom)
#                matches = app.forcefield._matchResidue(residue, t, bonded_to_atom)
                if matches is not None:
                    template = t
                    break
        #[template, matches] = forcefield._getResidueTemplateMatches(residue, bonded_to_atom)
        if matches is None:
            for generator in forcefield._templateGenerators:
                print(generator)
                if generator(forcefield, residue):
                    [template, matches] = forcefield._getResidueTemplateMatches(residue, bonded_to_atom)
                    if matches is None:
                        raise Exception('The residue handler %s indicated it had correctly parameterized residue %s, but the generated template did not match the residue signature.' % (generator.__class__.__name__, str(res)))
                    else:
                        break
        assert template is not None
        assert template.name == residue.name

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
            try:
                atom.index=k
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

            _match_template_and_res_name(current_modeller.topology, residue_map, pm_top_engine._ff)

            for k, atom in enumerate(current_modeller.topology.atoms()):
                try:
                    atom.index=k
                    atom_map[atom.index] = atom.old_index
                except AttributeError:
                    pass
            new_topology = current_modeller.topology

            try:
                new_system = pm_top_engine._ff.createSystem(new_topology)
            except:
                failed_mutants+=1
                for atom1, atom2 in current_modeller.topology._bonds:
                    if atom1.residue == residue or atom2.residue == residue:
                        print(atom1.name, atom1.residue.name, atom2.name, atom2.residue.name)
                raise Exception

            new_system = pm_top_engine._ff.createSystem(new_topology)
            pm_top_proposal = topology_proposal.PolymerTopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=current_system, old_positions=current_positions, logp_proposal=0.0, new_to_old_atom_map=atom_map, metadata=metadata)
        assert matching_amino_found == 1



if __name__ == "__main__":
#    test_run_point_mutation_propose()
#    test_run_point_mutation_engine()
    test_mutate_from_every_amino_to_every_other()




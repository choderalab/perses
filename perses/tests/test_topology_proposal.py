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
from nose.plugins.attrib import attr
from openmmtools.constants import kB

temperature = 300*unit.kelvin
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
    from collections import defaultdict
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
    import openeye.oechem as oechem
    list_of_smiles = ['CCCC','CCCCC','CCCCCC']
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    stats_dict = defaultdict(lambda: 0)
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])
    proposal_engine = topology_proposal.SmallMoleculeSetProposalEngine(list_of_smiles, system_generator)
    initial_molecule = generate_initial_molecule('CCCC')
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
        smiles = SmallMoleculeSetProposalEngine.canonicalize_smiles(oechem.OEMolToSmiles(oemol))
        assert smiles == proposal.new_chemical_state_key
        proposal = new_proposal

def test_two_molecule_proposal_engine():
    """
    Test TwoMoleculeSetProposalEngine
    """
    # Create a proposal engine for butane -> pentane
    old_mol = generate_initial_molecule('CCCC')
    new_mol = generate_initial_molecule('CCCCC')
    from perses.rjmc import topology_proposal
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])
    from perses.rjmc.topology_proposal import TwoMoleculeSetProposalEngine
    proposal_engine = TwoMoleculeSetProposalEngine(old_mol, new_mol, system_generator)
    initial_system, initial_positions, initial_topology = oemol_to_omm_ff(old_mol, "MOL")
    # Propose a transformation
    proposal = proposal_engine.propose(initial_system, initial_topology)
    # Check proposal
    assert (proposal.new_system.getNumParticles() == 17), "new_system (pentane) should have 17 atoms (actual: %d)" % proposal.new_system.getNumParticles()
    assert (proposal.old_system.getNumParticles() == 14), "old_system (butane) should have 14 atoms (actual: %d)" % proposal.old_system.getNumParticles()
    assert len(proposal.new_alchemical_atoms) == 17, "new_alchemical_atoms should be 17 (actual: %d)" % proposal.new_alchemical_atoms
    assert len(proposal.old_alchemical_atoms) == 14, "old_alchemical_atoms should be 14 (actual: %d)" % proposal.old_alchemical_atoms
    assert len(proposal.new_environment_atoms) == 0, "new_environment_atoms should be 0 (actual: %d)" % proposal.new_environment_atoms
    assert len(proposal.old_environment_atoms) == 0, "old_environment_atoms should be 0 (actual: %d)" % proposal.old_environment_atoms

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

@attr('advanced')
def test_specify_allowed_mutants():
    """
    Make sure proposals can be made using optional argument allowed_mutations

    This test has three possible insulin systems: wild type, Q5E, and Q5N/Y14F
    """
    import perses.rjmc.topology_proposal as topology_proposal

    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    chain_id = 'A'

    allowed_mutations = [[('5','GLU')],[('5','ASN'),('14','PHE')]]

    system_generator = topology_proposal.SystemGenerator([ff_filename])

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, allowed_mutations=allowed_mutations)

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

@attr('advanced')
def test_propose_self():
    """
    Propose a mutation to remain at WT in insulin
    """
    import perses.rjmc.topology_proposal as topology_proposal

    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    ff_filename = "amber99sbildn.xml"

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    chain_id = 'A'

    for chain in modeller.topology.chains():
        if chain.id == chain_id:
            residues = chain._residues
    mutant_res = np.random.choice(residues)
    allowed_mutations = [[(mutant_res.id,mutant_res.name)]]
    system_generator = topology_proposal.SystemGenerator([ff_filename])

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, allowed_mutations=allowed_mutations, verbose=True)
    print('Self mutation:')
    pm_top_proposal = pm_top_engine.propose(system, modeller.topology)
    assert pm_top_proposal.old_topology == pm_top_proposal.new_topology
    assert pm_top_proposal.old_system == pm_top_proposal.new_system
    assert pm_top_proposal.old_chemical_state_key == pm_top_proposal.new_chemical_state_key

@attr('advanced')
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

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, max_point_mutants=max_point_mutants)
    pm_top_proposal = pm_top_engine.propose(system, modeller.topology)

@attr('advanced')
def test_alanine_dipeptide_map():
    pdb_filename = resource_filename('openmmtools', 'data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb')
    from simtk.openmm.app import PDBFile
    pdbfile = PDBFile(pdb_filename)
    import perses.rjmc.topology_proposal as topology_proposal
    modeller = app.Modeller(pdbfile.topology, pdbfile.positions)

    ff_filename = "amber99sbildn.xml"
    allowed_mutations = [[('2', 'PHE')]]

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)
    chain_id = ' '

    metadata = dict()
    system_generator = topology_proposal.SystemGenerator([ff_filename])
    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, proposal_metadata=metadata, allowed_mutations=allowed_mutations, always_change=True)

    proposal = pm_top_engine.propose(system, modeller.topology)

    new_topology = proposal.new_topology
    new_system = proposal.new_system
    old_topology = proposal.old_topology
    old_system = proposal.old_system
    atom_map = proposal.old_to_new_atom_map

    for k, atom in enumerate(old_topology.atoms()):
        atom_idx = atom.index
        if atom_idx in atom_map.keys():
            atom2_idx = atom_map[atom_idx]
            for l, atom2 in enumerate(new_topology.atoms()):
                if atom2.index == atom2_idx:
                    new_atom = atom2
                    break
            old_name = atom.name
            new_name = new_atom.name
            print('\n%s to %s' % (str(atom.residue), str(new_atom.residue)))
            print('old_atom.index vs index in topology: %s %s' % (atom_idx, k))
            print('new_atom.index vs index in topology: %s %s' % (atom2_idx, l))
            print('Who was matched: old %s to new %s' % (old_name, new_name))
            if atom2_idx != l:
                mass_by_map = system.getParticleMass(atom2_idx)
                mass_by_sys = system.getParticleMass(l)
                print('Should have matched %s actually got %s' % (mass_by_map, mass_by_sys))

@attr('advanced')
def test_mutate_from_every_amino_to_every_other():
    """
    Make sure mutations are successful between every possible pair of before-and-after residues
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

    metadata = dict()

    system_generator = topology_proposal.SystemGenerator([ff_filename])

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, proposal_metadata=metadata, max_point_mutants=max_point_mutants, always_change=True)

    current_system = system
    current_topology = modeller.topology
    current_positions = modeller.positions

    pm_top_engine._allowed_mutations = [list()]
    for k, proposed_amino in enumerate(aminos):
        pm_top_engine._allowed_mutations[0].append((str(k+2),proposed_amino))
    pm_top_proposal = pm_top_engine.propose(current_system, current_topology)
    current_system = pm_top_proposal.new_system
    current_topology = pm_top_proposal.new_topology

    for chain in current_topology.chains():
        if chain.id == chain_id:
            # num_residues : int
            num_residues = len(chain._residues)
            break
    new_sequence = list()
    for residue in current_topology.residues():
        if residue.index == 0:
            continue
        if residue.index == (num_residues -1):
            continue
        if residue.name in ['HID','HIE']:
            residue.name = 'HIS'
        new_sequence.append(residue.name)
    for i in range(len(aminos)):
        assert new_sequence[i] == aminos[i]


    pm_top_engine = topology_proposal.PointMutationEngine(current_topology, system_generator, chain_id, proposal_metadata=metadata, max_point_mutants=max_point_mutants)

    from perses.rjmc.topology_proposal import append_topology
    old_topology = app.Topology()
    append_topology(old_topology, current_topology)
    new_topology = app.Topology()
    append_topology(new_topology, current_topology)

    old_chemical_state_key = pm_top_engine.compute_state_key(old_topology)


    for chain in new_topology.chains():
        if chain.id == chain_id:
            # num_residues : int
            num_residues = len(chain._residues)
            break
    for proposed_location in range(1, num_residues-1):
        print('Making mutations at residue %s' % proposed_location)
        original_residue_name = chain._residues[proposed_location].name
        matching_amino_found = 0
        for proposed_amino in aminos:
            pm_top_engine._allowed_mutations = [[(str(proposed_location+1),proposed_amino)]]
            new_topology = app.Topology()
            append_topology(new_topology, current_topology)
            old_system = current_system
            old_topology_natoms = sum([1 for atom in old_topology.atoms()])
            old_system_natoms = old_system.getNumParticles()
            if old_topology_natoms != old_system_natoms:
                msg = 'PolymerProposalEngine: old_topology has %d atoms, while old_system has %d atoms' % (old_topology_natoms, old_system_natoms)
                raise Exception(msg)
            metadata = dict()

            for atom in new_topology.atoms():
                atom.old_index = atom.index

            index_to_new_residues, metadata = pm_top_engine._choose_mutant(new_topology, metadata)
            if len(index_to_new_residues) == 0:
                matching_amino_found+=1
                continue
            print('Mutating %s to %s' % (original_residue_name, proposed_amino))

            residue_map = pm_top_engine._generate_residue_map(new_topology, index_to_new_residues)
            for res_pair in residue_map:
                residue = res_pair[0]
                name = res_pair[1]
                assert residue.index in index_to_new_residues.keys()
                assert index_to_new_residues[residue.index] == name
                assert residue.name+'-'+str(residue.id)+'-'+name in metadata['mutations']

            new_topology, missing_atoms = pm_top_engine._delete_excess_atoms(new_topology, residue_map)
            new_topology = pm_top_engine._add_new_atoms(new_topology, missing_atoms, residue_map)
            for res_pair in residue_map:
                residue = res_pair[0]
                name = res_pair[1]
                assert residue.name == name

            atom_map = pm_top_engine._construct_atom_map(residue_map, old_topology, index_to_new_residues, new_topology)
            templates = pm_top_engine._ff.getMatchingTemplates(new_topology)
            assert [templates[index].name == residue.name for index, (residue, name) in enumerate(residue_map)]

            new_chemical_state_key = pm_top_engine.compute_state_key(new_topology)
            new_system = pm_top_engine._system_generator.build_system(new_topology)
            pm_top_proposal = topology_proposal.TopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=old_system, old_chemical_state_key=old_chemical_state_key, new_chemical_state_key=new_chemical_state_key, logp_proposal=0.0, new_to_old_atom_map=atom_map)

        assert matching_amino_found == 1

@attr('advanced')
def test_limiting_allowed_residues():
    """
    Test example system with certain mutations allowed to mutate
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
    modeller.addHydrogens()

    ff_filename = "amber99sbildn.xml"

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)

    system_generator = topology_proposal.SystemGenerator([ff_filename])

    max_point_mutants = 1
    residues_allowed_to_mutate = ['903','904','905']

    pl_top_library = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, max_point_mutants=max_point_mutants, residues_allowed_to_mutate=residues_allowed_to_mutate)
    pl_top_proposal = pl_top_library.propose(system, modeller.topology)

@attr('advanced')
def test_always_change():
    """
    Test 'always_change' argument in topology proposal
    Allowing one residue to mutate, must change to a different residue each
    of 50 iterations
    """
    import perses.rjmc.topology_proposal as topology_proposal

    pdbid = "1G3F"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)

    chain_id = 'B'
    to_delete = list()
    for chain in modeller.topology.chains():
        if chain.id != chain_id:
            to_delete.append(chain)
    modeller.delete(to_delete)
    modeller.addHydrogens()

    ff_filename = "amber99sbildn.xml"

    ff = app.ForceField(ff_filename)
    system = ff.createSystem(modeller.topology)

    system_generator = topology_proposal.SystemGenerator([ff_filename])

    max_point_mutants = 1
    residues_allowed_to_mutate = ['903']

    for residue in modeller.topology.residues():
        if residue.id in residues_allowed_to_mutate:
            print('Old residue: %s' % residue.name)
            old_res_name = residue.name
    pl_top_library = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, max_point_mutants=max_point_mutants, residues_allowed_to_mutate=residues_allowed_to_mutate, always_change=True)
    topology = modeller.topology
    for i in range(50):
        pl_top_proposal = pl_top_library.propose(system, topology)
        for residue in pl_top_proposal.new_topology.residues():
            if residue.id in residues_allowed_to_mutate:
                print('Iter %s New residue: %s' % (i, residue.name))
                new_res_name = residue.name
        assert(old_res_name != new_res_name)
        old_res_name = new_res_name
        topology = pl_top_proposal.new_topology
        system = pl_top_proposal.new_system

@attr('advanced')
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
    modeller.addHydrogens()

    ff_filename = "amber99sbildn.xml"

    ff = app.ForceField(ff_filename)
    ff.loadFile("tip3p.xml")
    modeller.addSolvent(ff)
    system = ff.createSystem(modeller.topology)

    system_generator = topology_proposal.SystemGenerator([ff_filename])
    library = ['AVILMFYQP','RHKDESTNQ','STNQCFGPL']

    pl_top_library = topology_proposal.PeptideLibraryEngine(system_generator, library, chain_id)

    pl_top_proposal = pl_top_library.propose(system, modeller.topology)

def test_molecular_atom_mapping(initial_smiles="naphthalene", final_smiles="benzene"):
    """
    Test the creation of atom maps between pairs of molecules.

    """
    from openeye import oechem
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
    from perses.tests.utils import createOEMolFromSMILES
    from perses.tests.utils import render_atom_mapping

    # Test mappings for JACS dataset ligands
    for dataset_name in ['CDK2', 'p38', 'Tyk2', 'Thrombin', 'PTP1B', 'MCL1', 'Jnk1', 'Bace']:
        # Read molecules
        dataset_path = 'data/schrodinger-jacs-datasets/%s_ligands.sdf' % dataset_name
        mol2_filename = resource_filename('perses', dataset_path)
        ifs = oechem.oemolistream(mol2_filename)
        molecules = list()
        for mol in ifs.GetOEGraphMols():
            molecules.append(oechem.OEGraphMol(mol))
        nmolecules = len(molecules)

        # Build atom map for transformations to molecule 0
        for index in range(1, nmolecules):
            # Build atom map
            molecule1 = molecules[0]
            molecule2 = molecules[index]
            atom_map = SmallMoleculeSetProposalEngine._get_mol_atom_map(molecule1, molecule2)
            # Make sure we aren't mapping hydrogens onto anything else
            atoms1 = [atom for atom in molecule1.GetAtoms()]
            atoms2 = [atom for atom in molecule2.GetAtoms()]
            for (index1, index2) in atom_map.items():
                atom1, atom2 = atoms1[index1], atoms2[index2]
                if (atom1.GetAtomicNum()==1) != (atom2.GetAtomicNum()==1):
                    filename = 'mapping-error.png'
                    render_atom_mapping(filename, molecule1, molecule2, atom_map)
                    msg = 'Atom atomic number %d is being mapped to atomic number %d\n' % (atom1.GetAtomicNum(), atom2.GetAtomicNum())
                    msg += 'molecule 1 : %s\n' % oechem.OECreateIsoSmiString(molecule1)
                    msg += 'molecule 2 : %s\n' % oechem.OECreateIsoSmiString(molecule2)
                    msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
                    msg += str(atom_map)
                    raise Exception(msg)

if __name__ == "__main__":

    test_run_point_mutation_propose()
#    test_mutate_from_every_amino_to_every_other()
#    test_specify_allowed_mutants()
#    test_propose_self()
#    test_limiting_allowed_residues()
#    test_run_peptide_library_engine()
#    test_small_molecule_proposals()
#    test_alanine_dipeptide_map()
#    test_always_change()

import simtk.openmm.app as app
import simtk.openmm as openmm
import simtk.unit as unit
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
from perses.utils.openeye import OEMol_to_omm_ff, smiles_to_oemol
from perses.utils.smallmolecules import render_atom_mapping
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
from perses.rjmc import topology_proposal
from collections import defaultdict
import openeye.oechem as oechem
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule
from openmoltools.forcefield_generators import generateOEMolFromTopologyResidue

#default arguments for SystemGenerators
barostat = None
forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus}
nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff}
small_molecule_forcefield = 'gaff-2.11'

temperature = 300*unit.kelvin
# Compute kT and inverse temperature.
kT = kB * temperature
beta = 1.0 / kT
ENERGY_THRESHOLD = 1e-6
PROHIBITED_RESIDUES = ['CYS']

running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

def test_small_molecule_proposals():
    """
    Make sure the small molecule proposal engine generates molecules
    """
    list_of_smiles = ['CCCC','CCCCC','CCCCCC']
    list_of_mols = []
    for smi in list_of_smiles:
        mol = smiles_to_oemol(smi)
        list_of_mols.append(mol)
    molecules = [Molecule.from_openeye(mol) for mol in list_of_mols]
    stats_dict = defaultdict(lambda: 0)
    system_generator = SystemGenerator(forcefields = forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs, nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                                         small_molecule_forcefield = small_molecule_forcefield, molecules=molecules, cache=None)
    proposal_engine = topology_proposal.SmallMoleculeSetProposalEngine(list_of_mols, system_generator)
    initial_system, initial_positions, initial_topology,  = OEMol_to_omm_ff(list_of_mols[0], system_generator)

    proposal = proposal_engine.propose(initial_system, initial_topology)

    for i in range(50):
        # Positions are ignored here, and we don't want to run the geometry engine
        new_proposal = proposal_engine.propose(proposal.old_system, proposal.old_topology)
        stats_dict[new_proposal.new_chemical_state_key] += 1
        # Check that the molecule it generated is actually the smiles we expect
        matching_molecules = [res for res in proposal.new_topology.residues() if res.name=='MOL']
        if len(matching_molecules) != 1:
            raise ValueError("More than one residue with the same name!")
        mol_res = matching_molecules[0]
        oemol = generateOEMolFromTopologyResidue(mol_res)
        smiles = SmallMoleculeSetProposalEngine.canonicalize_smiles(oechem.OEMolToSmiles(oemol))
        assert smiles == proposal.new_chemical_state_key
        proposal = new_proposal

def test_mapping_strength_levels(pairs_of_smiles=[('Cc1ccccc1','c1ccc(cc1)N'),('CC(c1ccccc1)','O=C(c1ccccc1)'),('Oc1ccccc1','Sc1ccccc1')],test=True):

    correct_results = {0:{'default': (3,2), 'weak':(3,2), 'strong':(4,3)},
                       1:{'default': (7,3), 'weak':(6,2), 'strong':(7,3)},
                       2:{'default': (1,1), 'weak':(1,1), 'strong':(2,2)}}

    mapping = ['weak','default','strong']

    for example in mapping:
        for index, (lig_a, lig_b) in enumerate(pairs_of_smiles):
            print(f"conducting {example} mapping with ligands {lig_a}, {lig_b}")
            initial_molecule = smiles_to_oemol(lig_a)
            proposed_molecule = smiles_to_oemol(lig_b)
            molecules = [Molecule.from_openeye(mol) for mol in [initial_molecule, proposed_molecule]]
            system_generator = SystemGenerator(forcefields = forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs,nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                                                 small_molecule_forcefield = 'gaff-1.81', molecules=molecules, cache=None)
            proposal_engine = SmallMoleculeSetProposalEngine([initial_molecule, proposed_molecule], system_generator)
            initial_system, initial_positions, initial_topology = OEMol_to_omm_ff(initial_molecule, system_generator)
            print(f"running now with map strength {example}")
            proposal = proposal_engine.propose(initial_system, initial_topology, map_strength = example)
            print(lig_a, lig_b,'length OLD and NEW atoms',len(proposal.unique_old_atoms), len(proposal.unique_new_atoms))
            if test:
                render_atom_mapping(f'{index}-{example}.png', initial_molecule, proposed_molecule, proposal._new_to_old_atom_map)
                assert ( (len(proposal.unique_old_atoms), len(proposal.unique_new_atoms)) == correct_results[index][example]), f"the mapping failed, correct results are {correct_results[index][example]}"
                print(f"the mapping worked!!!")
            print()


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

def create_simple_protein_system_generator():
    from openmmforcefields.generators import SystemGenerator
    barostat = None
    forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus}
    nonperiodic_forcefield_kwargs={'nonbondedMethod': app.NoCutoff}

    system_generator = SystemGenerator(forcefields = forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs, nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                                         small_molecule_forcefield = 'gaff-2.11', molecules=None, cache=None)
    return system_generator

def create_insulin_topology_engine(chain_id = 'A', allowed_mutations = None, pdbid = "2HIU"):
    import perses.rjmc.topology_proposal as topology_proposal
    from openmmforcefields.generators import SystemGenerator

    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass
    modeller.delete([chain])
    system_generator = create_simple_protein_system_generator()
    system = system_generator.create_system(modeller.topology)

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, allowed_mutations=allowed_mutations)

    return pm_top_engine, system, topology, modeller.positions


def generate_atp(phase = 'vacuum'):
    """
    modify the AlanineDipeptideVacuum test system to be parametrized with amber14ffsb in vac or solvent (tip3p)
    """
    import openmmtools.testsystems as ts
    from openmmforcefields.generators import SystemGenerator
    atp = ts.AlanineDipeptideVacuum(constraints = app.HBonds, hydrogenMass = 4 * unit.amus)


    forcefield_files = ['gaff.xml', 'amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']

    if phase == 'vacuum':
        barostat = None
        system_generator = SystemGenerator(forcefield_files,
                                       barostat=barostat,
                                       forcefield_kwargs={'removeCMMotion': False,
                                                            'ewaldErrorTolerance': 1e-4,
                                                            'constraints' : app.HBonds,
                                                            'hydrogenMass' : 4 * unit.amus},
                                        nonperiodic_forcefield_kwargs={'nonbondedMethod': app.NoCutoff},
                                        small_molecule_forcefield='gaff-2.11',
                                        molecules=None,
                                        cache=None)

        atp.system = system_generator.create_system(atp.topology) # Update the parametrization scheme to amberff14sb

    elif phase == 'solvent':
        barostat = openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 50)

        system_generator = SystemGenerator(forcefield_files,
                                   barostat=barostat,
                                   forcefield_kwargs={'removeCMMotion': False,
                                                        'ewaldErrorTolerance': 1e-4,
                                                        'constraints' : app.HBonds,
                                                        'hydrogenMass' : 4 * unit.amus},
                                    periodic_forcefield_kwargs={'nonbondedMethod': app.PME},
                                    small_molecule_forcefield='gaff-2.11',
                                    molecules=None,
                                    cache=None)

    if phase == 'solvent':
        modeller = app.Modeller(atp.topology, atp.positions)
        modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=9*unit.angstroms, ionicStrength=0.15*unit.molar)
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()

        # Canonicalize the solvated positions: turn tuples into np.array
        atp.positions = unit.quantity.Quantity(value=np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit=unit.nanometers)
        atp.topology = solvated_topology

        atp.system = system_generator.create_system(atp.topology)


    return atp, system_generator

def generate_dipeptide_top_pos_sys(topology,
                                   new_res,
                                   system,
                                   positions,
                                   system_generator,
                                   conduct_geometry_prop = True,
                                   conduct_htf_prop = False,
                                   validate_energy_bookkeeping=True,
                                   repartitioned=False,
                                   endstate=None,
                                   flatten_torsions=False,
                                   flatten_exceptions=False,
                                   validate_endstate_energy=True
                                   ):
    """generate point mutation engine, geometry_engine, and conduct topology proposal, geometry propsal, and hybrid factory generation"""
    from perses.tests.utils import validate_endstate_energies
    if conduct_htf_prop:
        assert conduct_geometry_prop, f"the htf prop can only be conducted if there is a geometry proposal"
    # Create the point mutation engine
    from perses.rjmc.topology_proposal import PointMutationEngine
    point_mutation_engine = PointMutationEngine(wildtype_topology=topology,
                                                system_generator=system_generator,
                                                chain_id='1', # Denote the chain id allowed to mutate (it's always a string variable)
                                                max_point_mutants=1,
                                                residues_allowed_to_mutate=['2'], # The residue ids allowed to mutate
                                                allowed_mutations=[('2', new_res)], # The residue ids allowed to mutate with the three-letter code allowed to change
                                                aggregate=True) # Always allow aggregation

    # Create a top proposal
    print(f"making topology proposal")
    topology_proposal = point_mutation_engine.propose(current_system=system, current_topology=topology)

    if not conduct_geometry_prop:
        return topology_proposal

    if conduct_geometry_prop:
        # Create a geometry engine
        print(f"generating geometry engine")
        from perses.rjmc.geometry import FFAllAngleGeometryEngine
        geometry_engine = FFAllAngleGeometryEngine(metadata=None,
                                               use_sterics=False,
                                               n_bond_divisions=100,
                                               n_angle_divisions=180,
                                               n_torsion_divisions=360,
                                               verbose=True,
                                               storage=None,
                                               bond_softening_constant=1.0,
                                               angle_softening_constant=1.0,
                                               neglect_angles = False,
                                               use_14_nonbondeds = True)


        # Make a geometry proposal forward
        print(f"making geometry proposal from {list(topology.residues())[1].name} to {new_res}")
        forward_new_positions, logp_proposal = geometry_engine.propose(topology_proposal, positions, beta, validate_energy_bookkeeping=validate_energy_bookkeeping)
        logp_reverse = geometry_engine.logp_reverse(topology_proposal, forward_new_positions, positions, beta, validate_energy_bookkeeping=validate_energy_bookkeeping)

    if not conduct_htf_prop:
        return (topology_proposal, forward_new_positions, logp_proposal, logp_reverse)

    if conduct_htf_prop:
        # Create a hybrid topology factory
        if not repartitioned:
            from perses.annihilation.relative import HybridTopologyFactory
            factory = HybridTopologyFactory
        else:
            from perses.annihilation.relative import RepartitionedHybridTopologyFactory
            factory = RepartitionedHybridTopologyFactory

        forward_htf = factory(topology_proposal=topology_proposal,
                     current_positions=positions,
                     new_positions=forward_new_positions,
                     use_dispersion_correction=False,
                     functions=None,
                     softcore_alpha=None,
                     bond_softening_constant=1.0,
                     angle_softening_constant=1.0,
                     soften_only_new=False,
                     neglected_new_angle_terms=[],
                     neglected_old_angle_terms=[],
                     softcore_LJ_v2=True,
                     softcore_electrostatics=True,
                     softcore_LJ_v2_alpha=0.85,
                     softcore_electrostatics_alpha=0.3,
                     softcore_sigma_Q=1.0,
                     interpolate_old_and_new_14s=flatten_exceptions,
                     omitted_terms=None,
                     endstate=endstate,
                     flatten_torsions=flatten_torsions)

        if not validate_endstate_energy:
            return forward_htf
        else:
            if not topology_proposal.unique_new_atoms:
                assert geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
                assert geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
                vacuum_added_valence_energy = 0.0
            else:
                added_valence_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential

            if not topology_proposal.unique_old_atoms:
                assert geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
                assert geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
                subtracted_valence_energy = 0.0
            else:
                subtracted_valence_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential

            zero_state_error, one_state_error = validate_endstate_energies(forward_htf._topology_proposal,
                                                                           forward_htf,
                                                                           added_valence_energy,
                                                                           subtracted_valence_energy,
                                                                           beta = 1.0/(kB*temperature),
                                                                           ENERGY_THRESHOLD = ENERGY_THRESHOLD,
                                                                           platform = openmm.Platform.getPlatformByName('Reference'),
                                                                           repartitioned_endstate=endstate)
            print(f"zero state error : {zero_state_error}")
            print(f"one state error : {one_state_error}")

            return forward_htf


def test_mutate_from_alanine():
    """
    generate alanine dipeptide system (vacuum) and mutating to every other amino acid as a sanity check...
    """
    # TODO: run the full pipeline for all of the aminos; at the moment, large perturbations (i.e. to ARG have the potential of
    #      generating VERY large nonbonded energies, to which numerical precision cannot achieve a proper threshold of 1e-6.
    #      in the future, we can look to use sterics or something fancy.  At the moment, we recommend conservative transforms
    #      or transforms that have more unique _old_ atoms than new
    aminos = ['ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','SER','THR','TRP','TYR','VAL']
    attempt_full_pipeline_aminos = ['CYS', 'ILE', 'SER', 'THR', 'VAL'] #let's omit rings and large perturbations for now

    ala, system_generator = generate_atp()

    for amino in aminos:
        if amino in attempt_full_pipeline_aminos:
            _ = generate_dipeptide_top_pos_sys(ala.topology, amino, ala.system, ala.positions, system_generator, conduct_htf_prop=True)
        else:
            _ = generate_dipeptide_top_pos_sys(ala.topology, amino, ala.system, ala.positions, system_generator, conduct_geometry_prop=False)

#@attr('advanced')
def test_specify_allowed_mutants():
    """
    Make sure proposals can be made using optional argument allowed_mutations

    This test has three possible insulin systems: wild type, Q5E, and Q5N/Y14F
    """
    chain_id = 'A'
    allowed_mutations = [('5','GLU'),('5','ASN'),('14','PHE')]
    import perses.rjmc.topology_proposal as topology_proposal

    pdbid = "2HIU"
    topology, positions = load_pdbid_to_openmm(pdbid)
    modeller = app.Modeller(topology, positions)
    for chain in modeller.topology.chains():
        pass

    modeller.delete([chain])

    system_generator = create_simple_protein_system_generator()

    system = system_generator.create_system(modeller.topology)
    chain_id = 'A'

    for chain in modeller.topology.chains():
        if chain.id == chain_id:
            residues = chain._residues
    mutant_res = np.random.choice(residues[1:-1])

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, allowed_mutations=allowed_mutations)


    ntrials = 10
    for trian in range(ntrials):
        pm_top_proposal = pm_top_engine.propose(system, modeller.topology)
        # Check to make sure no out-of-bounds atoms are present in new_to_old_atom_map
        natoms_old = pm_top_proposal.n_atoms_old
        natoms_new = pm_top_proposal.n_atoms_new
        if not set(pm_top_proposal.new_to_old_atom_map.values()).issubset(range(natoms_old)):
            msg = "Some old atoms in TopologyProposal.new_to_old_atom_map are not in span of old atoms (1..%d):\n" % natoms_old
            msg += str(pm_top_proposal.new_to_old_atom_map)
            raise Exception(msg)
        if not set(pm_top_proposal.new_to_old_atom_map.keys()).issubset(range(natoms_new)):
            msg = "Some new atoms in TopologyProposal.new_to_old_atom_map are not in span of old atoms (1..%d):\n" % natoms_new
            msg += str(pm_top_proposal.new_to_old_atom_map)
            raise Exception(msg)

#@attr('advanced')
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

    system_generator = create_simple_protein_system_generator()

    system = system_generator.create_system(modeller.topology)
    chain_id = 'A'

    for chain in modeller.topology.chains():
        if chain.id == chain_id:
            residues = [res for res in chain._residues if res.name not in PROHIBITED_RESIDUES]
    mutant_res = np.random.choice(residues[1:-1])
    allowed_mutations = [(mutant_res.id,mutant_res.name)]

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, allowed_mutations=allowed_mutations)
    pm_top_proposal = pm_top_engine.propose(system, modeller.topology)
    assert pm_top_proposal.old_topology == pm_top_proposal.new_topology
    assert pm_top_proposal.old_system == pm_top_proposal.new_system
    assert pm_top_proposal.old_chemical_state_key == pm_top_proposal.new_chemical_state_key

#@attr('advanced')
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

    max_point_mutants = 1
    chain_id = 'A'

    # Pull the allowable mutatable residues..
    _chain = [chain for chain in modeller.topology.chains() if chain.id == chain_id][0]
    residue_ids = [residue.id for residue in _chain.residues() if residue.name != 'CYS'][1:-1]

    system_generator = create_simple_protein_system_generator()
    system = system_generator.create_system(modeller.topology)

    pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, max_point_mutants=max_point_mutants, residues_allowed_to_mutate=residue_ids)
    pm_top_proposal = pm_top_engine.propose(system, modeller.topology)

#@attr('advanced')
def test_alanine_dipeptide_map():
    pdb_filename = resource_filename('openmmtools', 'data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb')
    from simtk.openmm.app import PDBFile
    pdbfile = PDBFile(pdb_filename)
    import perses.rjmc.topology_proposal as topology_proposal
    modeller = app.Modeller(pdbfile.topology, pdbfile.positions)

    allowed_mutations = [('2', 'PHE')]
    system_generator = create_simple_protein_system_generator()
    system = system_generator.create_system(modeller.topology)
    chain_id = ' '

    metadata = dict()
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
                raise Exception(f"there is an atom mismatch")

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

    pm_top_engine._allowed_mutations = list()
    for k, proposed_amino in enumerate(aminos):
        pm_top_engine._allowed_mutations.append((str(k+2),proposed_amino))
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
            pm_top_engine._allowed_mutations = [(str(proposed_location+1),proposed_amino)]
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
            pm_top_proposal = topology_proposal.TopologyProposal(new_topology=new_topology,
                                                                 new_system=new_system,
                                                                 old_topology=old_topology,
                                                                 old_system=old_system,
                                                                 old_chemical_state_key=old_chemical_state_key,
                                                                 new_chemical_state_key=new_chemical_state_key,
                                                                 logp_proposal=0.0,
                                                                 new_to_old_atom_map=atom_map)

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

    pl_top_library = topology_proposal.PointMutationEngine(modeller.topology,
                                                           system_generator,
                                                           chain_id,
                                                           max_point_mutants=max_point_mutants,
                                                           residues_allowed_to_mutate=residues_allowed_to_mutate)
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
    pl_top_library = topology_proposal.PointMutationEngine(modeller.topology,
                                                           system_generator,
                                                           chain_id,
                                                           max_point_mutants=max_point_mutants,
                                                           residues_allowed_to_mutate=residues_allowed_to_mutate,
                                                           always_change=True)
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

def test_ring_breaking_detection():
    """
    Test the detection of ring-breaking transformations.

    """
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, AtomMapper
    from openmoltools.openeye import iupac_to_oemol, generate_conformers
    molecule1 = iupac_to_oemol("naphthalene")
    molecule2 = iupac_to_oemol("benzene")
    molecule1 = generate_conformers(molecule1,max_confs=1)
    molecule2 = generate_conformers(molecule2,max_confs=1)

    # Allow ring breaking
    new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecule1, molecule2, allow_ring_breaking=True)
    if not len(new_to_old_atom_map) > 0:
        filename = 'mapping-error.png'
        #render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
        msg = 'Napthalene -> benzene transformation with allow_ring_breaking=True is not returning a valid mapping\n'
        msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
        msg += str(new_to_old_atom_map)
        raise Exception(msg)

    new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecule1, molecule2, allow_ring_breaking=False)
    if new_to_old_atom_map is not None: # atom mapper should not retain _any_ atoms in default mode
        filename = 'mapping-error.png'
        #render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
        msg = 'Napthalene -> benzene transformation with allow_ring_breaking=False is erroneously allowing ring breaking\n'
        msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
        msg += str(new_to_old_atom_map)
        raise Exception(msg)

def test_molecular_atom_mapping():
    """
    Test the creation of atom maps between pairs of molecules from the JACS benchmark set.

    """
    from openeye import oechem
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, AtomMapper
    from itertools import combinations

    # Test mappings for JACS dataset ligands
    for dataset_name in ['CDK2']: #, 'p38', 'Tyk2', 'Thrombin', 'PTP1B', 'MCL1', 'Jnk1', 'Bace']:
        # Read molecules
        dataset_path = 'data/schrodinger-jacs-datasets/%s_ligands.sdf' % dataset_name
        mol2_filename = resource_filename('perses', dataset_path)
        ifs = oechem.oemolistream(mol2_filename)
        molecules = list()
        for mol in ifs.GetOEGraphMols():
            molecules.append(oechem.OEGraphMol(mol))

        # Build atom map for some transformations.
        #for (molecule1, molecule2) in combinations(molecules, 2): # too slow
        molecule1 = molecules[0]
        for i, molecule2 in enumerate(molecules[1:]):
            new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecule1, molecule2)
            # Make sure we aren't mapping hydrogens onto anything else
            atoms1 = [atom for atom in molecule1.GetAtoms()]
            atoms2 = [atom for atom in molecule2.GetAtoms()]
            #for (index2, index1) in new_to_old_atom_map.items():
            #    atom1, atom2 = atoms1[index1], atoms2[index2]
            #    if (atom1.GetAtomicNum()==1) != (atom2.GetAtomicNum()==1):
            filename = 'mapping-error-%d.png' % i
            render_atom_mapping(filename, molecule1, molecule2, new_to_old_atom_map)
            #msg = 'Atom atomic number %d is being mapped to atomic number %d\n' % (atom1.GetAtomicNum(), atom2.GetAtomicNum())
            msg = 'molecule 1 : %s\n' % oechem.OECreateIsoSmiString(molecule1)
            msg += 'molecule 2 : %s\n' % oechem.OECreateIsoSmiString(molecule2)
            msg += 'Wrote atom mapping to %s for inspection; please check this.' % filename
            msg += str(new_to_old_atom_map)
            print(msg)
            #        raise Exception(msg)

def test_map_strategy():
    """
    Test the creation of atom maps between pairs of molecules from the JACS benchmark set.

    """
    from openeye import oechem
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, AtomMapper
    from itertools import combinations

    # Test mappings for JACS dataset ligands
    for dataset_name in ['Jnk1']:
        # Read molecules
        dataset_path = 'data/schrodinger-jacs-datasets/%s_ligands.sdf' % dataset_name
        mol2_filename = resource_filename('perses', dataset_path)
        ifs = oechem.oemolistream(mol2_filename)
        molecules = list()
        for mol in ifs.GetOEGraphMols():
            molecules.append(oechem.OEGraphMol(mol))

        atom_expr = oechem.OEExprOpts_IntType
        bond_expr = oechem.OEExprOpts_RingMember

        # the 0th and 1st Jnk1 ligand have meta substituents that face opposite eachother
        # in the active site. Using `map_strategy=matching_criterion` should align these groups, and put them
        # both in the core. Using `map_strategy=geometry` should see that the orientations differ and chose
        # to unmap (i.e. put both these groups in core) such as to get the geometry right at the expense of
        # mapping fewer atoms
        new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecules[0], molecules[1],atom_expr=atom_expr,bond_expr=bond_expr)
        assert len(new_to_old_atom_map) == 37, 'Expected meta groups methyl C to map onto ethyl O'

        new_to_old_atom_map = AtomMapper._get_mol_atom_map(molecules[0], molecules[1],atom_expr=atom_expr,bond_expr=bond_expr,map_strategy='geometry')
        assert len(new_to_old_atom_map) == 35,  'Expected meta groups methyl C to NOT map onto ethyl O as they are distal in cartesian space'


def test_simple_heterocycle_mapping(iupac_pairs = [('benzene', 'pyridine')]):
    """
    Test the ability to map conjugated heterocycles (that preserves all rings).  Will assert that the number of ring members in both molecules is the same.
    """
    # TODO: generalize this to test for ring breakage and closure.
    from openmoltools.openeye import iupac_to_oemol
    from openeye import oechem
    from perses.rjmc.topology_proposal import AtomMapper

    for iupac_pair in iupac_pairs:
        old_oemol, new_oemol = iupac_to_oemol(iupac_pair[0]), iupac_to_oemol(iupac_pair[1])
        new_to_old_map = AtomMapper._get_mol_atom_map(old_oemol, new_oemol, allow_ring_breaking=False)

        # Assert that the number of ring members is consistent in the mapping...
        num_hetero_maps = 0
        for new_index, old_index in new_to_old_map.items():
            old_atom, new_atom = old_oemol.GetAtom(oechem.OEHasAtomIdx(old_index)), new_oemol.GetAtom(oechem.OEHasAtomIdx(new_index))
            if old_atom.IsInRing() and new_atom.IsInRing():
                if old_atom.GetAtomicNum() != new_atom.GetAtomicNum():
                    num_hetero_maps += 1

        assert num_hetero_maps > 0, f"there are no differences in atomic number mappings in {iupac_pair}"

def test_protein_counterion_topology_fix_positive():
    """
    mutate alanine dipeptide into ASP dipeptide and assert that the appropriate number of water indices are identified
    """
    from perses.rjmc.topology_proposal import PolymerProposalEngine
    new_res = 'ASP'
    charge_diff = 1

    # Make a vacuum system
    atp, system_generator = generate_atp(phase='vacuum')

    # Make a solvated system/topology/positions with modeller
    modeller = app.Modeller(atp.topology, atp.positions)
    modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=9*unit.angstroms, ionicStrength=0.15*unit.molar)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()

    # Canonicalize the solvated positions: turn tuples into np.array
    atp.positions = unit.quantity.Quantity(value=np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit=unit.nanometers)
    atp.topology = solvated_topology

    atp.system = system_generator.create_system(atp.topology)

    # Make a topology proposal and generate new positions
    top_proposal, new_pos, _, _ = generate_dipeptide_top_pos_sys(topology = atp.topology,
                                   new_res = new_res,
                                   system = atp.system,
                                   positions = atp.positions,
                                   system_generator = system_generator,
                                   conduct_geometry_prop = True,
                                   conduct_htf_prop = False,
                                   validate_energy_bookkeeping=True,
                                   )

    # Get the charge difference
    charge_diff_test = PolymerProposalEngine._get_charge_difference(top_proposal._old_topology.residue_topology.name,
                                                                top_proposal._new_topology.residue_topology.name)
    assert charge_diff_test == charge_diff

    # Get the array of water indices (w.r.t. new topology) to turn into ions
    water_indices = PolymerProposalEngine.get_water_indices(charge_diff = charge_diff_test,
                                             new_positions = new_pos,
                                             new_topology = top_proposal._new_topology,
                                             radius=0.8)

    assert len(water_indices) == 3

def test_protein_counterion_topology_fix_negitive():
    """
    mutate alanine dipeptide into ARG dipeptide and assert that the appropriate number of water indices are identified
    """
    from perses.rjmc.topology_proposal import PolymerProposalEngine
    new_res = 'ARG'
    charge_diff = -1

    # Make a vacuum system
    atp, system_generator = generate_atp(phase='vacuum')

    # Make a solvated system/topology/positions with modeller
    modeller = app.Modeller(atp.topology, atp.positions)
    modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=9*unit.angstroms, ionicStrength=0.15*unit.molar)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()

    # Canonicalize the solvated positions: turn tuples into np.array
    atp.positions = unit.quantity.Quantity(value=np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit=unit.nanometers)
    atp.topology = solvated_topology

    atp.system = system_generator.create_system(atp.topology)

    # Make a topology proposal and generate new positions
    top_proposal, new_pos, _, _ = generate_dipeptide_top_pos_sys(topology = atp.topology,
                                   new_res = new_res,
                                   system = atp.system,
                                   positions = atp.positions,
                                   system_generator = system_generator,
                                   conduct_geometry_prop = True,
                                   conduct_htf_prop = False,
                                   validate_energy_bookkeeping=True,
                                   )

    # Get the charge difference
    charge_diff_test = PolymerProposalEngine._get_charge_difference(top_proposal._old_topology.residue_topology.name,
                                                                top_proposal._new_topology.residue_topology.name)
    assert charge_diff_test == charge_diff

    # Get the array of water indices (w.r.t. new topology) to turn into ions
    water_indices = PolymerProposalEngine.get_water_indices(charge_diff = charge_diff_test,
                                             new_positions = new_pos,
                                             new_topology = top_proposal._new_topology,
                                             radius=0.8)

    assert len(water_indices) == 3


def test_protein_counterion_topology_fix_zero():
    """
    mutate alanine dipeptide into ASN dipeptide and assert that the appropriate number of water indices are identified
    """
    from perses.rjmc.topology_proposal import PolymerProposalEngine
    new_res = 'ASN'
    charge_diff = 0

    # Make a vacuum system
    atp, system_generator = generate_atp(phase='vacuum')

    # Make a solvated system/topology/positions with modeller
    modeller = app.Modeller(atp.topology, atp.positions)
    modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=9*unit.angstroms, ionicStrength=0.15*unit.molar)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()

    # Canonicalize the solvated positions: turn tuples into np.array
    atp.positions = unit.quantity.Quantity(value=np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit=unit.nanometers)
    atp.topology = solvated_topology

    atp.system = system_generator.create_system(atp.topology)

    # Make a topology proposal and generate new positions
    top_proposal, new_pos, _, _ = generate_dipeptide_top_pos_sys(topology = atp.topology,
                                   new_res = new_res,
                                   system = atp.system,
                                   positions = atp.positions,
                                   system_generator = system_generator,
                                   conduct_geometry_prop = True,
                                   conduct_htf_prop = False,
                                   validate_energy_bookkeeping=True,
                                   )

    # Get the charge difference
    charge_diff_test = PolymerProposalEngine._get_charge_difference(top_proposal._old_topology.residue_topology.name,
                                                                top_proposal._new_topology.residue_topology.name)
    assert charge_diff_test == charge_diff

    # Get the array of water indices (w.r.t. new topology) to turn into ions
    water_indices = PolymerProposalEngine.get_water_indices(charge_diff = charge_diff_test,
                                             new_positions = new_pos,
                                             new_topology = top_proposal._new_topology,
                                             radius=0.8)

    assert len(water_indices) == 0

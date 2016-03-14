__author__ = 'Patrick B. Grinaway'

import simtk.openmm as openmm
import openeye.oechem as oechem
import openmoltools
import openeye.oeiupac as oeiupac
import openeye.oeomega as oeomega
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
import parmed
import copy
from pkg_resources import resource_filename
try:
    from urllib.request import urlopen
    from io import StringIO
except:
    from urllib2 import urlopen
    from cStringIO import StringIO
import os

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

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

def generate_molecule_from_smiles(smiles):
    """
    Generate oemol with geometry from smiles
    """
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega(mol)
    return mol

def generate_initial_molecule(iupac_name):
    """
    Generate an oemol with a geometry
    """
    mol = oechem.OEMol()
    oeiupac.OEParseIUPACName(mol, iupac_name)
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega(mol)
    return mol

def oemol_to_openmm_system(oemol, molecule_name):
    from perses.rjmc import topology_proposal
    from openmoltools import forcefield_generators
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename], forcefield_kwargs={'constraints' : None})
    topology = forcefield_generators.generateTopologyFromOEMol(oemol)
    system = system_generator.build_system(topology)
    positions = extractPositionsFromOEMOL(oemol)
    return system, positions, topology

def extractPositionsFromOEMOL(molecule):
    positions = unit.Quantity(np.zeros([molecule.NumAtoms(), 3], np.float32), unit.angstroms)
    coords = molecule.GetCoords()
    for index in range(molecule.NumAtoms()):
        positions[index,:] = unit.Quantity(coords[index], unit.angstroms)
    return positions

def oemol_to_openmm_system_amber(oemol, molecule_name):
    """
    Create an openmm system out of an oemol

    Returns
    -------
    system : openmm.System object
        the system from the molecule
    positions : [n,3] np.array of floats
    """

    _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(oemol, tripos_mol2_filename=molecule_name + '.tripos.mol2', conformer=0, residue_name='MOL')
    gaff_mol2, frcmod = openmoltools.amber.run_antechamber(molecule_name, tripos_mol2_filename)
    prmtop_file, inpcrd_file = openmoltools.amber.run_tleap(molecule_name, gaff_mol2, frcmod)
    from parmed.amber import AmberParm
    prmtop = AmberParm(prmtop_file)
    system = prmtop.createSystem(implicitSolvent=None, removeCMMotion=False)
    crd = app.AmberInpcrdFile(inpcrd_file)
    return system, crd.getPositions(asNumpy=True), prmtop.topology

def align_molecules(mol1, mol2):
    """
    MCSS two OEmols. Return the mapping of new : old atoms
    """
    mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
    atomexpr = oechem.OEExprOpts_AtomicNumber
    bondexpr = oechem.OEExprOpts_BondOrder
    mcs.Init(mol1, atomexpr, bondexpr)
    mcs.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())
    unique = True
    match = [m for m in mcs.Match(mol2, unique)][0]
    new_to_old_atom_mapping = {}
    for matchpair in match.GetAtoms():
        old_index = matchpair.pattern.GetIdx()
        new_index = matchpair.target.GetIdx()
        new_to_old_atom_mapping[new_index] = old_index
    return new_to_old_atom_mapping

def test_mutate_from_every_amino_to_every_other():
    """
    Make sure mutations are successful between every possible pair of before-and-after residues
    Mutate Ecoli F-ATPase alpha subunit to all 20 amino acids (test going FROM all possibilities)
    Mutate each residue to all 19 alternatives
    """
    import perses.rjmc.topology_proposal as topology_proposal
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine()

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

    pm_top_engine = topology_proposal.PointMutationEngine(system_generator, chain_id, max_point_mutants=max_point_mutants)

    current_system = system
    current_topology = modeller.topology
    current_positions = modeller.positions
    minimize_integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    minimize_context = openmm.Context(current_system, minimize_integrator, platform)
    minimize_context.setPositions(current_positions)
    initial_state = minimize_context.getState(getEnergy=True)
    initial_potential = initial_state.getPotentialEnergy()
    openmm.LocalEnergyMinimizer.minimize(minimize_context)
    final_state = minimize_context.getState(getEnergy=True)
    final_potential = final_state.getPotentialEnergy()

    print("Minimized initial structure from %s to %s" % (str(initial_potential), str(final_potential)))

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
        print("Proposing %s from %s" % (proposed_amino, original_residue.name))
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
        new_topology = modeller.topology
        new_system = pm_top_engine._ff.createSystem(new_topology)
        pm_top_proposal = topology_proposal.TopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=current_system, old_chemical_state_key=original_residue.name, new_chemical_state_key=proposed_amino, logp_proposal=0.0, new_to_old_atom_map=atom_map, metadata=metadata)
        new_positions, logp = geometry_engine.propose(pm_top_proposal, current_positions, beta)
        if np.isnan(logp):
            raise Exception("NaN in the logp")
        integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(new_system, integrator, platform)
        context.setPositions(new_positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()
        potential_without_units = potential / potential.unit
        print(str(potential))
        if np.isnan(potential_without_units):
            raise Exception("Energy after proposal is NaN")

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

def test_run_geometry_engine(index=0):
    """
    Run the geometry engine a few times to make sure that it actually runs
    without exceptions. Convert n-pentane to 2-methylpentane
    """
    import copy
    molecule_name_1 = 'erlotinib'
    molecule_name_2 = 'nilotinib'
    #molecule_name_1 = 'benzene'
    #molecule_name_2 = 'biphenyl'

    molecule1 = generate_initial_molecule(molecule_name_1)
    molecule2 = generate_initial_molecule(molecule_name_2)
    new_to_old_atom_mapping = align_molecules(molecule1, molecule2)

    sys1, pos1, top1 = oemol_to_openmm_system(molecule1, molecule_name_1)
    sys2, pos2, top2 = oemol_to_openmm_system(molecule2, molecule_name_2)

    import perses.rjmc.geometry as geometry
    import perses.rjmc.topology_proposal as topology_proposal
    from perses.tests.utils import compute_potential_components

    sm_top_proposal = topology_proposal.TopologyProposal(new_topology=top2, new_system=sys2, old_topology=top1, old_system=sys1,
                                                                      old_chemical_state_key='',new_chemical_state_key='', logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_mapping, metadata={'test':0.0})
    sm_top_proposal._beta = beta
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true', 'reference_positions':pos2})
    # Turn on PDB file writing.
    geometry_engine.write_proposal_pdb = True
    geometry_engine.pdb_filename_prefix = 'geometry-proposal'
    test_pdb_file = open("nilotinib_from_erlotinib_%d_3.pdb" % index, 'w')

    valence_system = copy.deepcopy(sys2)
    valence_system.removeForce(3)
    valence_system.removeForce(3)
    integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    integrator_1 = openmm.VerletIntegrator(1*unit.femtoseconds)
    ctx_1 = openmm.Context(sys1, integrator_1)
    ctx_1.setPositions(pos1)
    ctx_1.setVelocitiesToTemperature(300*unit.kelvin)
    integrator_1.step(1000)
    pos1_new = ctx_1.getState(getPositions=True).getPositions(asNumpy=True)
    context = openmm.Context(sys2, integrator)
    context.setPositions(pos2)
    state = context.getState(getEnergy=True)
    print("Energy before proposal is: %s" % str(state.getPotentialEnergy()))

    new_positions, logp_proposal = geometry_engine.propose(sm_top_proposal, pos1_new, beta)
    geometry_engine.logp_reverse(sm_top_proposal, new_positions, pos1, beta)

    app.PDBFile.writeFile(top2, new_positions, file=test_pdb_file)
    test_pdb_file.close()
    context.setPositions(new_positions)
    state2 = context.getState(getEnergy=True)
    print("Energy after proposal is: %s" %str(state2.getPotentialEnergy()))
    print(compute_potential_components(context))

    valence_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    valence_ctx = openmm.Context(valence_system, valence_integrator, platform)
    valence_ctx.setPositions(new_positions)
    vstate = valence_ctx.getState(getEnergy=True)
    print("Valence energy after proposal is %s " % str(vstate.getPotentialEnergy()))

def test_existing_coordinates():
    """
    for each torsion, calculate position of atom1
    """
    molecule_name_2 = 'butane'
    molecule2 = generate_initial_molecule(molecule_name_2)
    sys, pos, top = oemol_to_openmm_system(molecule2, molecule_name_2)
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    structure = parmed.openmm.load_topology(top, sys)
    torsions = [torsion for torsion in structure.dihedrals if not torsion.improper]
    for torsion in torsions:
        atom1_position = pos[torsion.atom1.idx]
        atom2_position = pos[torsion.atom2.idx]
        atom3_position = pos[torsion.atom3.idx]
        atom4_position = pos[torsion.atom4.idx]
        _internal_coordinates, _ = geometry_engine._cartesian_to_internal(atom1_position, atom2_position, atom3_position, atom4_position)
        internal_coordinates = internal_in_unit(_internal_coordinates)
        recalculated_atom1_position, _ = geometry_engine._internal_to_cartesian(atom2_position, atom3_position, atom4_position, internal_coordinates[0], internal_coordinates[1], internal_coordinates[2])
        n = np.linalg.norm(atom1_position-recalculated_atom1_position)
        print(n)

def internal_in_unit(internal_coords):
    r = internal_coords[0]*unit.nanometers if type(internal_coords[0]) != unit.Quantity else internal_coords[0]
    theta = internal_coords[1]*unit.radians if type(internal_coords[1]) != unit.Quantity else internal_coords[1]
    phi = internal_coords[2]*unit.radians if type(internal_coords[2]) != unit.Quantity else internal_coords[2]
    return [r, theta, phi]

def test_coordinate_conversion():
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    example_coordinates = unit.Quantity(np.random.normal(size=[100,3]), unit=unit.nanometers)
    #try to transform random coordinates to and from
    for i in range(200):
        indices = np.random.randint(100, size=4)
        atom_position = unit.Quantity(np.array([ 0.80557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
        bond_position = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
        angle_position = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
        torsion_position = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)
        rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
        r = rtp[0]*unit.nanometers
        theta = rtp[1]*unit.radians
        phi = rtp[2]*unit.radians
        xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        assert np.linalg.norm(xyz-atom_position) < 1.0e-12

def test_dihedral_potential():
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    molecule_name = 'ethane'
    molecule2 = generate_initial_molecule(molecule_name)
    sys, pos, top = oemol_to_openmm_system(molecule2, molecule_name)
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    structure = parmed.openmm.load_topology(top, sys)

def test_openmm_dihedral():
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    import simtk.openmm as openmm
    integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
    sys = openmm.System()
    force = openmm.CustomTorsionForce("theta")
    for i in range(4):
        sys.addParticle(1.0*unit.amu)
    force.addTorsion(0,1,2,3,[])
    sys.addForce(force)
    atom_position = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
    bond_position = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
    angle_position = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
    torsion_position = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)
    rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(sys, integrator, platform)
    positions = [atom_position, bond_position, angle_position, torsion_position]
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()

    #rotate about the torsion:
    n_divisions = 100
    phis = unit.Quantity(np.arange(0, 2.0*np.pi, (2.0*np.pi)/n_divisions), unit=unit.radians)
    omm_phis = np.zeros(n_divisions)
    for i, phi in enumerate(phis):
        xyz_atom1, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, rtp[0]*unit.nanometers, rtp[1]*unit.radians, phi)
        context.setPositions([xyz_atom1, bond_position, angle_position, torsion_position])
        state = context.getState(getEnergy=True)
        omm_phis[i] = state.getPotentialEnergy()/unit.kilojoule_per_mole

    return 0

def test_try_random_itoc():

    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    import simtk.openmm as openmm
    integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
    sys = openmm.System()
    force = openmm.CustomTorsionForce("theta")
    for i in range(4):
        sys.addParticle(1.0*unit.amu)
    force.addTorsion(0,1,2,3,[])
    sys.addForce(force)
    atom_position = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
    bond_position = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
    angle_position = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
    torsion_position = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)
    for i in range(10):
        atom_position += unit.Quantity(np.random.normal(size=3), unit=unit.nanometers)
        r, theta, phi = _get_internal_from_omm(atom_position, bond_position, angle_position, torsion_position)
        r = (r/r.unit)*unit.nanometers
        theta = (theta/theta.unit)*unit.radians
        phi = (phi/phi.unit)*unit.radians
        recomputed_xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        new_r, new_theta, new_phi = _get_internal_from_omm(recomputed_xyz,bond_position, angle_position, torsion_position)
        crtp = geometry_engine._cartesian_to_internal(recomputed_xyz,bond_position, angle_position, torsion_position)
        print(atom_position-recomputed_xyz)

def test_logp_reverse():
    """
    Make sure logp_reverse and logp_forward are consistent
    """
    np.seterr(all='raise')
    molecule_name_1 = 'erlotinib'
    molecule_name_2 = 'imatinib'
    #molecule_name_1 = 'benzene'
    #molecule_name_2 = 'biphenyl'

    molecule1 = generate_initial_molecule(molecule_name_1)
    molecule2 = generate_initial_molecule(molecule_name_2)
    new_to_old_atom_mapping = align_molecules(molecule1, molecule2)

    sys1, pos1, top1 = oemol_to_openmm_system(molecule1, molecule_name_1)
    sys2, pos2, top2 = oemol_to_openmm_system(molecule2, molecule_name_2)
    test_pdb_file = open("reverse_test1.pdb", 'w')
    app.PDBFile.writeFile(top1, pos1, file=test_pdb_file)
    test_pdb_file.close()

    import perses.rjmc.geometry as geometry
    import perses.rjmc.topology_proposal as topology_proposal

    sm_top_proposal = topology_proposal.TopologyProposal(new_topology=top2, new_system=sys2, old_topology=top1, old_system=sys1,
                                                                    logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_mapping, new_chemical_state_key="CCC", old_chemical_state_key="CC", metadata={'test':0.0})
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    new_positions, logp_proposal = geometry_engine.propose(sm_top_proposal, pos1, beta)

    logp_reverse = geometry_engine.logp_reverse(sm_top_proposal, pos2, pos1, beta)
    print(logp_proposal)
    print(logp_reverse)
    print(logp_reverse-logp_proposal)

def _get_internal_from_omm(atom_coords, bond_coords, angle_coords, torsion_coords):
    import copy
    #master system, will be used for all three
    sys = openmm.System()
    platform = openmm.Platform.getPlatformByName("Reference")
    for i in range(4):
        sys.addParticle(1.0*unit.amu)

    #first, the bond length:
    bond_sys = openmm.System()
    bond_sys.addParticle(1.0*unit.amu)
    bond_sys.addParticle(1.0*unit.amu)
    bond_force = openmm.CustomBondForce("r")
    bond_force.addBond(0, 1, [])
    bond_sys.addForce(bond_force)
    bond_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    bond_context = openmm.Context(bond_sys, bond_integrator, platform)
    bond_context.setPositions([atom_coords, bond_coords])
    bond_state = bond_context.getState(getEnergy=True)
    r = bond_state.getPotentialEnergy()
    del bond_sys, bond_context, bond_integrator

    #now, the angle:
    angle_sys = copy.deepcopy(sys)
    angle_force = openmm.CustomAngleForce("theta")
    angle_force.addAngle(0,1,2,[])
    angle_sys.addForce(angle_force)
    angle_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    angle_context = openmm.Context(angle_sys, angle_integrator, platform)
    angle_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
    angle_state = angle_context.getState(getEnergy=True)
    theta = angle_state.getPotentialEnergy()
    del angle_sys, angle_context, angle_integrator

    #finally, the torsion:
    torsion_sys = copy.deepcopy(sys)
    torsion_force = openmm.CustomTorsionForce("theta")
    torsion_force.addTorsion(0,1,2,3,[])
    torsion_sys.addForce(torsion_force)
    torsion_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    torsion_context = openmm.Context(torsion_sys, torsion_integrator, platform)
    torsion_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
    torsion_state = torsion_context.getState(getEnergy=True)
    phi = torsion_state.getPotentialEnergy()
    del torsion_sys, torsion_context, torsion_integrator

    return r, theta, phi

if __name__=="__main__":
    #test_coordinate_conversion()
    #test_run_geometry_engine()
    #test_existing_coordinates()
    #test_openmm_dihedral()
    #test_try_random_itoc()
    #test_angle()
    #test_logp_reverse()
    test_mutate_from_every_amino_to_every_other()
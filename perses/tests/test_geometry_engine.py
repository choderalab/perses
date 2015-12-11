__author__ = 'Patrick B. Grinaway'

import simtk.openmm as openmm
import openeye.oechem as oechem
import openmoltools
import openeye.oeiupac as oeiupac
import openeye.oeomega as oeomega
import simtk.openmm.app as app
import simtk.unit as units
import numpy as np
import parmed

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
beta = 1.0/ (300.0*units.kelvin*kB)

def generate_molecule_from_smiles(smiles):
    """
    Generate oemol with geometry from smiles
    """
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    oechem.OEAddExplicitHydrogens(mol)
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
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega(mol)
    return mol

def oemol_to_openmm_system(oemol, molecule_name):
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
    system = prmtop.createSystem(implicitSolvent=app.OBC1)
    crd = app.AmberInpcrdFile(inpcrd_file)
    return system, crd.getPositions(asNumpy=True), prmtop.topology

def align_molecules(mol1, mol2):
    """
    MCSS two OEmols. Return the mapping of new : old atoms
    """
    mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
    atomexpr = oechem.OEExprOpts_AtomicNumber
    bondexpr = 0
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


def test_run_geometry_engine():
    """
    Run the geometry engine a few times to make sure that it actually runs
    without exceptions. Convert n-pentane to 2-methylpentane
    """
    molecule_name_1 = 'pentane'
    molecule_name_2 = 'hexane'

    molecule1 = generate_initial_molecule(molecule_name_1)
    molecule2 = generate_initial_molecule(molecule_name_2)
    new_to_old_atom_mapping = align_molecules(molecule1, molecule2)

    sys1, pos1, top1 = oemol_to_openmm_system(molecule1, molecule_name_1)
    sys2, pos2, top2 = oemol_to_openmm_system(molecule2, molecule_name_2)

    import perses.rjmc.geometry as geometry
    import perses.rjmc.topology_proposal as topology_proposal

    sm_top_proposal = topology_proposal.SmallMoleculeTopologyProposal(new_topology=top2, new_system=sys2, old_topology=top1, old_system=sys1,
                                                                      old_positions=pos1, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_mapping, metadata={'test':0.0})
    sm_top_proposal._beta = beta
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    #test_pdb_file = open("erlotinib_gefitinib_after_1.pdb", 'w')


    integrator = openmm.VerletIntegrator(1*units.femtoseconds)
    context = openmm.Context(sys2, integrator)
    context.setPositions(pos2)
    context.setVelocitiesToTemperature(300*units.kelvin)
    state = context.getState(getEnergy=True)
    print("Energy before proposal is: %s" % str(state.getPotentialEnergy()))

    new_positions, logp_proposal = geometry_engine.propose(sm_top_proposal)
    #app.PDBFile.writeFile(top2, new_positions, file=test_pdb_file)
    context.setPositions(new_positions)
    state2 = context.getState(getEnergy=True)
    print("Energy after proposal is: %s" %str(state2.getPotentialEnergy()))

    integrator.step(1000)
    state3 = context.getState(getEnergy=True, getPositions=True)
    after_dynamics_positions = state3.getPositions()
    #app.PDBFile.writeFile(top2, after_dynamics_positions, file=test_pdb_file)
    #test_pdb_file.close()


    print("Energy after 1000 steps is %s" % str(state3.getPotentialEnergy()))

def test_existing_coordinates():
    """
    predUS
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
        internal_coordinates = internal_in_units(_internal_coordinates)
        recalculated_atom1_position, _ = geometry_engine._internal_to_cartesian(atom2_position, atom3_position, atom4_position, internal_coordinates[0], internal_coordinates[1], internal_coordinates[2])
        n = np.linalg.norm(atom1_position-recalculated_atom1_position)
        print(n)

def internal_in_units(internal_coords):
    r = internal_coords[0]*units.nanometers if type(internal_coords[0]) != units.Quantity else internal_coords[0]
    theta = internal_coords[1]*units.radians if type(internal_coords[1]) != units.Quantity else internal_coords[1]
    phi = internal_coords[2]*units.radians if type(internal_coords[2]) != units.Quantity else internal_coords[2]
    return [r, theta, phi]

def test_coordinate_conversion():
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    example_coordinates = units.Quantity(np.random.normal(size=[100,3]), unit=units.nanometers)
    #try to transform random coordinates to and from
    for i in range(200):
        indices = np.random.randint(100, size=4)
        atom_position = units.Quantity(np.array([ 0.80557722 ,-1.10424644 ,-1.08578826]), unit=units.nanometers)
        bond_position = units.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=units.nanometers)
        angle_position = units.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=units.nanometers)
        torsion_position = units.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=units.nanometers)
        rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
        r = rtp[0]*units.nanometers
        theta = rtp[1]*units.radians
        phi = rtp[2]*units.radians
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
    integrator = openmm.VerletIntegrator(1.0*units.femtoseconds)
    sys = openmm.System()
    force = openmm.CustomTorsionForce("theta")
    for i in range(4):
        sys.addParticle(1.0*units.amu)
    force.addTorsion(0,1,2,3)
    sys.addForce(force)
    atom_position = units.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=units.nanometers)
    bond_position = units.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=units.nanometers)
    angle_position = units.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=units.nanometers)
    torsion_position = units.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=units.nanometers)
    rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(sys, integrator, platform)
    positions = [atom_position, bond_position, angle_position, torsion_position]
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()

    #rotate about the torsion:
    n_divisions = 100
    phis = units.Quantity(np.arange(0, 2.0*np.pi, (2.0*np.pi)/n_divisions), unit=units.radians)
    omm_phis = np.zeros(n_divisions)
    for i, phi in enumerate(phis):
        xyz_atom1, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, rtp[0]*units.nanometers, rtp[1]*units.radians, phi)
        context.setPositions([xyz_atom1, bond_position, angle_position, torsion_position])
        state = context.getState(getEnergy=True)
        omm_phis[i] = state.getPotentialEnergy()/units.kilojoule_per_mole

    return 0

def test_try_random_itoc():

    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    import simtk.openmm as openmm
    integrator = openmm.VerletIntegrator(1.0*units.femtoseconds)
    sys = openmm.System()
    force = openmm.CustomTorsionForce("theta")
    for i in range(4):
        sys.addParticle(1.0*units.amu)
    force.addTorsion(0,1,2,3)
    sys.addForce(force)
    atom_position = units.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=units.nanometers)
    bond_position = units.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=units.nanometers)
    angle_position = units.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=units.nanometers)
    torsion_position = units.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=units.nanometers)
    for i in range(10):
        atom_position += units.Quantity(np.random.normal(size=3), unit=units.nanometers)
        r, theta, phi = _get_internal_from_omm(atom_position, bond_position, angle_position, torsion_position)
        r = (r/r.unit)*units.nanometers
        theta = (theta/theta.unit)*units.radians
        phi = (phi/phi.unit)*units.radians
        recomputed_xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        new_r, new_theta, new_phi = _get_internal_from_omm(recomputed_xyz,bond_position, angle_position, torsion_position)
        crtp = geometry_engine._cartesian_to_internal(recomputed_xyz,bond_position, angle_position, torsion_position)
        print(atom_position-recomputed_xyz)


def _get_internal_from_omm(atom_coords, bond_coords, angle_coords, torsion_coords):
    import copy
    #master system, will be used for all three
    sys = openmm.System()
    platform = openmm.Platform.getPlatformByName("Reference")
    for i in range(4):
        sys.addParticle(1.0*units.amu)

    #first, the bond length:
    bond_sys = openmm.System()
    bond_sys.addParticle(1.0*units.amu)
    bond_sys.addParticle(1.0*units.amu)
    bond_force = openmm.CustomBondForce("r")
    bond_force.addBond(0, 1)
    bond_sys.addForce(bond_force)
    bond_integrator = openmm.VerletIntegrator(1*units.femtoseconds)
    bond_context = openmm.Context(bond_sys, bond_integrator, platform)
    bond_context.setPositions([atom_coords, bond_coords])
    bond_state = bond_context.getState(getEnergy=True)
    r = bond_state.getPotentialEnergy()
    del bond_sys, bond_context, bond_integrator

    #now, the angle:
    angle_sys = copy.deepcopy(sys)
    angle_force = openmm.CustomAngleForce("theta")
    angle_force.addAngle(0,1,2)
    angle_sys.addForce(angle_force)
    angle_integrator = openmm.VerletIntegrator(1*units.femtoseconds)
    angle_context = openmm.Context(angle_sys, angle_integrator, platform)
    angle_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
    angle_state = angle_context.getState(getEnergy=True)
    theta = angle_state.getPotentialEnergy()
    del angle_sys, angle_context, angle_integrator

    #finally, the torsion:
    torsion_sys = copy.deepcopy(sys)
    torsion_force = openmm.CustomTorsionForce("theta")
    torsion_force.addTorsion(0,1,2,3)
    torsion_sys.addForce(torsion_force)
    torsion_integrator = openmm.VerletIntegrator(1*units.femtoseconds)
    torsion_context = openmm.Context(torsion_sys, torsion_integrator, platform)
    torsion_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
    torsion_state = torsion_context.getState(getEnergy=True)
    phi = torsion_state.getPotentialEnergy()
    del torsion_sys, torsion_context, torsion_integrator

    return r, theta, phi

def test_angle():
    """
    Test the _calculate_angle function in the geometry engine to make sure it gets the same number as openmm
    """
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    example_coordinates = units.Quantity(np.random.normal(size=[4,3]))
    r, theta, phi = _get_internal_from_omm(example_coordinates[0], example_coordinates[1], example_coordinates[2], example_coordinates[3])
    theta_g = geometry_engine._calculate_angle(example_coordinates[0], example_coordinates[1], example_coordinates[2])
    assert abs(theta / theta.unit - theta_g) < 1.0e-12

def _omm_torsion_system(torsion):
    """

    Parameters
    ----------
    torsion : parmed.Torsion object
        A parmed.Torsion object representing a proper torsion
    Returns
    -------
    torsion_system : openmm.System object
        An OpenMM system that contains 4 particles with a single torsion
    """
    torsion_system = openmm.System()
    for i in range(4):
        torsion_system.addParticle(1*units.amu)
    torsion_force = openmm.PeriodicTorsionForce()
    torsion_system.addForce(torsion_force)
    force_constant = torsion.type.phi_k
    periodicity = torsion.type.per
    phase = torsion.type.phase
    torsion_force.addTorsion(0,1,2,3, periodicity, phase.in_units_of(units.radians), force_constant.in_units_of(units.kilojoule_per_mole))
    return torsion_system

def _omm_angle_system(angle):
    """
    Return a system that just has a harmonic angle based on the angle parameters

    Parameters
    ----------
    angle : parmed.Angle object
        parmed angle that has the united parameters

    Returns
    -------
    angle_system : openmm.System object
        System with appropriate angle
    """
    angle_system = openmm.System()
    for i in range(3):
        angle_system.addParticle(1.0*units.amu)
    angle_force = openmm.HarmonicAngleForce()
    angle_system.addForce(angle_force)
    force_constant = angle.type.k
    theteq = angle.type.thet

def test_molecule_torsion_potential():
    """
    Test the calculation of a torsion potential, and ensure it matches OpenMM
    """
    molecule_name_2 = 'butane'
    molecule2 = generate_initial_molecule(molecule_name_2)
    sys, pos, top = oemol_to_openmm_system(molecule2, molecule_name_2)
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    structure = parmed.openmm.load_topology(top, sys)
    torsions = [geometry_engine._add_torsion_units(torsion) for torsion in structure.dihedrals if not torsion.improper]
    platform = openmm.Platform.getPlatformByName("Reference")
    for torsion in torsions:
        atom_position = pos[torsion.atom1.idx]
        bond_position = pos[torsion.atom2.idx]
        angle_position = pos[torsion.atom3.idx]
        torsion_position = pos[torsion.atom4.idx]
        torsion_system = _omm_torsion_system(torsion)
        integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        context = openmm.Context(torsion_system, integrator, platform)
        context.setPositions([atom_position, bond_position, angle_position, torsion_position])
        state = context.getState(getEnergy=True)
        omm_logq = -beta*state.getPotentialEnergy()
        internal_coords, _ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
        geometry_logq = geometry_engine._torsion_logq(torsion, internal_coords[2]*units.radians, beta)
        assert np.abs(omm_logq-geometry_logq) < 1.0e-12

def test_arbitrary_torsion_potential():
    """
    Test the torsion potential by rotating an arbitrary particle
    """
    n_divisions = 100
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    periodicity = 1
    force_constant = 1.0
    phase = 0.0
    atom_position = units.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=units.nanometers)
    bond_position = units.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=units.nanometers)
    angle_position = units.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=units.nanometers)
    torsion_position = units.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=units.nanometers)
    phis = units.Quantity(np.arange(0, 2.0*np.pi, (2.0*np.pi)/n_divisions), unit=units.radians)
    r = 1.0*units.nanometers
    theta = np.pi*units.radians
    dihedral_type = parmed.DihedralType(force_constant, periodicity, phase)
    torsion = parmed.Dihedral(parmed.Atom(),parmed.Atom(),parmed.Atom(),parmed.Atom(), type=dihedral_type)
    torsion = geometry_engine._add_torsion_units(torsion)
    platform = openmm.Platform.getPlatformByName("Reference")
    for phi in phis:
        geometry_logq = geometry_engine._torsion_logq(torsion, phi, beta)
        xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        system  = _omm_torsion_system(torsion)
        integrator = openmm.VerletIntegrator(1.0*units.femtoseconds)
        context = openmm.Context(system, integrator, platform)
        context.setPositions([xyz, bond_position, angle_position, torsion_position])
        state = context.getState(getEnergy=True)
        omm_logq = -beta*state.getPotentialEnergy()
        print(np.abs(omm_logq-geometry_logq))









if __name__=="__main__":
    #test_coordinate_conversion()
    #test_run_geometry_engine()
    #test_existing_coordinates()
    #test_openmm_dihedral()
    #test_try_random_itoc()
    #test_angle()
    #test_molecule_torsion_potential()
    test_arbitrary_torsion_potential()
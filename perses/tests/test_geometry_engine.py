__author__ = 'Patrick B. Grinaway'

import simtk.openmm as openmm
import openeye.oechem as oechem
import openmoltools
import openeye.oeiupac as oeiupac
import openeye.oeomega as oeomega
import simtk.openmm.app as app
import simtk.unit as units
import numpy as np

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
beta = 1.0/ (300.0*units.kelvin*kB)
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
    mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
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
    molecule_name_1 = 'n-pentane'
    molecule_name_2 = '2-methylpentane'

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
    test_pdb_file = open("npentanehexane.pdb", 'w')


    integrator = openmm.VerletIntegrator(1*units.femtoseconds)
    context = openmm.Context(sys2, integrator)
    context.setPositions(pos2)
    state = context.getState(getEnergy=True)
    print("Energy before proposal is: %s" % str(state.getPotentialEnergy()))

    new_positions, logp_proposal = geometry_engine.propose(sm_top_proposal)
    app.PDBFile.writeFile(top2, new_positions, file=test_pdb_file)
    test_pdb_file.close()
    context.setPositions(new_positions)
    state2 = context.getState(getEnergy=True)
    print("Energy after proposal is: %s" %str(state2.getPotentialEnergy()))

def test_coordinate_conversion():

    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    example_coordinates = units.Quantity(np.random.normal(size=[100,3]), unit=units.nanometers)
    #try to transform random coordinates to and from
    for i in range(20):
        indices = np.random.randint(100, size=4)
        atom_position = example_coordinates[indices[0]]
        bond_position = example_coordinates[indices[1]]
        angle_position = example_coordinates[indices[2]]
        torsion_position = example_coordinates[indices[3]]
        rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
        r = rtp[0]*units.nanometers
        theta = rtp[1]*units.radians
        phi = rtp[2]*units.radians
        xyz, _ = geometry_engine._internal_to_cartesian(indices[1], indices[2], indices[3], r, theta, phi, example_coordinates)




if __name__=="__main__":
    test_coordinate_conversion()
    test_run_geometry_engine()

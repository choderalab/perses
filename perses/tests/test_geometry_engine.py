__author__ = 'Patrick B. Grinaway'

#for now, just make the geometry engine run

import simtk.openmm as openmm
import openeye.oechem as oechem
import openmoltools
import openeye.oeiupac as oeiupac
import openeye.oeomega as oeomega
import simtk.openmm.app as app
import simtk.unit as units


def generate_initial_molecule(iupac_name):
    """
    Generate an oemol with a geometry
    """
    mol = oechem.OEMol()
    oeiupac.OEParseIUPACName(mol, "n-pentane")
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
    openmoltools.openeye.enter_temp_directory()
    _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(oemol, tripos_mol2_filename=molecule_name + '.tripos.mol2', conformer=0, residue_name='MOL')
    gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(molecule_name, tripos_mol2_filename)
    prmtop_file, inpcrd_file = openmoltools.utils.run_tleap(molecule_name, gaff_mol2, frcmod)
    prmtop = app.AmberPrmtopFile(prmtop_file)
    system = prmtop.createSystem(implicitSolvent=app.OBC1)
    crd = app.AmberInpcrdFile(inpcrd_file)
    return system, crd.positions

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
    matches = mcs.Match(mol2, unique)
    match = matches[0]
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

    sys1, pos1 = oemol_to_openmm_system(molecule1, molecule_name_1)
    sys2, pos2 = oemol_to_openmm_system(molecule2, molecule_name_2)

    #copy the positions to openmm manually (not sure what happens to units otherwise)
    for atom in molecule1.GetAtoms():
        (x, y, z) = molecule1.GetCoords(atom)
        index = atom.GetIdx()
        pos1[index, 0] = x * units.angstrom
        pos1[index, 1] = y * units.angstrom
        pos1[index, 2] = z * units.angstrom

    for atom in molecule2.GetAtoms():
        (x, y, z) = molecule1.GetCoords(atom)
        index = atom.GetIdx()
        pos2[index, 0] = x * units.angstrom
        pos2[index, 1] = y * units.angstrom
        pos2[index, 2] = z * units.angstrom

    #propose(self, new_to_old_atom_map, new_system, old_system, old_positions)
    from . import geometry

    geometry_engine = geometry.FFGeometryEngine({'test': 'true'})

    for i in range(10):
        geometry_engine.propose(new_to_old_atom_mapping, sys2, sys1, pos1)
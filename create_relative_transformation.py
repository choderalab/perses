#!/bin/bash
"""
Testbed for creating relative alchemical transformation.

"""

import gaff2xml.openeye
import openeye.oechem as oe
import simtk.openmm as mm
import simtk.openmm.app as app

def generate_openmm_system(molecule):
    trajs, ffxmls = gaff2xml.openeye.oemols_to_ffxml([molecule])
    ff = app.ForceField(ffxmls)
    # Get OpenMM Topology.
    topology = trajs[0].top.to_openmm()
    # Create OpenMM System object.
    system = ff.createSystem(topology)
    return [system, topology]

def create_relative_alchemical_transformation(system, topology, molecule1_indices_in_system, molecule1, molecule2):
    """
    Create an OpenMM System object to handle the alchemical transformation from molecule1 to molecule2.

    system : simtk.openmm.System
       The system to be modified, already contains molecule1
    topology : simtk.openmm.app.Topology
       The topology object corresponding to System.
    molecule1_indices_in_system : list of int
       Indices of molecule1 in system
    molecule1 : openeye.oechem.OEMol
       Molecule already present in system, mapped to atoms in the system.
    molecule2 : openeye.oechem.OEMol
       Molecule that molecule1 will be transformed into.

    """
    # TODO: Normalize molecules.
    oe.OEPerceiveChiral(molecule1)
    oe.OEPerceiveChiral(molecule2)

    # Make copies to not destroy original objects.
    import copy
    system = copy.deepcopy(system)
    topology = copy.deepcopy(topology)

    # Create OpenMM Topology and System objects.
    [system1, topology1] = generate_openmm_system(molecule1)
    [system2, topology2] = generate_openmm_system(molecule2)

    # Create lists of corresponding atoms for common substructure and groups specific to molecules 1 and 2.
    #atomexpr = oe.OEExprOpts_EqONS
    atomexpr = oe.OEExprOpts_DefaultAtoms
    bondexpr = oe.OEExprOpts_BondOrder | oe.OEExprOpts_EqSingleDouble | oe.OEExprOpts_EqAromatic
    mcss = oe.OEMCSSearch(molecule1, atomexpr, bondexpr, oe.OEMCSType_Exhaustive)
    # This modifies scoring function to prefer keeping cycles complete.
    mcss.SetMCSFunc( oe.OEMCSMaxAtomsCompleteCycles() )
    # TODO: Set initial substructure size?
    # mcss.SetMinAtoms( some_number )
    # We only need one match.
    mcss.SetMaxMatches(1)
    # Determine common atoms in second molecule.
    matches = [ match for match in mcss.Match(molecule2, True) ]
    match = matches[0] # we only need the first match

    # Make a list of the atoms in common, molecule1 only, and molecule2 only
    common1 = list() # list of atom indices in molecule1 that also appear in molecule2
    common2 = list() # list of atom indices in molecule2 that also appear in molecule1
    unique1 = list() # list of atom indices in molecule1 that DO NOT appear in molecule2
    unique2 = list() # list of atom indices in molecule2 that DO NOT appear in molecule1
    mapping1 = dict() # mapping of atoms in molecule1 to molecule2
    mapping2 = dict() # mapping of atoms in molecule2 to molecule1
    for matchpair in match.GetAtoms():
        index1 = matchpair.pattern.GetIdx()
        index2 = matchpair.target.GetIdx()
        mapping1[ index1 ] = index2
        mapping2[ index2 ] = index1
    all1 = range(molecule1.NumAtoms())
    all2 = range(molecule2.NumAtoms())
    common1 = sorted(mapping1.keys())
    common2 = sorted(mapping2.keys())
    unique1 = sorted(list(set(all1) - set(common1)))
    unique2 = sorted(list(set(all2) - set(common2)))

    # DEBUG
    print "list of atoms common to both molecules:"
    print "molecule1: %s" % str(common1)
    print "molecule2: %s" % str(common2)
    print "list of atoms unqiue to individual molecules:"
    print "molecule1: %s" % str(unique1)
    print "molecule2: %s" % str(unique2)
    print "MAPPING FROM MOLECULE1 TO MOLECULE2"
    for atom1 in mapping1.keys():
        atom2 = mapping1[atom1]
        print "%5d => %5d" % (atom1, atom2)


    #
    # Start building combined OpenMM System object.
    #

    molecule1_atoms = [ atom for atom in molecule1.GetAtoms() ]
    molecule2_atoms = [ atom for atom in molecule2.GetAtoms() ]

    molecule2_indices_in_system = dict()

    # Build mapping of common substructure for molecule 2.
    for atom2 in common2:
        molecule2_indices_in_system[atom2] = molecule1_indices_in_system[mapping2[atom2]]

    # Handle additional particles.
    print "Adding particles from system2..."
    for atom2 in unique2:
        atom = molecule2_atoms[atom2]
        name = atom.GetName()
        atomic_number = atom.GetAtomicNum()
        element = app.Element.getByAtomicNumber(atomic_number)
        mass = system2.getParticleMass(atom2)
        print [name, element, mass]
        index = system.addParticle(mass)
        molecule2_indices_in_system[atom2] = index
        #residue = 'XXX' # TODO: Fix this later
        #topology.addAtom(name, element, residue)

    # Turn molecule2_indices_in_system into list
    molecule2_indices_in_system = [ molecule2_indices_in_system[atom2] for atom2 in range(molecule2.NumAtoms()) ]

    print "Atom mappings into System object"
    print "molecule1: %s" % str(molecule1_indices_in_system)
    print "molecule2: %s" % str(molecule2_indices_in_system)

    # Handle constraints.
    # TODO: What happens if constraints change? Raise Exception then.
    print "Adding constraints from system2..."
    for index in range(system2.getNumConstraints()):
        # Extract constraint distance from system2.
        [atom2_i, atom2_j, distance] = system.getConstraintParameters(index)
        # Map atoms from system2 into system.
        atom_i = molecule2_indices_in_system[atom2_i]
        atom_j = molecule2_indices_in_system[atom2_j]
        # Add constraint to system.
        system.addConstraint(atom_i, atom_j, distance)

    # Build a list of Force objects in system.
    forces = [ system.getForce(index) for index in range(system.getNumForces()) ]
    forces1 = { system1.getForce(index).__class__.__name__ : system1.getForce(index) for index in range(system1.getNumForces()) }
    forces2 = { system2.getForce(index).__class__.__name__ : system2.getForce(index) for index in range(system2.getNumForces()) }
    # Dispatch forces.
    for force in forces:
        # Get force name.
        force_name = force.__class__.__name__
        force1 = forces1[force_name]
        force2 = forces2[force_name]
        print force_name
        if force_name == 'HarmonicBondForce':
            # Add bonds that exist between molecules that are unique to molecule 2.
            for index in range(force2.getNumBonds()):
                [atom2_i, atom2_j, length, K] = force2.getBondParameters(index)
                if (atom2_i in set(unique2)) or (atom2_j in set(unique2)):
                    atom_i = molecule2_indices_in_system[atom2_i]
                    atom_j = molecule2_indices_in_system[atom2_j]
                    force.addBond(atom_i, atom_j, length, K)

            pass
        else:
            raise Exception("Force type %s unknown." % force_name)

    return [system, topology]


    return

def create_molecule(iupac_name):
    molecule = gaff2xml.openeye.iupac_to_oemol(iupac_name)
    import openeye.oeomega as om
    omega = om.OEOmega()
    omega.SetMaxConfs(1)
    omega(molecule)
    return molecule

if __name__ == '__main__':
    molecule1 = create_molecule('toluene')
    molecule2 = create_molecule('methoxytoluene')

    # Write molecules to mol2 files.
    gaff2xml.openeye.molecule_to_mol2(molecule1, tripos_mol2_filename='molecule1.mol2')
    gaff2xml.openeye.molecule_to_mol2(molecule2, tripos_mol2_filename='molecule2.mol2')

    [system1, topology1] = generate_openmm_system(molecule1)

    natoms = molecule1.NumAtoms()
    [system, topology] = create_relative_alchemical_transformation(system1, topology1, range(natoms), molecule1, molecule2)

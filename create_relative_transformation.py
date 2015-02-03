#!/usr/bin/env python
"""
Testbed for creating relative alchemical transformations.

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
       The system to be modified, already containing molecule1 whose atoms correspond to molecule1_indices_in_system.
    topology : simtk.openmm.app.Topology
       The topology object corresponding to system.
    molecule1_indices_in_system : list of int
       Indices of molecule1 in system, with atoms in same order.
    molecule1 : openeye.oechem.OEMol
       Molecule already present in system, where the atom mapping is given by molecule1_indices_in_system.
    molecule2 : openeye.oechem.OEMol
       New molecule that molecule1 will be transformed into as lambda parameter goes from 0 -> 1.

    Returns
    -------
    system : simtk.openmm.System
       Modified version of system in which old system is recovered for global context paramaeter `lambda` = 0 and new molecule is substituted for `lambda` = 1.
    topology : system.openmm.Topology
       Topology corresponding to system.

    """

    # Normalize molecules.
    # TODO: May need to do more normalization here.
    oe.OEPerceiveChiral(molecule1)
    oe.OEPerceiveChiral(molecule2)

    # Make copies to not destroy original objects.
    import copy
    system = copy.deepcopy(system)
    topology = copy.deepcopy(topology)

    # Create OpenMM Topology and System objects for given molecules using GAFF/AM1-BCC.
    # NOTE: This must generate the same forcefield parameters as occur in `system`.
    [system1, topology1] = generate_openmm_system(molecule1)
    [system2, topology2] = generate_openmm_system(molecule2)

    # Create lists of corresponding atoms for common substructure and groups specific to molecules 1 and 2.
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
    all1 = frozenset(range(molecule1.NumAtoms()))
    all2 = frozenset(range(molecule2.NumAtoms()))
    common1 = frozenset(mapping1.keys())
    common2 = frozenset(mapping2.keys())
    unique1 = all1 - common1
    unique2 = all2 - common2

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

        # TODO: Add new atoms to topology object as well.
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
            #
            # Process HarmonicBondForce
            #

            # Create index of bonds in system, system1, and system2.
            def unique(*args):
                if args[0] > args[-1]:
                    return tuple(reversed(args))
                else:
                    return tuple(args)

            def index_bonds(force):
                bonds = dict()
                for index in range(force.getNumBonds()):
                    [atom_i, atom_j, length, K] = force.getBondParameters(index)
                    key = unique(atom_i, atom_j) # unique tuple, possibly in reverse order
                    bonds[key] = index
                return bonds

            bonds  = index_bonds(force)   # index of bonds for system
            bonds1 = index_bonds(force1)  # index of bonds for system1
            bonds2 = index_bonds(force2)  # index of bonds for system2

            # Find bonds that are unique to each molecule.
            print "Finding bonds unique to each molecule..."
            unique_bonds1 = [ bonds1[atoms] for atoms in bonds1 if not set(atoms).issubset(common1) ]
            unique_bonds2 = [ bonds2[atoms] for atoms in bonds2 if not set(atoms).issubset(common2) ]

            # Build list of bonds shared among all molecules.
            print "Building a list of shared bonds..."
            shared_bonds = list()
            for atoms2 in bonds2:
                if set(atoms2).issubset(common2):
                    atoms  = tuple(molecule2_indices_in_system[atom2] for atom2 in atoms2)
                    atoms1 = tuple(mapping2[atom2] for atom2 in atoms2)
                    # Find bond index terms.
                    index  = bonds[unique(*atoms)]
                    index1 = bonds1[unique(*atoms1)]
                    index2 = bonds2[unique(*atoms2)]
                    # Store.
                    shared_bonds.append( (index, index1, index2) )

            # Add bonds that are unique to molecule2.
            print "Adding bonds unique to molecule2..."
            for index2 in unique_bonds2:
                [atom2_i, atom2_j, length2, K2] = force2.getBondParameters(index2)
                atom_i = molecule2_indices_in_system[atom2_i]
                atom_j = molecule2_indices_in_system[atom2_j]
                force.addBond(atom_i, atom_j, length2, K2)

            # Create a CustomBondForce to handle interpolated bond parameters.
            print "Creating CustomBondForce..."
            energy_expression  = '(K/2)*(r-length)**2;'
            energy_expression += 'K = (1-lambda)*K1 + lambda*K2;' # linearly interpolate spring constant
            energy_expression += 'length = (1-lambda)*length1 + lambda*length2;' # linearly interpolate bond length
            custom_force = mm.CustomBondForce(energy_expression)
            custom_force.addGlobalParameter('lambda', 0.0)
            custom_force.addPerBondParameter('length1') # molecule1 bond length
            custom_force.addPerBondParameter('K1') # molecule1 spring constant
            custom_force.addPerBondParameter('length2') # molecule2 bond length
            custom_force.addPerBondParameter('K2') # molecule2 spring constant
            system.addForce(custom_force)

            # Process bonds that are shared by molecule1 and molecule2.
            print "Translating shared bonds to CustomBondForce..."
            for (index, index1, index2) in shared_bonds:
                # Zero out standard bond force.
                [atom_i, atom_j, length, K] = force.getBondParameters(index)
                force.setBondParameters(index, atom_i, atom_j, length, K*0.0)
                # Create interpolated bond parameters.
                [atom1_i, atom1_j, length1, K1] = force1.getBondParameters(index1)
                [atom2_i, atom2_j, length2, K2] = force2.getBondParameters(index2)
                custom_force.addBond(atom_i, atom_j, [length1, K1, length2, K2])

        if force_name == 'HarmonicAngleForce':
            #
            # Process HarmonicAngleForce
            #

            # Create index of angles in system, system1, and system2.
            def unique(*args):
                if args[0] > args[-1]:
                    return tuple(reversed(args))
                else:
                    return tuple(args)

            def index_angles(force):
                angles = dict()
                for index in range(force.getNumAngles()):
                    [atom_i, atom_j, atom_k, angle, K] = force.getAngleParameters(index)
                    key = unique(atom_i, atom_j, atom_k) # unique tuple, possibly in reverse order
                    angles[key] = index
                return angles

            angles  = index_angles(force)   # index of angles for system
            angles1 = index_angles(force1)  # index of angles for system1
            angles2 = index_angles(force2)  # index of angles for system2

            # Find angles that are unique to each molecule.
            print "Finding angles unique to each molecule..."
            unique_angles1 = [ angles1[atoms] for atoms in angles1 if not set(atoms).issubset(common1) ]
            unique_angles2 = [ angles2[atoms] for atoms in angles2 if not set(atoms).issubset(common2) ]

            # Build list of angles shared among all molecules.
            print "Building a list of shared angles..."
            shared_angles = list()
            for atoms2 in angles2:
                if set(atoms2).issubset(common2):
                    atoms  = tuple(molecule2_indices_in_system[atom2] for atom2 in atoms2)
                    atoms1 = tuple(mapping2[atom2] for atom2 in atoms2)
                    # Find angle index terms.
                    index  = angles[unique(*atoms)]
                    index1 = angles1[unique(*atoms1)]
                    index2 = angles2[unique(*atoms2)]
                    # Store.
                    shared_angles.append( (index, index1, index2) )

            # Add angles that are unique to molecule2.
            print "Adding angles unique to molecule2..."
            for index2 in unique_angles2:
                [atom2_i, atom2_j, atom2_k, theta2, K2] = force2.getAngleParameters(index2)
                atom_i = molecule2_indices_in_system[atom2_i]
                atom_j = molecule2_indices_in_system[atom2_j]
                atom_k = molecule2_indices_in_system[atom2_k]
                force.addAngle(atom_i, atom_j, atom_k, theta2, K2)

            # Create a CustomAngleForce to handle interpolated angle parameters.
            print "Creating CustomAngleForce..."
            energy_expression  = '(K/2)*(theta-theta0)**2;'
            energy_expression += 'K = (1-lambda)*K_1 + lambda*K_2;' # linearly interpolate spring constant
            energy_expression += 'theta0 = (1-lambda)*theta0_1 + lambda*theta0_2;' # linearly interpolate equilibrium angle
            custom_force = mm.CustomAngleForce(energy_expression)
            custom_force.addGlobalParameter('lambda', 0.0)
            custom_force.addPerAngleParameter('theta0_1') # molecule1 equilibrium angle
            custom_force.addPerAngleParameter('K_1') # molecule1 spring constant
            custom_force.addPerAngleParameter('theta0_2') # molecule2 equilibrium angle
            custom_force.addPerAngleParameter('K_2') # molecule2 spring constant
            system.addForce(custom_force)

            # Process angles that are shared by molecule1 and molecule2.
            print "Translating shared angles to CustomAngleForce..."
            for (index, index1, index2) in shared_angles:
                # Zero out standard angle force.
                [atom_i, atom_j, atom_k, theta0, K] = force.getAngleParameters(index)
                force.setAngleParameters(index, atom_i, atom_j, atom_k, theta0, K*0.0)
                # Create interpolated angle parameters.
                [atom1_i, atom1_j, atom1_k, theta1, K1] = force1.getAngleParameters(index1)
                [atom2_i, atom2_j, atom2_k, theta2, K2] = force2.getAngleParameters(index2)
                custom_force.addAngle(atom_i, atom_j, atom_k, [theta1, K1, theta2, K2])

        if force_name == 'PeriodicTorsionForce':
            #
            # Process PeriodicTorsionForce
            #

            # Create index of torsions in system, system1, and system2.
            def unique(*args):
                if args[0] > args[-1]:
                    return tuple(reversed(args))
                else:
                    return tuple(args)

            def index_torsions(force):
                torsions = dict()
                for index in range(force.getNumTorsions()):
                    [atom_i, atom_j, atom_k, atom_l, periodicity, phase, K] = force.getTorsionParameters(index)
                    key = unique(atom_i, atom_j, atom_k, atom_l) # unique tuple, possibly in reverse order
                    torsions[key] = index
                return torsions

            torsions  = index_torsions(force)   # index of torsions for system
            torsions1 = index_torsions(force1)  # index of torsions for system1
            torsions2 = index_torsions(force2)  # index of torsions for system2

            # Find torsions that are unique to each molecule.
            print "Finding torsions unique to each molecule..."
            unique_torsions1 = [ torsions1[atoms] for atoms in torsions1 if not set(atoms).issubset(common1) ]
            unique_torsions2 = [ torsions2[atoms] for atoms in torsions2 if not set(atoms).issubset(common2) ]

            # Build list of torsions shared among all molecules.
            print "Building a list of shared torsions..."
            shared_torsions = list()
            for atoms2 in torsions2:
                if set(atoms2).issubset(common2):
                    atoms  = tuple(molecule2_indices_in_system[atom2] for atom2 in atoms2)
                    atoms1 = tuple(mapping2[atom2] for atom2 in atoms2)
                    # Find torsion index terms.
                    index  = torsions[unique(*atoms)]
                    index1 = torsions1[unique(*atoms1)]
                    index2 = torsions2[unique(*atoms2)]
                    # Store.
                    shared_torsions.append( (index, index1, index2) )

            # Add torsions that are unique to molecule2.
            print "Adding torsions unique to molecule2..."
            for index2 in unique_torsions2:
                [atom2_i, atom2_j, atom2_k, atom2_l, periodicity2, phase2, K2] = force2.getTorsionParameters(index2)
                atom_i = molecule2_indices_in_system[atom2_i]
                atom_j = molecule2_indices_in_system[atom2_j]
                atom_k = molecule2_indices_in_system[atom2_k]
                atom_l = molecule2_indices_in_system[atom2_l]
                force.addTorsion(atom_i, atom_j, atom_k, atom_l, periodicity2, phase2, K2)

            # Create a CustomTorsionForce to handle interpolated torsion parameters.
            print "Creating CustomTorsionForce..."
            energy_expression  = '(1-lambda)*U1 + lambda*U2;'
            energy_expression += 'U1 = K1*(1+cos(periodicity1*theta-phase1));'
            energy_expression += 'U2 = K2*(1+cos(periodicity2*theta-phase2));'
            custom_force = mm.CustomTorsionForce(energy_expression)
            custom_force.addGlobalParameter('lambda', 0.0)
            custom_force.addPerTorsionParameter('periodicity1') # molecule1 periodicity
            custom_force.addPerTorsionParameter('phase1') # molecule1 phase
            custom_force.addPerTorsionParameter('K1') # molecule1 spring constant
            custom_force.addPerTorsionParameter('periodicity2') # molecule2 periodicity
            custom_force.addPerTorsionParameter('phase2') # molecule2 phase
            custom_force.addPerTorsionParameter('K2') # molecule2 spring constant
            system.addForce(custom_force)

            # Process torsions that are shared by molecule1 and molecule2.
            print "Translating shared torsions to CustomTorsionForce..."
            for (index, index1, index2) in shared_torsions:
                # Zero out standard torsion force.
                [atom_i, atom_j, atom_k, atom_l, periodicity, phase, K] = force.getTorsionParameters(index)
                force.setTorsionParameters(index, atom_i, atom_j, atom_k, atom_l, periodicity, phase, K*0.0)
                # Create interpolated torsion parameters.
                [atom1_i, atom1_j, atom1_k, atom1_l, periodicity1, phase1, K1] = force1.getTorsionParameters(index1)
                [atom2_i, atom2_j, atom2_k, atom2_l, periodicity2, phase2, K2] = force2.getTorsionParameters(index2)
                custom_force.addTorsion(atom_i, atom_j, atom_k, atom_l, [periodicity1, phase1, K1, periodicity2, phase2, K2])
        else:
            #raise Exception("Force type %s unknown." % force_name)
            pass

    return [system, topology]

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

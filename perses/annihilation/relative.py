import simtk.openmm as mm
from simtk import unit
import simtk.openmm.app as app
import numpy as np
import copy

ONE_4PI_EPS0 = 138.935456 # OpenMM constant for Coulomb interactions (openmm/platforms/reference/include/SimTKOpenMMRealType.h) in OpenMM units

def unique(atom_list):
    if atom_list[0] > atom_list[-1]:
        return tuple(reversed(atom_list))
    else:
        return tuple(atom_list)

class HybridTopologyFactory(object):
    def __init__(self, system1, system2, topology1, topology2, positions1, positions2, atom_mapping_1to2):
        """
        Arguments:
            system1
            system2
            topology1
            topology2
            positions1
            positions2
            atom_mapping_1to2 : dict[atom_index_in_system1] = atom_index_in_system2
        """
        self.softcore_alpha=0.5
        self.softcore_beta=12*unit.angstrom**2
        self.system1 = copy.deepcopy(system1)
        self.system2 = copy.deepcopy(system2)

        self.return_self = False
        if topology1 == topology2:
            self.return_self = True
        self.topology1 = copy.deepcopy(topology1)
        self.topology2 = copy.deepcopy(topology2)

        system2_atoms = dict()
        for atom2 in self.topology2.atoms():
            system2_atoms[atom2.index] = atom2
        self.system2_atoms = system2_atoms
        system1_atoms = dict()
        for atom1 in self.topology1.atoms():
            system1_atoms[atom1.index] = atom1
        self.system1_atoms = system1_atoms

        self.positions1 = copy.deepcopy(positions1)
        self.positions2 = copy.deepcopy(positions2)
        self.atom_mapping_1to2 = copy.deepcopy(atom_mapping_1to2)
        keys_to_delete = list()
        for atom1idx, atom2idx in self.atom_mapping_1to2.items():
            atom1 = system1_atoms[atom1idx]
            atom2 = system2_atoms[atom2idx]
            if not atom1.name == atom2.name:
                if atom1.element == atom2.element and atom1.element == app.Element.getBySymbol('H'):
                    continue
                keys_to_delete.append(atom1idx)
        for key in keys_to_delete:
            del(self.atom_mapping_1to2[key])

        self.atom_mapping_2to1 = {old_atom : new_atom for new_atom, old_atom in self.atom_mapping_1to2.items()}
        self.unique_atoms1 = [atom for atom in range(topology1._numAtoms) if atom not in self.atom_mapping_1to2.keys()]
        self.unique_atoms2 = [atom for atom in range(topology2._numAtoms) if atom not in self.atom_mapping_1to2.values()]

        self.verbose = False

    def _handle_constraints(self, system, system2, sys2_indices_in_system):
        if self.verbose: print("Adding constraints from system2...")
        for index in range(system2.getNumConstraints()):
            # Extract constraint distance from system2.
            [atom2_i, atom2_j, distance] = system.getConstraintParameters(index)
                # Map atoms from system2 into system.
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            # Add constraint to system.
            system.addConstraint(atom_i, atom_j, distance)
        return system

    def _create_new_positions_array(self, topology, positions, sys1_indices_in_system, sys2_indices_in_system):
        natoms = positions.shape[0] + len(self.unique_atoms2) # new number of atoms
        positions = unit.Quantity(np.resize(positions/positions.unit, [natoms,3]), positions.unit)
        index_in_sys1 = { value: key for key, value in sys1_indices_in_system.items() }
        index_in_sys2 = { value: key for key, value in sys2_indices_in_system.items() }
        for atom in topology.atoms():
            if atom.index in index_in_sys1.keys():
                atomidx = index_in_sys1[atom.index]
                position = self.positions1[atomidx]
            elif atom.index in index_in_sys2.keys():
                atomidx = index_in_sys2[atom.index]
                position = self.positions2[atomidx]
            else:
                raise Exception('Atom not found to assign position')
            positions[atom.index] = position
        # julie debugging 8/3/16
        with open('alchemy.txt','w') as fo:
            for atom in topology.atoms():
                position = positions[atom.index]
                fo.write('%s, %s, %s\n' % (atom.residue, atom.name, position))
        app.PDBFile.writeFile(topology, positions, open('alchemy.pdb','w'))
        # end debug
        return positions

    #######################
    # HARMONIC BOND FORCE #
    #######################
    def _harmonic_bond_find_shared(self, common2, sys2_indices_in_system, mapping2, bonds, bonds1, bonds2):
        if self.verbose: print("Building a list of shared bonds...")
        shared_bonds = list()
        for atoms2 in bonds2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms1 = [mapping2[atom2] for atom2 in atoms2]
                # Find bond index terms.
                index  = bonds[unique(atoms1)]
                index1 = bonds1[unique(atoms1)]
                index2 = bonds2[unique(atoms2)]
                # Store.
                shared_bonds.append( (index, index1, index2) )
        return shared_bonds

    def _harmonic_bond_custom_force(self):
        # Create a CustomBondForce to handle interpolated bond parameters.
        if self.verbose: print("Creating CustomBondForce...")
        energy_expression  = '(K/2)*(r-length)^2;'
        energy_expression += 'K = (1-lambda_bonds)*K1 + lambda_bonds*K2;' # linearly interpolate spring constant
        energy_expression += 'length = (1-lambda_bonds)*length1 + lambda_bonds*length2;' # linearly interpolate bond length
        custom_force = mm.CustomBondForce(energy_expression)
        custom_force.addGlobalParameter('lambda_bonds', 0.0)
        custom_force.addPerBondParameter('length1') # molecule1 bond length
        custom_force.addPerBondParameter('K1') # molecule1 spring constant
        custom_force.addPerBondParameter('length2') # molecule2 bond length
        custom_force.addPerBondParameter('K2') # molecule2 spring constant
        return custom_force

    def _harmonic_bond_add_core(self, shared_bonds, sys2_indices_in_system, force1, force2, custom_force):
        # Process bonds that are shared by molecule1 and molecule2.
        if self.verbose: print("Translating shared bonds to CustomBondForce...")
        for (index, index1, index2) in shared_bonds:
            # Create interpolated bond parameters.
            [atom1_i, atom1_j, length1, K1] = force1.getBondParameters(index1)
            [atom2_i, atom2_j, length2, K2] = force2.getBondParameters(index2)
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            custom_force.addBond(atom_i, atom_j, [length1, K1, length2, K2])

    def _harmonic_bond_add_unique(self, unique_bonds2, unique_bonds1, force2, force1, sys2_indices_in_system, sys1_indices_in_system, custom_force):
        if self.verbose: print("Adding custom parameters to unique bonds...")
        for index2 in unique_bonds2:
            [atom2_i, atom2_j, length2, K2] = force2.getBondParameters(index2)
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            #custom_force.addBond(atom_i, atom_j, [length2, 0.1*K2, length2, K2])
            custom_force.addBond(atom_i, atom_j, [length2, 0.0*K2, length2, K2])
    
        for index1 in unique_bonds1:
            [atom1_i, atom1_j, length1, K1] = force1.getBondParameters(index1)
            atom_i = sys1_indices_in_system[atom1_i]
            atom_j = sys1_indices_in_system[atom1_j]
            #custom_force.addBond(atom_i, atom_j, [length1, K1, length1, 0.1*K1])
            custom_force.addBond(atom_i, atom_j, [length1, K1, length1, 0.0*K1])

    def _harmonic_bond_force(self, force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping2, system):
        def index_bonds(force):
            bonds = dict()
            for index in range(force.getNumBonds()):
                [atom_i, atom_j, length, K] = force.getBondParameters(index)
                key = unique([atom_i, atom_j]) # unique tuple, possibly in reverse order
                bonds[key] = index
            return bonds

        bonds  = index_bonds(force)   # index of bonds for system
        bonds1 = index_bonds(force1)  # index of bonds for system1
        bonds2 = index_bonds(force2)  # index of bonds for system2

        # Find bonds that are unique to each molecule.
        if self.verbose: print("Finding bonds unique to each molecule...")
        unique_bonds1 = [ bonds1[atoms] for atoms in bonds1 if not set(atoms).issubset(common1) ]
        unique_bonds2 = [ bonds2[atoms] for atoms in bonds2 if not set(atoms).issubset(common2) ]

        for atoms, index in bonds.items():
            [atom_i, atom_j, length, K] = force.getBondParameters(index)
            force.setBondParameters(index, atom_i, atom_j, length, 0*K)
 
        shared_bonds = self._harmonic_bond_find_shared(common2, sys2_indices_in_system, mapping2, bonds, bonds1, bonds2)
    
        custom_force = self._harmonic_bond_custom_force()
        system.addForce(custom_force)

        self._harmonic_bond_add_core(shared_bonds, sys2_indices_in_system, force1, force2, custom_force)    
        self._harmonic_bond_add_unique(unique_bonds2, unique_bonds1, force2, force1, sys2_indices_in_system, sys1_indices_in_system, custom_force)
        system.removeForce(0)

    ########################
    # HARMONIC ANGLE FORCE #
    ########################
    def _harmonic_angle_find_shared(self, common2, sys2_indices_in_system, mapping2, angles, angles1, angles2):
        # Build list of angles shared among all molecules.
        if self.verbose: print("Building a list of shared angles...")
        shared_angles = list()
        for atoms2 in angles2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms1 = [mapping2[atom2] for atom2 in atoms2]
                # Find angle index terms.
                index  = angles[unique(atoms1)]
                index1 = angles1[unique(atoms1)]
                index2 = angles2[unique(atoms2)]
                # Store.
                shared_angles.append( (index, index1, index2) )
        return shared_angles

    def _harmonic_angle_custom_force(self):
        # Create a CustomAngleForce to handle interpolated angle parameters.
        if self.verbose: print("Creating CustomAngleForce...")
        energy_expression  = '(K/2)*(theta-theta0)^2;'
        energy_expression += 'K = (1.0-lambda_angles)*K_1 + lambda_angles*K_2;' # linearly interpolate spring constant
        energy_expression += 'theta0 = (1.0-lambda_angles)*theta0_1 + lambda_angles*theta0_2;' # linearly interpolate equilibrium angle
        custom_force = mm.CustomAngleForce(energy_expression)
        custom_force.addGlobalParameter('lambda_angles', 0.0)
        custom_force.addPerAngleParameter('theta0_1') # molecule1 equilibrium angle
        custom_force.addPerAngleParameter('K_1') # molecule1 spring constant
        custom_force.addPerAngleParameter('theta0_2') # molecule2 equilibrium angle
        custom_force.addPerAngleParameter('K_2') # molecule2 spring constant
        return custom_force

    def _harmonic_angle_add_core(self, shared_angles, sys1_indices_in_system, force1, force2, custom_force):
        # Process angles that are shared by molecule1 and molecule2.
        if self.verbose: print("Translating shared angles to CustomAngleForce...")
        for (index, index1, index2) in shared_angles:
            # Create interpolated angle parameters.
            [atom1_i, atom1_j, atom1_k, theta1, K1] = force1.getAngleParameters(index1)
            [atom2_i, atom2_j, atom2_k, theta2, K2] = force2.getAngleParameters(index2)
            atom_i = sys1_indices_in_system[atom1_i]
            atom_j = sys1_indices_in_system[atom1_j]
            atom_k = sys1_indices_in_system[atom1_k]
            custom_force.addAngle(atom_i, atom_j, atom_k, [theta1, K1, theta2, K2])

    def _harmonic_angle_add_unique(self, unique_angles2, unique_angles1, force2, force1, sys2_indices_in_system, sys1_indices_in_system, custom_force):
        if self.verbose: print("Adding custom parameters to unique angles...")
        for index2 in unique_angles2:
            [atom2_i, atom2_j, atom2_k, theta2, K2] = force2.getAngleParameters(index2)
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            atom_k = sys2_indices_in_system[atom2_k]
#            custom_force.addAngle(atom_i, atom_j, atom_k, [theta2, 0.1*K2, theta2, K2])
            custom_force.addAngle(atom_i, atom_j, atom_k, [theta2, 0.0*K2, theta2, K2])
        for index1 in unique_angles1:
            [atom1_i, atom1_j, atom1_k, theta1, K1] = force1.getAngleParameters(index1)
            atom_i = sys1_indices_in_system[atom1_i]
            atom_j = sys1_indices_in_system[atom1_j]
            atom_k = sys1_indices_in_system[atom1_k]
#            custom_force.addAngle(atom_i, atom_j, atom_k, [theta1, K1, theta1, 0.1*K1])
            custom_force.addAngle(atom_i, atom_j, atom_k, [theta1, K1, theta1, 0.0*K1])

    def _harmonic_angle_force(self, force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping2, system):
        def index_angles(force):
            angles = dict()
            for index in range(force.getNumAngles()):
                [atom_i, atom_j, atom_k, angle, K] = force.getAngleParameters(index)
                key = unique([atom_i, atom_j, atom_k]) # unique tuple, possibly in reverse order
                angles[key] = index
            return angles

        angles  = index_angles(force)   # index of angles for system
        angles1 = index_angles(force1)  # index of angles for system1
        angles2 = index_angles(force2)  # index of angles for system2

        # Find angles that are unique to each molecule.
        if self.verbose: print("Finding angles unique to each molecule...")
        unique_angles1 = [ angles1[atoms] for atoms in angles1 if not set(atoms).issubset(common1) ]
        unique_angles2 = [ angles2[atoms] for atoms in angles2 if not set(atoms).issubset(common2) ]

        shared_angles = self._harmonic_angle_find_shared(common2, sys2_indices_in_system, mapping2, angles, angles1, angles2)

        if self.verbose: print("Removing existing angle parameters...")
        for index in range(force.getNumAngles()):
            [atom_i, atom_j, atom_k, angle, K] = force.getAngleParameters(index)
            force.setAngleParameters(index, atom_i, atom_j, atom_k, angle, 0*K)

        custom_force = self._harmonic_angle_custom_force() 
        system.addForce(custom_force)
    
        self._harmonic_angle_add_core(shared_angles, sys1_indices_in_system, force1, force2, custom_force)
        self._harmonic_angle_add_unique(unique_angles2, unique_angles1, force2, force1, sys2_indices_in_system, sys1_indices_in_system, custom_force)
        system.removeForce(0)

    ##########################
    # PERIODIC TORSION FORCE #
    ##########################
    def _periodic_torsion_find_shared(self, common2, unique_torsions1, unique_torsions2, sys1_indices_in_system, mapping2, torsions, torsions1, torsions2, system_atoms):
        # Build list of torsions shared among all molecules.
        if self.verbose: print("Building a list of shared torsions...")
        shared_torsions = list()
        for atoms2 in torsions2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms1 = [mapping2[atom2] for atom2 in atoms2]
                # Find torsion index terms.
                try:
                    index  = torsions[unique(atoms1)]
                except Exception as e:
                    if self.verbose: print("Warning: problem occurred in building a list of torsions common to all molecules -- SYSTEM.")
                    atom_names = [system_atoms[sys1_indices_in_system[atom]] for atom in atoms1]
                    if self.verbose: print(atom_names)
                    try:
                        index1 = torsions1[unique(atoms1)]
                        if self.verbose: print("ERROR: torsion present in SYSTEM 1, not copied to SYSTEM.")
                        if self.verbose: print("torsions :  %s" % str(unique(atoms1)))
                        if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                        if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                        raise(e)
                    except:
                        try:
                            index2 = torsions2[unique(atoms2)]
                            unique_torsions2.append(torsions2[unique(atoms2)]) # so this will never catch unique torsions from 1 that use core atoms?
                            continue
                        except:
                            if self.verbose: print("ERROR: the torsion does not exist.")
                            if self.verbose: print("torsions :  %s" % str(unique(atoms1)))
                            if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                            if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                            raise(e)
                try:
                    index1 = torsions1[unique(atoms1)]
                except Exception as e:
                    if self.verbose: print("Error occurred in building a list of torsions common to all molecules -- SYSTEM 1.")
                    if self.verbose: print("torsions :  %s" % str(unique(atoms1)))
                    if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                    if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                    raise(e)
                try:
                    index2 = torsions2[unique(atoms2)]
                except Exception as e:
                    if self.verbose: print("Error occurred in building a list of torsions common to all molecules -- SYSTEM 2.")
                    if self.verbose: print("torsions :  %s" % str(unique(atoms1)))
                    if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                    if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                    raise(e)

                shared_torsions.append( (index, index1, index2) )
        return shared_torsions, unique_torsions1, unique_torsions2

    def _periodic_torsion_custom_force(self):
        # Create a CustomTorsionForce to handle interpolated torsion parameters.
        if self.verbose: print("Creating CustomTorsionForce...")
        energy_expression  = '(1-lambda_torsions)*U1 + lambda_torsions*U2;'
        energy_expression += 'U1 = K1*(1+cos(periodicity1*theta-phase1));'
        energy_expression += 'U2 = K2*(1+cos(periodicity2*theta-phase2));'
        custom_force = mm.CustomTorsionForce(energy_expression)
        custom_force.addGlobalParameter('lambda_torsions', 0.0)
        custom_force.addPerTorsionParameter('periodicity1') # molecule1 periodicity
        custom_force.addPerTorsionParameter('phase1') # molecule1 phase
        custom_force.addPerTorsionParameter('K1') # molecule1 spring constant
        custom_force.addPerTorsionParameter('periodicity2') # molecule2 periodicity
        custom_force.addPerTorsionParameter('phase2') # molecule2 phase
        custom_force.addPerTorsionParameter('K2') # molecule2 spring constant
        return custom_force

    def _periodic_torsion_add_core(self, shared_torsions, sys1_indices_in_system, force1, force2, custom_force):
        # Process torsions that are shared by molecule1 and molecule2.
        if self.verbose: print("Translating shared torsions to CustomTorsionForce...")
        for (index, index1, index2) in shared_torsions:
            for ix1 in index1:
            # Create interpolated torsion parameters.
                [atom1_i, atom1_j, atom1_k, atom1_l, periodicity1, phase1, K1] = force1.getTorsionParameters(ix1)
                # only have to do this once -- it's the same set of atoms for both lists
                atom_i = sys1_indices_in_system[atom1_i]
                atom_j = sys1_indices_in_system[atom1_j]
                atom_k = sys1_indices_in_system[atom1_k]
                atom_l = sys1_indices_in_system[atom1_l]
                custom_force.addTorsion(atom_i, atom_j, atom_k, atom_l, [periodicity1, phase1, K1, periodicity1, phase1, 0.0*K1])
            for ix2 in index2:
                [atom2_i, atom2_j, atom2_k, atom2_l, periodicity2, phase2, K2] = force2.getTorsionParameters(ix2)
                custom_force.addTorsion(atom_i, atom_j, atom_k, atom_l, [periodicity2, phase2, 0.0*K2, periodicity2, phase2, K2])

    def _periodic_torsion_add_unique(self, unique_torsions2, unique_torsions1, force2, force1, sys2_indices_in_system, sys1_indices_in_system, custom_force):
        # Add torsions that are unique to molecule2.
        if self.verbose: print("Adding torsions unique to molecule2...")
        for index2 in unique_torsions2:
            for ix2 in index2:
                [atom2_i, atom2_j, atom2_k, atom2_l, periodicity2, phase2, K2] = force2.getTorsionParameters(ix2)
                atom_i = sys2_indices_in_system[atom2_i]
                atom_j = sys2_indices_in_system[atom2_j]
                atom_k = sys2_indices_in_system[atom2_k]
                atom_l = sys2_indices_in_system[atom2_l]
                custom_force.addTorsion(atom_i, atom_j, atom_k, atom_l, [periodicity2, phase2, 0.0, periodicity2, phase2, K2])
        for index1 in unique_torsions1:
            for ix1 in index1:
                [atom1_i, atom1_j, atom1_k, atom1_l, periodicity1, phase1, K1] = force1.getTorsionParameters(ix1)
                atom_i = sys1_indices_in_system[atom1_i]
                atom_j = sys1_indices_in_system[atom1_j]
                atom_k = sys1_indices_in_system[atom1_k]
                atom_l = sys1_indices_in_system[atom1_l]
                custom_force.addTorsion(atom_i, atom_j, atom_k, atom_l, [periodicity1, phase1, K1, periodicity1, phase1, 0.0])

    def _periodic_torsion_force(self, force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping2, system, system_atoms):
        def index_torsions(force):
            torsions = dict()
            for index in range(force.getNumTorsions()):
                [atom_i, atom_j, atom_k, atom_l, periodicity, phase, K] = force.getTorsionParameters(index)
                key = unique([atom_i, atom_j, atom_k, atom_l]) # unique tuple, possibly in reverse order
                if key not in torsions.keys():
                    torsions[key] = list()
                torsions[key].append(index)
            return torsions

        torsions  = index_torsions(force)   # index of torsions for system
        torsions1 = index_torsions(force1)  # index of torsions for system1
        torsions2 = index_torsions(force2)  # index of torsions for system2

        # Find torsions that are unique to each molecule.
        if self.verbose: print("Finding torsions unique to each molecule...")
        unique_torsions1 = [ torsions1[atoms] for atoms in torsions1 if not set(atoms).issubset(common1) ]
        unique_torsions2 = [ torsions2[atoms] for atoms in torsions2 if not set(atoms).issubset(common2) ]

        shared_torsions, unique_torsions1, unique_torsions2 = self._periodic_torsion_find_shared(common2, unique_torsions1, unique_torsions2, sys1_indices_in_system, mapping2, torsions, torsions1, torsions2, system_atoms)

        assert len(shared_torsions) + len(unique_torsions1) == len(torsions1.keys())
        assert len(shared_torsions) + len(unique_torsions2) == len(torsions2.keys())

        if self.verbose: print("Removing existing torsion parameters...")
        for index in range(force.getNumTorsions()):
            [atom1_i, atom1_j, atom1_k, atom1_l, periodicity, phase, K] = force.getTorsionParameters(index)
            atom_i = sys1_indices_in_system[atom1_i]
            atom_j = sys1_indices_in_system[atom1_j]
            atom_k = sys1_indices_in_system[atom1_k]
            atom_l = sys1_indices_in_system[atom1_l]
            force.setTorsionParameters(index, atom_i, atom_j, atom_k, atom_l, periodicity, phase, 0*K)
        custom_force = self._periodic_torsion_custom_force()
        system.addForce(custom_force)

        self._periodic_torsion_add_core(shared_torsions, sys1_indices_in_system, force1, force2, custom_force)
        self._periodic_torsion_add_unique(unique_torsions2, unique_torsions1, force2, force1, sys2_indices_in_system, sys1_indices_in_system, custom_force)
        system.removeForce(0)

    ###################
    # NONBONDED FORCE #
    ###################
    def _nonbonded_find_shared(self, common2, sys2_indices_in_system, mapping2, exceptions, exceptions1, exceptions2):
        # Build list of exceptions shared among all molecules.
        if self.verbose: print("Building a list of shared exceptions...")
        shared_exceptions = list()
        for atoms2 in exceptions2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms1 = tuple(mapping2[atom2] for atom2 in atoms2)
                # Find exception index terms.
                try:
                    index  = exceptions[unique(atoms1)]
                    index1 = exceptions1[unique(atoms1)]
                    index2 = exceptions2[unique(atoms2)]
                    # Store.
                    shared_exceptions.append( (index, index1, index2) )
                except:
                    pass
        return shared_exceptions

    def _nonbonded_custom_sterics_allmethods(self):
        # Add additional definitions common to all methods.
        sterics_addition = "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;" #interpolation
        sterics_addition += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
        sterics_addition += "softcore_alpha = %f;" % self.softcore_alpha
        sterics_addition += "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
        sterics_addition += "lambda_alpha = lambda_sterics*(1-lambda_sterics);"
        return sterics_addition

    def _nonbonded_custom_electro_allmethods(self):
        # Add additional definitions common to all methods.
        electrostatics_addition = "chargeprod = (1-lambda_electrostatics)*chargeprodA + lambda_electrostatics*chargeprodB;" #interpolation
        electrostatics_addition += "reff_electrostatics = sqrt(softcore_beta*lambda_beta + r^2);" # effective softcore distance for electrostatics
        electrostatics_addition += "softcore_beta = %f;" % (self.softcore_beta / self.softcore_beta.in_unit_system(unit.md_unit_system).unit)
        electrostatics_addition += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0 # already in OpenMM units
        electrostatics_addition += "lambda_beta = lambda_electrostatics*(1-lambda_electrostatics);"
        return electrostatics_addition

    def _nonbonded_custom_nocutoff(self):
        # soft-core Lennard-Jones
        sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
        # soft-core Coulomb
        electrostatics_energy_expression = "U_electrostatics = ONE_4PI_EPS0*chargeprod/reff_electrostatics;"
        return sterics_energy_expression, electrostatics_energy_expression

    def _nonbonded_custom_cutoff(self, force):
        # soft-core Lennard-Jones
        sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
        # reaction-field electrostatics
        epsilon_solvent = force.getReactionFieldDielectric()
        r_cutoff = force.getCutoffDistance()
        electrostatics_energy_expression = "U_electrostatics = ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);"
        k_rf = r_cutoff**(-3) * ((epsilon_solvent - 1) / (2*epsilon_solvent + 1))
        c_rf = r_cutoff**(-1) * ((3*epsilon_solvent) / (2*epsilon_solvent + 1))
        electrostatics_energy_expression += "k_rf = %f;" % (k_rf / k_rf.in_unit_system(unit.md_unit_system).unit)
        electrostatics_energy_expression += "c_rf = %f;" % (c_rf / c_rf.in_unit_system(unit.md_unit_system).unit)
        return sterics_energy_expression, electrostatics_energy_expression

    def _nonbonded_custom_ewald(self, force):
        # soft-core Lennard-Jones
        sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
        # Ewald direct-space electrostatics
        [alpha_ewald, nx, ny, nz] = force.getPMEParameters()
        if alpha_ewald == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
            delta = force.getEwaldErrorTolerance()
            r_cutoff = force.getCutoffDistance()
            alpha_ewald = np.sqrt(-np.log(2*delta)) / r_cutoff
        electrostatics_energy_expression = "U_electrostatics = ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
        electrostatics_energy_expression += "alpha_ewald = %f;" % (alpha_ewald / alpha_ewald.in_unit_system(unit.md_unit_system).unit)
        # TODO: Handle reciprocal-space electrostatics
        return sterics_energy_expression, electrostatics_energy_expression

    def _nonbonded_custom_mixing_rules(self):
        # Define mixing rules.
        sterics_mixing_rules = "epsilonA = sqrt(epsilonA1*epsilonA2);" # mixing rule for epsilon
        sterics_mixing_rules += "epsilonB = sqrt(epsilonB1*epsilonB2);" # mixing rule for epsilon
        sterics_mixing_rules += "sigmaA = 0.5*(sigmaA1 + sigmaA2);" # mixing rule for sigma
        sterics_mixing_rules += "sigmaB = 0.5*(sigmaB1 + sigmaB2);" # mixing rule for sigma
        electrostatics_mixing_rules = "chargeprodA = chargeA1*chargeA2;" # mixing rule for charges
        electrostatics_mixing_rules += "chargeprodB = chargeB1*chargeB2;" # mixing rule for charges
        return sterics_mixing_rules, electrostatics_mixing_rules

    def _nonbonded_custom_force(self, force):
        """
        Create 2 custom force instances (steric and electrostatic)
        depending on the nonbonded method of the system
        """
        # Create a CustomNonbondedForce to handle alchemically interpolated nonbonded parameters.
        # Select functional form based on nonbonded method.
        method = force.getNonbondedMethod()
        if method in [mm.NonbondedForce.NoCutoff]:
            sterics_energy_expression, electrostatics_energy_expression = self._nonbonded_custom_nocutoff()
        elif method in [mm.NonbondedForce.CutoffPeriodic, mm.NonbondedForce.CutoffNonPeriodic]:
            sterics_energy_expression, electrostatics_energy_expression = self._nonbonded_custom_cutoff(force)
        elif method in [mm.NonbondedForce.PME, mm.NonbondedForce.Ewald]:
            sterics_energy_expression, electrostatics_energy_expression = self._nonbonded_custom_ewald(force)
        else:
            raise Exception("Nonbonded method %s not supported yet." % str(method))
        sterics_energy_expression += self._nonbonded_custom_sterics_allmethods()
        electrostatics_energy_expression += self._nonbonded_custom_electro_allmethods()

        sterics_mixing_rules, electrostatics_mixing_rules = self._nonbonded_custom_mixing_rules()

        # Create CustomNonbondedForce to handle interactions between alchemically-modified atoms and rest of system.
        electrostatics_custom_nonbonded_force = mm.CustomNonbondedForce("U_electrostatics;" + electrostatics_energy_expression + electrostatics_mixing_rules)
        electrostatics_custom_nonbonded_force.addGlobalParameter("lambda_electrostatics", 0.0);
        electrostatics_custom_nonbonded_force.addPerParticleParameter("chargeA") # partial charge initial
        electrostatics_custom_nonbonded_force.addPerParticleParameter("chargeB") # partial charge final
        sterics_custom_nonbonded_force = mm.CustomNonbondedForce("U_sterics;" + sterics_energy_expression + sterics_mixing_rules)
        sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics", 0.0);
        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaA") # Lennard-Jones sigma initial
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonA") # Lennard-Jones epsilon initial
        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaB") # Lennard-Jones sigma final
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonB") # Lennard-Jones epsilon final
        return electrostatics_custom_nonbonded_force, sterics_custom_nonbonded_force

    def _nonbonded_add_core(self, common1, mapping1, sys1_indices_in_system, force, force1, force2, sterics_custom_nonbonded_force, electrostatics_custom_nonbonded_force):
        """
        Define the intra-core alchemical interactions
        """
        # Copy over the right number of non-interacting particles
        for particle_index in range(force.getNumParticles()):
            sterics_custom_nonbonded_force.addParticle([1.0, 0.0, 1.0, 0.0])
            electrostatics_custom_nonbonded_force.addParticle([0.0, 0.0])
        # Copy over parameters for common substructure.
        for atom1 in common1:
            atom2 = mapping1[atom1] # index into system2
            index = sys1_indices_in_system[atom1] # index into system
            [charge1, sigma1, epsilon1] = force1.getParticleParameters(atom1)
            [charge2, sigma2, epsilon2] = force2.getParticleParameters(atom2)
            sterics_custom_nonbonded_force.setParticleParameters(index, [sigma1, epsilon1, sigma2, epsilon2])
            electrostatics_custom_nonbonded_force.setParticleParameters(index, [charge1, charge2])
        core = [sys1_indices_in_system[atom1] for atom1 in common1]
        sterics_custom_nonbonded_force.addInteractionGroup(core, core)
        electrostatics_custom_nonbonded_force.addInteractionGroup(core, core)

    def _nonbonded_add_unique(self, common1, force1, force2, sys2_indices_in_system, sys1_indices_in_system, sterics_custom_nonbonded_force, electrostatics_custom_nonbonded_force):
        """
        For the custom forces, unique atoms interact only with core atoms
        unique atoms with each other are dealt with in the original force
        """
        core = [sys1_indices_in_system[atom1] for atom1 in common1]
        alchemical1 = [sys1_indices_in_system[atom1] for atom1 in self.unique_atoms1]
        alchemical2 = [sys2_indices_in_system[atom2] for atom2 in self.unique_atoms2]
        alchemicals = alchemical1 + alchemical2
        sterics_custom_nonbonded_force.addInteractionGroup(core, alchemicals)
        electrostatics_custom_nonbonded_force.addInteractionGroup(core, alchemicals)
        for atom1 in self.unique_atoms1:
            index = sys1_indices_in_system[atom1] # index into system
            [charge1, sigma1, epsilon1] = force1.getParticleParameters(atom1)
            sterics_custom_nonbonded_force.setParticleParameters(index, [sigma1, epsilon1, sigma1, 0*epsilon1])
            electrostatics_custom_nonbonded_force.setParticleParameters(index, [charge1, 0*charge1])
        for atom2 in self.unique_atoms2:
            index = sys2_indices_in_system[atom2] # index into system
            [charge2, sigma2, epsilon2] = force2.getParticleParameters(atom2)
            sterics_custom_nonbonded_force.setParticleParameters(index, [sigma2, 0*epsilon2, sigma2, epsilon2])
            electrostatics_custom_nonbonded_force.setParticleParameters(index, [0*charge2, charge2])

    def _nonbonded_exclude_uniques(self, force, sys2_indices_in_system, sys1_indices_in_system):#, electrostatics_custom_nonbonded_force, sterics_custom_nonbonded_force):
        """
        The intra-unique bits of each molecule maintain original
        nonbonded forces; should not see each other
        Not needed for custom forces, because they are never
        in the same InteractionGroup
        """
        # Add exclusions between unique parts of molecule1 and molecule2 so they do not interact.
        if self.verbose: print("Add exclusions between unique parts of molecule1 and molecule2 that should not interact...")
        for atom1_i in self.unique_atoms1:
            for atom2_j in self.unique_atoms2:
                atom_i = sys1_indices_in_system[atom1_i]
                atom_j = sys2_indices_in_system[atom2_j]
                #electrostatics_custom_nonbonded_force.addExclusion(atom_i, atom_j)
                #sterics_custom_nonbonded_force.addExclusion(atom_i, atom_j)
                force.addException(atom_i, atom_j, 0.0, 1.0, 0.0, replace=True)

    def _nonbonded_fix_noncustom(self, force, force1, force2, unique_exceptions1, unique_exceptions2, sys1_indices_in_system, sys2_indices_in_system):
        """
        Want to keep the unique atoms in a regular (non-custom)
        nonbonded force, but first have to add the right number of
        particles, then zero out everything and re-add the unique
        parameters to correct for the re-indexing
        """
        # add additional particles for unique atoms in sys2
        for atom in self.unique_atoms2:
            [charge, sigma, epsilon] = force2.getParticleParameters(atom)
            force.addParticle(charge, sigma, epsilon)
        # Zero out everything
        for atom in range(force.getNumParticles()):
            [charge, sigma, epsilon] = force.getParticleParameters(atom)
            force.setParticleParameters(atom, 0*charge, sigma, 0*epsilon)
        for index in range(force.getNumExceptions()):
            [atom1_i, atom1_j, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
            force.addException(atom1_i, atom1_j, 0*chargeProd, sigma, 0*epsilon, replace=True)
        # Add unique atom parameters back
        for atom1 in self.unique_atoms1:
            [charge, sigma, epsilon] = force1.getParticleParameters(atom1)
            atom = sys1_indices_in_system[atom1]
            force.setParticleParameters(atom, charge, sigma, epsilon)
        for atom2 in self.unique_atoms2:
            [charge, sigma, epsilon] = force2.getParticleParameters(atom2)
            atom = sys2_indices_in_system[atom2]
            force.setParticleParameters(atom, charge, sigma, epsilon)
        # Add exceptions.
        if self.verbose: print("Adding exceptions unique to molecule1...")
        for index1 in unique_exceptions1:
            [atom1_i, atom1_j, chargeProd, sigma, epsilon] = force1.getExceptionParameters(index1)
            atom_i = sys1_indices_in_system[atom1_i]
            atom_j = sys1_indices_in_system[atom1_j]
            force.addException(atom_i, atom_j, chargeProd, sigma, epsilon, replace=True)
        if self.verbose: print("Adding exceptions unique to molecule2...")
        for index2 in unique_exceptions2:
            [atom2_i, atom2_j, chargeProd, sigma, epsilon] = force2.getExceptionParameters(index2)
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            force.addException(atom_i, atom_j, chargeProd, sigma, epsilon, replace=True)

    def _nonbonded_force(self, force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping1, mapping2, system):
        """
        Will add 3 forces to the system:
            NonbondedForce --> intra-unique atoms
                               exceptions to eliminate unique1-unique2 interactions
            CustomNonbondedForces --> 2 interaction groups:
                                      intra-core atoms
                                      core to all alchemical atoms
                Sterics
                Electrostatics
        """
        # Create index of exceptions in system, system1, and system2.
        def index_exceptions(force):
            exceptions = dict()
            for index in range(force.getNumExceptions()):
                [atom_i, atom_j, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
                key = unique([atom_i, atom_j]) # unique tuple, possibly in reverse order
                exceptions[key] = index
            return exceptions

        exceptions  = index_exceptions(force)   # index of exceptions for system
        exceptions1 = index_exceptions(force1)  # index of exceptions for system1
        exceptions2 = index_exceptions(force2)  # index of exceptions for system2

        # Find exceptions that are unique to each molecule.
        if self.verbose: print("Finding exceptions unique to each molecule...")
        unique_exceptions1 = [ exceptions1[atoms] for atoms in exceptions1 if not set(atoms).issubset(common1) ]
        unique_exceptions2 = [ exceptions2[atoms] for atoms in exceptions2 if not set(atoms).issubset(common2) ]

        shared_exceptions = self._nonbonded_find_shared(common2, sys2_indices_in_system, mapping2, exceptions, exceptions1, exceptions2)

        self._nonbonded_fix_noncustom(force, force1, force2, unique_exceptions1, unique_exceptions2, sys1_indices_in_system, sys2_indices_in_system)

        electrostatics_custom_nonbonded_force, sterics_custom_nonbonded_force = self._nonbonded_custom_force(force)
        self._nonbonded_exclude_uniques(force, sys2_indices_in_system, sys1_indices_in_system)#, electrostatics_custom_nonbonded_force, sterics_custom_nonbonded_force)

        # Add custom forces to system.
        system.addForce(sterics_custom_nonbonded_force)
        system.addForce(electrostatics_custom_nonbonded_force)
    
        self._nonbonded_add_core(common1, mapping1, sys1_indices_in_system, force, force1, force2, sterics_custom_nonbonded_force, electrostatics_custom_nonbonded_force)
        self._nonbonded_add_unique(common1, force1, force2, sys2_indices_in_system, sys1_indices_in_system, sterics_custom_nonbonded_force, electrostatics_custom_nonbonded_force)
    ################################
    # END CUSTOM FORCE DEFINITIONS #
    ################################

    def createPerturbedSystem(self):

        sys1_indices_in_system = { a:a for a in self.system1_atoms.keys() }

        if self.return_self:
            return [self.system1, self.topology1, self.positions1, self.atom_mapping_2to1, sys1_indices_in_system]

        topology1 = copy.deepcopy(self.topology1)
        self.topology1 = topology1
        topology2 = copy.deepcopy(self.topology2)
        self.topology2 = topology2

        system2_atoms = self.system2_atoms
        system1_atoms = self.system1_atoms

        system = copy.deepcopy(self.system1)
        topology = copy.deepcopy(self.topology1)
        positions = copy.deepcopy(self.positions1)

        system_atoms = dict()
        for atom in topology.atoms():
            system_atoms[atom.index] = atom

        system1 = self.system1
        system2 = self.system2
        mapping1 = self.atom_mapping_1to2
        mapping2 = self.atom_mapping_2to1
        common1 = mapping1.keys()
        common2 = mapping2.keys()
        assert len(common1) == len(common2)

        sys2_indices_in_system = copy.deepcopy(self.atom_mapping_2to1)

        residues_2_to_sys = dict()
        for index2, index in sys2_indices_in_system.items():
            atom = system_atoms[index]
            atom2 = system2_atoms[index2]
            residues_2_to_sys[atom2.residue] = atom.residue

        for atom2idx in self.unique_atoms2:
            atom2 = system2_atoms[atom2idx]
            name = atom2.name
            # CURRENTLY NOT POSSIBLE TO CREATE A NEW RESIDUE
            residue = residues_2_to_sys[atom2.residue]
            element = atom2.element
            mass = self.system2.getParticleMass(atom2idx)
            index = system.addParticle(mass)
            sys2_indices_in_system[atom2idx] = index
            topology.addAtom(name, element, residue)

        # RENUMBERING OF ATOM.INDEX:
        #     INDEX MUST MATCH POSITION IN TOPOLOGY.ATOMS()
        #     FOR POSITIONS TO BE APPLIED PROPERLY
        system_atoms = dict()
        sys_index_in_sys1 = { value: key for key, value in sys1_indices_in_system.items() }
        sys_index_in_sys2 = { value: key for key, value in sys2_indices_in_system.items() }
        for k, atom in enumerate(topology.atoms()):
            system_atoms[k] = atom
            if atom.index in sys_index_in_sys1.keys():
                atom1idx = sys_index_in_sys1[atom.index]
                sys1_indices_in_system[atom1idx] = k
            if atom.index in sys_index_in_sys2.keys():
                atom2idx = sys_index_in_sys2[atom.index]
                sys2_indices_in_system[atom2idx] = k
            atom.index = k

        # Handle constraints.
        system = self._handle_constraints(system, system2, sys2_indices_in_system)

        # Create new positions array.
        # julie debugging 8/3/16
#        positions = self._create_new_positions_array(positions, sys2_indices_in_system)
        positions = self._create_new_positions_array(topology, positions, sys1_indices_in_system, sys2_indices_in_system)
 
        # Build a list of Force objects in system.
        forces = [ system.getForce(index) for index in range(system.getNumForces()) ]
        forces1 = { system1.getForce(index).__class__.__name__ : system1.getForce(index) for index in range(system1.getNumForces()) }
        forces2 = { system2.getForce(index).__class__.__name__ : system2.getForce(index) for index in range(system2.getNumForces()) }
    
        # Process forces.
        for force in forces:
            # Get force name.
            force_name = force.__class__.__name__
            force1 = forces1[force_name]
            force2 = forces2[force_name]
            if self.verbose: print(force_name)
            if force_name == 'HarmonicBondForce':
                self._harmonic_bond_force(force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping2, system)
            elif force_name == 'HarmonicAngleForce':
                self._harmonic_angle_force(force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping2, system)
            elif force_name == 'PeriodicTorsionForce':
                self._periodic_torsion_force(force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping2, system, system_atoms)
            elif force_name == 'NonbondedForce':
                self._nonbonded_force(force, force1, force2, common1, common2, sys1_indices_in_system, sys2_indices_in_system, mapping1, mapping2, system)
            else:
                pass

        # juile debugging 8/3/16
        print('Forces remaining in the hybrid system:')
        for force in [ system.getForce(index) for index in range(system.getNumForces()) ]:
            force_name = force.__class__.__name__
            print(force_name)


        return [system, topology, positions, sys2_indices_in_system, sys1_indices_in_system]

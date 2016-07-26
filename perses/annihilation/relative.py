"""
Place holder
"""
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

    def _create_new_positions_array(self, positions, sys2_indices_in_system):
        natoms = positions.shape[0] + len(self.unique_atoms2) # new number of atoms
        positions = unit.Quantity(np.resize(positions/positions.unit, [natoms,3]), positions.unit)
        for atom2 in self.unique_atoms2:
            pos_atom2 = self.positions2[atom2,:]
            index = sys2_indices_in_system[atom2]
            positions[index] = pos_atom2
        return positions

    def _harmonic_bond_force(self, force, force1, force2, common1, common2, sys2_indices_in_system, mapping2, system):
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
 
        # Build list of bonds shared among all molecules.
        if self.verbose: print("Building a list of shared bonds...")
        shared_bonds = list()
        for atoms2 in bonds2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms  = [sys2_indices_in_system[atom2] for atom2 in atoms2]
                atoms1 = [mapping2[atom2] for atom2 in atoms2]
                # Find bond index terms.
                index  = bonds[unique(atoms)]
                index1 = bonds1[unique(atoms1)]
                index2 = bonds2[unique(atoms2)]
                # Store.
                shared_bonds.append( (index, index1, index2) )
    
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
        system.addForce(custom_force)
    
        # Process bonds that are shared by molecule1 and molecule2.
        if self.verbose: print("Translating shared bonds to CustomBondForce...")
        for (index, index1, index2) in shared_bonds:
            # Zero out standard bond force.
            [atom_i, atom_j, length, K] = force.getBondParameters(index)
            force.setBondParameters(index, atom_i, atom_j, length, K*0.0)
            # Create interpolated bond parameters.
            [atom1_i, atom1_j, length1, K1] = force1.getBondParameters(index1)
            [atom2_i, atom2_j, length2, K2] = force2.getBondParameters(index2)
            custom_force.addBond(atom_i, atom_j, [length1, K1, length2, K2])

        if self.verbose: print("Adding custom parameters to unique bonds...")
        for index2 in unique_bonds2:
            [atom2_i, atom2_j, length2, K2] = force2.getBondParameters(index2)
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            custom_force.addBond(atom_i, atom_j, [length2, 0.1*K2, length2, K2])
    
        for index1 in unique_bonds1:
            [atom_i, atom_j, length1, K1] = force1.getBondParameters(index1)
            custom_force.addBond(atom_i, atom_j, [length1, K1, length1, 0.1*K1])

    def _harmonic_angle_force(self, force, force1, force2, common1, common2, sys2_indices_in_system, mapping2, system):
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

        # Build list of angles shared among all molecules.
        if self.verbose: print("Building a list of shared angles...")
        shared_angles = list()
        for atoms2 in angles2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms  = [sys2_indices_in_system[atom2] for atom2 in atoms2]
                atoms1 = [mapping2[atom2] for atom2 in atoms2]
                # Find angle index terms.
                index  = angles[unique(atoms)]
                index1 = angles1[unique(atoms1)]
                index2 = angles2[unique(atoms2)]
                # Store.
                shared_angles.append( (index, index1, index2) )

        if self.verbose: print("Removing existing angle parameters...")
        for index in range(force.getNumAngles()):
            [atom_i, atom_j, atom_k, angle, K] = force.getAngleParameters(index)
            force.setAngleParameters(index, atom_i, atom_j, atom_k, angle, 0*K)
    
        # Create a CustomAngleForce to handle interpolated angle parameters.
        if self.verbose: print("Creating CustomAngleForce...")
        energy_expression  = '(K/2)*(theta-theta0)^2;'
        energy_expression += 'K = (1.1-lambda_angles)*K_1 + lambda_angles*K_2;' # linearly interpolate spring constant
        energy_expression += 'theta0 = (1.1-lambda_angles)*theta0_1 + lambda_angles*theta0_2;' # linearly interpolate equilibrium angle
        custom_force = mm.CustomAngleForce(energy_expression)
        custom_force.addGlobalParameter('lambda_angles', 0.0)
        custom_force.addPerAngleParameter('theta0_1') # molecule1 equilibrium angle
        custom_force.addPerAngleParameter('K_1') # molecule1 spring constant
        custom_force.addPerAngleParameter('theta0_2') # molecule2 equilibrium angle
        custom_force.addPerAngleParameter('K_2') # molecule2 spring constant
        system.addForce(custom_force)
    
        # Process angles that are shared by molecule1 and molecule2.
        if self.verbose: print("Translating shared angles to CustomAngleForce...")
        for (index, index1, index2) in shared_angles:
            # Zero out standard angle force.
            [atom_i, atom_j, atom_k, theta0, K] = force.getAngleParameters(index)
            force.setAngleParameters(index, atom_i, atom_j, atom_k, theta0, K*0.0)
            # Create interpolated angle parameters.
            [atom1_i, atom1_j, atom1_k, theta1, K1] = force1.getAngleParameters(index1)
            [atom2_i, atom2_j, atom2_k, theta2, K2] = force2.getAngleParameters(index2)
            custom_force.addAngle(atom_i, atom_j, atom_k, [theta1, K1, theta2, K2])

    def _periodic_torsion_force(self, force, force1, force2, common1, common2, sys2_indices_in_system, mapping2, system):
        def index_torsions(force):
            torsions = dict()
            for index in range(force.getNumTorsions()):
                [atom_i, atom_j, atom_k, atom_l, periodicity, phase, K] = force.getTorsionParameters(index)
                key = unique([atom_i, atom_j, atom_k, atom_l]) # unique tuple, possibly in reverse order
                torsions[key] = index
            return torsions

        torsions  = index_torsions(force)   # index of torsions for system
        torsions1 = index_torsions(force1)  # index of torsions for system1
        torsions2 = index_torsions(force2)  # index of torsions for system2

        # Find torsions that are unique to each molecule.
        if self.verbose: print("Finding torsions unique to each molecule...")
        unique_torsions1 = [ torsions1[atoms] for atoms in torsions1 if not set(atoms).issubset(common1) ]
        unique_torsions2 = [ torsions2[atoms] for atoms in torsions2 if not set(atoms).issubset(common2) ]
 
        # Build list of torsions shared among all molecules.
        if self.verbose: print("Building a list of shared torsions...")
        shared_torsions = list()
        for atoms2 in torsions2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms  = [sys2_indices_in_system[atom2] for atom2 in atoms2]
                atoms1 = [mapping2[atom2] for atom2 in atoms2]
                # Find torsion index terms.
                try:
                    index  = torsions[unique(atoms)]
                except Exception as e:
                    if self.verbose: print("Warning: problem occurred in building a list of torsions common to all molecules -- SYSTEM.")
                    atom_names = [system_atoms[atom] for atom in atoms]
                    if self.verbose: print(atom_names)
                    try:
                        index1 = torsions1[unique(atoms1)]
                        if self.verbose: print("ERROR: torsion present in SYSTEM 1, not copied to SYSTEM.")
                        if self.verbose: print("torsions :  %s" % str(unique(atoms)))
                        if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                        if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                        raise(e)
                    except:
                        try:
                            index2 = torsions2[unique(atoms2)]
                            unique_torsions2.append(torsions2[unique(atoms2)]) # so this will never catch unique torsions from 1 that use core atoms?
                            continue
#                            if self.verbose: print("ERROR: torsion present in SYSTEM 2 but not in SYSTEM 1.")
                        except:
                            if self.verbose: print("ERROR: the torsion does not exist.")
                            if self.verbose: print("torsions :  %s" % str(unique(atoms)))
                            if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                            if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                            raise(e)
                try:
                    index1 = torsions1[unique(atoms1)]
                except Exception as e:
                    if self.verbose: print("Error occurred in building a list of torsions common to all molecules -- SYSTEM 1.")
                    if self.verbose: print("torsions :  %s" % str(unique(atoms)))
                    if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                    if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                    raise(e)
                try:
                    index2 = torsions2[unique(atoms2)]
                except Exception as e:
                    if self.verbose: print("Error occurred in building a list of torsions common to all molecules -- SYSTEM 2.")
                    if self.verbose: print("torsions :  %s" % str(unique(atoms)))
                    if self.verbose: print("torsions1:  %s" % str(unique(atoms1)))
                    if self.verbose: print("torsions2:  %s" % str(unique(atoms2)))
                    raise(e)

                # Store.
                shared_torsions.append( (index, index1, index2) )
 
        # Add torsions that are unique to molecule2.
        if self.verbose: print("Adding torsions unique to molecule2...")
        for index2 in unique_torsions2:
            [atom2_i, atom2_j, atom2_k, atom2_l, periodicity2, phase2, K2] = force2.getTorsionParameters(index2)
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            atom_k = sys2_indices_in_system[atom2_k]
            atom_l = sys2_indices_in_system[atom2_l]
            force.addTorsion(atom_i, atom_j, atom_k, atom_l, periodicity2, phase2, K2)

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
        system.addForce(custom_force)

        # Process torsions that are shared by molecule1 and molecule2.
        if self.verbose: print("Translating shared torsions to CustomTorsionForce...")
        for (index, index1, index2) in shared_torsions:
            # Zero out standard torsion force.
            [atom_i, atom_j, atom_k, atom_l, periodicity, phase, K] = force.getTorsionParameters(index)
            force.setTorsionParameters(index, atom_i, atom_j, atom_k, atom_l, periodicity, phase, K*0.0)
            # Create interpolated torsion parameters.
            [atom1_i, atom1_j, atom1_k, atom1_l, periodicity1, phase1, K1] = force1.getTorsionParameters(index1)
            [atom2_i, atom2_j, atom2_k, atom2_l, periodicity2, phase2, K2] = force2.getTorsionParameters(index2)
            custom_force.addTorsion(atom_i, atom_j, atom_k, atom_l, [periodicity1, phase1, K1, periodicity2, phase2, K2])

    def _nonbonded_force(self, force, force1, force2, common1, common2, sys2_indices_in_system, system1_atoms, mapping1, mapping2, system):
        softcore_alpha = self.softcore_alpha
        softcore_beta = self.softcore_beta

        for atom in self.unique_atoms2:
            [charge, sigma, epsilon] = force2.getParticleParameters(atom)
            force.addParticle(charge, sigma, epsilon)

        # Zero out nonbonded entries for molecule1.
        for atom, atom_obj in system1_atoms.items():
            [charge, sigma, epsilon] = force.getParticleParameters(atom)
            force.setParticleParameters(atom, 0*charge, sigma, 0*epsilon)
        # Zero out nonbonded entries for molecule2.
        for atom in sys2_indices_in_system.values():
            [charge, sigma, epsilon] = force.getParticleParameters(atom)
            force.setParticleParameters(atom, 0*charge, sigma, 0*epsilon)

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

        # Build list of exceptions shared among all molecules.
        if self.verbose: print("Building a list of shared exceptions...")
        shared_exceptions = list()
        for atoms2 in exceptions2:
            atoms2 = list(atoms2)
            if set(atoms2).issubset(common2):
                atoms  = tuple(sys2_indices_in_system[atom2] for atom2 in atoms2)
                atoms1 = tuple(mapping2[atom2] for atom2 in atoms2)
                # Find exception index terms.
                try:
                    index  = exceptions[unique(atoms)]
                    index1 = exceptions1[unique(atoms1)]
                    index2 = exceptions2[unique(atoms2)]
                    # Store.
                    shared_exceptions.append( (index, index1, index2) )
                except:
                    pass 

        # Add exceptions that are unique to molecule2.
        if self.verbose: print("Adding exceptions unique to molecule2...")
        for index2 in unique_exceptions2:
            [atom2_i, atom2_j, chargeProd, sigma, epsilon] = force2.getExceptionParameters(index2)
            atom_i = sys2_indices_in_system[atom2_i]
            atom_j = sys2_indices_in_system[atom2_j]
            force.addException(atom_i, atom_j, chargeProd, sigma, epsilon)

        # Create list of alchemically modified atoms in system.
        alchemical_atom_indices = list(set([index for index in system1_atoms.keys()]).union(set(sys2_indices_in_system.values())))

        # Create atom groups.
        natoms = system.getNumParticles()
        atomset1 = set(alchemical_atom_indices) # only alchemically-modified atoms
        atomset2 = set(range(system.getNumParticles())) # all atoms, including alchemical region

        # CustomNonbondedForce energy expression.
        sterics_energy_expression = ""
        electrostatics_energy_expression = ""

        # Create a CustomNonbondedForce to handle alchemically interpolated nonbonded parameters.
        # Select functional form based on nonbonded method.
        method = force.getNonbondedMethod()
        if method in [mm.NonbondedForce.NoCutoff]:
            # soft-core Lennard-Jones
            sterics_energy_expression += "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            # soft-core Coulomb
            electrostatics_energy_expression += "U_electrostatics = ONE_4PI_EPS0*chargeprod/reff_electrostatics;"
        elif method in [mm.NonbondedForce.CutoffPeriodic, mm.NonbondedForce.CutoffNonPeriodic]:
            # soft-core Lennard-Jones
            sterics_energy_expression += "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            # reaction-field electrostatics
            epsilon_solvent = force.getReactionFieldDielectric()
            r_cutoff = force.getCutoffDistance()
            electrostatics_energy_expression += "U_electrostatics = ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);"
            k_rf = r_cutoff**(-3) * ((epsilon_solvent - 1) / (2*epsilon_solvent + 1))
            c_rf = r_cutoff**(-1) * ((3*epsilon_solvent) / (2*epsilon_solvent + 1))
            electrostatics_energy_expression += "k_rf = %f;" % (k_rf / k_rf.in_unit_system(unit.md_unit_system).unit)
            electrostatics_energy_expression += "c_rf = %f;" % (c_rf / c_rf.in_unit_system(unit.md_unit_system).unit)
        elif method in [mm.NonbondedForce.PME, mm.NonbondedForce.Ewald]:
            # soft-core Lennard-Jones
            sterics_energy_expression += "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            # Ewald direct-space electrostatics
            [alpha_ewald, nx, ny, nz] = force.getPMEParameters()
            if alpha_ewald == 0.0:
                # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
                delta = force.getEwaldErrorTolerance()
                r_cutoff = force.getCutoffDistance()
                alpha_ewald = np.sqrt(-np.log(2*delta)) / r_cutoff
            electrostatics_energy_expression += "U_electrostatics = ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
            electrostatics_energy_expression += "alpha_ewald = %f;" % (alpha_ewald / alpha_ewald.in_unit_system(unit.md_unit_system).unit)
            # TODO: Handle reciprocal-space electrostatics
        else:
            raise Exception("Nonbonded method %s not supported yet." % str(method))

        # Add additional definitions common to all methods.
        sterics_energy_expression += "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;" #interpolation
        sterics_energy_expression += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
        sterics_energy_expression += "softcore_alpha = %f;" % softcore_alpha
        sterics_energy_expression += "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
        # TODO: We may have to ensure that softcore_degree is 1 if we are close to an alchemically-eliminated endpoint.
        sterics_energy_expression += "lambda_alpha = lambda_sterics*(1-lambda_sterics);"
        electrostatics_energy_expression += "chargeprod = (1-lambda_electrostatics)*chargeprodA + lambda_electrostatics*chargeprodB;" #interpolation
        electrostatics_energy_expression += "reff_electrostatics = sqrt(softcore_beta*lambda_beta + r^2);" # effective softcore distance for electrostatics
        electrostatics_energy_expression += "softcore_beta = %f;" % (softcore_beta / softcore_beta.in_unit_system(unit.md_unit_system).unit)
        electrostatics_energy_expression += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0 # already in OpenMM units
        # TODO: We may have to ensure that softcore_degree is 1 if we are close to an alchemically-eliminated endpoint.
        electrostatics_energy_expression += "lambda_beta = lambda_electrostatics*(1-lambda_electrostatics);"

        # Define mixing rules.
        sterics_mixing_rules = ""
        sterics_mixing_rules += "epsilonA = sqrt(epsilonA1*epsilonA2);" # mixing rule for epsilon
        sterics_mixing_rules += "epsilonB = sqrt(epsilonB1*epsilonB2);" # mixing rule for epsilon
        sterics_mixing_rules += "sigmaA = 0.5*(sigmaA1 + sigmaA2);" # mixing rule for sigma
        sterics_mixing_rules += "sigmaB = 0.5*(sigmaB1 + sigmaB2);" # mixing rule for sigma
        electrostatics_mixing_rules = ""
        electrostatics_mixing_rules += "chargeprodA = chargeA1*chargeA2;" # mixing rule for charges
        electrostatics_mixing_rules += "chargeprodB = chargeB1*chargeB2;" # mixing rule for charges

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

        # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
        # TODO: Exclude intra-alchemical region if we are separately handling that through a separate CustomNonbondedForce for decoupling.
        sterics_custom_nonbonded_force.addInteractionGroup(atomset1, atomset2)
        electrostatics_custom_nonbonded_force.addInteractionGroup(atomset1, atomset2)

        # Add exclusions between unique parts of molecule1 and molecule2 so they do not interact.
        if self.verbose: print("Add exclusions between unique parts of molecule1 and molecule2 that should not interact...")
        for atom1_i in self.unique_atoms1:
            for atom2_j in self.unique_atoms2:
                atom_i = atom1_i
                atom_j = sys2_indices_in_system[atom2_j]
                electrostatics_custom_nonbonded_force.addExclusion(atom_i, atom_j)
                sterics_custom_nonbonded_force.addExclusion(atom_i, atom_j)

        # Add custom forces to system.
        system.addForce(sterics_custom_nonbonded_force)
        system.addForce(electrostatics_custom_nonbonded_force)
    
        # Copy over all Nonbonded parameters for normal atoms to Custom*Force objects.
        for particle_index in range(force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = force.getParticleParameters(particle_index)
            # Add parameters to custom force handling interactions between alchemically-modified atoms and rest of system.
            sterics_custom_nonbonded_force.addParticle([sigma, epsilon, sigma, epsilon])
            electrostatics_custom_nonbonded_force.addParticle([charge, charge])

        # Copy over parameters for common substructure.
        for atom1 in common1:
            atom2 = mapping1[atom1] # index into system2
            index = atom1 # index into system
            [charge1, sigma1, epsilon1] = force1.getParticleParameters(atom1)
            [charge2, sigma2, epsilon2] = force2.getParticleParameters(atom2)
            sterics_custom_nonbonded_force.setParticleParameters(index, [sigma1, epsilon1, sigma2, epsilon2])
            electrostatics_custom_nonbonded_force.setParticleParameters(index, [charge1, charge2])

        # Copy over parameters for molecule1 unique atoms.
        for atom1 in self.unique_atoms1:
            index = atom1 # index into system
            [charge1, sigma1, epsilon1] = force1.getParticleParameters(atom1)
            sterics_custom_nonbonded_force.setParticleParameters(index, [sigma1, epsilon1, sigma1, 0*epsilon1])
            electrostatics_custom_nonbonded_force.setParticleParameters(index, [charge1, 0*charge1])

        # Copy over parameters for molecule2 unique atoms.
        for atom2 in self.unique_atoms2:
            index = sys2_indices_in_system[atom2] # index into system
            [charge2, sigma2, epsilon2] = force2.getParticleParameters(atom2)
            sterics_custom_nonbonded_force.setParticleParameters(index, [sigma2, 0*epsilon2, sigma2, epsilon2])
            electrostatics_custom_nonbonded_force.setParticleParameters(index, [0*charge2, charge2])

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
            #if not atom.name == atom2.name:
                #print(atom.name, atom2.name, atom.residue, atom2.residue)
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

        for atom in topology.atoms():
            if not atom.index in system_atoms.keys():
                system_atoms[atom.index] = atom

        # Handle constraints.
        system = self._handle_constraints(system, system2, sys2_indices_in_system)
 
        # Create new positions array.
        positions = self._create_new_positions_array(positions, sys2_indices_in_system)

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
                self._harmonic_bond_force(force, force1, force2, common1, common2, sys2_indices_in_system, mapping2, system)
            elif force_name == 'HarmonicAngleForce':
                self._harmonic_angle_force(force, force1, force2, common1, common2, sys2_indices_in_system, mapping2, system)
            elif force_name == 'PeriodicTorsionForce':
                self._periodic_torsion_force(force, force1, force2, common1, common2, sys2_indices_in_system, mapping2, system)
            elif force_name == 'NonbondedForce':
                self._nonbonded_force(force, force1, force2, common1, common2, sys2_indices_in_system, system1_atoms, mapping1, mapping2, system)
            else:
                pass

        return [system, topology, positions, sys2_indices_in_system, sys1_indices_in_system]

"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple
import perses.rjmc.topology_proposal as topology_proposal
import parmed
import simtk.unit as units
import logging
import numpy as np
import coordinate_tools
import simtk.openmm as openmm
from collections import OrderedDict

GeometryProposal = namedtuple('GeometryProposal',['new_positions','logp'])


class GeometryEngine(object):
    """
    This is the base class for the geometry engine.
    
    Arguments
    ---------
    metadata : dict
        GeometryEngine-related metadata as a dict
    """
    
    def __init__(self, metadata):
        pass

    def propose(self, top_proposal):
        """
        Make a geometry proposal for the appropriate atoms.
        
        Arguments
        ----------
        top_proposal : TopologyProposal object
            Object containing the relevant results of a topology proposal

        Returns
        -------
        new_positions : [n, 3] ndarray
            The new positions of the system
        """
        return np.array([0.0,0.0,0.0])

    def logp(self, top_proposal, new_coordinates, old_coordinates):
        """
        Calculate the logp for the given geometry proposal

        Arguments
        ----------
        top_proposal : TopologyProposal object
            Object containing the relevant results of a topology proposal
        new_coordinates : [n, 3] np.ndarray
            The coordinates of the system after the proposal
        old_coordiantes : [n, 3] np.ndarray
            The coordinates of the system before the proposal
        direction : str, either 'forward' or 'reverse'
            whether the transformation is for the forward NCMC move or the reverse

        Returns
        -------
        logp : float
            The log probability of the proposal for the given transformation
        """
        return 0.0

class FFGeometryEngine(GeometryEngine):
    """
    This class is a base class for GeometryEngines which rely on forcefield information for
    making matching proposals
    """
    def __init__(self, metadata):
        self._metadata = metadata

    def propose(self, top_proposal):
        """
        Make a geometry proposal for the appropriate atoms.

        Arguments
        ----------
        top_proposal : TopologyProposal object
            Object containing the relevant results of a topology proposal

        Returns
        -------
        new_positions : [n, 3] ndarray
            The new positions of the system
        logp_proposal : float
            The log probability of the forward-only proposal
        """
        beta = top_proposal.beta
        logp_proposal = 0.0
        structure = parmed.openmm.load_topology(top_proposal.new_topology, top_proposal.new_system)
        new_atoms = [structure.atoms[idx] for idx in top_proposal.unique_new_atoms]
        atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(top_proposal.n_atoms_new) if atom_idx not in top_proposal.unique_new_atoms]
        new_positions = units.Quantity(np.zeros([top_proposal.n_atoms_new, 3]), unit=units.nanometers)
        #copy positions
        for atom in atoms_with_positions:
            old_index = top_proposal.new_to_old_atom_map[atom.idx]
            new_positions[atom.idx] = top_proposal.old_positions[old_index]

        #maintain a running list of the atoms still needing positions
        while(len(new_atoms)>0):
            atoms_for_proposal = self._atoms_eligible_for_proposal(new_atoms, structure, atoms_with_positions)
            for atom in atoms_for_proposal:
                torsion, logp_choice = self._choose_torsion(atoms_with_positions, atom)

                if torsion.atom1 == atom:
                    bond_atom = torsion.atom2
                    angle_atom = torsion.atom3
                    torsion_atom = torsion.atom4
                else:
                    bond_atom = torsion.atom3
                    angle_atom = torsion.atom2
                    torsion_atom = torsion.atom1

                #propose a bond and calculate its probability
                bond = self._get_relevant_bond(atom, bond_atom)
                r_proposed = self._propose_bond(bond, beta)
                bond_k = bond.type.k
                sigma_r = units.sqrt(1/(beta*bond_k))
                logZ_r = np.log((np.sqrt(2*np.pi)*sigma_r/sigma_r.unit))
                logp_r = self._bond_logq(r_proposed, bond, beta) - logZ_r

                #propose an angle and calculate its probability
                propose_angle_separately = True
                if propose_angle_separately:
                    angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
                    theta_proposed = self._propose_angle(angle, beta)
                    angle_k = angle.type.k
                    sigma_theta = units.sqrt(1/(beta*angle_k))
                    logZ_theta = np.log((np.sqrt(2*np.pi)*sigma_theta/sigma_theta.unit))
                    logp_theta = self._angle_logq(theta_proposed, angle, beta) - logZ_theta

                    #propose a torsion angle and calcualate its probability
                    phi_proposed, logp_phi = self._propose_torsion(atom, r_proposed, theta_proposed, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, new_positions, beta)
                else:
                    theta_proposed, phi_proposed, logp_theta_phi, xyz = self._joint_torsion_angle_proposal(atom, r_proposed, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, new_positions, beta)
                #convert to cartesian
                xyz, detJ = self._internal_to_cartesian(new_positions[bond_atom.idx], new_positions[angle_atom.idx], new_positions[torsion_atom.idx], r_proposed, theta_proposed, phi_proposed)
                detJ=0.0
                rtp_test = self._cartesian_to_internal(xyz, new_positions[bond_atom.idx], new_positions[angle_atom.idx], new_positions[torsion_atom.idx])

                #add new position to array of new positions
                new_positions[atom.idx] = xyz
                #accumulate logp
                logp_proposal = logp_proposal + logp_choice + logp_r + logp_theta +logp_phi + np.log(detJ+1)

                atoms_with_positions.append(atom)
                new_atoms.remove(atom)
        return new_positions, logp_proposal


    def logp_reverse(self, top_proposal, new_coordinates, old_coordinates):
        """
        Calculate the logp for the given geometry proposal

        Arguments
        ----------
        top_proposal : TopologyProposal object
            Object containing the relevant results of a topology proposal
        new_coordinates : [n, 3] np.ndarray
            The coordinates of the system after the proposal
        old_coordiantes : [n, 3] np.ndarray
            The coordinates of the system before the proposal

        Returns
        -------
        logp : float
            The log probability of the proposal for the given transformation
        """
        beta = top_proposal.beta
        logp = 0.0
        top_proposal = topology_proposal.SmallMoleculeTopologyProposal()
        structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
        atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(top_proposal.n_atoms_old) if atom_idx not in top_proposal.unique_old_atoms]
        new_atoms = [structure.atoms[idx] for idx in top_proposal.unique_old_atoms]
        #we'll need to copy the current positions of the core to the old system
        #In the case of C-A --> C/-A -> C/-B ---> C-B these are the same
        reverse_proposal_coordinates = units.Quantity(np.zeros([top_proposal.n_atoms_new, 3]), unit=units.nanometers)
        for atom in atoms_with_positions:
            new_index = top_proposal.old_to_new_atom_map[atom.idx]
            reverse_proposal_coordinates[atom.idx] = new_coordinates[new_index]

        #maintain a running list of the atoms still needing logp
        while(len(new_atoms)>0):
            atoms_for_proposal = self._atoms_eligible_for_proposal(new_atoms, structure, atoms_with_positions)
            for atom in atoms_for_proposal:
                torsion, logp_choice = self._choose_torsion(atoms_with_positions, atom)
                if torsion.atom1 == atom:
                    bond_atom = torsion.atom2
                    angle_atom = torsion.atom3
                    torsion_atom = torsion.atom4
                else:
                    bond_atom = torsion.atom3
                    angle_atom = torsion.atom2
                    torsion_atom = torsion.atom1
                #get the internal coordinate representation
                #we want to see the probability of old_atom where all other coordinates are new_atom
                atom_coords = old_coordinates[atom.idx]
                bond_coords = reverse_proposal_coordinates[bond_atom.idx]
                angle_coords = reverse_proposal_coordinates[angle_atom.idx]
                torsion_coords = reverse_proposal_coordinates[torsion_atom.idx]
                internal_coordinates, detJ = self._cartesian_to_internal(atom_coords, bond_coords, angle_coords, torsion_coords)

                #propose a bond and calculate its probability
                bond = self._get_relevant_bond(atom, bond_atom)
                bond_k = bond.type.k
                sigma_r = units.sqrt(1/(beta*bond_k))
                logZ_r = np.log((np.sqrt(2*np.pi)*sigma_r/sigma_r.unit)) #need to eliminate unit to allow numpy log
                logp_r = self._bond_logq(internal_coordinates[0], bond, beta) - logZ_r

                #propose an angle and calculate its probability
                angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
                angle_k = angle.type.k
                sigma_theta = units.sqrt(1/(beta*angle_k))
                logZ_theta = np.log((np.sqrt(2*np.pi)*sigma_theta/sigma_theta.unit)) #need to eliminate unit to allow numpy log
                logp_theta = self._angle_logq(internal_coordinates[1], angle, beta) - logZ_theta

                #calculate torsion probability
                logp_phi = self._torsion_logp(atom, atom_coords, torsion, atoms_with_positions, reverse_proposal_coordinates, beta)
                logp = logp + logp_choice + logp_r + logp_theta + logp_phi + np.log(detJ)

                atoms_with_positions.append(atom)
                new_atoms.remove(atom)
        return logp

    def _get_relevant_bond(self, atom1, atom2):
        """
        utility function to get the bond connecting atoms 1 and 2

        Arguments
        ---------
        atom1 : parmed atom object
             One of the atoms in the bond
        atom2 : parmed.atom object
             The other atom in the bond

        Returns
        -------
        bond : bond object
            Bond connecting the two atoms
        """
        bonds_1 = set(atom1.bonds)
        bonds_2 = set(atom2.bonds)
        relevant_bond_set = bonds_1.intersection(bonds_2)
        relevant_bond = relevant_bond_set.pop()
        relevant_bond_with_units = self._add_bond_units(relevant_bond)
        return relevant_bond_with_units

    def _get_relevant_angle(self, atom1, atom2, atom3):
        """
        Get the angle containing the 3 given atoms
        """
        atom1_angles = set(atom1.angles)
        atom2_angles = set(atom2.angles)
        atom3_angles = set(atom3.angles)
        relevant_angle_set = atom1_angles.intersection(atom2_angles, atom3_angles)
        relevant_angle = relevant_angle_set.pop()
        if type(relevant_angle.type.k) != units.Quantity:
            relevant_angle_with_units = self._add_angle_units(relevant_angle)
        else:
            relevant_angle_with_units = relevant_angle
        return relevant_angle_with_units

    def _add_bond_units(self, bond):
        """
        Add the correct units to a harmonic bond

        Arguments
        ---------
        bond : parmed bond object
            The bond to get units

        Returns
        -------

        """
        if type(bond.type.k)==units.Quantity:
            return bond
        bond.type.req = units.Quantity(bond.type.req, unit=units.angstrom)
        bond.type.k = units.Quantity(2.0*bond.type.k, unit=units.kilocalorie_per_mole/units.angstrom**2)
        return bond

    def _add_angle_units(self, angle):
        """
        Add the correct units to a harmonic angle

        Arguments
        ----------
        angle : parmed angle object
             the angle to get unit-ed

        Returns
        -------
        angle_with_units : parmed angle
            The angle, but with units on its parameters
        """
        if type(angle.type.k)==units.Quantity:
            return angle
        angle.type.theteq = units.Quantity(angle.type.theteq, unit=units.degree)
        angle.type.k = units.Quantity(2.0*angle.type.k, unit=units.kilocalorie_per_mole/units.radian**2)
        return angle

    def _add_torsion_units(self, torsion):
        """
        Add the correct units to a torsion

        Arguments
        ---------
        torsion : parmed.dihedral object
            The torsion needing units

        Returns
        -------
        torsion : parmed.dihedral object
            Torsion but with units added
        """
        if type(torsion.type.phi_k)==units.Quantity:
            return torsion
        torsion.type.phi_k = units.Quantity(torsion.type.phi_k, unit=units.kilocalorie_per_mole)
        torsion.type.phase = units.Quantity(torsion.type.phase, unit=units.degree)
        return torsion

    def _get_torsions(self, atoms_with_positions, new_atom):
        """
        Get the torsions that the new atom_index participates in, where all other
        atoms in the torsion have positions.

        Arguments
        ---------
        atoms_with_positions : list
            list of atoms with valid positions
        atom_index : parmed atom object
           parmed atom object

        Returns
        ------
        torsions : list of parmed.Dihedral objects
            list of the torsions meeting the criteria
        """
        torsions = new_atom.dihedrals
        eligible_torsions = []
        for torsion in torsions:
            if torsion.improper:
                continue
            if torsion.atom1 == new_atom:
                torsion_partners = [torsion.atom2, torsion.atom3, torsion.atom4]
            elif torsion.atom4 == new_atom:
                torsion_partners = [torsion.atom1, torsion.atom2, torsion.atom3]
            else:
                continue
            if set(atoms_with_positions).issuperset(set(torsion_partners)):
                eligible_torsions.append(torsion)
        eligible_torsions_with_units = [self._add_torsion_units(torsion) for torsion in eligible_torsions]
        return eligible_torsions_with_units

    def _get_valid_angles(self, atoms_with_positions, new_atom):
        """
        Get the angles that involve other atoms with valid positions

        Arguments
        ---------
        structure : parmed.Structure
            Structure object containing topology and parameters for the system
        atoms_with_positions : list
            list of atoms with valid positions
        atom_index : int
            Index of the new atom of interest

        Returns
        -------
        angles : list of parmed.Angle objects
            list of the angles meeting the criteria
        """
        atoms_with_positions = set(atoms_with_positions)
        eligible_angles = []
        angles = new_atom.angles
        for angle in angles:
            #check to make sure both other two atoms in angle also have positions
            if angle.atom1 == new_atom:
                if atoms_with_positions.issuperset(set([angle.atom2, angle.atom3])):
                    eligible_angles.append(angle)
            elif angle.atom2 == new_atom:
                if atoms_with_positions.issuperset(set([angle.atom1, angle.atom3])):
                    eligible_angles.append(angle)
            else:
                if atoms_with_positions.issuperset(set([angle.atom1, angle.atom2])):
                    eligible_angles.append(angle)
        eligible_angles_with_units = [self._add_angle_units(angle) for angle in eligible_angles]
        return eligible_angles_with_units

    def _rotation_matrix(self, axis, angle):
        """
        This method produces a rotation matrix given an axis and an angle.
        """
        axis = axis/np.linalg.norm(axis)
        axis_squared = np.square(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rot_matrix_row_one = np.array([cos_angle+axis_squared[0]*(1-cos_angle),
                                       axis[0]*axis[1]*(1-cos_angle) - axis[2]*sin_angle,
                                       axis[0]*axis[2]*(1-cos_angle)+axis[1]*sin_angle])

        rot_matrix_row_two = np.array([axis[1]*axis[0]*(1-cos_angle)+axis[2]*sin_angle,
                                      cos_angle+axis_squared[1]*(1-cos_angle),
                                      axis[1]*axis[2]*(1-cos_angle) - axis[0]*sin_angle])

        rot_matrix_row_three = np.array([axis[2]*axis[0]*(1-cos_angle)-axis[1]*sin_angle,
                                        axis[2]*axis[1]*(1-cos_angle)+axis[0]*sin_angle,
                                        cos_angle+axis_squared[2]*(1-cos_angle)])

        rotation_matrix = np.array([rot_matrix_row_one, rot_matrix_row_two, rot_matrix_row_three])
        return rotation_matrix

    def _cartesian_to_internal(self, atom_position, bond_position, angle_position, torsion_position):
        """
        Cartesian to internal function
        """
        #ensure we have the correct units, then remove them
        atom_position = atom_position.in_units_of(units.nanometers)/units.nanometers
        bond_position = bond_position.in_units_of(units.nanometers)/units.nanometers
        angle_position = angle_position.in_units_of(units.nanometers)/units.nanometers
        torsion_position = torsion_position.in_units_of(units.nanometers)/units.nanometers

        internal_coords = coordinate_tools._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)


        return internal_coords, internal_coords[0]**2*np.sin(internal_coords)


    def _internal_to_cartesian(self, bond_position, angle_position, torsion_position, r, theta, phi):
        """
        Calculate the cartesian coordinates given the internal, as well as abs(detJ)
        """
        r = r.in_units_of(units.nanometers)/units.nanometers
        theta = theta.in_units_of(units.radians)/units.radians
        phi = phi.in_units_of(units.radians)/units.radians
        bond_position = bond_position.in_units_of(units.nanometers)/units.nanometers
        angle_position = angle_position.in_units_of(units.nanometers)/units.nanometers
        torsion_position = torsion_position.in_units_of(units.nanometers)/units.nanometers
        xyz = coordinate_tools._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        xyz = units.Quantity(xyz, unit=units.nanometers)
        return xyz, r**2*np.sin(theta)


    def _bond_logq(self, r, bond, beta):
        """
        Calculate the log-probability of a given bond at a given inverse temperature

        Arguments
        ---------
        r : float
            bond length, in nanometers
        r0 : float
            equilibrium bond length, in nanometers
        k_eq : float
            Spring constant of bond
        beta : float
            1/kT or inverse temperature
        """
        k_eq = bond.type.k
        r0 = bond.type.req
        logq = -beta*0.5*k_eq*(r-r0)**2
        return logq

    def _angle_logq(self, theta, angle, beta):
        """
        Calculate the log-probability of a given bond at a given inverse temperature

        Arguments
        ---------
        theta : float
            bond angle, in randians
        angle : parmed angle object
            Bond angle object containing parameters
        beta : float
            1/kT or inverse temperature
        """
        k_eq = angle.type.k
        theta0 = angle.type.theteq
        logq = -beta*k_eq*0.5*(theta-theta0)**2
        return logq

    def _choose_torsion(self, atoms_with_positions, atom_for_proposal):
        """
        Pick an eligible torsion uniformly
        """
        eligible_torsions = self._get_torsions(atoms_with_positions, atom_for_proposal)
        torsion_idx = np.random.randint(0, len(eligible_torsions))
        torsion_selected = eligible_torsions[torsion_idx]
        return torsion_selected, np.log(1.0/len(eligible_torsions))

    def _atoms_eligible_for_proposal(self, new_atoms, structure, atoms_with_positions):
        """
        Get the set of atoms eligible for proposal
        """
        eligible_atoms = []
        for atom in new_atoms:
            #get array of booleans to see if a bond partner has a position
            has_bonded_position = [a in atoms_with_positions for a in atom.bond_partners]
            #if at least one does, then the atom is ready to be proposed.
            if np.sum(has_bonded_position) > 0:
                eligible_atoms.append(atom)
        return eligible_atoms

    def _propose_bond(self, bond, beta):
        """
        Bond length proposal
        """
        r0 = bond.type.req
        k = bond.type.k
        sigma_r = units.sqrt(1.0/(beta*k))
        r = sigma_r*np.random.random() + r0
        return r

    def _propose_angle(self, angle, beta):
        """
        Bond angle proposal
        """
        theta0 = angle.type.theteq
        k = angle.type.k
        sigma_theta = units.sqrt(1.0/(beta*k))
        theta = sigma_theta*np.random.random() + theta0
        return theta

    def _torsion_logq(self, torsion, phi, beta):
        """
        Calculate the log-unnormalized probability
        of the torsion
        """
        gamma = torsion.type.phase
        V = torsion.type.phi_k
        n = torsion.type.per
        logq = -beta*V*(1+units.cos(n*phi-gamma))
        return logq

    def _torsion_logp(self, atom, xyz, torsion, atoms_with_positions, positions, beta):
        pass

    def _propose_torsion(self, atom, r, theta, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, positions, beta):
        pass

    def _joint_torsion_angle_proposal(self, atom, r, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, new_positions, beta):
        return 0, 0, 0

class FFAllAngleGeometryEngine(FFGeometryEngine):
    """
    This is a forcefield-based geometry engine that takes all relevant angles
    and torsions into account when proposing a given torsion. it overrides the torsion_proposal
    and torsion_p methods of the base.
    """

    def _torsion_and_angle_logq(self, xyz, atom, positions, involved_angles, involved_torsions, beta):
        """
        Calculate the potential resulting from torsions and angles
        at a given cartesian coordinate
        """
        logq_angles = 0.0
        logq_torsions= 0.0
        if type(xyz) != units.Quantity:
            xyz = units.Quantity(xyz, units.nanometers)
        for angle in involved_angles:
            atom_position = xyz if angle.atom1 == atom else positions[angle.atom1.idx]
            bond_atom_position = xyz if angle.atom2 == atom else positions[angle.atom2.idx]
            angle_atom_position = xyz if angle.atom3 == atom else positions[angle.atom3.idx]
            theta = self._calculate_angle(atom_position, bond_atom_position, angle_atom_position)
            logq_i = self._angle_logq(theta*units.radians, angle, beta)
            if np.isnan(logq_i):
                raise Exception("Angle logq is nan")
            logq_angles+=logq_i
        for torsion in involved_torsions:
            atom_position = xyz if torsion.atom1 == atom else positions[torsion.atom1.idx]
            bond_atom_position = xyz if torsion.atom2 == atom else positions[torsion.atom2.idx]
            angle_atom_position = xyz if torsion.atom3 == atom else positions[torsion.atom3.idx]
            torsion_atom_position = xyz if torsion.atom4 == atom else positions[torsion.atom4.idx]
            internal_coordinates, _ = self._cartesian_to_internal(atom_position, bond_atom_position, angle_atom_position, torsion_atom_position)
            phi = internal_coordinates[2]*units.radians
            logq_i = self._torsion_logq(torsion, phi, beta)
            if np.isnan(logq_i):
                raise Exception("Torsion logq is nan")
            logq_torsions+=logq_i

        return logq_angles+logq_torsions


    def _normalize_torsion_proposal(self, atom, r, theta, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=5000):
        """
        Calculates the array of normalized proposal probabilities for this torsion

        Arguments
        ---------
        atom : parmed atom object
            Atom whose position is being proposed
        structure : parmed structure
            structure of the molecule undergoing a geometry proposal
        torsion : parmed dihedral
            Parmed object containing the dihedral in question
        atoms_with_positions : list
            List of atoms with positions
        Returns
        -------
        p : np.array of float
            normalized torsion probabilities
        Z : float
            torsion normalizing constant, estimated using deterministic integration
        """

        involved_angles = self._get_valid_angles(atoms_with_positions, atom)
        involved_torsions = self._get_torsions(atoms_with_positions, atom)

        #get an array of [0,2pi)
        phis = units.Quantity(np.arange(0, 2.0*np.pi, (2.0*np.pi)/n_divisions), unit=units.radians)
        xyzs = np.zeros([len(phis), 3])

        #rotate atom about torsion angle, calculating an xyz for each
        for i, phi in enumerate(phis):
            xyzs[i], _ = self._internal_to_cartesian(positions[bond_atom.idx], positions[angle_atom.idx], positions[torsion_atom.idx], r, theta, phi)

        #set up arrays for energies from angles and torsions
        logq = np.zeros(n_divisions)
        for i, xyz in enumerate(xyzs):
            logq_i = self._torsion_and_angle_logq(xyz, atom, positions, involved_angles, involved_torsions, beta)
            if np.isnan(logq_i):
                raise Exception("logq_i was NaN")
            logq[i]+=logq_i
        logq -= max(logq)

        #exponentiate to get the unnormalized probability
        q = np.exp(logq)

        #estimate the normalizing constant
        Z = np.sum(q)

        #get the normalized probabilities for torsions
        logp = logq - np.log(Z)

        return logp, Z, q, phis

    def _calculate_angle(self, atom_position, bond_atom_position, angle_atom_position):
        """
        Calculate the angle theta between 3 atoms 1-2-3
        """
        a = bond_atom_position - atom_position
        b = angle_atom_position - bond_atom_position
        a_u = a / np.linalg.norm(a)
        b_u = b / np.linalg.norm(b)
        cos_theta = np.dot(-a_u, b_u)
        if cos_theta > 1.0:
            cos_theta = 1.0
        elif cos_theta < -1.0:
            cos_theta = -1.0
        theta = np.arccos(cos_theta)
        return theta

    def _torsion_logp(self, atom, xyz, torsion, atoms_with_positions, positions, beta):
        """
        Calculate the log-probability of a given torsion. This is calculated via a distribution
        that includes angle and other torsion potentials.
        """
        if torsion.atom1 == atom:
            bond_atom = torsion.atom2
            angle_atom = torsion.atom3
            torsion_atom = torsion.atom4
        else:
            bond_atom = torsion.atom3
            angle_atom = torsion.atom2
            torsion_atom = torsion.atom1
        internal_coordinates = self._cartesian_to_internal(xyz, positions[bond_atom.idx], positions[angle_atom.idx], positions[torsion_atom.idx])
        logp, Z, q, phis = self._normalize_torsion_proposal(atom, internal_coordinates[0], internal_coordinates[1], bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=5000)
        #find the phi that's closest to the internal_coordinate phi:
        phi_idx, phi = min(enumerate(phis), key=lambda x: abs(x[1]-internal_coordinates[2]))
        return logp[phi_idx]

    def _propose_torsion(self, atom, r, theta, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, positions, beta):
        """
        Propose a torsion angle, including energetic contributions from other torsions and angles
        """
        #first, let's get the normalizing constant of this distribution
        logp, Z, q, phis = self._normalize_torsion_proposal(atom, r, theta, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=100)
        #choose from the set of possible torsion angles
        phi_idx = np.random.choice(range(len(phis)), p=np.exp(logp))
        logp = logp[phi_idx]
        phi = phis[phi_idx]
        return phi, logp

    def _normalize_torsion_and_angle(self, atom, r, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=5000):
        """
        Construct a grid of points on the surface of a sphere bounded by the bond length
        """
        involved_angles = self._get_valid_angles(atoms_with_positions, atom)
        involved_torsions = self._get_torsions(atoms_with_positions, atom)

        xyz_grid = units.Quantity(np.zeros([n_divisions, n_divisions, 3]), unit=units.nanometers)
        thetas = units.Quantity(np.arange(0, np.pi, np.pi/n_divisions), unit=units.radians)
        phis = units.Quantity(np.arange(0, 2.0*np.pi, (2.0*np.pi)/n_divisions), unit=units.radians)

        #now, for every theta, phi combination, get the xyz coordinates:
        for i in range(len(thetas)):
            for j in range(len(phis)):
                xyz_grid[i, j, :], _ = self._internal_to_cartesian(positions[bond_atom.idx], positions[angle_atom.idx], positions[torsion_atom.idx], r, thetas[i], phis[j])

        logq = np.zeros([n_divisions, n_divisions])

        #now, for each xyz point in the grid, calculate the log q(theta, phi)
        for i in range(len(thetas)):
            for j in range(len(phis)):
                logq[i, j] = self._torsion_and_angle_logq(xyz_grid[i, j], atom, positions, involved_angles, involved_torsions, beta)
        logq -= np.max(logq)
        q = np.exp(logq)
        Z = np.sum(q)
        logp = logq - np.log(Z)
        return logp, Z, q, thetas, phis, xyz_grid

    def _joint_torsion_angle_proposal(self, atom, r, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, positions, beta):
        n_divisions = 50
        logp, Z, q, thetas, phis, xyz_grid = self._normalize_torsion_and_angle(atom, r, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=n_divisions)
        logp_flat = np.ravel(logp)
        theta_phi_idx = np.random.choice(range(len(logp_flat)), p=np.exp(logp_flat))
        theta_idx, phi_idx = np.unravel_index(theta_phi_idx, [n_divisions, n_divisions])
        xyz = xyz_grid[theta_idx, phi_idx]
        logp = logp_flat[theta_phi_idx]
        theta = thetas[theta_idx]
        phi = phis[phi_idx]
        return theta, phi, logp, xyz

class SystemFactory(object):
    """
    This class generates OpenMM systems that allow certain atomic interactions to be turned on/off
    """
    _HarmonicBondForceEnergy = "step(growth_idx - {})*(K/2)*(r-r0)^2"
    _HarmonicAngleForceEnergy = "step(growth_idx - {})*(K/2)*(theta-theta0)^2;"
    _PeriodicTorsionForceEnergy = "step(growth_idx - {})*k*(1+cos(periodicity*theta-phase))"


    def create_modified_system(self, reference_system, growth_indices, parameter_name, force_names=None, force_parameters=None):
        """
        Create a modified system with parameter_name parameter. When 0, only core atoms are interacting;
        for each integer above 0, an additional atom is made interacting, with order determined by growth_index

        Parameters
        ----------
        reference_system : simtk.openmm.System object
            The system containing the relevant forces and particles
        growth_indices : list of int
            The order in which the atom indices will be proposed
        parameter_name : str
            The name of the global context parameter
        force_names : list of str
            A list of the names of forces that will be included in this system
        force_parameters : dict
            Options for the forces (e.g., NonbondedMethod : 'CutffNonPeriodic')

        Returns
        -------
        growth_system : simtk.openmm.System object
            System with the appropriate modifications
        """
        reference_forces = {reference_system.getForce(index).__class__.__name__ : reference_system.getForce(index) for index in range(reference_system.getNumForces())}
        growth_system = openmm.System()
        #create the forces:
        modified_bond_force = openmm.CustomBondForce(self._HarmonicBondForceEnergy.format(parameter_name))
        modified_bond_force.addPerBondParameter("r0")
        modified_bond_force.addPerBondParameter("K")
        modified_bond_force.addPerBondParameter("growth_idx")
        modified_bond_force.addGlobalParameter(parameter_name, 0)

        modified_angle_force = openmm.CustomAngleForce(self._HarmonicAngleForceEnergy.format(parameter_name))
        modified_angle_force.addPerAngleParameter("theta0")
        modified_angle_force.addPerAngleParameter("K")
        modified_angle_force.addPerAngleParameter("growth_idx")
        modified_angle_force.addGlobalParameter(parameter_name, 0)

        modified_torsion_force = openmm.CustomTorsionForce(self._PeriodicTorsionForceEnergy.format(parameter_name))
        modified_torsion_force.addPerTorsionParameter("periodicity")
        modified_torsion_force.addPerTorsionParameter("phase")
        modified_torsion_force.addPerTorsionParameter("k")
        modified_torsion_force.addPerTorsionParameter("growth_idx")
        modified_angle_force.addGlobalParameter(parameter_name, 0)

        growth_system.addForce(modified_bond_force)
        growth_system.addForce(modified_angle_force)
        growth_system.addForce(modified_torsion_force)

        #copy the particles over
        for i in range(reference_system.getNumParticles()):
            growth_system.addParticle(reference_system.getParticleMass(i))

        #copy each bond, adding the per-particle parameter as well
        reference_bond_force = reference_forces['HarmonicBondForce']
        for bond in range(reference_bond_force.getNumBonds()):
            bond_parameters = reference_bond_force.getBondParameters(bond)
            growth_idx = self._calculate_growth_idx(bond_parameters[:2], growth_indices)
            modified_bond_force.addBond(bond_parameters[0], bond_parameters[1], [bond_parameters[2], bond_parameters[3], growth_idx])

        #copy each angle, adding the per particle parameter as well
        reference_angle_force = reference_forces['HarmonicAngleForce']
        for angle in range(reference_angle_force.getNumAngles()):
            angle_parameters = reference_angle_force.getAngleParameters(angle)
            growth_idx = self._calculate_growth_idx(angle_parameters[:3], growth_indices)
            modified_angle_force.addAngle(angle_parameters[0], angle_parameters[1], angle_parameters[2], [angle_parameters[3], angle_parameters[4], growth_idx])

        #copy each torsion, adding the per particle parameter as well
        reference_torsion_force = reference_forces['PeriodicTorsionForce']
        for torsion in range(reference_torsion_force.getNumTorsions()):
            torsion_parameters = reference_torsion_force.getTorsionParameters(torsion)
            growth_idx = self._calculate_growth_idx(torsion_parameters[:4], growth_indices)
            modified_torsion_force.addTorsion(torsion_parameters[0], torsion_parameters[1], torsion_parameters[2], torsion_parameters[3], [torsion_parameters[4], torsion_parameters[5], torsion_parameters[6], growth_idx])

        return growth_system


    def _calculate_growth_idx(self, particle_indices, growth_indices):
        """
        Utility function to calculate the growth index of a particular force.
        For each particle index, it will check to see if it is in growth_indices.
        If not, 0 is added to an array, if yes, the index in growth_indices is added.
        Finally, the method returns the max of the accumulated array

        Parameters
        ----------
        particle_indices : list of int
            The indices of particles involved in this force
        growth_indices : list of int
            The ordered list of indices for atom position proposals

        Returns
        -------
        growth_idx : int
            The growth_idx parameter
        """
        particle_indices_set = set(particle_indices)
        growth_indices_set = set(growth_indices)
        new_atoms_in_force = particle_indices_set.intersection(growth_indices_set)
        if len(new_atoms_in_force) == 0:
            return 0
        new_atom_growth_order = [growth_indices.index(atom_idx)+1 for atom_idx in new_atoms_in_force]
        return max(new_atom_growth_order)


class TopologyParameterTools(object):
    """
    This class contains utilities to extract parameters and information about the topology relevant
    to making proposals for new atomic positions. It can harvest both relevant forces and relevant
    parameters.
    """

    def __init__(self, system, topology, new_atoms, new_to_old_atom_map):
        self._structure = parmed.openmm.load_topology(topology, system)
        self._new_to_old_atom_map = new_to_old_atom_map
        self._new_atoms = new_atoms

    def _calculate_atom_proposal_order(self):
        """
        Calculate the order in which new atoms will be proposed

        Returns
        -------
        atom_order : list of int
            list of the atom indices in order of proposal
        """
        atoms_with_positions_idx = self._new_to_old_atom_map.keys()
        atoms_with_positions = [self._structure.atoms[atom_idx] for atom_idx in atoms_with_positions_idx]
        new_atoms = self._new_atoms
        atom_proposal_order = OrderedDict()
        while len(new_atoms) > 0:
            for atom in new_atoms:
                eligible_atom_torsions = self._eligible_torsions(atom, atoms_with_positions, self._structure)
                if len(eligible_atom_torsions) > 0:
                    atom_proposal_order[atom.idx] = eligible_atom_torsions



    def _eligible_torsions(self, atom, atoms_with_positions, structure):
        """
        Determine the list of torsions that could be used to propose a given
        atomic position.

        Parameters
        ----------
        atom : parmed.Atom object
            The atom needing a position
        atoms_with_positions : list of parmed.Atom objects
            Atoms that already have positions
        structure : parmed.Structure object
            Structure of system getting new positions

        Returns
        -------
        eligible_torsions : list of parmed.Dihedral
            A list of the torsions that are valid for proposal, with units
        """
        involved_dihedrals = atom.dihedrals
        atoms_with_positions_set = set(atoms_with_positions)
        eligible_torsions = []
        for dihedral in involved_dihedrals:
            if dihedral.improper:
                continue
            if dihedral.atom1 == atom:
                if {dihedral.atom2, dihedral.atom3, dihedral.atom4}.issubset(atoms_with_positions_set):
                    eligible_torsions.append(dihedral)
            elif dihedral.atom4 == atom:
                if {dihedral.atom3, dihedral.atom2, dihedral.atom1}.issubset(atoms_with_positions_set):
                    eligible_torsions.append(dihedral)
        return [self._add_torsion_units(torsion) for torsion in eligible_torsions]


    def _add_bond_units(self, bond):
        """
        Add the correct units to a harmonic bond

        Arguments
        ---------
        bond : parmed bond object
            The bond to get units

        Returns
        -------

        """
        if type(bond.type.k)==units.Quantity:
            return bond
        bond.type.req = units.Quantity(bond.type.req, unit=units.angstrom)
        bond.type.k = units.Quantity(2.0*bond.type.k, unit=units.kilocalorie_per_mole/units.angstrom**2)
        return bond

    def _add_angle_units(self, angle):
        """
        Add the correct units to a harmonic angle

        Arguments
        ----------
        angle : parmed angle object
             the angle to get unit-ed

        Returns
        -------
        angle_with_units : parmed angle
            The angle, but with units on its parameters
        """
        if type(angle.type.k)==units.Quantity:
            return angle
        angle.type.theteq = units.Quantity(angle.type.theteq, unit=units.degree)
        angle.type.k = units.Quantity(2.0*angle.type.k, unit=units.kilocalorie_per_mole/units.radian**2)
        return angle

    def _add_torsion_units(self, torsion):
        """
        Add the correct units to a torsion

        Arguments
        ---------
        torsion : parmed.dihedral object
            The torsion needing units

        Returns
        -------
        torsion : parmed.dihedral object
            Torsion but with units added
        """
        if type(torsion.type.phi_k)==units.Quantity:
            return torsion
        torsion.type.phi_k = units.Quantity(torsion.type.phi_k, unit=units.kilocalorie_per_mole)
        torsion.type.phase = units.Quantity(torsion.type.phase, unit=units.degree)
        return torsion






"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple
import perses.rjmc.topology_proposal as topology_proposal
import numpy as np
import parmed
import simtk.unit as units
import logging



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
        beta = top_proposal.beta.in_units_of(units.kilocalorie_per_mole**-1)
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
                bond_k = bond.type.k*units.kilocalorie_per_mole/units.angstrom**2
                sigma_r = units.sqrt(1/beta*bond_k)
                logZ_r = np.log((np.sqrt(2*np.pi)*sigma_r/sigma_r.unit))
                logp_r = self._bond_logq(r_proposed, bond, beta) - logZ_r

                #propose an angle and calculate its probability
                angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
                theta_proposed = self._propose_angle(angle, beta)
                angle_k = angle.type.k*units.kilocalorie_per_mole/units.angstrom**2
                sigma_theta = units.sqrt(1/beta*angle_k)
                logZ_theta = np.log((np.sqrt(2*np.pi)*sigma_theta/sigma_theta.unit))
                logp_theta = self._angle_logq(theta_proposed, angle, beta) - logZ_theta

                #propose a torsion angle and calcualate its probability
                phi_proposed, logp_phi = self._propose_torsion(atom, r_proposed, theta_proposed, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, new_positions, beta)
                #convert to cartesian
                xyz, detJ = self._autograd_itoc(bond_atom.idx, angle_atom.idx, torsion_atom.idx, r_proposed, theta_proposed, phi_proposed, new_positions)

                #add new position to array of new positions
                new_positions[atom.idx] = xyz
                #accumulate logp
                logp_proposal = logp_proposal + logp_choice + logp_r + logp_theta + logp_phi + np.log(detJ)

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
        beta = top_proposal.beta.in_units_of(units.kilocalorie_per_mole**-1)
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
                internal_coordinates, detJ = self._autograd_ctoi(atom_coords, bond_coords, angle_coords, torsion_coords)

                #propose a bond and calculate its probability
                bond = self._get_relevant_bond(atom, bond_atom)
                bond_k = bond.type.k*units.kilocalorie_per_mole/units.angstrom**2
                sigma_r = units.sqrt(1/(beta*bond_k))
                logZ_r = np.log((np.sqrt(2*np.pi)*sigma_r/sigma_r.unit)) #need to eliminate unit to allow numpy log
                logp_r = self._bond_logq(internal_coordinates[0], bond, beta) - logZ_r

                #propose an angle and calculate its probability
                angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
                angle_k = angle.type.k*units.kilocalorie_per_mole/(units.degrees**2)
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
        relevant_bond = bonds_1.intersection(bonds_2)
        return relevant_bond.pop()

    def _get_relevant_angle(self, atom1, atom2, atom3):
        """
        Get the angle containing the 3 given atoms
        """
        atom1_angles = set(atom1.angles)
        atom2_angles = set(atom2.angles)
        atom3_angles = set(atom3.angles)
        relevant_angle = atom1_angles.intersection(atom2_angles, atom3_angles)
        return relevant_angle.pop()

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
                torsion_partners = [torsion.atom2.idx, torsion.atom3.idx, torsion.atom4.idx]
            elif torsion.atom4 == new_atom:
                torsion_partners = [torsion.atom1.idx, torsion.atom2.idx, torsion.atom3.idx]
            else:
                continue
            if set(atoms_with_positions).issuperset(set(torsion_partners)):
                eligible_torsions.append(torsion)
        return torsions

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
        return eligible_angles

    def _autograd_ctoi(self, atom_position, bond_position, angle_position, torsion_position, calculate_jacobian=True):
        import autograd
        import autograd.numpy as np

        atom_position = atom_position/atom_position.unit
        bond_position = bond_position/bond_position.unit
        angle_position = angle_position/angle_position.unit
        torsion_position = torsion_position/torsion_position.unit


        def _cartesian_to_internal(xyz):
            """
            Autograd-based jacobian of transformation from cartesian to internal.
            Returns without units!
            """
            a = bond_position - xyz
            b = angle_position - bond_position
            #3-4 bond
            c = angle_position - torsion_position
            a_u = a / np.linalg.norm(a)
            b_u = b / np.linalg.norm(b)
            c_u = c / np.linalg.norm(c)

            #bond length
            r = np.linalg.norm(a)

            #bond angle
            theta = np.arccos(np.dot(-a_u, b_u))

            #torsion angle
            plane1 = np.cross(a, b)
            plane2 = np.cross(b, c)

            phi = np.arccos(np.dot(plane1, plane2)/(np.linalg.norm(plane1)*np.linalg.norm(plane2)))

            if np.dot(np.cross(plane1, plane2), b_u) < 0:
                phi = -phi

            return np.array([r, theta, phi])
        internal_coords = _cartesian_to_internal(atom_position)
        if calculate_jacobian:
            j = autograd.jacobian(_cartesian_to_internal)
            jacobian_det = np.linalg.det(j(atom_position))
        else:
            jacobian_det = 0.0
        return internal_coords, np.abs(jacobian_det)

    def _autograd_itoc(self, bond, angle, torsion, r, theta, phi, positions, calculate_jacobian=True):
        """
        Autograd based coordinate conversion internal -> cartesian

        Arguments
        ---------
        bond : int
            index of the bonded atom
        angle : int
            index of the angle atom
        torsion : int
            index of the torsion atom
        r : float, Quantity nm
            bond length
        theta : float, Quantity rad
            bond angle
        phi : float, Quantity rad
            torsion angle
        positions : [n, 3] np.array Quantity nm
            positions of the atoms in the molecule
        calculate_jacobian : boolean
            Whether to calculate a jacobian--default True

        Returns
        -------
        atom_xyz : [1, 3] np.array Quantity nm
            The atomic positions in cartesian space
        detJ : float
            The absolute value of the determinant of the jacobian
        """
        import autograd
        import autograd.numpy as np
        positions = positions/positions.unit
        r = r/r.unit
        theta = theta/theta.unit
        phi = phi/phi.unit
        rtp = np.array([r, theta, phi])

        def _internal_to_cartesian(rthetaphi):

            a = positions[angle] - positions[bond]
            b = positions[angle] - positions[torsion]

            a_u = a / np.linalg.norm(a)
            b_u = b / np.linalg.norm(b)


            d_r = rthetaphi[0]*a_u

            normal = np.cross(a, b)

            #construct the angle rotation matrix
            axis_angle = normal / np.linalg.norm(normal)
            a = np.cos(rthetaphi[1]/2)
            b, c, d = -axis_angle*np.sin(rthetaphi[1]/2)
            angle_rotation_matrix =  np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                            [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                            [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
            #apply it
            d_ang = np.dot(angle_rotation_matrix, d_r)

            #construct the torsion rotation matrix and apply it
            axis = a_u
            a = np.cos(rthetaphi[2]/2)
            b, c, d = -axis*np.sin(rthetaphi[2]/2)
            torsion_rotation_matrix = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                            [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                            [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
            #apply it
            d_torsion = np.dot(torsion_rotation_matrix, d_ang)

            #add the positions of the bond atom
            xyz = positions[bond] + d_torsion

            return xyz

        atom_xyz = _internal_to_cartesian(rtp)
        if calculate_jacobian:
            j = autograd.jacobian(_internal_to_cartesian)
            jacobian_det = np.linalg.det(j(rtp))
            logging.debug("detJ is %f" %(jacobian_det))
        else:
            jacobian_det = 0.0
        #detj_spherical = r**2*np.sin(theta)
        #logging.debug("The spherical detJ is %f" % detj_spherical)
        return units.Quantity(atom_xyz, unit=units.nanometers), np.abs(jacobian_det)

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
        k_eq = bond.type.k*units.kilocalories_per_mole/(units.angstrom**2)
        r0 = bond.type.req*units.nanometers
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
        k_eq = angle.type.k*units.kilocalories_per_mole/(units.degrees**2)
        theta0 = angle.type.theteq*units.degrees
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
        r0 = bond.type.req*units.angstrom
        k = bond.type.k*units.kilocalorie_per_mole/units.angstrom**2
        sigma_r = units.sqrt(1.0/(beta*k))
        r = sigma_r*np.random.random() + r0
        return r

    def _propose_angle(self, angle, beta):
        """
        Bond angle proposal
        """
        theta0 = angle.type.theteq*units.degrees
        k = angle.type.k*units.kilocalorie_per_mole/units.radian**2
        sigma_theta = units.sqrt(1.0/(beta*k))
        theta = sigma_theta*np.random.random() + theta0
        return theta

    def _torsion_potential(self, torsion, phi, beta):
        """
        Calculate the log-unnormalized probability
        of the torsion
        """
        phi = phi/phi.unit
        beta = beta/beta.unit
        gamma = torsion.type.phase
        V = torsion.type.phi_k
        n = torsion.type.per
        q = -beta*(V/2.0)*(1+np.cos(n*phi-gamma))
        return q

    def _torsion_logp(self, atom, xyz, torsion, atoms_with_positions, positions, beta):
        pass

    def _propose_torsion(self, atom, r, theta, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, positions, beta):
        pass

class FFAllAngleGeometryEngine(FFGeometryEngine):
    """
    This is a forcefield-based geometry engine that takes all relevant angles
    and torsions into account when proposing a given torsion. it overrides the torsion_proposal
    and torsion_p methods of the base.
    """

    def _torsion_and_angle_potential(self, xyz, atom, positions, involved_angles, involved_torsions, beta):
        """
        Calculate the potential resulting from torsions and angles
        at a given cartesian coordinate
        """
        ub_angles = 0.0
        ub_torsions = 0.0
        if type(xyz) != units.Quantity:
            xyz = units.Quantity(xyz, units.nanometers)
        for angle in involved_angles:
            atom_position = xyz if angle.atom1 == atom else positions[angle.atom1.idx]
            bond_atom_position = xyz if angle.atom2 == atom else positions[angle.atom2.idx]
            angle_atom_position = xyz if angle.atom3 == atom else positions[angle.atom3.idx]
            theta = self._calculate_angle(atom_position, bond_atom_position, angle_atom_position)
            ub_angles += self._angle_logq(theta*units.radians, angle, beta)
        for torsion in involved_torsions:
            atom_position = xyz if torsion.atom1 == atom else positions[torsion.atom1.idx]
            bond_atom_position = xyz if torsion.atom2 == atom else positions[torsion.atom2.idx]
            angle_atom_position = xyz if torsion.atom3 == atom else positions[torsion.atom3.idx]
            torsion_atom_position = xyz if torsion.atom4 == atom else positions[torsion.atom4.idx]
            internal_coordinates, _ = self._autograd_ctoi(atom_position, bond_atom_position, angle_atom_position, torsion_atom_position, calculate_jacobian=False)
            phi = internal_coordinates[2]*units.radians
            ub_torsions += self._torsion_potential(torsion, phi, beta)
        return ub_angles+ub_torsions


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
        phis = units.Quantity(np.linspace(0, 2.0*np.pi, n_divisions), unit=units.radians)
        xyzs = np.zeros([n_divisions, 3])

        #rotate atom about torsion angle, calculating an xyz for each
        for i, phi in enumerate(phis):
            xyzs[i], _ = self._autograd_itoc(bond_atom.idx, angle_atom.idx, torsion_atom.idx, r, theta, phi, positions, calculate_jacobian=False)

        #set up arrays for energies from angles and torsions
        loq= np.zeros(n_divisions)
        for i, xyz in enumerate(xyzs):
            ub_i = self._torsion_and_angle_potential(xyz, atom, positions, involved_angles, involved_torsions, beta)
            if np.isnan(ub_i):
                ub_i = np.inf
            loq[i]+=ub_i

        #exponentiate to get the unnormalized probability
        q = np.exp(loq)

        #estimate the normalizing constant
        Z = np.trapz(q, phis)

        #get the normalized probabilities for torsions
        p = q / Z

        return p, Z, q, phis

    def _calculate_angle(self, atom_position, bond_atom_position, angle_atom_position):
        """
        Calculate the angle theta between 3 atoms 1-2-3
        """
        a = bond_atom_position - atom_position
        b = angle_atom_position - bond_atom_position
        a_u = a / np.linalg.norm(a)
        b_u = b / np.linalg.norm(b)
        theta = np.arccos(np.dot(a_u, b_u))
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
        involved_angles = self._get_valid_angles(atoms_with_positions, atom)
        involved_torsions = self._get_torsions(atoms_with_positions, atom)
        internal_coordinates = self._autograd_ctoi(xyz, positions[bond_atom.idx], positions[angle_atom.idx], positions[torsion_atom.idx])
        p, Z, q, phis = self._normalize_torsion_proposal(atom, internal_coordinates[0], internal_coordinates[1], bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=5000)
        #find the phi that's closest to the internal_coordinate phi:
        phi_idx, phi = min(enumerate(phis), key=lambda x: abs(x[1]-internal_coordinates[2]))
        p_phi = p[phi_idx]
        return np.log(p_phi)

    def _propose_torsion(self, atom, r, theta, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, positions, beta):
        """
        Propose a torsion angle, including energetic contributions from other torsions and angles
        """
        #first, let's get the normalizing constant of this distribution
        p, Z, q, phis = self._normalize_torsion_proposal(atom, r, theta, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=5000)
        #choose from the set of possible torsion angles
        phi_idx = np.random.choice(range(len(phis)), p=p)
        logp = np.log(p[phi_idx])
        phi = phis[phi_idx]
        return phi, logp

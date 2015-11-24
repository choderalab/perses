"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple
import perses.rjmc.topology_proposal as topology_proposal
import numpy as np
import scipy.stats as stats
import numexpr as ne
import parmed
import simtk.unit as units
import logging
import simtk.openmm.app as app
import simtk.openmm as openmm


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
        logp_proposal = 0.0
        top_proposal = topology_proposal.SmallMoleculeTopologyProposal()
        new_atoms = top_proposal.unique_new_atoms
        structure = parmed.openmm.load_topology(top_proposal.new_topology, top_proposal.new_system)
        atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(top_proposal.n_atoms_new) if atom_idx not in top_proposal.unique_new_atoms]
        new_positions = units.Quantity(np.zeros([top_proposal.n_atoms_new, 3]), unit=units.nanometers)
        #copy positions
        for atom in atoms_with_positions:
            old_index = top_proposal.new_to_old_atom_map[atom.idx]
            new_positions[atom.idx] = top_proposal.old_positions[old_index]

        #maintain a running list of the atoms still needing positions
        while(len(new_atoms)>0):
            atoms_for_proposal = self._atoms_eligible_for_proposal(structure, atoms_with_positions)
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
                r_proposed = self._propose_bond(bond, top_proposal.beta)
                logp_r = self._bond_logp(r_proposed, bond, top_proposal.beta)

                #propose an angle and calculate its probability
                angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
                theta_proposed = self._propose_angle(angle, top_proposal.beta)
                logp_theta = self._angle_logp(theta_proposed, angle, top_proposal.beta)

                #propose a torsion angle and calcualate its probability
                phi_proposed = self._propose_torsion(atom, structure, torsion)
                logp_phi = self._torsion_logp(atom, structure, torsion, phi_proposed)

                #convert to cartesian
                xyz, detJ = self._autograd_itoc(bond_atom.idx, angle_atom.idx, torsion_atom.idx, r_proposed, theta_proposed, phi_proposed, new_positions)

                #add new position to array of new positions
                new_positions[atom.idx] = xyz
                #accumulate logp
                logp_proposal = logp_proposal + logp_choice + logp_r + logp_theta + logp_phi + np.log(detJ)

                atoms_with_positions.append(atom)
                new_atoms.remove(atom.idx)
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
        logp = 0.0
        top_proposal = topology_proposal.SmallMoleculeTopologyProposal()
        structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
        atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(top_proposal.n_atoms_old) if atom_idx not in top_proposal.unique_old_atoms]
        new_atoms_idx = top_proposal.unique_old_atoms
        #we'll need to copy the current positions of the core to the old system
        #In the case of C-A --> C/-A -> C/-B ---> C-B these are the same
        reverse_proposal_coordinates = units.Quantity(np.zeros([top_proposal.n_atoms_new, 3]), unit=units.nanometers)
        for atom in atoms_with_positions:
            new_index = top_proposal.old_to_new_atom_map[atom.idx]
            reverse_proposal_coordinates[atom.idx] = new_coordinates[new_index]

        #maintain a running list of the atoms still needing logp
        while(len(new_atoms_idx)>0):
            atoms_for_proposal = self._atoms_eligible_for_proposal(structure, atoms_with_positions)
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
                internal_coordinates, detJ = self._autograd_ctoi(atom.idx, bond_atom.idx, angle_atom.idx, torsion_atom.idx, reverse_proposal_coordinates)

                #propose a bond and calculate its probability
                bond = self._get_relevant_bond(atom, bond_atom)
                logp_r = self._bond_logp(internal_coordinates[0], bond, top_proposal.beta)

                #propose an angle and calculate its probability
                angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
                logp_theta = self._angle_logp(internal_coordinates[1], angle, top_proposal.beta)

                #propose a torsion angle and calcualate its probability
                logp_phi = self._torsion_logp(atom, structure, torsion, internal_coordinates[2])
                logp = logp + logp_choice + logp_r + logp_theta + logp_phi + np.log(detJ)

                atoms_with_positions.append(atom)
                new_atoms_idx.remove(atom.idx)
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
        return relevant_bond

    def _get_relevant_angle(self, atom1, atom2, atom3):
        """
        Get the angle containing the 3 given atoms
        """
        atom1_angles = set(atom1.angles)
        atom2_angles = set(atom2.angles)
        atom3_angles = set(atom3.angles)
        relevant_angle = atom1_angles.intersection(atom2_angles, atom3_angles)
        return relevant_angle

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
        torsions = new_atom.torsions
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

    def _autograd_ctoi(self, atom_position, bond_position, angle_position, torsion_position):
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

        j = autograd.jacobian(_cartesian_to_internal)
        internal_coords = _cartesian_to_internal(atom_position)
        jacobian_det = np.linalg.det(j(atom_position))
        return internal_coords, np.abs(jacobian_det)

    def _autograd_itoc(self, bond, angle, torsion, r, theta, phi, positions):
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

        j = autograd.jacobian(_internal_to_cartesian)
        atom_xyz = _internal_to_cartesian(rtp)
        jacobian_det = np.linalg.det(j(rtp))
        logging.debug("detJ is %f" %(jacobian_det))
        detj_spherical = r**2*np.sin(theta)
        logging.debug("The spherical detJ is %f" % detj_spherical)
        return units.Quantity(atom_xyz, unit=units.nanometers), np.abs(jacobian_det)

    def _bond_logp(self, r, bond, beta):
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
        k_eq = bond.type.k*units.kilojoule_per_mole/(units.nanometers**2)
        r0 = bond.type.req*units.nanometers
        sigma = beta*2.0/np.sqrt(2.0*k_eq/k_eq.unit)
        logp = stats.distributions.norm.logpdf(r/r.unit, r0/r0.unit, sigma)
        return logp

    def _angle_logp(self, theta, angle, beta):
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
        k_eq = angle.type.keq*units.kilojoule_per_mole/(units.radians**2)
        theta0 = angle.type.theteq*units.radians
        sigma = beta*2.0/np.sqrt(2.0*k_eq/k_eq.unit)
        logp = stats.distributions.norm.logpdf(theta/theta.unit, theta0/theta0.unit, sigma)
        return logp

    def _torsion_logp(self, phi, torsion, beta):
        """
        Utility function for calculating the unnormalized probability of a torsion angle
        """
        beta = beta/beta.unit
        gamma = torsion.type.phase
        V = torsion.type.phi_k
        n = torsion.type.per
        q = np.exp(-beta*(V/2.0)*(1+np.cos(n*phi-gamma)))
        return q

    def _choose_torsion(self, atoms_with_positions, atom_for_proposal):
        """
        Pick an eligible torsion uniformly
        """
        eligible_torsions = self._get_torsions(atoms_with_positions, atom_for_proposal.idx)
        torsion_idx = np.random.randint(0, len(eligible_torsions))
        torsion_selected = eligible_torsions[torsion_idx]
        return torsion_selected, np.log(1/len(eligible_torsions))

    def _atoms_eligible_for_proposal(self, structure, atoms_with_positions):
        """
        Get the set of atoms eligible for proposal
        """
        eligible_atoms = []
        for atom in structure.atoms:
            #get array of booleans to see if a bond partner has a position
            has_bonded_position = [a.idx in atoms_with_positions for a in atom.bond_partners]
            #if at least one does, then the atom is ready to be proposed.
            if np.sum(has_bonded_position) > 0:
                eligible_atoms.append(atom)
        return eligible_atoms

    def _propose_bond(self, bond, beta):
        """
        Bond length proposal
        """
        k_eq = bond.type.k*units.kilojoule_per_mole/(units.nanometers**2)
        r0 = bond.type.req*units.nanometers
        sigma = beta*2.0/np.sqrt(2.0*k_eq/k_eq.unit)
        r = sigma*np.random.random()*r0.unit + r0
        return r

    def _propose_angle(self, angle, beta):
        """
        Bond angle proposal
        """
        theta0 = angle.type.theteq*units.radians
        k_eq = angle.type.keq*units.kilojoule_per_mole/(units.radians**2)
        sigma = beta*2.0/np.sqrt(2.0*k_eq/k_eq.unit)
        theta = sigma*np.random.random()*theta0.unit + theta0
        return theta

    def _propose_torsion(self, atom, structure, bond, angle, torsion, atoms_with_position, positionss):
        pass

class FFAllAngleGeometryEngine(FFGeometryEngine):
    """
    This is a forcefield-based geometry engine that takes all relevant angles
    and torsions into account when proposing a given torsion. it overrides the torsion_proposal
    and torsion_p methods of the base.
    """

    def _calc_torsion_proposal(self, atom, r, theta, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=5000):
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
        phi_proposed : float, radians
            proposed torsion angle
        """
        n_divisions = 5000
        #first, get the list of angles and torsions involved:
        involved_angles = self._get_valid_angles(atoms_with_positions, atom)
        involved_torsions = self._get_torsions(atoms_with_positions, atom)

        #get an array of [0,2pi)
        phis = np.linspace(0, 2.0*np.pi, n_divisions)
        xyzs = np.zeros([n_divisions, 3])

        #rotate atom about torsion angle, calculating an xyz for each
        for i, phi in enumerate(phis):
            xyzs[i], _ = self._autograd_itoc(bond_atom.idx, angle_atom.idx, torsion_atom.idx, r, theta, phi, positions)

        #set up arrays for energies from angles and torsions
        ub_angles = np.zeros(n_divisions)
        for i, xyz in enumerate(xyzs):
            for angle in involved_angles:
                atom_position = xyz if angle.atom1 == atom else positions[angle.atom1.idx]
                bond_atom_position = xyz if angle.atom2 == atom else positions[angle.atom2.idx]
                angle_atom_position = xyz if angle.atom3 == atom else positions[angle.atom3.idx]
                theta = self._calculate_angle(atom_position, bond_atom_position, angle_atom_position)
                ub_angles[i] += self._angle_logp(theta*units.radians, angle, beta)

        #now the torsions
        ub_torsions = np.zeros(n_divisions)
        for i, xyz in enumerate(xyzs):
            for torsion in involved_torsions:
                atom_position = xyz if torsion.atom1 == atom else positions[torsion.atom1.idx]
                bond_atom_position = xyz if torsion.atom2 == atom else positions[torsion.atom2.idx]
                angle_atom_position = xyz if torsion.atom3 == atom else positions[torsion.atom3.idx]
                torsion_atom_position = xyz if torsion.atom4 == atom else positions[torsion.atom4.idx]
                internal_coordinates, _ = self._autograd_ctoi(atom_position, bond_atom_position, angle_atom_position, torsion_atom_position, positions)
                phi = internal_coordinates[2]
                ub_torsions[i] += self._torsion_logp(phi, torsion, beta)

        #add the energetic contributions
        ub_total = ub_angles + ub_torsions

        #exponentiate to get the unnormalized probability
        q = np.exp(ub_total)

        #estimate the normalizing constant
        Z = 1.0 / np.trapz(q, phis)

        #get the normalized probabilities for torsions
        p = q / Z

        return p

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


class FFGeometryEngineOld(GeometryEngine):
    """
    This is an implementation of the GeometryEngine class which proposes new dimensions based on the forcefield parameters
    """


    def propose(self, top_proposal):
        """
        Propose a new geometry for the appropriate atoms using forcefield parameters

        Arguments
        ----------
        topology_proposal : TopologyProposal object
            Object containing the relevant results of a topology proposal

        Returns
        -------
        new_positions : [m, 3] np.array of floats
            The positions of the m atoms in the new system (if not propose_positions, this is the old positions)
        logp : float
            The logp of the proposal, including the jacobian
        """

        #get the mapping between new and old atoms
        n_atoms_new = top_proposal.n_atoms_new
        n_atoms_old = top_proposal.n_atoms_old

        #get a list of the new atoms
        new_atoms = top_proposal.unique_new_atoms

        #get a list of the old atoms
        old_atoms = top_proposal.unique_old_atoms

        #create a new array with the appropriate size
        new_positions = np.zeros([n_atoms_new, 3])

        new_to_old_atom_map = top_proposal.new_to_old_atom_map

        #transfer known positions--be careful about units!
        for atom in new_to_old_atom_map.keys():
            new_positions[atom] = top_proposal.old_positions[new_to_old_atom_map[atom]].in_units_of(units.nanometers)

        #restore units
        new_positions = new_positions*units.nanometers
        #get a new set of positions and a logp of that proposal
        final_new_positions, logp_forward = self._propose_new_positions(new_atoms, top_proposal.new_system, new_positions, new_to_old_atom_map)

        #get the probability of a reverse proposal
        #first get the atom map reversed
        old_to_new_atom_map = top_proposal.old_to_new_atom_map
        _, logp_reverse = self._propose_new_positions(old_atoms, top_proposal.old_system, top_proposal.old_positions, old_to_new_atom_map, propose_positions=False)

        #construct the return object
        geometry_proposal = GeometryProposal(final_new_positions, logp_reverse - logp_forward)

        return geometry_proposal

    def _propose_new_positions(self, new_atoms, new_system, new_positions, new_to_old_atom_map, propose_positions=True):
        """
        Propose new atomic positions, or get the probability of existing ones
        (used to evaluate the reverse probability, where the old atoms become the new atoms, etc)
        Arguments
        ---------
        new_atoms : list of int
            Indices of atoms that need positions
        topology_proposal : TopologyProposal namedtuple
            The result of the topology proposal, containing the atom mapping and topologies
        new_system : simtk.OpenMM.System object
            The new system
        new_positions : [m, 3] np.array of floats
            The positions of the m atoms in the new system, with unpositioned atoms having [0,0,0]

        Returns
        -------
        new_positions : [m, 3] np.array of floats
            The positions of the m atoms in the new system (if not propose_positions, this is the old positions)
        logp : float
            The logp of the proposal, including the jacobian
        """
        logp = 0.0
        atoms_with_positions = new_to_old_atom_map.keys()
        forces = {new_system.getForce(index).__class__.__name__ : new_system.getForce(index) for index in range(new_system.getNumForces())}
        torsion_force = forces['PeriodicTorsionForce']
        bond_force = forces['HarmonicBondForce']
        atomic_proposal_order = self._generate_proposal_order(new_atoms, new_system, new_to_old_atom_map)
        for atomset in atomic_proposal_order:
            for atom, possible_torsions in atomset.items():
                #randomly select a torsion from the available list, and calculate its log-probability
                selected_torsion_index = np.random.randint(0, len(possible_torsions))
                selected_torsion = possible_torsions[selected_torsion_index]
                logp_selected_torsion = - np.log(len(possible_torsions))
                torsion_parameters = torsion_force.getTorsionParameters(selected_torsion)
                #check to see whether the atom is the first or last in the torsion
                if atom == torsion_parameters[0]:
                    bonded_atom = torsion_parameters[1]
                    angle_atom = torsion_parameters[2]
                    torsion_atom = torsion_parameters[3]
                else:
                    bonded_atom = torsion_parameters[2]
                    angle_atom = torsion_parameters[1]
                    torsion_atom = torsion_parameters[0]

                #get a new position for the atom, if specified
                if propose_positions:
                    atomic_xyz, logp_atomic = self._propose_atomic_position(new_system, atom, bonded_atom, angle_atom, torsion_atom, torsion_parameters, new_positions)
                    new_positions[atom] = atomic_xyz
                else:
                    logp_atomic = 0.0
                    logp_atomic = self._calculate_logp_atomic_position(new_system, new_positions, atom, bonded_atom, angle_atom, torsion_atom, torsion_parameters)
                logp += logp_atomic + logp_selected_torsion

        return new_positions, logp

    def _generate_proposal_order(self, new_atoms, new_system, new_to_old_atom_map):
        """
        Generate the list of atoms (and corresponding torsions) to be proposed each round

        Arguments
        ---------
        new_atoms : list of int
            Indices of atoms that need positions
        topology_proposal : TopologyProposal namedtuple
            The result of the topology proposal, containing the atom mapping and topologies
        new_system : simtk.OpenMM.System object
            The new system
        new_positions : [m, 3] np.array of floats
            The positions of the m atoms in the new system, with unpositioned atoms having [0,0,0]

        Returns
        -------
        atom_proposal_list : list of dict
            List of dict of form {atom_index : [torsion_list]}. each entry in the list is a
            round of atoms eligible for proposal.
        """
        atoms_with_positions = new_to_old_atom_map.keys()
        forces = {new_system.getForce(index).__class__.__name__ : new_system.getForce(index) for index in range(new_system.getNumForces())}
        torsion_force = forces['PeriodicTorsionForce']
        bond_force = forces['HarmonicBondForce']
        atom_proposal_list = []
        while len(new_atoms) > 0:
            atom_torsion_proposals = self._atoms_eligible_for_proposal(torsion_force, bond_force, atoms_with_positions, new_atoms)
            #now loop through the list of atoms found to be eligible for positions this round
            atom_proposal_list.append(atom_torsion_proposals)
            for atom in atom_torsion_proposals.keys():
                #remove atom from list of atoms needing positions
                new_atoms.remove(atom)
                atoms_with_positions.append(atom)
        return atom_proposal_list



    def _propose_atomic_position(self, system, atom, bonded_atom, angle_atom, torsion_atom, torsion_parameters, positions):
        """
        Method to propose the position of a single atom.

        Arguments
        ---------
        system : simtk.openmm.System
            the system containing the atom of interest
        atom : int
            the index of the atom whose position should be proposed
        bonded_atom : int
            the index of the atom bonded to the atom of interest
        angle_atom : int
            the 1, 3 position atom index
        torsion_parameters : dict
            The parameters of the torsion that is being used for this proposal

        Returns
        -------
        atomic_xyz : [1,3] np.array of floats
            the cartesian coordinates of the atom
        logp_atomic_position : float
            the logp of this proposal, including jacobian correction
        """
        logging.debug("Proposing position %d with bond atom %d, angle atom %d, torsion atom %d" % (atom, bonded_atom, angle_atom, torsion_atom))
        #get bond parameters and draw bond length
        r0, k_eq = self._get_bond_parameters(system, atom, bonded_atom)
        r, logp_bond = self._propose_bond_length(r0, k_eq)

        #get angle parameters and draw bond angle
        theta0, k_eq = self._get_angle_parameters(system, atom, bonded_atom, angle_atom)
        theta, logp_angle = self._propose_bond_angle(theta0, k_eq)

        #propose torsion
        phi, logp_torsion = self._propose_torsion_angle(torsion_parameters[6], torsion_parameters[4], torsion_parameters[5])
        #convert spherical to cartesian coordinates
        spherical_coordinates = np.asarray([r, theta, phi])
        atomic_xyz, detJ = self._autograd_itoc(bonded_atom, angle_atom, torsion_atom, r, theta, phi, positions)
        #accumulate the forward logp with jacobian correction
        logp_atomic_position = (logp_angle + logp_bond + logp_torsion + np.log(detJ))
        logging.debug("Proposed (r, theta, phi) of (%s, %s, %s)" % (str(r), str(theta), str(phi)))
        return atomic_xyz, logp_atomic_position


    def _calculate_logp_atomic_position(self, system, positions, atom, bonded_atom, angle_atom, torsion_atom, torsion_parameters):
        """
        Calculate the log-probability of a given atom's position

        Arguments
        ---------
        system : simtk.openmm.System object
            The system whose atomic position needs a logp
        positions : [n, 3] np.ndarray of floats, Quantity nm
            The positions of the particles in the system
        atom : int
            The index of the atom of interest
        bonded_atom : int
            The index of the bonded atom
        angle_atom : int
            The index of the angle atom
        torsion_atom : int
            The index of the torsion atom
        torsion_parameters : array
            The parameters of the torsion

        Returns
        -------
        logp : float
            logp of position, including det(J) correction
        """
        #convert cartesian coordinates to spherical
        relative_xyz = positions[atom] - positions[bonded_atom]
        internal_coordinates, detJ = self._autograd_ctoi(atom, bonded_atom, angle_atom, torsion_atom, positions)

        #get bond parameters and calculate the probability of choosing the current positions
        r0, k_eq = self._get_bond_parameters(system, atom, bonded_atom)
        sigma_bond = 1.0/np.sqrt(2.0*k_eq/k_eq.unit)
        logp_bond = stats.distributions.norm.logpdf(internal_coordinates[0], r0/r0.unit, sigma_bond)

        #calculate the probability of choosing this bond angle
        theta0, k_eq = self._get_angle_parameters(system, atom, bonded_atom, angle_atom)
        sigma_angle = 1.0/np.sqrt(2.0*k_eq/k_eq.unit)
        logp_angle = stats.distributions.norm.logpdf(internal_coordinates[1], theta0/theta0.unit, sigma_angle)

        #calculate the probabilty of choosing this torsion angle
        torsion_Z, _ = self._torsion_normalizer(torsion_parameters[6], torsion_parameters[4], torsion_parameters[5])
        logp_torsion = self._torsion_p(torsion_Z,torsion_parameters[6], torsion_parameters[4], torsion_parameters[5], internal_coordinates[2])

        logp_atomic_position = (logp_angle + logp_bond + logp_torsion + np.log(detJ))

        return logp_atomic_position

    def _atoms_eligible_for_proposal(self, torsion_force, bond_force, atoms_with_positions, new_atoms):
        """
        Determine which atoms are eligible for position proposal this round.
        The criteria are that the atom must not yet have a position, and must
        be 1 or 4 in at least one torsion where the other atoms have positoins.

        Arguments
        ---------
        torsion_force : simtk.openmm.PeriodicTorsionForce
            The torsion force from the system
        bond_roce : simtk.openmm.HarmonicBondForce
            The HarmonicBondForce from the system
        atoms_with_positions : list of int
            A list of the indices of atoms that have positions

        Returns
        -------
        atom_torsion_proposals : dict
            Dictionary containing {eligible_atom : [list_of_torsions]}

        """
        current_atom_proposals = []
        atom_torsion_proposals = {atom_index: [] for atom_index in new_atoms}
        for torsion_index in range(torsion_force.getNumTorsions()):
            atom1 = torsion_force.getTorsionParameters(torsion_index)[0]
            atom2 = torsion_force.getTorsionParameters(torsion_index)[1]
            atom3 = torsion_force.getTorsionParameters(torsion_index)[2]
            atom4 = torsion_force.getTorsionParameters(torsion_index)[3]
            #Only take torsions where the "new" statuses of atoms 1 and 4 are not equal
            if (atom1 in atoms_with_positions) != (atom4 in atoms_with_positions):
                #make sure torsion is not improper:
                if not self._is_proper(torsion_force.getTorsionParameters(torsion_index), bond_force):
                    continue
                #only take torsions where 2 and 3 have known positions
                if (atom2 in atoms_with_positions) and (atom3 in atoms_with_positions):
                    #finally, append the torsion index to the list of possible atom sets for proposal
                    if atom1 in atoms_with_positions and (atom1 not in current_atom_proposals):
                        atom_torsion_proposals[atom4].append(torsion_index)
                        current_atom_proposals.append(atom4)
                    elif atom4 in atoms_with_positions and (atom4 not in current_atom_proposals):
                        atom_torsion_proposals[atom1].append(torsion_index)
                        current_atom_proposals.append(atom1)
                    else:
                        continue
        return {atom: torsionlist for atom, torsionlist in atom_torsion_proposals.items() if len(torsionlist) > 0}


    def _is_proper(self, torsion_parameters, bond_force):
        """
        Utility function to determine if torsion is proper
        """
        #get the atoms
        torsion_atoms = torsion_parameters[:4]
        is_proper = True
        for i in range(3):
            is_proper = is_proper and self._is_bond(torsion_atoms[i], torsion_atoms[i+1], bond_force)
        return is_proper

    def _is_bond(self, atom1, atom2, bond_force):
        """
        Utility function to determine if bond exists between two atoms
        """
        for bond_index in range(bond_force.getNumBonds()):
            parameters = bond_force.getBondParameters(bond_index)
            if (parameters[0] == atom1 and parameters[1] == atom2) or (parameters[1] == atom1 and parameters[0] == atom2):
                return True
        return False


    def _get_bond_parameters(self, system, atom1, atom2):
        """
        Utility function to get the bonded parameters for a pair of atoms

        Arguments
        ---------
        system : simtk.openmm.System
            the system containing the parameters of interest
        atom1 : int
            the index of the first atom in the bond
        atom2 : int
            the index of the second atom in the bond

        Returns
        -------
        r0 : float
            equilibrium bond length
        k_eq : float
            bond spring constant
        """
        forces = {system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces())}
        bonded_force = forces['HarmonicBondForce']
        for bond_index in range(bonded_force.getNumBonds()):
            parameters = bonded_force.getBondParameters(bond_index)
            if (parameters[0] == atom1 and parameters[1] == atom2) or (parameters[1] == atom1 and parameters[0] == atom2):
                return parameters[2], parameters[3]

    def _get_angle_parameters(self, system, atom1, atom2, atom3):
        """
        Utility function to retrieve harmonic angle parameters for
        a given set of atoms in a system

        Arguments
        ---------
        system : simtk.openmm.System
            the system with parameters
        atom1 : int
            the index of the first atom
        atom2 : int
            the index of the second atom
        atom3 : int
            the index of the third atom

        Returns
        -------
        theta0 : float
            equilibrium bond angle
        k_eq : float
            angle spring constant
        """
        list_of_angle_atoms = [atom1, atom2, atom3]
        #compare sorted lists of atoms
        forces = {system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces())}
        angle_force = forces['HarmonicAngleForce']
        for angle_index in range(angle_force.getNumAngles()):
            parameters = angle_force.getAngleParameters(angle_index)
            #the first three "parameters" are atom indices
            atoms = parameters[:3]
            if np.all(np.equal(list_of_angle_atoms, atoms)) or np.all(np.equal(list_of_angle_atoms[::-1], atoms)):
                return parameters[3], parameters[4]

    def _get_unique_atoms(self, new_to_old_map, new_atom_list, old_atom_list):
        """
        Get the set of atoms unique to both the new and old system.

        Arguments
        ---------
        new_to_old_map : dict
            Dictionary of the form {new_atom : old_atom}

        Returns
        -------
        unique_old_atoms : list of int
            A list of the indices of atoms unique to the old system
        unique_new_atoms : list of int
            A list of the indices of atoms unique to the new system
        """
        mapped_new_atoms = new_to_old_map.keys()
        unique_new_atoms = [atom for atom in new_atom_list not in mapped_new_atoms]
        mapped_old_atoms = new_to_old_map.values()
        unique_old_atoms = [atom for atom in old_atom_list not in mapped_old_atoms]
        return unique_old_atoms, unique_new_atoms

    def _propose_bond_length(self, r0, k_eq):
        """
        Draw a bond length from the equilibrium bond length distribution

        Arguments
        ---------
        r0 : float
            The equilibrium bond length
        k_eq : float
            The bond spring constant

        Returns
        --------
        r : float
            the proposed bond length
        logp : float
            the log-probability of the proposal
        """
        sigma = 2.0/np.sqrt(2.0*k_eq/k_eq.unit)
        r = sigma*np.random.random()*r0.unit + r0
        logp = stats.distributions.norm.logpdf(r/r.unit, r0/r0.unit, sigma)
        return (r, logp)

    def _propose_bond_angle(self, theta0, k_eq):
        """
        Draw a bond length from the equilibrium bond length distribution

        Arguments
        ---------
        theta0 : float
            The equilibrium bond angle
        k_eq : float
            The angle spring constant

        Returns
        --------
        r : float
            the proposed bond angle
        logp : float
            the log-probability of the proposal
        """
        sigma = 1.0/np.sqrt(2.0*k_eq/k_eq.unit)
        theta = sigma*np.random.random()*theta0.unit + theta0
        logp = stats.distributions.norm.logpdf(theta / theta.unit, theta0/theta0.unit, sigma)
        return (theta, logp)

    def _propose_torsion_angle(self, V, n, gamma):
        """
        Draws a torsion angle from the distribution of one torsion.
        Uses the functional form U(phi) = V/2[1+cos(n*phi-gamma)],
        as described by Amber.

        Arguments
        ---------
        V : float
            force constant
        n : int
            multiplicity of the torsion
        gamma : float
            phase angle

        Returns
        -------
        phi : float
            The proposed torsion angle
        logp : float
            The proposed log probability of the torsion angle
        """
        (Z, max_p) = self._torsion_normalizer(V, n, gamma)
        phi = 0.0
        logp = 0.0
        #sample from the distribution
        accepted = False
        #use rejection sampling
        while not accepted:
            phi_samp = np.random.uniform(0.0, 2*np.pi)
            runif = np.random.uniform(0.0, max_p+1.0)
            p_phi_samp = self._torsion_p(Z, V, n, gamma, phi_samp)
            if p_phi_samp > runif:
                phi = phi_samp
                logp = np.log(p_phi_samp)
                accepted = True
            else:
                continue
        return (phi*units.radians, logp)

    def _torsion_p(self, Z, V, n, gamma, phi):
        """
        Utility function for calculating the normalized probability of a torsion angle
        """
        #must remove units
        V = V/V.unit
        gamma = gamma/gamma.unit
        return (1.0/Z)*np.exp(-(V/2.0)*(1+np.cos(n*phi-gamma)))

    def _torsion_normalizer(self, V, n, gamma):
        """
        Utility function to numerically normalize torsion angles.
        Also return max_p to facilitate rejection sampling
        """
        #generate a grid of 5000 points from 0 < phi < 2*pi
        phis = np.linspace(0, 2.0*np.pi, 5000)
        #evaluate the unnormalized probability at each of those points
        #need to remove units--numexpr can't handle them otherwise
        V = V/V.unit
        gamma = gamma/gamma.unit
        phi_q = ne.evaluate("exp(-(V/2.0)*(1+cos(n*phis-gamma)))")
        #integrate the values
        Z = np.trapz(phi_q, phis)
        max_p = np.max(phi_q/Z)
        return Z, max_p



    def _autograd_ctoi(self, atom, bond, angle, torsion, positions):
        import autograd
        import autograd.numpy as np

        positions = positions/positions.unit
        atom_position = positions[atom]


        def _cartesian_to_internal(xyz):
            """
            Autograd-based jacobian of transformation from cartesian to internal.
            Returns without units!
            """
            a = positions[bond] - xyz
            b = positions[angle] - positions[bond]
            #3-4 bond
            c = positions[angle] - positions[torsion]
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

        j = autograd.jacobian(_cartesian_to_internal)
        internal_coords = _cartesian_to_internal(atom_position)
        jacobian_det = np.linalg.det(j(atom_position))
        return internal_coords, np.abs(jacobian_det)

    def _autograd_itoc(self, bond, angle, torsion, r, theta, phi, positions):
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

        j = autograd.jacobian(_internal_to_cartesian)
        atom_xyz = _internal_to_cartesian(rtp)
        jacobian_det = np.linalg.det(j(rtp))
        logging.debug("detJ is %f" %(jacobian_det))
        detj_spherical = r**2*np.sin(theta)
        logging.debug("The spherical detJ is %f" % detj_spherical)
        return units.Quantity(atom_xyz, unit=units.nanometers), np.abs(jacobian_det)


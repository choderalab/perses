"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
import parmed
import simtk.unit as units
import logging
import numpy as np
import copy
from perses.rjmc import coordinate_tools
import simtk.openmm as openmm




class GeometryEngine(object):
    """
    This is the base class for the geometry engine.

    Arguments
    ---------
    metadata : dict
        GeometryEngine-related metadata as a dict
    """

    def __init__(self, metadata=None):
        pass

    def propose(self, top_proposal, current_positions, beta):
        """
        Make a geometry proposal for the appropriate atoms.

        Arguments
        ----------
        top_proposal : TopologyProposal object
            Object containing the relevant results of a topology proposal
        beta : float
            The inverse temperature

        Returns
        -------
        new_positions : [n, 3] ndarray
            The new positions of the system
        """
        return np.array([0.0,0.0,0.0])

    def logp_reverse(self, top_proposal, new_coordinates, old_coordinates, beta):
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
        beta : float
            The inverse temperature

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
    def __init__(self, metadata=None):
        self._metadata = metadata

    def propose(self, top_proposal, current_positions, beta):
        """
        Make a geometry proposal for the appropriate atoms.

        Arguments
        ----------
        top_proposal : TopologyProposal object
            Object containing the relevant results of a topology proposal
        beta : float
            The inverse temperature

        Returns
        -------
        new_positions : [n, 3] ndarray
            The new positions of the system
        logp_proposal : float
            The log probability of the forward-only proposal
        """
        logp_proposal, new_positions = self._logp_propose(top_proposal, current_positions, beta, direction='forward')
        return new_positions, logp_proposal


    def logp_reverse(self, top_proposal, new_coordinates, old_coordinates, beta):
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
        beta : float
            The inverse temperature

        Returns
        -------
        logp : float
            The log probability of the proposal for the given transformation
        """
        logp_proposal, _ = self._logp_propose(top_proposal, old_coordinates, beta, new_positions=new_coordinates, direction='reverse')
        return logp_proposal

    def _logp_propose(self, top_proposal, old_positions, beta, new_positions=None, direction='forward'):
        """
        This is an INTERNAL function that handles both the proposal and the logp calculation,
        to reduce code duplication. Whether it proposes or just calculates a logp is based on
        the direction option. Note that with respect to "new" and "old" terms, "new" will always
        mean the direction we are proposing (even in the reverse case), so that for a reverse proposal,
        this function will still take the new coordinates as new_coordinates

        Parameters
        ----------
        top_proposal : topology_proposal.TopologyProposal object
            topology proposal containing the relevant information
        old_positions : np.ndarray [n,3] in nm
            The old coordinates.
        beta : float
            Inverse temperature
        new_positions : np.ndarray [n,3] in nm, optional for forward
            The new coordinates, if any. For proposal this is none
        direction : str
            Whether to make a proposal (forward) or just calculate logp (reverse)

        Returns
        -------
        logp_proposal : float
            the logp of the proposal
        new_positions : [n,3] np.ndarray
            The new positions (same as input if direction='reverse')
        """
        proposal_order_tool = ProposalOrderTools(top_proposal)
        growth_system_generator = GeometrySystemGenerator()
        growth_parameter_name = 'growth_stage'
        if direction=="forward":
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='forward')
            structure = parmed.openmm.load_topology(top_proposal.new_topology, top_proposal.new_system)
            logp_proposal = logp_choice

            #find and copy known positions
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(top_proposal.n_atoms_new) if atom_idx not in top_proposal.unique_new_atoms]
            new_positions = self._copy_positions(atoms_with_positions, top_proposal, old_positions)

            growth_system = growth_system_generator.create_modified_system(top_proposal.new_system, atom_proposal_order.keys(), growth_parameter_name)
        elif direction=='reverse':
            if new_positions is None:
                raise ValueError("For reverse proposals, new_positions must not be none.")
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='reverse')
            structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
            logp_proposal = logp_choice

            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.old_to_new_atom_map.keys()]

            #copy common atomic positions
            for atom in structure.atoms:
                if atom.idx in atoms_with_positions:
                    corresponding_new_index = top_proposal.old_to_new_atom_map[atom.idx]
                    old_positions[atom.idx] = new_positions[corresponding_new_index]
            growth_system = growth_system_generator.create_modified_system(top_proposal.old_system, atom_proposal_order.keys(), growth_parameter_name)
        else:
            raise ValueError("Parameter 'direction' must be forward or reverse")

        platform = openmm.Platform.getPlatformByName('CPU')
        integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        context = openmm.Context(growth_system, integrator, platform)
        growth_parameter_value = 0
        #now for the main loop:
        for atom, torsion in atom_proposal_order.items():

            context.setParameter(growth_parameter_name, growth_parameter_value)
            bond_atom = torsion.atom2
            angle_atom = torsion.atom3
            torsion_atom = torsion.atom4

            #get internal coordinates if direction is reverse
            if direction=='reverse':
                atom_coords = old_positions[atom.idx]
                bond_coords = old_positions[bond_atom.idx]
                angle_coords = old_positions[angle_atom.idx]
                torsion_coords = old_positions[torsion_atom.idx]
                internal_coordinates, detJ = self._cartesian_to_internal(atom_coords, bond_coords, angle_coords, torsion_coords)
                r = internal_coordinates[0]*atom_coords.unit
                theta = internal_coordinates[1]*units.radian
                phi = internal_coordinates[2]*units.radian

            bond = self._get_relevant_bond(atom, bond_atom)
            if bond is not None:
                if direction=='forward':
                    r = self._propose_bond(bond, beta)
                bond_k = bond.type.k
                sigma_r = units.sqrt(1/(beta*bond_k))
                logZ_r = np.log((np.sqrt(2*np.pi)*sigma_r/sigma_r.unit))
                logp_r = self._bond_logq(r, bond, beta) - logZ_r
            else:
                constraint = self._get_bond_constraint(atom, bond_atom, top_proposal.new_system)
                r = constraint #set bond length to exactly constraint
                logp_r = 0.0

            #propose an angle and calculate its probability
            angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
            if direction=='forward':
                theta = self._propose_angle(angle, beta)
            angle_k = angle.type.k
            sigma_theta = units.sqrt(1/(beta*angle_k))
            logZ_theta = np.log((np.sqrt(2*np.pi)*sigma_theta/sigma_theta.unit))
            logp_theta = self._angle_logq(theta, angle, beta) - logZ_theta

            #propose a torsion angle and calcualate its probability
            if direction=='forward':
                phi, logp_phi = self._propose_torsion(context, torsion, new_positions, r, theta, beta, n_divisions=5000)
                xyz, detJ = self._internal_to_cartesian(new_positions[bond_atom.idx], new_positions[angle_atom.idx], new_positions[torsion_atom.idx], r, theta, phi)
                new_positions[atom.idx] = xyz
            else:
                logp_phi = self._torsion_logp(context, torsion, old_positions, phi, beta, n_divisions=5000)

            #accumulate logp
            logp_proposal += logp_proposal + logp_r + logp_theta +logp_phi + np.log(detJ)
            growth_parameter_value += 1

        return logp_proposal, new_positions

    def _copy_positions(self, atoms_with_positions, top_proposal, current_positions):
        """
        Copy the current positions to an array that will also hold new positions
        Parameters
        ----------
        atoms_with_positions : list of parmed.Atom
            atoms that currently have positions
        top_proposal : topology_proposal.TopologyProposal
            topology proposal object
        current_positions : [n, 3] np.ndarray in nm
            Positions of the current system

        Returns
        -------
        new_positions : np.ndarray in nm
            Array for new positions with known positions filled in
        """
        new_positions = units.Quantity(np.zeros([top_proposal.n_atoms_new, 3]), unit=units.nanometers)
        current_positions = current_positions.in_units_of(units.nanometers)
        #copy positions
        for atom in atoms_with_positions:
            old_index = top_proposal.new_to_old_atom_map[atom.idx]
            new_positions[atom.idx] = current_positions[old_index]
        return new_positions

    def _get_relevant_bond(self, atom1, atom2):
        """
        utility function to get the bond connecting atoms 1 and 2.
        Returns either a bond object or None
        (since there is no constraint class)

        Arguments
        ---------
        atom1 : parmed atom object
             One of the atoms in the bond
        atom2 : parmed.atom object
             The other atom in the bond

        Returns
        -------
        bond : bond object
            Bond connecting the two atoms, if there is one. None if constrained or
            no bond.
        """
        bonds_1 = set(atom1.bonds)
        bonds_2 = set(atom2.bonds)
        relevant_bond_set = bonds_1.intersection(bonds_2)
        relevant_bond = relevant_bond_set.pop()
        if relevant_bond.type is None:
            return None
        relevant_bond_with_units = self._add_bond_units(relevant_bond)
        return relevant_bond_with_units

    def _get_bond_constraint(self, atom1, atom2, system):
        """
        Get the constraint parameters corresponding to the bond
        between the given atoms

        Parameters
        ----------
        atom1 : parmed.Atom object
           the first atom of the constrained bond
        atom2 : parmed.Atom object
           the second atom of the constrained bond
        system : openmm.System object
           The system containing the constraint

        Returns
        -------
        constraint : float, quantity nm
            the parameters of the bond constraint
        """
        atom_indices = {atom1.idx, atom2.idx}
        n_constraints = system.getNumConstraints()
        constraint = None
        for i in range(n_constraints):
            constraint_parameters = system.getConstraintParameters(i)
            constraint_atoms = set(constraint_parameters[:2])
            if len(constraint_atoms.intersection(atom_indices))==2:
                constraint = constraint_parameters[2]
        return constraint

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
        if type(torsion.type.phi_k) == units.Quantity:
            return torsion
        torsion.type.phi_k = units.Quantity(torsion.type.phi_k, unit=units.kilocalorie_per_mole)
        torsion.type.phase = units.Quantity(torsion.type.phase, unit=units.degree)
        return torsion

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
        atom_position = atom_position.value_in_unit(units.nanometers)
        bond_position = bond_position.value_in_unit(units.nanometers)
        angle_position = angle_position.value_in_unit(units.nanometers)
        torsion_position = torsion_position.value_in_unit(units.nanometers)

        internal_coords = coordinate_tools._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)


        return internal_coords, internal_coords[0]**2*np.sin(internal_coords[1])

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

    def _torsion_scan(self, torsion, positions, r, theta, n_divisions=5000):
        """
        Rotate the atom about the
        Parameters
        ----------
        torsion : parmed.Dihedral
            parmed Dihedral containing relevant atoms
        positions : [n,3] np.ndarray in nm
            positions of the atoms in the system
        r : float in nm
            bond length
        theta : float in radians
            bond angle

        Returns
        -------
        xyzs : np.ndarray, in nm
            The cartesian coordinates of each
        phis : np.ndarray, in radians
            The torsions angles at which a potential will be calculated
        """
        positions_copy = copy.deepcopy(positions)
        positions_copy = positions_copy.in_units_of(units.nanometers)
        r = r.in_units_of(units.nanometers)
        theta = theta.in_units_of(units.radians)
        bond_atom = torsion.atom2
        angle_atom = torsion.atom3
        torsion_atom = torsion.atom4
        phis = units.Quantity(np.arange(0, 2.0*np.pi, (2.0*np.pi)/n_divisions), unit=units.radians)
        xyzs = units.Quantity(np.zeros([len(phis), 3]), unit=units.nanometers)
        for i, phi in enumerate(phis):
            xyzs[i], _ = self._internal_to_cartesian(positions_copy[bond_atom.idx], positions_copy[angle_atom.idx], positions_copy[torsion_atom.idx], r, theta, phi)
        return xyzs, phis

    def _torsion_log_pmf(self, growth_context, torsion, positions, r, theta, beta, n_divisions=5000):
        """
        Calculate the torsion logp pmf using OpenMM

        Parameters
        ----------
        growth_context : openmm.Context
            Context containing the modified system and
        torsion : parmed.Dihedral
            parmed Dihedral containing relevant atoms
        positions : [n,3] np.ndarray in nm
            positions of the atoms in the system
        r : float in nm
            bond length
        theta : float in radians
            bond angle
        beta : float
            inverse temperature
        n_divisions : int, optional
            number of divisions for the torsion scan

        Returns
        -------
        logp_torsions : np.ndarray of float
            normalized probability of each of n_divisions of torsion
        phis : np.ndarray, in radians
            The torsions angles at which a potential was calculated
        """
        logq = np.zeros(n_divisions)
        atom_idx = torsion.atom1.idx
        xyzs, phis = self._torsion_scan(torsion, positions, r, theta, n_divisions=n_divisions)
        for i, xyz in enumerate(xyzs):
            positions[atom_idx] = xyz
            growth_context.setPositions(positions)
            state = growth_context.getState(getEnergy=True)
            logq[i] = -beta*state.getPotentialEnergy()
        logq -= max(logq)
        q = np.exp(logq)
        Z = np.sum(q)
        logp_torsions = logq - np.log(Z)
        return logp_torsions, phis

    def _propose_torsion(self, growth_context, torsion, positions, r, theta, beta, n_divisions=5000):
        """
        Propose a torsion using OpenMM

        Parameters
        ----------
        growth_context : openmm.Context
            Context containing the modified system and
        torsion : parmed.Dihedral
            parmed Dihedral containing relevant atoms
        positions : [n,3] np.ndarray in nm
            positions of the atoms in the system
        r : float in nm
            bond length
        theta : float in radians
            bond angle
        beta : float
            inverse temperature
        n_divisions : int, optional
            number of divisions for the torsion scan. default 5000

        Returns
        -------
        phi : float in radians
            The proposed torsion
        logp : float
            The log probability of the proposal.
        """
        logp_torsions, phis = self._torsion_log_pmf(growth_context, torsion, positions, r, theta, beta, n_divisions=5000)
        phi_idx = np.random.choice(range(len(phis)), p=np.exp(logp_torsions))
        logp = logp_torsions[phi_idx]
        phi = phis[phi_idx]
        return phi, logp

    def _torsion_logp(self, growth_context, torsion, positions, phi, beta, n_divisions=5000):
        """
        Calculate the logp of a torsion using OpenMM

        Parameters
        ----------
        growth_context : openmm.Context
            Context containing the modified system and
        torsion : parmed.Dihedral
            parmed Dihedral containing relevant atoms
        positions : [n,3] np.ndarray in nm
            positions of the atoms in the system
        phi : float, in radians
            The torsion angle
        beta : float
            inverse temperature
        n_divisions : int, optional
            number of divisions for logp calculation. default 5000.

        Returns
        -------
        torsion_logp : float
            the logp of this torsion
        """
        logp_torsions, phis = self._torsion_log_pmf(growth_context, torsion, positions, r, theta, beta, n_divisions=5000)
        phi_idx, phi = min(enumerate(phis), key=lambda x: abs(x[1]-phi))
        torsion_logp = logp_torsions[phi_idx]
        return torsion_logp

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
        internal_coordinates, _ = self._cartesian_to_internal(xyz, positions[bond_atom.idx], positions[angle_atom.idx], positions[torsion_atom.idx])
        r = internal_coordinates[0]*units.nanometers
        theta = internal_coordinates[1]*units.radians
        phi = internal_coordinates[2]*units.radians
        logp, Z, q, phis = self._normalize_torsion_proposal(atom, r, theta, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=50)
        #find the phi that's closest to the internal_coordinate phi:
        phi_idx, phi = min(enumerate(phis), key=lambda x: abs(x[1]-phi))
        return logp[phi_idx]

    def _propose_torsion(self, atom, r, theta, bond_atom, angle_atom, torsion_atom, torsion, atoms_with_positions, positions, beta):
        """
        Propose a torsion angle, including energetic contributions from other torsions and angles
        """
        #first, let's get the normalizing constant of this distribution
        logp, Z, q, phis= self._normalize_torsion_proposal(atom, r, theta, bond_atom, angle_atom, torsion_atom, atoms_with_positions, positions, beta, n_divisions=50)
        #choose from the set of possible torsion angles
        phi_idx = np.random.choice(range(len(phis)), p=np.exp(logp))
        logp = logp[phi_idx]
        phi = phis[phi_idx]
        return phi, logp


class GeometrySystemGenerator(object):
    """
    This is an internal utility class that generates OpenMM systems
    with only valence terms and special parameters to assist in
    geometry proposals.
    """
    _HarmonicBondForceEnergy = "step(growth_idx - {})*(K/2)*(r-r0)^2"
    _HarmonicAngleForceEnergy = "step(growth_idx - {})*(K/2)*(theta-theta0)^2;"
    _PeriodicTorsionForceEnergy = "step(growth_idx - {})*k*(1+cos(periodicity*theta-phase))"

    def __init__(self):
        pass

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

class ProposalOrderTools(object):
    """
    This is an internal utility class for determining the order of atomic position proposals.
    It encapsulates funcionality needed by the geometry engine.

    Parameters
    ----------
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        The topology proposal containing the relevant move.
    """

    def __init__(self, topology_proposal):
        self._topology_proposal = topology_proposal

    def determine_proposal_order(self, direction='forward'):
        """
        Determine the proposal order of this system pair.
        This includes the choice of a torsion. As such, a logp is returned.

        Parameters
        ----------
        direction : str, optional
            whether to determine the forward or reverse proposal order

        Returns
        -------
        atoms_torsions : dict
            parmed.Atom : parmed.Dihedral
        logp_torsion_choice : float
            log probability of the chosen torsions
        """
        logp_torsion_choice = 0.0
        atoms_torsions = {}
        if direction=='forward':
            structure = parmed.openmm.load_topology(self._topology_proposal.new_topology, self._topology_proposal.new_system)
            new_atoms = [structure.atoms[idx] for idx in self._topology_proposal.unique_new_atoms]
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(self._topology_proposal.n_atoms_new) if atom_idx not in self._topology_proposal.unique_new_atoms]
        elif direction=='reverse':
            structure = parmed.openmm.load_topology(self._topology_proposal.old_topology, self._topology_proposal.old_system)
            new_atoms = [structure.atoms[idx] for idx in self._topology_proposal.unique_old_atoms]
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in self._topology_proposal.old_to_new_atom_map.keys()]
        else:
            raise ValueError("direction parameter must be either forward or reverse.")

        while(len(new_atoms))>0:
            eligible_atoms = self._atoms_eligible_for_proposal(new_atoms, atoms_with_positions)
            for atom in eligible_atoms:
                chosen_torsion, logp_choice = self._choose_torsion(atoms_with_positions, atom)
                atoms_torsions[atom] = chosen_torsion
                logp_torsion_choice += logp_choice
                new_atoms.remove(atom)
                atoms_with_positions.append(atom)
        return atoms_torsions, logp_choice



    def _atoms_eligible_for_proposal(self, new_atoms, atoms_with_positions):
        """
        Get the set of atoms currently eligible for proposal

        Parameters
        ----------
        new_atoms : list of parmed.Atom
            the new atoms that need positions
        atoms_with_positions : list of parmed.Atom
            the atoms with positions
        """
        eligible_atoms = []
        for atom in new_atoms:
            #get array of booleans to see if a bond partner has a position
            has_bonded_position = [a in atoms_with_positions for a in atom.bond_partners]
            #if at least one does, then the atom is ready to be proposed.
            if np.sum(has_bonded_position) > 0:
                eligible_atoms.append(atom)
        return eligible_atoms

    def _choose_torsion(self, atoms_with_positions, atom_for_proposal):
        """
        Get a torsion from the set of possible topological torsions.

        Parameters
        ----------
        atoms_with_positions : list of parmed.Atom
            list of the atoms that already have positions
        atom_for_proposal : parmed.Atom
            atom that is being proposed now

        Returns
        -------
        torsion_selected, logp_torsion_choice : parmed.Dihedral, float
            The torsion that was selected, along with the logp of the choice.

        """
        eligible_torsions = self._get_topological_torsions(atoms_with_positions, atom_for_proposal)
        if len(eligible_torsions) == 0:
            raise Exception("No eligible torsions found for placing atom %s." % str(atom_for_proposal))
        torsion_idx = np.random.randint(0, len(eligible_torsions))
        torsion_selected = eligible_torsions[torsion_idx]
        return torsion_selected, np.log(1.0/len(eligible_torsions))

    def _get_topological_torsions(self, atoms_with_positions, new_atom):
        """
        Get the topological torsions involving new_atom. This includes
        torsions which don't have any parameters assigned to them.

        Parameters
        ----------
        atoms_with_positions : list
            list of atoms with a valid position
        new_atom : parmed.Atom object
            Atom object for the new atom
        Returns
        -------
        torsions : list of parmed.Dihedral objects with no "type"
            list of topological torsions including only atoms with positions
        """
        topological_torsions = []
        angles = new_atom.angles
        atoms_with_positions = set(atoms_with_positions)
        for angle in angles:
            if angle.atom1 is new_atom:
                if angle.atom2 in atoms_with_positions and angle.atom3 in atoms_with_positions:
                    bonds_to_angle = angle.atom3.bonds
                    for bond in bonds_to_angle:
                        bonded_atoms = {bond.atom1, bond.atom2}
                        if angle.atom2 in bonded_atoms:
                            continue
                        if bonded_atoms.issubset(atoms_with_positions):
                            bond_atom = angle.atom2
                            angle_atom = angle.atom3
                            torsion_atom = bond.atom1 if bond.atom2==angle.atom3 else bond.atom2
                            dihedral = parmed.Dihedral(new_atom, bond_atom, angle_atom, torsion_atom)
                            topological_torsions.append(dihedral)
            elif angle.atom3 is new_atom:
                if angle.atom1 in atoms_with_positions and angle.atom2 in atoms_with_positions:
                    bonds_to_angle = angle.atom1.bonds
                    for bond in bonds_to_angle:
                        bonded_atoms = {bond.atom1, bond.atom2}
                        if angle.atom2 in bonded_atoms:
                            continue
                        if bonded_atoms.issubset(atoms_with_positions):
                            bond_atom = angle.atom2
                            angle_atom = angle.atom1
                            torsion_atom = bond.atom1 if bond.atom2==angle.atom1 else bond.atom2
                            dihedral = parmed.Dihedral(new_atom, bond_atom, angle_atom, torsion_atom)
                            topological_torsions.append(dihedral)
        return topological_torsions

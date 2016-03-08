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
import collections


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

class FFAllAngleGeometryEngine(GeometryEngine):
    """
    This is an implementation of GeometryEngine which uses all valence terms and OpenMM
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
        current_positions = current_positions.in_units_of(units.nanometers)
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
        new_coordinates = new_coordinates.in_units_of(units.nanometers)
        old_coordinates = old_coordinates.in_units_of(units.nanometers)
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

            #find and copy known positions
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.new_to_old_atom_map.keys()]
            new_positions = self._copy_positions(atoms_with_positions, top_proposal, old_positions)

            growth_system = growth_system_generator.create_modified_system(top_proposal.new_system, atom_proposal_order.keys(), growth_parameter_name)
        elif direction=='reverse':
            if new_positions is None:
                raise ValueError("For reverse proposals, new_positions must not be none.")
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='reverse')
            structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
            growth_system = growth_system_generator.create_modified_system(top_proposal.old_system, atom_proposal_order.keys(), growth_parameter_name)
        else:
            raise ValueError("Parameter 'direction' must be forward or reverse")

        logp_proposal = logp_choice

        platform = openmm.Platform.getPlatformByName('Reference')
        integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        context = openmm.Context(growth_system, integrator, platform)
        growth_parameter_value = 1
        #now for the main loop:
        for atom, torsion in atom_proposal_order.items():

            context.setParameter(growth_parameter_name, growth_parameter_value)
            bond_atom = torsion.atom2
            angle_atom = torsion.atom3
            torsion_atom = torsion.atom4

            if atom != torsion.atom1:
                raise Exception('atom != torsion.atom1')

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
                logZ_r = np.log((np.sqrt(2*np.pi)*(sigma_r/units.angstroms))) # CHECK DOMAIN AND UNITS
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
            logZ_theta = np.log((np.sqrt(2*np.pi)*(sigma_theta/units.radians))) # CHECK DOMAIN AND UNITS
            logp_theta = self._angle_logq(theta, angle, beta) - logZ_theta

            #propose a torsion angle and calcualate its probability
            if direction=='forward':
                phi, logp_phi = self._propose_torsion(context, torsion, new_positions, r, theta, beta, n_divisions=360)
                xyz, detJ = self._internal_to_cartesian(new_positions[bond_atom.idx], new_positions[angle_atom.idx], new_positions[torsion_atom.idx], r, theta, phi)
                new_positions[atom.idx] = xyz
            else:
                old_positions_for_torsion = copy.deepcopy(old_positions)
                logp_phi = self._torsion_logp(context, torsion, old_positions_for_torsion, r, theta, phi, beta, n_divisions=360)

            #accumulate logp
            if direction == 'reverse':
                print('%8d logp_r %12.3f | logp_theta %12.3f | logp_phi %12.3f | log(detJ) %12.3f' % (atom.idx, logp_r, logp_theta, logp_phi, np.log(detJ)))
            logp_proposal += logp_r + logp_theta + logp_phi + np.log(detJ)
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
        # Workaround for CustomAngleForce NaNs: Create random non-zero positions for new atoms.
        new_positions = units.Quantity(np.random.random([top_proposal.n_atoms_new, 3]), unit=units.nanometers)

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

    def _get_internal_from_omm(self, atom_coords, bond_coords, angle_coords, torsion_coords):
        import copy
        #master system, will be used for all three
        sys = openmm.System()
        platform = openmm.Platform.getPlatformByName("Reference")
        for i in range(4):
            sys.addParticle(1.0*units.amu)

        #first, the bond length:
        bond_sys = openmm.System()
        bond_sys.addParticle(1.0*units.amu)
        bond_sys.addParticle(1.0*units.amu)
        bond_force = openmm.CustomBondForce("r")
        bond_force.addBond(0, 1, [])
        bond_sys.addForce(bond_force)
        bond_integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        bond_context = openmm.Context(bond_sys, bond_integrator, platform)
        bond_context.setPositions([atom_coords, bond_coords])
        bond_state = bond_context.getState(getEnergy=True)
        r = bond_state.getPotentialEnergy()
        del bond_sys, bond_context, bond_integrator

        #now, the angle:
        angle_sys = copy.deepcopy(sys)
        angle_force = openmm.CustomAngleForce("theta")
        angle_force.addAngle(0,1,2,[])
        angle_sys.addForce(angle_force)
        angle_integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        angle_context = openmm.Context(angle_sys, angle_integrator, platform)
        angle_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
        angle_state = angle_context.getState(getEnergy=True)
        theta = angle_state.getPotentialEnergy()
        del angle_sys, angle_context, angle_integrator

        #finally, the torsion:
        torsion_sys = copy.deepcopy(sys)
        torsion_force = openmm.CustomTorsionForce("theta")
        torsion_force.addTorsion(0,1,2,3,[])
        torsion_sys.addForce(torsion_force)
        torsion_integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        torsion_context = openmm.Context(torsion_sys, torsion_integrator, platform)
        torsion_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
        torsion_state = torsion_context.getState(getEnergy=True)
        phi = torsion_state.getPotentialEnergy()
        del torsion_sys, torsion_context, torsion_integrator

        return r, theta, phi

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
        r = sigma_r*np.random.randn() + r0
        return r

    def _propose_angle(self, angle, beta):
        """
        Bond angle proposal
        """
        theta0 = angle.type.theteq
        k = angle.type.k
        sigma_theta = units.sqrt(1.0/(beta*k))
        theta = sigma_theta*np.random.randn() + theta0
        return theta

    def _torsion_scan(self, torsion, positions, r, theta, n_divisions=360):
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
        phis = units.Quantity(np.arange(-np.pi, +np.pi, (2.0*np.pi)/n_divisions), unit=units.radians) # changed to [-pi,+pi) to make it easier to compare with openmm-derived torsions]
        xyzs = units.Quantity(np.zeros([len(phis), 3]), unit=units.nanometers)
        for i, phi in enumerate(phis):
            xyzs[i], _ = self._internal_to_cartesian(positions_copy[bond_atom.idx], positions_copy[angle_atom.idx], positions_copy[torsion_atom.idx], r, theta, phi)
        return xyzs, phis

    def _torsion_log_pmf(self, growth_context, torsion, positions, r, theta, beta, n_divisions=360):
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
            positions[atom_idx,:] = xyz
            growth_context.setPositions(positions)
            state = growth_context.getState(getEnergy=True)
            logq_i = -beta*state.getPotentialEnergy()
            logq[i] = logq_i

        if np.sum(np.isnan(logq)) == n_divisions:
            raise Exception("All %d torsion energies in torsion PMF are NaN." % n_divisions)
        logq[np.isnan(logq)] = -np.inf
        logq -= max(logq)
        q = np.exp(logq)
        Z = np.sum(q)
        logp_torsions = logq - np.log(Z)
        return logp_torsions, phis

    def _propose_torsion(self, growth_context, torsion, positions, r, theta, beta, n_divisions=360):
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
            number of divisions for the torsion scan. default 360

        Returns
        -------
        phi : float in radians
            The proposed torsion
        logp : float
            The log probability of the proposal.
        """
        logp_torsions, phis = self._torsion_log_pmf(growth_context, torsion, positions, r, theta, beta, n_divisions=n_divisions)
        phi_idx = np.random.choice(range(len(phis)), p=np.exp(logp_torsions))
        logp = logp_torsions[phi_idx] - np.log(2*np.pi / n_divisions) # convert from probability mass function to probability density function so that sum(dphi*p) = 1, with dphi = (2*pi)/n_divisions.
        phi = phis[phi_idx]
        return phi, logp

    def _torsion_logp(self, growth_context, torsion, positions, r, theta, phi, beta, n_divisions=360):
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
        r : float in nm
            Bond length
        theta : float in radians
            Bond angle
        phi : float, in radians
            The torsion angle
        beta : float
            inverse temperature
        n_divisions : int, optional
            number of divisions for logp calculation. default 360.

        Returns
        -------
        torsion_logp : float
            the logp of this torsion
        """
        logp_torsions, phis = self._torsion_log_pmf(growth_context, torsion, positions, r, theta, beta, n_divisions=n_divisions)
        phi_idx = np.argmin(np.abs(phi-phis)) # WARNING: This assumes both phi and phis have domain of [-pi,+pi)
        torsion_logp = logp_torsions[phi_idx] - np.log(2*np.pi / n_divisions) # convert from probability mass function to probability density function so that sum(dphi*p) = 1, with dphi = (2*pi)/n_divisions.
        return torsion_logp

class GeometrySystemGenerator(object):
    """
    This is an internal utility class that generates OpenMM systems
    with only valence terms and special parameters to assist in
    geometry proposals.
    """
    _HarmonicBondForceEnergy = "select(step({} - growth_idx), (K/2)*(r-r0)^2, 0);"
    _HarmonicAngleForceEnergy = "select(step({} - growth_idx), (K/2)*(theta-theta0)^2, 0);"
    _PeriodicTorsionForceEnergy = "select(step({} - growth_idx), k*(1+cos(periodicity*theta-phase)), 0);"

    def __init__(self):
        pass

    def create_modified_system(self, reference_system, growth_indices, parameter_name, add_extra_torsions=True, force_names=None, force_parameters=None):
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
        add_extra_torsions : bool, optional
            Whether to add additional torsions to keep rings flat. Default true.
        force_names : list of str
            A list of the names of forces that will be included in this system
        force_parameters : dict
            Options for the forces (e.g., NonbondedMethod : 'CutffNonPeriodic')
        Returns
        -------
        growth_system : simtk.openmm.System object
            System with the appropriate modifications
        """
        if add_extra_torsions:
            pass
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
        modified_torsion_force.addGlobalParameter(parameter_name, 0)

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

    def _determine_extra_torsions(self, torsion_force, reference_topology, growth_indices):
        """
        Determine which atoms need an extra torsion. First figure out which residue is
        covered by the new atoms, then determine the rotatable bonds. Finally, construct
        the residue in omega and measure the appropriate torsions, and generate relevant parameters.
        ONLY ONE RESIDUE SHOULD BE CHANGING!

        Parameters
        ----------
        torsion_force : openmm.CustomTorsionForce object
            the new/old torsion force if forward/backward
        reference_topology : openmm.app.Topology object
            the new/old topology if forward/backward
        growth_indices : list of int
            The list of new atoms and the order in which they will be added.

        Returns
        -------
        torsion_force : openmm.CustomTorsionForce
            The torsion force with extra torsions added appropriately.
        """
        import simtk.openmm.app as app
        import openmoltools.forcefield_generators as forcefield_generators
        reference_topology = app.Topology()
        atoms = list(reference_topology.atoms())

        #get residue from first atom
        residue = atoms[growth_indices[0]].residue
        try:
            oemol = forcefield_generators.generateOEMolFromTopologyResidue(residue)
        except Exception as e:
            print("Could not generate an oemol from the residue.")
            print(e)

        #get the omega geometry of the molecule:
        import openeye.oeomega as oeomega
        import openeye.oechem as oechem
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(1)
        omega(oemol)

        #get the list of torsions in the molecule that are not about a rotatable bond
        relevant_torsion_list = list(oechem.OEGetTorsions(oemol, oechem.OEIsRotor(False)))


        #now, for each torsion, extract the set of indices and






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
        growth_indices_list = [atom.idx for atom in list(growth_indices)]
        particle_indices_set = set(particle_indices)
        growth_indices_set = set(growth_indices_list)
        new_atoms_in_force = particle_indices_set.intersection(growth_indices_set)
        if len(new_atoms_in_force) == 0:
            return 0
        new_atom_growth_order = [growth_indices_list.index(atom_idx)+1 for atom_idx in new_atoms_in_force]
        return max(new_atom_growth_order)

class ProposalOrderTools(object):
    """
    This is an internal utility class for determining the order of atomic position proposals.
    It encapsulates funcionality needed by the geometry engine.

    Parameters
    ----------
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        The topology proposal containing the relevant move.
    add_extra_torsions : bool, optional
        Whether to add additional torsions to keep rings flat. Default true.
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
        atoms_torsions = collections.OrderedDict()
        if direction=='forward':
            structure = parmed.openmm.load_topology(self._topology_proposal.new_topology, self._topology_proposal.new_system)
            new_atoms = [structure.atoms[idx] for idx in self._topology_proposal.unique_new_atoms]
            #atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(self._topology_proposal.n_atoms_new) if atom_idx not in self._topology_proposal.unique_new_atoms]
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in self._topology_proposal.new_to_old_atom_map.keys()]
        elif direction=='reverse':
            structure = parmed.openmm.load_topology(self._topology_proposal.old_topology, self._topology_proposal.old_system)
            new_atoms = [structure.atoms[idx] for idx in self._topology_proposal.unique_old_atoms]
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in self._topology_proposal.old_to_new_atom_map.keys()]
        else:
            raise ValueError("direction parameter must be either forward or reverse.")


        logp_choice = 0
        while(len(new_atoms))>0:
            eligible_atoms = self._atoms_eligible_for_proposal(new_atoms, atoms_with_positions)
            if (len(new_atoms) > 0) and (len(eligible_atoms) == 0):
                raise Exception('new_atoms (%s) has remaining atoms to place, but eligible_atoms is empty.' % str(new_atoms))
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

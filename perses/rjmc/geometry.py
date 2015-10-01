"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
from collections import namedtuple
import numpy as np
import scipy.stats as stats
import numexpr as ne

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

    def propose(self, new_to_old_atom_map, new_system, old_system, old_positions):
        """
        Make a geometry proposal for the appropriate atoms.
        
        Arguments
        ----------
        new_to_old_atom_map : dict
            mapping of the new to old atoms
        new_system : simtk.openmm.System
            The new system object
        old_system : simtk.openmm.System
            The old system object
        old_positions : [m, 3] np.array of floats
            The positions of the old m atoms

        Returns
        -------
        proposal : GeometryProposal namedtuple
             Contains the new positions as well as the logp ratio
             of the proposal.
        """
        return GeometryProposal(np.array([0.0,0.0,0.0]), 0)


class FFGeometryEngine(GeometryEngine):
    """
    This is an implementation of the GeometryEngine class which proposes new dimensions based on the forcefield parameters
    """


    def propose(self, new_to_old_atom_map, new_system, old_system, old_positions):
        """
        Propose a new geometry for the appropriate atoms using forcefield parameters

        Arguments
        ----------
        new_to_old_atom_map : dict
            mapping of the new to old atoms
        new_system : simtk.openmm.System
            The new system object
        old_system : simtk.openmm.System
            The old system object
        old_positions : [m, 3] np.array of floats
            The positions of the old m atoms

        Returns
        -------
        proposal : GeometryProposal namedtuple
             Contains the new positions as well as the logp ratio
             of the proposal.
        """

        #get the mapping between new and old atoms
        n_atoms_new = new_system.getNumParticles()
        n_atoms_old = old_system.getNumParticles()

        #get a list of the new atoms
        new_atoms = [atom for atom in range(n_atoms_new) if atom not in new_to_old_atom_map.keys()]

        #get a list of the old atoms
        old_atoms = [atom for atom in range(n_atoms_old) if atom not in new_to_old_atom_map.values()]

        #create a new array with the appropriate size
        new_positions = np.zeros([n_atoms_new, 3])

        #transfer known positions
        for atom in new_to_old_atom_map.keys():
            new_positions[atom] = old_positions[new_to_old_atom_map[atom]]

        #get a new set of positions and a logp of that proposal
        final_new_positions, logp_forward = self._propose_new_positions(new_atoms, new_system, new_positions, new_to_old_atom_map)

        #get the probability of a reverse proposal
        logp_reverse = self._reverse_proposal_logp(old_atoms, old_system, old_positions, new_to_old_atom_map)

        #construct the return object
        geometry_proposal = GeometryProposal(final_new_positions, logp_reverse - logp_forward)

        return geometry_proposal

    def _propose_new_positions(self, new_atoms, new_system, new_positions, new_to_old_atom_map):
        """
        Propose new atomic positions

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
            The positions of the m atoms in the new system
        logp_forward : float
            The logp of the forward proposal, including the jacobian
        """
        logp_forward = 0.0
        atoms_with_positions = new_to_old_atom_map.keys()
        forces = {new_system.getForce(index).__class__.__name__ : new_system.getForce(index) for index in range(new_system.getNumForces())}
        torsion_force = forces['PeriodicTorsionForce']
        bond_force = forces['HarmonicBondForce']
        atom_torsion_proposals = {atom_index: [] for atom_index in new_atoms}
        while len(new_atoms) > 0:
            #find atoms to propose
            current_atom_proposals = []
            atom_torsion_proposals = self._atoms_eligible_for_proposal(torsion_force, bond_force, atoms_with_positions, new_atoms)
            #now loop through the list of atoms found to be eligible for positions this round
            for atom, possible_torsions in atom_torsion_proposals.items():
                #randomly select a torsion from the available list, and calculate its log-probability
                selected_torsion = np.random.randint(0, len(possible_torsions))
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

                #get bond parameters and draw bond length
                r0, k_eq = self._get_bond_parameters(new_system, atom, bonded_atom)
                r, logp_bond = self._propose_bond_length(r0, k_eq)

                #get angle parameters and draw bond angle
                theta0, k_eq = self._get_angle_parameters(new_system, atom, bonded_atom, angle_atom)
                theta, logp_angle = self._propose_bond_angle(theta0, k_eq)

                #propose torsion
                phi, logp_torsion = self._propose_torsion_angle(torsion_parameters[6], torsion_parameters[4], torsion_parameters[5])

                #convert spherical to cartesian coordinates
                spherical_coordinates = np.array([r, theta, phi])
                atomic_xyz, detJ = self._spherical_to_cartesian(spherical_coordinates)

                #accumulate the forward logp with jacobian correction
                logp_forward += (logp_selected_torsion + logp_angle + logp_bond + logp_torsion + np.log(detJ))

                #add new position to atom relative to bonded atom
                new_positions[atom] = atomic_xyz + new_positions[bonded_atom]
                atoms_with_positions.append(atom)

                #remove atom from list of atoms needing positions
                new_atoms.remove(atom)

        return new_positions, logp_forward

    def _propose_atomic_position(self, system, atom, bonded_atom, angle_atom, torsion_parameters):
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

        """
        #get bond parameters and draw bond length
        r0, k_eq = self._get_bond_parameters(system, atom, bonded_atom)
        r, logp_bond = self._propose_bond_length(r0, k_eq)

        #get angle parameters and draw bond angle
        theta0, k_eq = self._get_angle_parameters(system, atom, bonded_atom, angle_atom)
        theta, logp_angle = self._propose_bond_angle(theta0, k_eq)

        #propose torsion
        phi, logp_torsion = self._propose_torsion_angle(torsion_parameters[6], torsion_parameters[4], torsion_parameters[5])

        #convert spherical to cartesian coordinates
        spherical_coordinates = np.array([r, theta, phi])
        atomic_xyz, detJ = self._spherical_to_cartesian(spherical_coordinates)

        #accumulate the forward logp with jacobian correction
        logp_atomic_position = (logp_angle + logp_bond + logp_torsion + np.log(detJ))

        return atomic_xyz, logp_atomic_position


    def _reverse_proposal_logp(self, old_atoms, old_system, old_positions, new_to_old_atom_map):
        """
        Calculate the log-probability of the proposal of the unique old atoms (reverse proposal).

        Arguments
        ---------
        old_atoms : list of int
            List of indices of the old unique atoms
        old_system : simtk.openmm.System
            Old system object
        old_positions : [m, 3] np.array of floats
            Positions of the old system

        Returns
        -------
        logp_reverse : float
            Log-propbability of the reverse transformation
        """
        logp_reverse = 0.0
        atoms_with_positions = new_to_old_atom_map.values()
        forces = {old_system.getForce(index).__class__.__name__ : old_system.getForce(index) for index in range(old_system.getNumForces())}
        torsion_force = forces['PeriodicTorsionForce']
        bond_force = forces['HarmonicBondForce']
        atom_torsion_proposals = {atom_index: [] for atom_index in old_atoms}
        while len(old_atoms) > 0:
            #find atoms to propose
            current_atom_proposals = []
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
                        if atom1 in atoms_with_positions:
                            atom_torsion_proposals[atom1].append(torsion_index)
                            current_atom_proposals.append(atom1)
                        else:
                            atom_torsion_proposals[atom4].append(torsion_index)
                            current_atom_proposals.append(atom4)
            #now loop through the list of atoms found to be eligible for positions this round
            for atom in current_atom_proposals:
                possible_torsions = atom_torsion_proposals[atom]
                #randomly select a torsion from the available list, and calculate its log-probability
                selected_torsion = np.random.randint(0, len(possible_torsions))
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

                #convert cartesian coordinates to spherical
                relative_xyz = old_positions[atom] - old_positions[bonded_atom]
                spherical_coordinates, detJ = self._cartesian_to_spherical(relative_xyz)

                #get bond parameters and calculate the probability of choosing the current positions
                r0, k_eq = self._get_bond_parameters(old_system, atom, bonded_atom)
                sigma_bond = 1.0/np.sqrt(2.0*k_eq)
                logp_bond = stats.distributions.norm.logpdf(spherical_coordinates[0], r0, sigma_bond)

                #calculate the probability of choosing this bond angle
                theta0, k_eq = self._get_angle_parameters(old_system, atom, bonded_atom, angle_atom)
                sigma_angle = 1.0/np.sqrt(2.0*k_eq)
                logp_angle = stats.distributions.norm.logpdf(spherical_coordinates[1], theta0, sigma_angle)

                #calculate the probabilty of choosing this torsion angle
                torsion_Z = self._torsion_normalizer(torsion_parameters[6], torsion_parameters[4], torsion_parameters[5])
                logp_torsion = self._torsion_p(torsion_Z,torsion_parameters[6], torsion_parameters[4], torsion_parameters[5], spherical_coordinates[2])

                #accumulate the reverse logp with jacobian correction
                logp_reverse += (logp_selected_torsion + logp_angle + logp_bond + logp_torsion + np.log(detJ))

                #add new position to atom relative to bonded atom

                atoms_with_positions.append(atom)

                #remove atom from list of atoms needing positions
                old_atoms.remove(atom)
        return logp_reverse


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
        return {atom: torsionlist for atom, torsionlist in atom_torsion_proposals if len(torsionlist) > 0}


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
            if (parameters[0] == atom1 and parameters[1] == atom2) or (parameters[1] == atom1 and parameters[0] == atom1):
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
        list_of_angle_atoms.sort()
        forces = {system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces())}
        angle_force = forces['HarmonicAngleForce']
        for angle_index in range(angle_force.getNumAngles()):
            parameters = angle_force.getAngleParameters(angle_index)
            #the first three "parameters" are atom indices
            atoms = parameters[:3]
            if atoms.sort() == list_of_angle_atoms:
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
        r = sigma*np.random.random() + r0
        logp = stats.distributions.norm.logpdf(r, r0, sigma)
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
        theta = sigma*np.random.random() + theta0
        logp = stats.distributions.norm.logpdf(theta, theta0, sigma)
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
        return (phi, logp)

    def _torsion_p(self, Z, V, n, gamma, phi):
        """
        Utility function for calculating the normalized probability of a torsion angle
        """
        return (1.0/Z)*np.exp(-(V/2.0)*(1+np.cos(n*phi-gamma)))

    def _torsion_normalizer(self, V, n, gamma):
        """
        Utility function to numerically normalize torsion angles.
        Also return max_p to facilitate rejection sampling
        """
        #generate a grid of 5000 points from 0 < phi < 2*pi
        phis = np.linspace(0, 2.0*np.pi, 5000)
        #evaluate the unnormalized probability at each of those points
        phi_q = ne.evaluate("exp(-(V/2.0)*(1+cos(n*phis-gamma)))")
        #integrate the values
        Z = np.trapz(phi_q, phis)
        max_p = np.max(phi_q/Z)
        return Z, max_p


    def _spherical_to_cartesian(self, spherical):
        """
        Convert spherical coordinates to cartesian coordinates, and get jacobian

        Arguments
        ---------
        spherical : 1x3 np.array of floats
            r, theta, phi

        Returns
        -------
        [x,y,z] : np.array 1x3
            the transformed cartesian coordinates
        detJ : float
            the determinant of the jacobian of the transformation
        """
        xyz = np.zeros(3)
        xyz[0] = spherical[0]*np.cos(spherical[1])
        xyz[1] = spherical[0]*np.sin(spherical[1])
        xyz[2] = spherical[0]*np.cos(spherical[2])
        detJ = spherical[0]**2*np.sin(spherical[2])
        return xyz, detJ

    def _cartesian_to_spherical(self, xyz):
        """
        Convert cartesian coordinates to spherical coordinates

        Arguments
        ---------
        xyz : 1x3 np.array of floats
            the cartesian coordinates to convert

        Returns
        -------
        spherical : 1x3 np.array of floats
            the spherical coordinates
        detJ : float
            The determinant of the jacobian of the transformation
        """
        spherical = np.zeros(3)
        spherical[0] = np.linalg.norm(xyz)
        spherical[1] = np.arccos(xyz[2]/spherical[0])
        spherical[2] = np.arctan(xyz[1]/xyz[0])
        detJ = spherical[0]**2*np.sin(spherical[2])
        return spherical, detJ

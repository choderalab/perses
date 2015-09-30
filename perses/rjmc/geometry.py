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

    def propose(self, topology_proposal, new_system, old_positions):
        """
        Make a geometry proposal for the appropriate atoms.
        
        Arguments
        ----------
        topology_proposal : TopologyProposal namedtuple
            The result of the topology proposal, containing the atom mapping and topologies.
        sampler_state : SamplerState namedtuple
        The current state of the sampler

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
        atom_torsion_proposals = {atom_index  : [] for atom_index in new_atoms}
        while len(new_atoms) > 0:
            #find atoms to propose
            current_atom_proposals = []
            for torsion_index in range(torsion_force.getNumTorsions()):
                atom1 = torsion_force.getTorsionParameters(torsion_index)[0]
                atom2 = torsion_force.getTorsionParameters(torsion_index)[1]
                atom3 = torsion_force.getTorsionParameters(torsion_index)[2]
                atom4 = torsion_force.getTorsionParameters(torsion_index)[3]
                #Only take torsions where the "new" statuses of atoms 1 and 4 are not equal
                if (atom1 in atoms_with_positions) != (atom4 in atoms_with_positions):
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
                    bonded_atom = torsion_parameters[3]
                    angle_atom = torsion_parameters[2]
                    torsion_atom = torsion_parameters[1]
                #get bond parameters and draw bond length
                r0, k_eq = self._get_bond_parameters(new_system, atom, bonded_atom)
                r, logp_bond = self._propose_bond_length(r0, k_eq)
                #get angle parameters and draw bond angle





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
        sigma = 1.0/np.sqrt(2.0*k_eq)
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
        sigma = 1.0/np.sqrt(2.0*k_eq)
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

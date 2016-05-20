"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
import parmed
import simtk.unit as units
import logging
import numpy as np
import copy
from perses.rjmc import coordinate_numba
import simtk.openmm as openmm
import collections
import openeye.oechem as oechem
import openeye.oeomega as oeomega
import simtk.openmm.app as app
import time

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
        self.write_proposal_pdb = False # if True, will write PDB for sequential atom placements
        self.pdb_filename_prefix = 'geometry-proposal' # PDB file prefix for writing sequential atom placements
        self.nproposed = 0 # number of times self.propose() has been called
        self._energy_time = 0.0
        self._torsion_coordinate_time = 0.0
        self._position_set_time = 0.0
        self.verbose = False

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
        if not top_proposal.unique_new_atoms:
            structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.new_to_old_atom_map.keys()]
            new_positions = self._copy_positions(atoms_with_positions, top_proposal, current_positions)
            return new_positions, 0.0
        logp_proposal, new_positions = self._logp_propose(top_proposal, current_positions, beta, direction='forward')
        self.nproposed += 1
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
        if not top_proposal.unique_old_atoms:
            return 0.0
        new_coordinates = new_coordinates.in_units_of(units.nanometers)
        old_coordinates = old_coordinates.in_units_of(units.nanometers)
        logp_proposal, _ = self._logp_propose(top_proposal, old_coordinates, beta, new_positions=new_coordinates, direction='reverse')
        return logp_proposal

    def _write_partial_pdb(self, pdbfile, topology, positions, atoms_with_positions, model_index):
        """
        Write the subset of the molecule for which positions are defined.

        """
        from simtk.openmm.app import Modeller
        modeller = Modeller(topology, positions)
        atom_indices_with_positions = [ atom.idx for atom in atoms_with_positions ]
        atoms_to_delete = [ atom for atom in modeller.topology.atoms() if (atom.index not in atom_indices_with_positions) ]
        modeller.delete(atoms_to_delete)

        pdbfile.write('MODEL %5d\n' % model_index)
        from simtk.openmm.app import PDBFile
        PDBFile.writeFile(modeller.topology, modeller.positions, file=pdbfile)
        pdbfile.flush()
        pdbfile.write('ENDMDL\n')

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
        initial_time = time.time()
        proposal_order_tool = ProposalOrderTools(top_proposal)
        proposal_order_time = time.time() - initial_time
        growth_system_generator = GeometrySystemGenerator()
        growth_parameter_name = 'growth_stage'
        if direction=="forward":
            forward_init = time.time()
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='forward')
            proposal_order_forward = time.time() - forward_init
            structure = parmed.openmm.load_topology(top_proposal.new_topology, top_proposal.new_system)

            #find and copy known positions
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.new_to_old_atom_map.keys()]
            new_positions = self._copy_positions(atoms_with_positions, top_proposal, old_positions)
            system_init = time.time()
            growth_system = growth_system_generator.create_modified_system(top_proposal.new_system, atom_proposal_order.keys(), growth_parameter_name, reference_topology=top_proposal.new_topology)
            growth_system_time = time.time() - system_init
        elif direction=='reverse':
            if new_positions is None:
                raise ValueError("For reverse proposals, new_positions must not be none.")
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='reverse')
            structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.old_to_new_atom_map.keys()]
            growth_system = growth_system_generator.create_modified_system(top_proposal.old_system, atom_proposal_order.keys(), growth_parameter_name, reference_topology=top_proposal.old_topology)
        else:
            raise ValueError("Parameter 'direction' must be forward or reverse")

        logp_proposal = logp_choice

        if self.write_proposal_pdb:
            # DEBUG: Write growth stages
            from simtk.openmm.app import PDBFile
            prefix = '%s-%d-%s' % (self.pdb_filename_prefix, self.nproposed, direction)
            if direction == 'forward':
                pdbfile = open('%s-initial.pdb' % prefix, 'w')
                PDBFile.writeFile(top_proposal.old_topology, old_positions, file=pdbfile)
                pdbfile.close()
                pdbfile = open("%s-stages.pdb" % prefix, 'w')
                self._write_partial_pdb(pdbfile, top_proposal.new_topology, new_positions, atoms_with_positions, 0)
            else:
                pdbfile = open('%s-initial.pdb' % prefix, 'w')
                PDBFile.writeFile(top_proposal.new_topology, new_positions, file=pdbfile)
                pdbfile.close()
                pdbfile = open("%s-stages.pdb" % prefix, 'w')
                self._write_partial_pdb(pdbfile, top_proposal.old_topology, old_positions, atoms_with_positions, 0)

        platform = openmm.Platform.getPlatformByName('Reference')
        integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        context = openmm.Context(growth_system, integrator, platform)
        debug = False
        if debug:
            context.setPositions(self._metadata['reference_positions'])
            context.setParameter(growth_parameter_name, len(atom_proposal_order.keys()))
            state = context.getState(getEnergy=True)
            print("The potential of the valence terms is %s" % str(state.getPotentialEnergy()))
        growth_parameter_value = 1
        #now for the main loop:
        logging.debug("There are %d new atoms" % len(atom_proposal_order.items()))
        for atom, torsion in atom_proposal_order.items():
            context.setParameter(growth_parameter_name, growth_parameter_value)
            bond_atom = torsion.atom2
            angle_atom = torsion.atom3
            torsion_atom = torsion.atom4
            print("Proposing atom %s from torsion %s" %(str(atom), str(torsion)))

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
                if self.verbose: print('%8d logp_r %12.3f | logp_theta %12.3f | logp_phi %12.3f | log(detJ) %12.3f' % (atom.idx, logp_r, logp_theta, logp_phi, np.log(detJ)))
            logp_proposal += logp_r + logp_theta + logp_phi + np.log(detJ)
            growth_parameter_value += 1

            # DEBUG: Write PDB file for placed atoms
            atoms_with_positions.append(atom)
            if self.write_proposal_pdb:
                if direction=='forward':
                    self._write_partial_pdb(pdbfile, top_proposal.new_topology, new_positions, atoms_with_positions, growth_parameter_value)
                else:
                    self._write_partial_pdb(pdbfile, top_proposal.old_topology, old_positions, atoms_with_positions, growth_parameter_value)

        if self.write_proposal_pdb:
            pdbfile.close()

            prefix = '%s-%d-%s' % (self.pdb_filename_prefix, self.nproposed, direction)
            if direction == 'forward':
                pdbfile = open('%s-final.pdb' % prefix, 'w')
                PDBFile.writeFile(top_proposal.new_topology, new_positions, file=pdbfile)
                pdbfile.close()
        total_time = time.time() - initial_time
        if direction=='forward':
            logging.log(logging.DEBUG, "Proposal order time: %f s | Growth system generation: %f s | Total torsion scan time %f s | Total energy computation time %f s | Position set time %f s| Total time %f s" % (proposal_order_time, growth_system_time , self._torsion_coordinate_time, self._energy_time, self._position_set_time, total_time))
        self._torsion_coordinate_time = 0.0
        self._energy_time = 0.0
        self._position_set_time = 0.0
        return logp_proposal, new_positions

    @staticmethod
    def _oemol_from_residue(res, verbose=False):
        """
        Get an OEMol from a residue, even if that residue
        is polymeric. In the latter case, external bonds
        are replaced by hydrogens.

        Parameters
        ----------
        res : app.Residue
            The residue in question
        verbose : bool, optional, default=False
            If True, will print verbose output.

        Returns
        -------
        oemol : openeye.oechem.OEMol
            an oemol representation of the residue with topology indices
        """
        from openmoltools.forcefield_generators import generateOEMolFromTopologyResidue
        external_bonds = list(res.external_bonds())
        for bond in external_bonds:
            if verbose: print(bond)
        new_atoms = {}
        highest_index = 0
        if external_bonds:
            new_topology = app.Topology()
            new_chain = new_topology.addChain(0)
            new_res = new_topology.addResidue("new_res", new_chain)
            for atom in res.atoms():
                new_atom = new_topology.addAtom(atom.name, atom.element, new_res, atom.id)
                new_atom.index = atom.index
                new_atoms[atom] = new_atom
                highest_index = max(highest_index, atom.index)
            for bond in res.internal_bonds():
                new_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]])
            for bond in res.external_bonds():
                internal_atom = [atom for atom in bond if atom.residue==res][0]
                if verbose:
                    print('internal atom')
                    print(internal_atom)
                highest_index += 1
                if internal_atom.name=='N':
                    if verbose: print('Adding H to N')
                    new_atom = new_topology.addAtom("H2", app.Element.getByAtomicNumber(1), new_res, -1)
                    new_atom.index = -1
                    new_topology.addBond(new_atoms[internal_atom], new_atom)
                if internal_atom.name=='C':
                    if verbose: print('Adding OH to C')
                    new_atom = new_topology.addAtom("O2", app.Element.getByAtomicNumber(8), new_res, -1)
                    new_atom.index = -1
                    new_topology.addBond(new_atoms[internal_atom], new_atom)
                    highest_index += 1
                    new_hydrogen = new_topology.addAtom("HO", app.Element.getByAtomicNumber(1), new_res, -1)
                    new_hydrogen.index = -1
                    new_topology.addBond(new_hydrogen, new_atom)
            res_to_use = new_res
            external_bonds = list(res_to_use.external_bonds())
        else:
            res_to_use = res
        oemol = generateOEMolFromTopologyResidue(res_to_use, geometry=False)
        oechem.OEAddExplicitHydrogens(oemol)
        return oemol

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
        atom_position = atom_position.value_in_unit(units.nanometers).astype(np.float64)
        bond_position = bond_position.value_in_unit(units.nanometers).astype(np.float64)
        angle_position = angle_position.value_in_unit(units.nanometers).astype(np.float64)
        torsion_position = torsion_position.value_in_unit(units.nanometers).astype(np.float64)

        internal_coords = coordinate_numba.cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)


        return internal_coords, internal_coords[0]**2*np.sin(internal_coords[1])

    def _internal_to_cartesian(self, bond_position, angle_position, torsion_position, r, theta, phi):
        """
        Calculate the cartesian coordinates given the internal, as well as abs(detJ)
        """
        r = r.value_in_unit(units.nanometers)
        theta = theta.value_in_unit(units.radians)
        phi = phi.value_in_unit(units.radians)
        bond_position = bond_position.value_in_unit(units.nanometers).astype(np.float64)
        angle_position = angle_position.value_in_unit(units.nanometers).astype(np.float64)
        torsion_position = torsion_position.value_in_unit(units.nanometers).astype(np.float64)
        xyz = coordinate_numba.internal_to_cartesian(bond_position, angle_position, torsion_position, np.array([r, theta, phi], dtype=np.float64))
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
        torsion_scan_init = time.time()
        positions_copy = copy.deepcopy(positions)
        positions_copy = positions_copy.in_units_of(units.nanometers)
        positions_copy = positions_copy / units.nanometers
        positions_copy = positions_copy.astype(np.float64)
        r = r.value_in_unit(units.nanometers)
        theta = theta.value_in_unit(units.radians)
        bond_atom = torsion.atom2
        angle_atom = torsion.atom3
        torsion_atom = torsion.atom4
        phis = np.arange(-np.pi, +np.pi, (2.0*np.pi)/n_divisions) # Can't use units here.
        xyzs = coordinate_numba.torsion_scan(positions_copy[bond_atom.idx], positions_copy[angle_atom.idx], positions_copy[torsion_atom.idx], np.array([r, theta, 0.0]), phis)
        xyzs_quantity = units.Quantity(xyzs, unit=units.nanometers) #have to put the units back now
        phis = units.Quantity(phis, unit=units.radians)
        torsion_scan_time = time.time() - torsion_scan_init
        self._torsion_coordinate_time += torsion_scan_time
        return xyzs_quantity, phis

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
        xyzs = xyzs.value_in_unit_system(units.md_unit_system)
        positions = positions.value_in_unit_system(units.md_unit_system)
        for i, xyz in enumerate(xyzs):
            positions[atom_idx,:] = xyz
            position_set = time.time()
            growth_context.setPositions(positions)
            position_time = time.time() - position_set
            self._position_set_time += position_time
            energy_computation_init = time.time()
            state = growth_context.getState(getEnergy=True)
            energy_computation_time = time.time() - energy_computation_init
            self._energy_time += energy_computation_time
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
        self._plot_torsion(phis, logp_torsions, torsion)
        logp = logp_torsions[phi_idx] - np.log(2*np.pi / n_divisions) # convert from probability mass function to probability density function so that sum(dphi*p) = 1, with dphi = (2*pi)/n_divisions.
        phi = phis[phi_idx]
        return phi, logp

    def _plot_torsion(self, phis, logp_torsions, torsion):
        import matplotlib.pyplot as pyplot
        p_torsion = np.exp(logp_torsions)
        pyplot.plot(phis, p_torsion)

        pyplot.xlabel("torsion phi")
        pyplot.ylabel("P(phi)")
        pyplot.title("Probability of %d-%d-%d-%d" %(torsion.atom1.idx, torsion.atom2.idx, torsion.atom3.idx, torsion.atom4.idx))
        pyplot.grid(False)
        pyplot.savefig("%d-%d-%d-%d.eps" % (torsion.atom1.idx, torsion.atom2.idx, torsion.atom3.idx, torsion.atom4.idx))
        pyplot.close()

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


class OmegaFFGeometryEngine(FFAllAngleGeometryEngine):
    """
    Instead of using the forcefield to propose torsion angles, use Omega geometries as a reference
    """

    def __init__(self, torsion_kappa=8.0, max_confs=1, n_trials=10, strict_stereo=False):
        self._kappa = torsion_kappa
        self._oemols = {}
        self._max_confs = max_confs
        self._omega = oeomega.OEOmega()
        self._omega.SetMaxConfs(max_confs)
        self._omega.SetStrictStereo(strict_stereo)
        self.nproposed = 0
        self._n_trials = n_trials
        self.verbose = False
        self.write_proposal_pdb = False

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
        initial_time = time.time()
        proposal_order_tool = ProposalOrderTools(top_proposal)
        proposal_order_time = time.time() - initial_time
        growth_system_generator = GeometrySystemGenerator()
        growth_parameter_name = "growth_stage"
        if direction=="forward":
            forward_init = time.time()
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='forward')
            proposal_order_forward = time.time() - forward_init
            structure = parmed.openmm.load_topology(top_proposal.new_topology, top_proposal.new_system)
            #find and copy known positions
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.new_to_old_atom_map.keys()]
            new_positions = self._copy_positions(atoms_with_positions, top_proposal, old_positions)
            new_residue_atom_idx = top_proposal.unique_new_atoms[0]
            atoms = list(top_proposal.new_topology.atoms())
            new_residue = atoms[new_residue_atom_idx].residue
            res_mol = self._oemol_from_residue(new_residue)
            oechem.OECanonicalOrderAtoms(res_mol)
            oechem.OETriposAtomNames(res_mol)
            res_smiles = oechem.OEMolToSmiles(res_mol)
            growth_system = growth_system_generator.create_modified_system(top_proposal.new_system, atom_proposal_order.keys(), growth_parameter_name, use_sterics=True, add_extra_torsions=False, reference_topology=top_proposal.new_topology)
        elif direction=='reverse':
            if new_positions is None:
                raise ValueError("For reverse proposals, new_positions must not be none.")
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='reverse')
            structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.old_to_new_atom_map.keys()]
            old_residue_atom_idx = top_proposal.unique_old_atoms[0]
            atoms = list(top_proposal.old_topology.atoms())
            old_residue = atoms[old_residue_atom_idx].residue
            res_mol = self._oemol_from_residue(old_residue)
            oechem.OECanonicalOrderAtoms(res_mol)
            oechem.OETriposAtomNames(res_mol)
            res_smiles = oechem.OEMolToSmiles(res_mol)
            growth_system = growth_system_generator.create_modified_system(top_proposal.old_system, atom_proposal_order.keys(), growth_parameter_name, use_sterics=True, add_extra_torsions=False, reference_topology=top_proposal.old_topology)
        else:
            raise ValueError("Parameter 'direction' must be forward or reverse")

        logp_proposal = logp_choice

        #choose conformation from omega:
        if res_smiles not in self._oemols.keys():
            mol_conf = self._generate_conformations(res_smiles)
            self._oemols[res_smiles] = mol_conf
        else:
            mol_conf = self._oemols[res_smiles]


        #ostream = oechem.oemolostream("Conf1.pdb")
        #oechem.OEWriteMolecule(ostream, oeconf)
        if self.write_proposal_pdb:
            # DEBUG: Write growth stages
            from simtk.openmm.app import PDBFile
            prefix = '%s-%d-%s' % (self.pdb_filename_prefix, self.nproposed, direction)
            if direction == 'forward':
                pdbfile = open('%s-initial.pdb' % prefix, 'w')
                PDBFile.writeFile(top_proposal.old_topology, old_positions, file=pdbfile)
                pdbfile.close()
                pdbfile = open("%s-stages.pdb" % prefix, 'w')
                self._write_partial_pdb(pdbfile, top_proposal.new_topology, new_positions, atoms_with_positions, 0)
            else:
                pdbfile = open('%s-initial.pdb' % prefix, 'w')
                PDBFile.writeFile(top_proposal.new_topology, new_positions, file=pdbfile)
                pdbfile.close()
                pdbfile = open("%s-stages.pdb" % prefix, 'w')
                self._write_partial_pdb(pdbfile, top_proposal.old_topology, old_positions, atoms_with_positions, 0)

        logging.debug("There are %d new atoms" % len(atom_proposal_order.items()))
        growth_parameter_value = 1
        platform = openmm.Platform.getPlatformByName('Reference')
        integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        context = openmm.Context(growth_system, integrator, platform)
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
                positions = copy.deepcopy(new_positions)
                phi, logp_phi = self._propose_mtm_torsion(atom, torsion, res_mol, mol_conf, positions, r, theta, context, beta)
                phi_unit = units.Quantity(phi, unit=units.radian)
                xyz, detJ = self._internal_to_cartesian(new_positions[bond_atom.idx], new_positions[angle_atom.idx], new_positions[torsion_atom.idx], r, theta, phi_unit)
                if detJ <= 0.0:
                    detJ = 0.0
                new_positions[atom.idx] = xyz
            else:
                positions = copy.deepcopy(old_positions)
                _, logp_phi = self._propose_mtm_torsion(atom, torsion, res_mol, mol_conf, positions, r, theta, context, beta, phi=phi)
            #accumulate logp
            if direction == 'reverse':
                if self.verbose: print('%8d logp_r %12.3f | logp_theta %12.3f | logp_phi %12.3f | log(detJ) %12.3f' % (atom.idx, logp_r, logp_theta, logp_phi, np.log(detJ)))
            logp_proposal += logp_r + logp_theta + logp_phi + np.log(detJ)

            # DEBUG: Write PDB file for placed atoms
            atoms_with_positions.append(atom)
            if self.write_proposal_pdb:
                if direction=='forward':
                    self._write_partial_pdb(pdbfile, top_proposal.new_topology, new_positions, atoms_with_positions, growth_parameter_value)
                else:
                    self._write_partial_pdb(pdbfile, top_proposal.old_topology, old_positions, atoms_with_positions, growth_parameter_value)
            growth_parameter_value += 1

        if self.write_proposal_pdb:
            pdbfile.close()

            prefix = '%s-%d-%s' % (self.pdb_filename_prefix, self.nproposed, direction)
            if direction == 'forward':
                pdbfile = open('%s-final.pdb' % prefix, 'w')
                PDBFile.writeFile(top_proposal.new_topology, new_positions, file=pdbfile)
                pdbfile.close()
        total_time = time.time() - initial_time
        print("total time: %f" % total_time)
        growth_parameter_value += 1
        return logp_proposal, new_positions

    def _generate_conformations(self, smiles):
        """
        Generate an oemol with up to max_confs conformations.

        Parameters
        ----------
        smiles : str
            The SMILES string for the molecule

        Returns
        -------
        conf_mol : oechem.OEMol with confs
        """

        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        oechem.OEAddExplicitHydrogens(mol)
        oechem.OECanonicalOrderAtoms(mol)
        oechem.OETriposAtomNames(mol)
        self._omega(mol)
        return mol

    def _get_omega_torsions(self, mol_conf, res_mol, torsion):
        """
        Utility function to get a particular torsion from the conformation.
        Note that all atoms in the OEConf must have a topology index defined.

        Parameters
        ----------
        mol_conf : openeye.OEMol
            The conformations of the residue of interest
        res_mol : oechem.OEMol
            The OEMol representation with old topology indexes
        torsion : parmed.Dihedral
            The chosen torsion for this proposal

        Returns
        -------
        torsion_phis : list of float, in radians
            The angles in the oeconf geometries
        """
        from perses.tests.utils import extractPositionsFromOEMOL
        atom_1_name = res_mol.GetAtom(PredAtomTopologyIndex(torsion.atom1.idx)).GetName()
        atom_2_name = res_mol.GetAtom(PredAtomTopologyIndex(torsion.atom2.idx)).GetName()
        atom_3_name = res_mol.GetAtom(PredAtomTopologyIndex(torsion.atom3.idx)).GetName()
        atom_4_name = res_mol.GetAtom(PredAtomTopologyIndex(torsion.atom4.idx)).GetName()
        torsion_phis = []
        for oeconf in mol_conf.GetConfs():
            positions = extractPositionsFromOEMOL(oeconf)
            #then, retrieve the atoms from the reference conformation
            atom_1_index = oeconf.GetAtom(oechem.OEHasAtomName(atom_1_name)).GetIdx()
            atom_2_index = oeconf.GetAtom(oechem.OEHasAtomName(atom_2_name)).GetIdx()
            atom_3_index = oeconf.GetAtom(oechem.OEHasAtomName(atom_3_name)).GetIdx()
            atom_4_index = oeconf.GetAtom(oechem.OEHasAtomName(atom_4_name)).GetIdx()


            internal_coords, _ = self._cartesian_to_internal(positions[atom_1_index], positions[atom_2_index], positions[atom_3_index], positions[atom_4_index])
            torsion_phis.append(internal_coords[2])

        return torsion_phis

    def _propose_torsion_oeconf(self, torsion, res_mol, mol_conf):
        """
        Propose a torsion based on a von mises distribution
        about the reference geometry in oeconf
        Parameters
        ----------
        torsion : parmed.Dihedral
            torsion of interest
        res_mol : oechem.OEMol
            OEMol of the new residue with tripos names and
            topology_index
        mol_conf : oechem.OEMol
            reference geometries

        Returns
        -------
        proposed_torsion_angle : simtk.unit.Quantity radians
            the proposed torsion angle
        logp_torsion : the log-probability of the choice
        """
        reference_angle_list = self._get_omega_torsions(mol_conf, res_mol, torsion)
        reference_angle = np.random.choice(reference_angle_list)
        logp_choice = - np.log(len(reference_angle_list))
        adjusted_reference = reference_angle
        proposed_torsion_angle = np.random.vonmises(adjusted_reference, self._kappa)
        #print("Proposing %s-%s-%s-%s with angle %f" % (str(torsion.atom1), str(torsion.atom2), str(torsion.atom3), str(torsion.atom4), proposed_torsion_angle))
        #print("With an unadjusted reference of %s" % str(reference_angle))
        logp_torsion = self._torsion_vm_logp(proposed_torsion_angle, adjusted_reference) + logp_choice
        return proposed_torsion_angle, logp_torsion

    def _propose_mtm_torsion(self, atom, torsion, res_mol, mol_conf, positions, r, theta, context, beta, phi=None, use_oeconf=False):
        """
        Use the multiple-try/CBMC method to propose a torsion angle. Omega geometries are used as the proposal distribution
        Parameters
        ----------
        atom
        torsion
        res_mol
        mol_conf
        positions
        r
        theta
        context
        beta
        phi

        Returns
        -------

        """
        internal_coordinates = np.zeros([3])
        proposed_torsions = np.zeros([self._n_trials])
        proposal_logps = np.zeros([self._n_trials])
        log_proposal_weights = np.zeros([self._n_trials])
        bond_position = positions[torsion.atom2.idx].value_in_unit(units.nanometers)
        angle_position = positions[torsion.atom3.idx].value_in_unit(units.nanometers)
        torsion_position = positions[torsion.atom4.idx].value_in_unit(units.nanometers)
        internal_coordinates[0] = r.value_in_unit(units.nanometers)
        internal_coordinates[1] = theta.value_in_unit(units.radians)
        if phi:
            reference_angles = self._get_omega_torsions(mol_conf, res_mol, torsion)
            #TODO: consider more numerically stable thing here
            p_phi = 0.0
            for i, reference_angle in enumerate(reference_angles):
                p_phi += np.exp(self._torsion_vm_logp(phi.value_in_unit(units.radian), reference_angle))
            logp_phi_proposal = np.log(p_phi) - np.log(len(reference_angles))
            phi = phi.value_in_unit(units.radians)
            proposed_torsions[0] = phi
            proposal_logps[0] = logp_phi_proposal
            trial_range = range(1, self._n_trials)
        else:
            trial_range = range(self._n_trials)

        for trial_idx in trial_range:
            if use_oeconf:
                proposed_torsions[trial_idx], proposal_logps[trial_idx] = self._propose_torsion_oeconf(torsion, res_mol, mol_conf)
            else:
                proposed_torsions[trial_idx] = np.random.uniform(-np.pi, np.pi)
                proposal_logps[trial_idx] = -np.log(self._n_trials)


        trial_xyzs = self.torsion_scan(bond_position, angle_position, torsion_position, internal_coordinates, proposed_torsions)

        trial_xyzs = units.Quantity(trial_xyzs, unit=units.nanometers)

        for i, xyz in enumerate(range(self._n_trials)):
            new_positions = copy.deepcopy(positions)
            new_positions[atom.idx] = units.Quantity(xyz, unit=units.nanometer)
            context.setPositions(new_positions)
            state = context.getState(getEnergy=True)
            potential = state.getPotentialEnergy()
            unnormalized_log_p = - beta * potential
            log_proposal_weights[i] = unnormalized_log_p - proposal_logps[i]

        normalized_log_weights = self._normalize_log_weights(log_proposal_weights)
        weights = np.exp(normalized_log_weights)

        if phi:
            logp_torsion = np.log(weights[0])
            return phi, logp_torsion
        else:
            phi_idx = np.random.choice(range(self._n_trials), p=weights)
            return proposed_torsions[phi_idx], np.log(weights[phi_idx])

    def torsion_scan(self, bond_position, angle_position, torsion_position, internal_coordinates, proposed_torsions):
        """
        A wrapper of the coordinate_numba version, promotes everything to np.float64
        """
        from coordinate_numba import torsion_scan
        bond_position = bond_position.astype(np.float64)
        angle_position = angle_position.astype(np.float64)
        torsion_position = torsion_position.astype(np.float64)
        internal_coordinates = internal_coordinates.astype(np.float64)
        proposed_torsions = proposed_torsions.astype(np.float64)

        xyzs = torsion_scan(bond_position, angle_position, torsion_position, internal_coordinates, proposed_torsions)

        return xyzs

    def _normalize_log_weights(self, unnormalized_log_weights):
        adjusted_log_weights = unnormalized_log_weights - max(unnormalized_log_weights)
        unnormalized_weights = np.exp(adjusted_log_weights)
        normalized_log_weights = adjusted_log_weights - np.log(np.sum(unnormalized_weights))
        return normalized_log_weights



    def _torsion_vm_logp(self, torsion_angle, mean):
        """
        Calculate the logp of the given torsion according to the von mises
        distribution with the kappa parameter set in the constructor

        Parameters
        ----------
        torsion_angle : float, in radians
            The angle whose logp is desired
        mean : float, in radians
            The mean of the distribution

        Returns
        -------
        logp_torsion : float
            the logp of the torsion angle
        """
        import scipy.stats as stats
        logp_torsion = stats.vonmises.logpdf(torsion_angle, self._kappa, mean)
        return logp_torsion

    def _logp_torsion_reverse(self, positions, torsion, oeconf):
        """
        Calculate the logp_reverse of the given torsion

        Parameters
        ----------
        positions : [n, 3] np.array of float
            the positions of all the atoms in the system
        torsion : parmed.Dihedral
            the torsion of interest
        oeconf

        Returns
        -------

        """
        pass

class PredAtomTopologyIndex(oechem.OEUnaryAtomPred):

    def __init__(self, topology_index):
        super(PredAtomTopologyIndex, self).__init__()
        self._topology_index = topology_index

    def __call__(self, atom):
        atom_data = atom.GetData()
        if 'topology_index' in atom_data.keys():
            if atom_data['topology_index'] == self._topology_index:
                return True
        return False


class BootstrapParticleFilter(object):
    """
    Implements a Bootstrap Particle Filter (BPF)
    to sample from the appropriate degrees of freedom.
    Designed for use with the dimension-matching scheme
    of Perses.
    """

    def __init__(self, growth_context, atom_torsions, initial_positions, beta, n_particles=18, resample_frequency=10):
        """

        Parameters
        ----------
        growth_context : simtk.openmm.Context object
            Context containing appropriate "growth system"
        atom_torsions : dict
            parmed.Atom : parmed.Dihedral dict that specifies
            what torsion to use to propose each atom
        initial_positions : np.ndarray [n,3]
            The positions of existing atoms.
        beta : simtk.unit.Quantity
            The inverse temperature, with units
        n_particles : int, optional
            The number of particles in the BPF (note that this
            is NOT the number of atoms). Default 18.
        resample_frequency : int, optional
            How often to resample particles. default 10
        """

        raise NotImplementedError
        self._system = growth_context.getSystem()
        self._beta = beta
        self._growth_stage = 0
        self._growth_context = growth_context
        self._atom_torsions = atom_torsions
        self._n_particles = n_particles
        self._resample_frequency = resample_frequency
        self._n_new_atoms = len(self._atom_torsions)
        self._initial_positions = initial_positions
        self._new_indices = [atom.idx for atom in self._atom_torsions.keys()]
        #create a matrix for log weights (n_particles, n_stages)
        self._Wij = np.zeros([self._n_particles, self._n_new_atoms])
        #create an array for positions--only store new positions to avoid
        #consuming way too much memory
        self._new_positions = np.zeros([self._n_particles, self._n_new_atoms, 3])
        self._generate_configurations()

    def _internal_to_cartesian(self, bond_position, angle_position, torsion_position, r, theta, phi):
        """
        Calculate the cartesian coordinates given the internal, as well as abs(detJ)
        """
        r = r.value_in_unit(units.nanometers)
        theta = theta.value_in_unit(units.radians)
        phi = phi.value_in_unit(units.radians)
        bond_position = bond_position.astype(np.float64)
        angle_position = angle_position.astype(np.float64)
        torsion_position = torsion_position.astype(np.float64)
        xyz = coordinate_numba.internal_to_cartesian(bond_position, angle_position, torsion_position, np.array([r, theta, phi], dtype=np.float64))
        return xyz, r**2*np.sin(theta)

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

    def _log_unnormalized_target(self, new_positions):
        """
        Given a set of new positions (not all positions!) and a growth
        stage, return the log unnormalized probability.

        Parameters
        ----------
        new_positions :  np.array
            Array containing m 3D coordinates of new atoms

        Returns
        -------
        log_unnormalized_probability : float
            The unnormalized probability of this configuration
        """
        positions = copy.deepcopy(self._initial_positions)
        positions[self._new_indices] = new_positions
        self._growth_context.setParameter('growth_stage', self._growth_stage)
        self._growth_context.setPositions(positions)
        energy = self._growth_context.getState(getEnergy=True).getPotentialEnergy()
        return -self._beta*energy

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
        relevant_bond_with_units : parmed.Bond
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

    def _bond_logq(self, r, bond):
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
        beta : simtk.unit.Quantity
            1/kT or inverse temperature
        """
        k_eq = bond.type.k
        r0 = bond.type.req
        logq = -self._beta*0.5*k_eq*(r-r0)**2
        return logq

    def _angle_logq(self, theta, angle):
        """
        Calculate the log-probability of a given bond at a given inverse temperature

        Arguments
        ---------
        theta : float
            bond angle, in randians
        angle : parmed angle object
            Bond angle object containing parameters
        beta : simtk.unit.Quantity
            1/kT or inverse temperature
        """
        k_eq = angle.type.k
        theta0 = angle.type.theteq
        logq = -self._beta*k_eq*0.5*(theta-theta0)**2
        return logq

    def _propose_bond(self, bond):
        """
        Bond length proposal
        """
        r0 = bond.type.req
        k = bond.type.k
        sigma_r = units.sqrt(1.0/(self._beta*k))
        r = sigma_r*np.random.randn() + r0
        return r

    def _propose_angle(self, angle):
        """
        Bond angle proposal
        """
        theta0 = angle.type.theteq
        k = angle.type.k
        sigma_theta = units.sqrt(1.0/(self._beta*k))
        theta = sigma_theta*np.random.randn() + theta0
        return theta

    def _propose_atom(self, atom, torsion, new_positions):
        """
        Propose a set of internal coordinates (r, theta, phi) and transform
        to cartesian coordinates (with jacobian correction).
        for the given atom. R and theta are drawn from their respective
        equilibrium distributions, whereas phi is simply a uniform sample.

        Parameters
        ----------
        atom : parmed.Atom
            atom that will have its position proposed
        torsion : parmed.Dihedral
            torsion that contains relevant information for atom
        new_positions : [m, 3] np.array
            array of just the new positions (not existing atoms)
        Returns
        -------
        xyz : [1,3] np.array of float
            The proposed cartesian coordinates
        logp : float
            The log probability with jacobian correction
        """
        positions = copy.deepcopy(self._initial_positions)
        positions[self._new_indices] = new_positions
        bond_atom = torsion.atom2
        angle_atom = torsion.atom3
        torsion_atom = torsion.atom4

        if atom != torsion.atom1:
            raise Exception('atom != torsion.atom1')

        bond = self._get_relevant_bond(atom, bond_atom)

        if bond is not None:
            r = self._propose_bond(bond)
            bond_k = bond.type.k
            sigma_r = units.sqrt(1/(self._beta*bond_k))
            logZ_r = np.log((np.sqrt(2*np.pi)*(sigma_r/units.angstroms))) # CHECK DOMAIN AND UNITS
            logp_r = self._bond_logq(r, bond) - logZ_r
        else:
            constraint = self._get_bond_constraint(atom, bond_atom, self._system)
            r = constraint #set bond length to exactly constraint
            logp_r = 0.0

        #propose an angle and calculate its probability
        angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
        theta = self._propose_angle(angle)
        angle_k = angle.type.k
        sigma_theta = units.sqrt(1/(self._beta*angle_k))
        logZ_theta = np.log((np.sqrt(2*np.pi)*(sigma_theta/units.radians))) # CHECK DOMAIN AND UNITS
        logp_theta = self._angle_logq(theta, angle) - logZ_theta

        #propose a torsion angle uniformly (this can be dramatically improved)
        phi = np.random.uniform(-np.pi, np.pi)
        logp_phi = -np.log(2*np.pi)

        #get the new cartesian coordinates and detJ:
        new_xyz, detJ = self._internal_to_cartesian(positions[bond_atom.idx], positions[angle_atom.idx], positions[torsion_atom.idx], r, theta, phi)
        #accumulate logp
        logp_proposal = logp_r + logp_theta + logp_phi + np.log(np.abs(detJ))

        return new_xyz, logp_proposal

    def _resample(self):
        """
        Resample from the current set of weights and positions.
        """
        particle_indices = range(self._n_particles)
        new_indices = np.random.choice(particle_indices, size=self._n_particles, p=self._Wij[:, self._growth_stage-1])
        for particle_index in particle_indices:
            self._new_positions[particle_index, :, :] = self._new_positions[new_indices[particle_index], :, :]
        self._Wij[:, self._growth_stage-1] = -np.log(self._n_particles) #set particle weights to be equal

    def _generate_configurations(self):
        """
        Generate the ensemble of configurations of the new atoms, approximately
        from p(x_new | x_common).
        """
        for i, atom_torsion in enumerate(self._atom_torsions.items()):
            self._growth_stage = i+1
            for particle_index in range(self._n_particles):
                proposed_xyz, logp_proposal = self._propose_atom(atom_torsion[0], atom_torsion[1])
                self._new_positions[particle_index, i, :] = proposed_xyz
                unnormalized_log_target = self._log_unnormalized_target(self._new_positions[particle_index, :,:])
                if i > 0:
                    self._Wij = [particle_index, i] = (unnormalized_log_target - logp_proposal) + self._Wij[particle_index, i-1]
                else:
                    self._Wij = [particle_index, i] = unnormalized_log_target - logp_proposal
            sum_log_weights = np.sum(np.exp(self._Wij[:,i]))
            self._Wij -= np.log(sum_log_weights)
            if i % self._resample_frequency == 0 and i != 0:
                self._resample()


class OmegaGeometryEngine(GeometryEngine):
    """
    This class proposes new small molecule geometries based on a set of precomputed
    omega geometries.
    """

    def __init__(self, n_omega_references=1, proposal_sigma=1.0, metadata=None):
        self._n_omega_references = n_omega_references
        self._proposal_sigma = 1.0
        self._reference_oemols = {}
        self._metadata = metadata
        raise NotImplementedError

    def propose(self, top_proposal, current_positions, beta):
        """
        Propose positions of new atoms according to a selected omega geometry.

        Parameters
        ----------
        top_proposal : TopologyProposal object
            TopologyProposal object generated by the proposal engine
        current_positions : [n, 3] np.array of float
            Positions of the current system
        beta : float
            inverse temperature

        Returns
        -------
        new_positions : [m, 3] np.array of float
            The positions of the new system
        logp_propose : float
            The log-probability of the proposal
        """
        pass

    def logp_reverse(self, top_proposal, new_coordinates, old_coordinates, beta):
        """

        Parameters
        ----------
        top_proposal
        new_coordinates
        old_coordinates
        beta

        Returns
        -------

        """
        pass


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
        self._stericsNonbondedEnergy = "select(step({}-max(growth_idx1, growth_idx2)), U_sterics_active, 0);"
        self._stericsNonbondedEnergy += "U_sterics_active = 4*epsilon*x*(x-1.0); x = (sigma/r)^6;"
        self._stericsNonbondedEnergy += "epsilon = sqrt(epsilon1*epsilon2); sigma = 0.5*(sigma1 + sigma2);"




    def create_modified_system(self, reference_system, growth_indices, parameter_name, add_extra_torsions=True, reference_topology=None, use_sterics=True, force_names=None, force_parameters=None):
        """
        Create a modified system with parameter_name parameter. When 0, only core atoms are interacting;
        for each integer above 0, an additional atom is made interacting, with order determined by growth_index
        Parameters
        ----------
        reference_system : simtk.openmm.System object
            The system containing the relevant forces and particles
        growth_indices : list of atom
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

        modified_sterics_force = openmm.CustomNonbondedForce(self._stericsNonbondedEnergy.format(parameter_name))
        modified_sterics_force.addPerParticleParameter("sigma")
        modified_sterics_force.addPerParticleParameter("epsilon")
        modified_sterics_force.addPerParticleParameter("growth_idx")
        modified_sterics_force.addGlobalParameter(parameter_name, 0)

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
            if growth_idx==0:
                continue
            modified_bond_force.addBond(bond_parameters[0], bond_parameters[1], [bond_parameters[2], bond_parameters[3], growth_idx])

        #copy each angle, adding the per particle parameter as well
        reference_angle_force = reference_forces['HarmonicAngleForce']
        for angle in range(reference_angle_force.getNumAngles()):
            angle_parameters = reference_angle_force.getAngleParameters(angle)
            growth_idx = self._calculate_growth_idx(angle_parameters[:3], growth_indices)
            if growth_idx==0:
                continue
            modified_angle_force.addAngle(angle_parameters[0], angle_parameters[1], angle_parameters[2], [angle_parameters[3], angle_parameters[4], growth_idx])

        #copy each torsion, adding the per particle parameter as well
        reference_torsion_force = reference_forces['PeriodicTorsionForce']
        for torsion in range(reference_torsion_force.getNumTorsions()):
            torsion_parameters = reference_torsion_force.getTorsionParameters(torsion)
            growth_idx = self._calculate_growth_idx(torsion_parameters[:4], growth_indices)
            if growth_idx==0:
                continue
            modified_torsion_force.addTorsion(torsion_parameters[0], torsion_parameters[1], torsion_parameters[2], torsion_parameters[3], [torsion_parameters[4], torsion_parameters[5], torsion_parameters[6], growth_idx])

        #copy parameters for sterics parameters in nonbonded force
        if 'NonbondedForce' in reference_forces.keys() and use_sterics:
            modified_sterics_force = openmm.CustomNonbondedForce(self._stericsNonbondedEnergy.format(parameter_name))
            modified_sterics_force.addPerParticleParameter("sigma")
            modified_sterics_force.addPerParticleParameter("epsilon")
            modified_sterics_force.addPerParticleParameter("growth_idx")
            modified_sterics_force.addGlobalParameter(parameter_name, 0)
            growth_system.addForce(modified_sterics_force)
            reference_nonbonded_force = reference_forces['NonbondedForce']
            for particle_index in range(reference_nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = reference_nonbonded_force.getParticleParameters(particle_index)
                growth_idx = growth_indices.index(particle_index) + 1 if particle_index in growth_indices else 0
                modified_sterics_force.addParticle([sigma, epsilon, growth_idx])
            new_particle_indices = [atom.idx for atom in growth_indices]
            old_particle_indices = [idx for idx in range(reference_nonbonded_force.getNumParticles()) if idx not in new_particle_indices]
            modified_sterics_force.addInteractionGroup(set(new_particle_indices), set(old_particle_indices))
            modified_sterics_force.addInteractionGroup(set(new_particle_indices), set(new_particle_indices))



        if add_extra_torsions:
            if reference_topology==None:
                raise ValueError("Need to specify topology in order to add extra torsions.")
            self._determine_extra_torsions(modified_torsion_force, reference_topology, growth_indices)

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
        growth_indices : list of atom
            The list of new atoms and the order in which they will be added.

        Returns
        -------
        torsion_force : openmm.CustomTorsionForce
            The torsion force with extra torsions added appropriately.
        """
        # Do nothing if there are no atoms to grow.
        if len(growth_indices) == 0:
            return torsion_force

        import openmoltools.forcefield_generators as forcefield_generators
        atoms = list(reference_topology.atoms())
        growth_indices = list(growth_indices)
        #get residue from first atom
        residue = atoms[growth_indices[0].idx].residue
        try:
            oemol = FFAllAngleGeometryEngine._oemol_from_residue(residue)
        except Exception as e:
            print("Could not generate an oemol from the residue.")
            print(e)

        #get the omega geometry of the molecule:
        import openeye.oeomega as oeomega
        import openeye.oechem as oechem
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(1)
        omega.SetStrictStereo(False) #TODO: fix stereochem
        omega(oemol)

        #get the list of torsions in the molecule that are not about a rotatable bond
        rotor = oechem.OEIsRotor()
        torsion_predicate = oechem.OENotBond(rotor)
        non_rotor_torsions = list(oechem.OEGetTorsions(oemol, torsion_predicate))
        relevant_torsion_list = self._select_torsions_without_h(non_rotor_torsions)


        #now, for each torsion, extract the set of indices and the angle
        periodicity = 1
        k = 40.0*units.kilojoule_per_mole
        #print([atom.name for atom in growth_indices])
        for torsion in relevant_torsion_list:
            #make sure to get the atom index that corresponds to the topology
            atom_indices = [torsion.a.GetData("topology_index"), torsion.b.GetData("topology_index"), torsion.c.GetData("topology_index"), torsion.d.GetData("topology_index")]
            # Determine phase in [-pi,+pi) interval
            #phase = (np.pi)*units.radians+angle
            phase = torsion.radians + np.pi
            while (phase >= np.pi):
                phase -= 2*np.pi
            while (phase < -np.pi):
                phase += 2*np.pi
            phase *= units.radian
            #print('PHASE>>>> ' + str(phase)) # DEBUG
            growth_idx = self._calculate_growth_idx(atom_indices, growth_indices)
            atom_names = [torsion.a.GetName(), torsion.b.GetName(), torsion.c.GetName(), torsion.d.GetName()]
            #print("Adding torsion with atoms %s and growth index %d" %(str(atom_names), growth_idx))
            torsion_force.addTorsion(atom_indices[0], atom_indices[1], atom_indices[2], atom_indices[3], [periodicity, phase, k, 0])

        return torsion_force

    def _select_torsions_without_h(self, torsion_list):
        """
        Return only torsions that do not contain hydrogen

        Parameters
        ----------
        torsion_list : list of oechem.OETorsion

        Returns
        -------
        heavy_torsions : list of oechem.OETorsion
        """
        heavy_torsions = []
        for torsion in torsion_list:
            is_h_present = torsion.a.IsHydrogen() + torsion.b.IsHydrogen() + torsion.c.IsHydrogen() + torsion.d.IsHydrogen()
            if not is_h_present:
                heavy_torsions.append(torsion)
        return heavy_torsions


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
        growth_indices : list of atom
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

class PredHBond(oechem.OEUnaryBondPred):
    """
    Example elaborating usage on:
    https://docs.eyesopen.com/toolkits/python/oechemtk/predicates.html#section-predicates-match
    """
    def __call__(self, bond):
        atom1 = bond.GetBgn()
        atom2 = bond.GetEnd()
        if atom1.IsHydrogen() or atom2.IsHydrogen():
            return True
        else:
            return False

class ProposalOrderTools(object):
    """
    This is an internal utility class for determining the order of atomic position proposals.
    It encapsulates funcionality needed by the geometry engine. Atoms can be proposed without
    torsions or even angles, though this may not be recommended. Default is to require torsions.

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
        atoms_torsions : ordereddict
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
        return atoms_torsions, logp_torsion_choice


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
        if not eligible_torsions:
            raise NoTorsionError("No eligible torsions found for placing atom %s." % str(atom_for_proposal))
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

class BondOnlyProposalOrder(ProposalOrderTools):
    """
    A class similar to ProposalOrderTools, but only uses bonds (no angles or torsions)
    to decide proposal order and bond choice.
    """

    def determine_proposal_order(self, direction='forward'):
        """
        Determine the proposal order of this system pair.
        This includes the choice of a bond As such, a logp is returned,
        but this is typically 0 except in rings

        Parameters
        ----------
        direction : str, optional
            whether to determine the forward or reverse proposal order

        Returns
        -------
        atoms_bonds : OrderedDict
            parmed.Atom : parmed.Atom atom : chosen_bonded_atom
        logp_bond_choice : float
            log probability of the chosen torsions
        """
        atoms_bonds = collections.OrderedDict()
        logp_bond_choice = 0.0
        if direction=='forward':
            structure = parmed.openmm.load_topology(self._topology_proposal.new_topology, self._topology_proposal.new_system)
            new_atoms = [structure.atoms[idx] for idx in self._topology_proposal.unique_new_atoms]
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in self._topology_proposal.new_to_old_atom_map.keys()]
        elif direction=='reverse':
            structure = parmed.openmm.load_topology(self._topology_proposal.old_topology, self._topology_proposal.old_system)
            new_atoms = [structure.atoms[idx] for idx in self._topology_proposal.unique_old_atoms]
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in self._topology_proposal.old_to_new_atom_map.keys()]
        else:
            raise ValueError("direction parameter must be either forward or reverse.")

        while(len(new_atoms))>0:
            eligible_atoms = self._atoms_eligible_for_proposal(new_atoms, atoms_with_positions)
            if (len(new_atoms) > 0) and (len(eligible_atoms) == 0):
                raise Exception('new_atoms (%s) has remaining atoms to place, but eligible_atoms is empty.' % str(new_atoms))
            for atom in eligible_atoms:
                chosen_bond_atom, logp_choice = self._choose_bond_partner(atoms_with_positions, atom)
                atoms_bonds[atom] = chosen_bond_atom
                logp_bond_choice += logp_choice
                new_atoms.remove(atom)
                atoms_with_positions.append(atom)
        return atoms_bonds, logp_bond_choice

    def _choose_bond_partner(self, atoms_with_positions, atom):
        """
        Choose a bond partner to atom that has positions.

        Parameters
        ----------
        atoms_with_positoins
        atom

        Returns
        -------

        """
        potential_bond_partners = [a for a in atom.bond_partners if a in atoms_with_positions]
        logp_choice = np.log(1.0/len(potential_bond_partners))
        chosen_bonded_atom = np.random.choice(potential_bond_partners)
        return chosen_bonded_atom, logp_choice


class NoTorsionError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(NoTorsionError, self).__init__(message)

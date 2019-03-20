"""
This contains the base class for the geometry engine, which proposes new positions
for each additional atom that must be added.
"""
import parmed
import simtk.unit as units
import logging
import numpy as np
import copy
#from perses.rjmc import coordinate_numba
import simtk.openmm as openmm
import collections
import openeye.oechem as oechem
import openeye.oeomega as oeomega
import simtk.openmm.app as app
import time
import logging
from perses.storage import NetCDFStorage, NetCDFStorageView
from openmmtools.constants import kB
_logger = logging.getLogger("geometry")


class GeometryEngine(object):
    """
    This is the base class for the geometry engine.

    Arguments
    ---------
    metadata : dict
        GeometryEngine-related metadata as a dict
    """

    def __init__(self, metadata=None, storage=None):
        # TODO: Either this base constructor should be called by subclasses, or we should remove its arguments.
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

    Parameters
    ----------
    use_sterics : bool, optional, default=False
        If True, sterics will be used in proposals to minimize clashes.
        This may significantly slow down the simulation, however.

    """
    def __init__(self, metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0):
        self._metadata = metadata
        self.write_proposal_pdb = False # if True, will write PDB for sequential atom placements
        self.pdb_filename_prefix = 'geometry-proposal' # PDB file prefix for writing sequential atom placements
        self.nproposed = 0 # number of times self.propose() has been called
        self._energy_time = 0.0
        self._torsion_coordinate_time = 0.0
        self._position_set_time = 0.0
        self.verbose = verbose
        self.use_sterics = use_sterics
        self._n_bond_divisions = n_bond_divisions
        self._n_angle_divisions = n_angle_divisions
        self._n_torsion_divisions = n_torsion_divisions
        self._bond_softening_constant = bond_softening_constant
        self._angle_softening_constant = angle_softening_constant
        if storage:
            self._storage = NetCDFStorageView(modname="GeometryEngine", storage=storage)
        else:
            self._storage = None

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
            growth_system_generator = GeometrySystemGenerator(top_proposal.new_system, atom_proposal_order.keys(), growth_parameter_name, reference_topology=top_proposal.new_topology, use_sterics=self.use_sterics)
            growth_system = growth_system_generator.get_modified_system()
            growth_system_time = time.time() - system_init
        elif direction=='reverse':
            if new_positions is None:
                raise ValueError("For reverse proposals, new_positions must not be none.")
            atom_proposal_order, logp_choice = proposal_order_tool.determine_proposal_order(direction='reverse')
            structure = parmed.openmm.load_topology(top_proposal.old_topology, top_proposal.old_system)
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in top_proposal.old_to_new_atom_map.keys()]
            growth_system_generator = GeometrySystemGenerator(top_proposal.old_system, atom_proposal_order.keys(), growth_parameter_name, reference_topology=top_proposal.old_topology, use_sterics=self.use_sterics)
            growth_system = growth_system_generator.get_modified_system()
        else:
            raise ValueError("Parameter 'direction' must be forward or reverse")

        logp_proposal = logp_choice
        if self._storage:
            self._storage.write_object("{}_proposal_order".format(direction), proposal_order_tool, iteration=self.nproposed)

        if self.use_sterics:
            platform_name = 'CPU'
        else:
            platform_name = 'Reference'

        platform = openmm.Platform.getPlatformByName(platform_name)
        integrator = openmm.VerletIntegrator(1*units.femtoseconds)
        context = openmm.Context(growth_system, integrator, platform)
        growth_system_generator.set_growth_parameter_index(len(atom_proposal_order.keys())+1, context)
        growth_parameter_value = 1

        #now for the main loop:
        logging.debug("There are %d new atoms" % len(atom_proposal_order.items()))
        atom_placements = []
        #atom_number=0
        for atom, torsion in atom_proposal_order.items():
            growth_system_generator.set_growth_parameter_index(growth_parameter_value, context=context)
            bond_atom = torsion.atom2
            angle_atom = torsion.atom3
            torsion_atom = torsion.atom4
            if self.verbose: _logger.info("Proposing atom %s from torsion %s" %(str(atom), str(torsion)))

            if atom != torsion.atom1:
                raise Exception('atom != torsion.atom1')

            #get internal coordinates if direction is reverse
            if direction=='reverse':
                atom_coords = old_positions[atom.idx]
                bond_coords = old_positions[bond_atom.idx]
                angle_coords = old_positions[angle_atom.idx]
                torsion_coords = old_positions[torsion_atom.idx]
                internal_coordinates, detJ = self._cartesian_to_internal(atom_coords, bond_coords, angle_coords, torsion_coords)
                r, theta, phi = internal_coordinates[0], internal_coordinates[1], internal_coordinates[2]


            bond = self._get_relevant_bond(atom, bond_atom)
            if bond is not None:
                if direction=='forward':
                    r = self._propose_bond(bond, beta, n_divisions=self._n_bond_divisions)

                logp_r=self._bond_logq(r, bond, beta, self._n_bond_divisions)
            else:
                if direction == 'forward':
                    constraint = self._get_bond_constraint(atom, bond_atom, top_proposal.new_system)
                    if constraint is None:
                        raise ValueError("Structure contains a topological bond [%s - %s] with no constraint or bond information." % (str(atom), str(bond_atom)))
                    r = constraint.value_in_unit_system(units.md_unit_system) #set bond length to exactly constraint
                logp_r = 0.0

            #propose an angle and calculate its probability
            angle = self._get_relevant_angle(atom, bond_atom, angle_atom)
            if direction=='forward':
                theta = self._propose_angle(angle, beta, n_divisions=self._n_angle_divisions)

            logp_theta=self._angle_logq(theta, angle, beta, self._n_angle_divisions)

            #propose a torsion angle and calcualate its probability
            if direction=='forward':
                phi, logp_phi = self._propose_torsion(context, torsion, new_positions, r, theta, beta, n_divisions=self._n_torsion_divisions)
                xyz, detJ = self._internal_to_cartesian(new_positions[bond_atom.idx], new_positions[angle_atom.idx], new_positions[torsion_atom.idx], r, theta, phi)
                new_positions[atom.idx] = xyz
            else:
                old_positions_for_torsion = copy.deepcopy(old_positions)
                logp_phi = self._torsion_logp(context, torsion, old_positions_for_torsion, r, theta, phi, beta, n_divisions=self._n_torsion_divisions)

            #accumulate logp
            #if direction == 'reverse':
            if self.verbose: _logger.info('%8d logp_r %12.3f | logp_theta %12.3f | logp_phi %12.3f | log(detJ) %12.3f' % (atom.idx, logp_r, logp_theta, logp_phi, np.log(detJ)))

            atom_placement_array = np.array([atom.idx,r,
                                             theta,
                                             phi,
                                             logp_r, logp_theta, logp_phi, np.log(detJ)])
            atom_placements.append(atom_placement_array)

            logp_proposal += logp_r + logp_theta + logp_phi - np.log(detJ)
            growth_parameter_value += 1

            # DEBUG: Write PDB file for placed atoms
            atoms_with_positions.append(atom)
        total_time = time.time() - initial_time

        #use a new array for each placement, since the variable size will be different.
        if self._storage:
            self._storage.write_array("atom_placement_logp_{}_{}".format(direction, self.nproposed), np.stack(atom_placements))

        if direction=='forward':
            logging.log(logging.DEBUG, "Proposal order time: %f s | Growth system generation: %f s | Total torsion scan time %f s | Total energy computation time %f s | Position set time %f s| Total time %f s" % (proposal_order_time, growth_system_time , self._torsion_coordinate_time, self._energy_time, self._position_set_time, total_time))
        self._torsion_coordinate_time = 0.0
        self._energy_time = 0.0
        self._position_set_time = 0.0
        return logp_proposal, new_positions

    @staticmethod
    def _oemol_from_residue(res, verbose=True):
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
        # TODO: This seems to be broken. Can we fix it?
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

        # DEBUG
        if len(relevant_angle_set) == 0:
            print('atom1_angles:')
            print(atom1_angles)
            print('atom2_angles:')
            print(atom2_angles)
            print('atom3_angles:')
            print(atom3_angles)
            raise Exception('Atoms %s-%s-%s do not share a parmed Angle term' % (atom1, atom2, atom3))

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
        from perses.rjmc import coordinate_numba
        #ensure we have the correct units, then remove them
        atom_position = atom_position.value_in_unit(units.nanometers).astype(np.float64)
        bond_position = bond_position.value_in_unit(units.nanometers).astype(np.float64)
        angle_position = angle_position.value_in_unit(units.nanometers).astype(np.float64)
        torsion_position = torsion_position.value_in_unit(units.nanometers).astype(np.float64)

        internal_coords = coordinate_numba.cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)


        return internal_coords, np.abs(internal_coords[0]**2*np.sin(internal_coords[1]))

    def _internal_to_cartesian(self, bond_position, angle_position, torsion_position, r, theta, phi):
        """
        Calculate the cartesian coordinates of a newly placed atom in terms of internal coordinates,
        along with the absolute value of the determinant of the Jacobian.

        Parameters
        ----------
        bond_position : simtk.unit.Quantity wrapped numpy array of shape (natoms,) with units compatible with nanometers
            Position of atom separated from newly placed atom with bond length ``r``
        angle_position : simtk.unit.Quantity wrapped numpy array of shape (natoms,) with units compatible with nanometers
            Position of atom separated from newly placed atom with angle ``theta``
        torsion_position : simtk.unit.Quantity wrapped numpy array of shape (natoms,) with units compatible with nanometers
            Position of atom separated from newly placed atom with torsion ``phi``
        r : simtk.unit.Quantity with units compatible with nanometers
            Bond length distance from ``bond_position`` to newly placed atom
        theta : simtk.unit.Quantity with units compatible with radians
            Angle formed by ``(angle_position, bond_position, new_atom)``
        phi : simtk.unit.Quantity with units compatible with radians
            Torsion formed by ``(torsion_position, angle_position, bond_position, new_atom)``

        Returns
        -------
        xyz : simtk.unit.Quantity wrapped numpy array of shape (3,) with units compatible with nanometers
            The position of the newly placed atom
        detJ : float
            The absolute value of the determinant of the Jacobian transforming from (r,theta,phi) to (x,y,z)
            .. todo :: Clarify the direction of the Jacobian

        """
        # TODO: Refine API to always use unit-bearing quantities on input and output

        # Ensure (r,theta,phi) are already dimensionless
        assert isinstance

        # Ensure positions all have correct dimensionality

        # Transform into unitless values in distance units
        length_unit = units.nanometers
        bond_position = bond_position.value_in_unit(length_unit).astype(np.float64)
        angle_position = angle_position.value_in_unit(length_unit).astype(np.float64)
        torsion_position = torsion_position.value_in_unit(length_unit).astype(np.float64)
        # Compute Cartesian coordinates from internal coordinates using all-dimensionless quantities
        from perses.rjmc import coordinate_numba
        xyz = coordinate_numba.internal_to_cartesian(bond_position, angle_position, torsion_position, np.array([r, theta, phi], dtype=np.float64))
        # Transform position of new atom back into unit-bearing Quantity
        xyz = units.Quantity(xyz, unit=length_unit)
        # Compute abs det Jacobian
        detJ = np.abs(r**2*np.sin(theta))

        return xyz, detJ

    def _bond_logq(self, r, bond, beta, n_divisions):
        """
        Calculate the log-probability of a given bond at a given inverse temperature

        Arguments
        ---------
        r : nanometers
            bond length, in nanometers
        r0 : nanometers
            equilibrium bond length, in nanometers
        k_eq : kcal / (mol * nanometers**2)
            Spring constant of bond
        beta : mol/kcal
            1/kT or inverse temperature
        n_divisions : int
            number of discretizations of space
        """
        from simtk.unit.quantity import is_dimensionless

        r0 = bond.type.req
        k = bond.type.k * self._bond_softening_constant
        sigma_r = units.sqrt((1.0/(beta*k)))

        for quant, unit_divisor in zip( [beta, r0, k], [1./units.kilocalories_per_mole, units.nanometers, units.kilocalories_per_mole/(units.nanometers**2)]):
            assert is_dimensionless(quant / unit_divisor), "{} is not dimensionless".format(quant)

        r0, k, sigma_r = r0.value_in_unit_system(units.md_unit_system), k.value_in_unit_system(units.md_unit_system), sigma_r.value_in_unit_system(units.md_unit_system)
        lower_bound, upper_bound = max(0.,r0-6*sigma_r), r0+6*sigma_r

        f = lambda x: (x)**2 * np.exp(-(0.5/sigma_r**2)*(x-r0)**2)
        r_array = np.linspace(lower_bound, upper_bound, n_divisions)
        division_size = (upper_bound - lower_bound) / (n_divisions - 1)
        r_index = np.argmin(np.abs(r - r_array))
        Z = sum(f(r_array))
        logp = 2 * np.log((r_array[r_index])) - (0.5/sigma_r**2)*(r_array[r_index]-r0)**2 - np.log(division_size) - np.log(Z)

        return logp

    def _angle_logq(self, theta, angle, beta, n_divisions):
        """
        Calculate the log-probability of a given bond at a given inverse temperature

        Arguments
        ---------
        theta : radians
            bond angle, in radians
        angle : radians
            Bond angle object containing parameters
        beta : mol / kcal
            1/kT or inverse temperature
        """
        from simtk.unit.quantity import is_dimensionless

        theta0 = angle.type.theteq
        k = angle.type.k * self._angle_softening_constant
        sigma_theta = units.sqrt(1.0/(beta * k))

        for quant, unit_divisor in zip([beta, theta0, k], [1./units.kilocalories_per_mole, units.radians, units.kilocalories_per_mole/units.radians**2]):
            assert is_dimensionless(quant / unit_divisor), "{} is not dimensionless".format(quant)

        theta0, k, sigma_theta = theta0.value_in_unit_system(units.md_unit_system), k.value_in_unit_system(units.md_unit_system), sigma_theta.value_in_unit_system(units.md_unit_system)
        lower_bound, upper_bound=0., np.pi

        #'exact' probability
        f = lambda x: np.sin(x) * np.exp(-(0.5/sigma_theta**2) * (x-theta0)**2)
        theta_array = np.linspace(0, np.pi, n_divisions)
        division_size = np.pi/(n_divisions-1)
        theta_index = np.argmin(np.abs(theta-theta_array))
        Z = sum(f(theta_array))
        logp = np.log(np.sin(theta_array[theta_index])) - (0.5/sigma_theta**2) * (theta_array[theta_index]-theta0)**2-np.log(division_size)-np.log(Z)
        return logp

    def _propose_bond(self, bond, beta, n_divisions=1000):
        """
        Bond length proposal
        """
        import scipy.integrate as integrate
        from simtk.unit.quantity import is_dimensionless

        r0 = bond.type.req
        k = bond.type.k * self._bond_softening_constant
        sigma_r = units.sqrt((1.0/(beta*k)))

        for quant, unit_divisor in zip( [beta, r0, k], [1./units.kilocalories_per_mole, units.nanometers, units.kilocalories_per_mole/(units.nanometers**2)]):
            assert is_dimensionless(quant / unit_divisor), "{} is not dimensionless".format(quant)

        r0, k, sigma_r = r0.value_in_unit_system(units.md_unit_system), k.value_in_unit_system(units.md_unit_system), sigma_r.value_in_unit_system(units.md_unit_system)

        lower_bound, upper_bound = max(0., r0-6 * sigma_r), r0+6 * sigma_r

        r_array = np.linspace(lower_bound, upper_bound, n_divisions)
        division_size = (upper_bound - lower_bound) / (n_divisions - 1)
        r_array_indices = range(n_divisions)


        r_probability_mass_function = (r_array)**2 * np.exp(-(0.5/sigma_r**2) * (r_array-r0)**2)
        r_probability_mass_function_Z = sum(r_probability_mass_function)
        r_index = np.random.choice(r_array_indices, p = r_probability_mass_function/r_probability_mass_function_Z)


        r_min = max(r_array[r_index]-division_size/2.0, 0.)
        r_max = r_array[r_index]+division_size/2.0

        r = np.random.uniform(r_min, r_max)

        return r


    def _propose_angle(self, angle, beta, n_divisions=180):
        """
        Bond angle proposal
        """
        import scipy.integrate as integrate
        from simtk.unit.quantity import is_dimensionless

        theta0 = angle.type.theteq
        k = angle.type.k * self._angle_softening_constant
        sigma_theta = units.sqrt(1.0/(beta * k))

        for quant, unit_divisor in zip([beta, theta0, k], [1./units.kilocalories_per_mole, units.radians, units.kilocalories_per_mole/units.radians**2]):
            assert is_dimensionless(quant / unit_divisor), "{} is not dimensionless".format(quant)

        theta0, k, sigma_theta = theta0.value_in_unit_system(units.md_unit_system), k.value_in_unit_system(units.md_unit_system), sigma_theta.value_in_unit_system(units.md_unit_system)

        lower_bound, upper_bound=0., np.pi
        theta_array=np.linspace(lower_bound, upper_bound,n_divisions)
        division_size=theta_array[1] - theta_array[0]
        theta_array_indices=range(n_divisions)

        theta_probability_mass_function=np.sin(theta_array)*np.exp(-(0.5/sigma_theta**2)*(theta_array-theta0)**2)
        theta_probability_mass_function_Z=sum(theta_probability_mass_function)
        theta_index=np.random.choice(theta_array_indices, p=theta_probability_mass_function/theta_probability_mass_function_Z)


        theta_min=max(0.,theta_array[theta_index]-division_size/2.)
        theta_max=min(theta_array[theta_index]+division_size/2., np.pi)

        theta=np.random.uniform(theta_min, theta_max)

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
        from perses.rjmc import coordinate_numba
        torsion_scan_init = time.time()
        positions_copy = copy.deepcopy(positions)
        positions_copy = positions_copy.value_in_unit(units.nanometers)
        positions_copy = positions_copy.astype(np.float64)

        bond_atom = torsion.atom2
        angle_atom = torsion.atom3
        torsion_atom = torsion.atom4
        phis = np.arange(-np.pi, +np.pi, (2.0*np.pi)/n_divisions) # Can't use units here.
        xyzs_input_array=np.array([r, theta, 0.0])
        xyzs = coordinate_numba.torsion_scan(positions_copy[bond_atom.idx], positions_copy[angle_atom.idx], positions_copy[torsion_atom.idx], xyzs_input_array, phis)
        xyzs_quantity = units.Quantity(xyzs, unit=units.nanometers) #have to put the units back now

        torsion_scan_time = time.time() - torsion_scan_init
        self._torsion_coordinate_time += torsion_scan_time
        return xyzs_quantity, phis

    def _torsion_log_probability_mass_function(self, growth_context, torsion, positions, r, theta, beta, n_divisions=360):
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
            potential_energy = state.getPotentialEnergy()
            energy_computation_time = time.time() - energy_computation_init
            self._energy_time += energy_computation_time
            logq_i = -beta*potential_energy
            logq[i] = logq_i

        if np.sum(np.isnan(logq)) == n_divisions:
            raise Exception("All %d torsion energies in torsion PMF are NaN." % n_divisions)
        logq[np.isnan(logq)] = -np.inf
        logq -= max(logq)
        q = np.exp(logq)
        Z = np.sum(q)
        logp_torsions = logq - np.log(Z)

        if hasattr(self, '_proposal_pdbfile'):
            # Write proposal probabilities to PDB file as B-factors for inert atoms
            f_i = -logp_torsions
            f_i -= f_i.min() # minimum free energy is zero
            f_i[f_i > 999.99] = 999.99
            self._proposal_pdbfile.write('MODEL\n')
            for i, xyz in enumerate(xyzs):
                self._proposal_pdbfile.write('ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n' % (i+1, ' Ar ', 'Ar ', ' ', atom_idx+1, 10*xyz[0], 10*xyz[1], 10*xyz[2], np.exp(logp_torsions[i]), f_i[i]))
            self._proposal_pdbfile.write('TER\n')
            self._proposal_pdbfile.write('ENDMDL\n')
            # TODO: Write proposal PMFs to storage
            # atom_proposal_indices[order]
            # atom_positions[order,k]
            # torsion_pmf[order, division_index]

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
        logp_torsions, phis = self._torsion_log_probability_mass_function(growth_context, torsion, positions, r, theta, beta, n_divisions=n_divisions)

        division=2*np.pi/n_divisions
        phi_median_idx = np.random.choice(range(len(phis)), p=np.exp(logp_torsions))
        phi_min = phis[phi_median_idx] - division/2.0
        phi_max = phis[phi_median_idx] + division/2.0

        phi=np.random.uniform(phi_min, phi_max)
        logp = logp_torsions[phi_median_idx] - np.log(2*np.pi / n_divisions) # convert from probability mass function to probability density function so that sum(dphi*p) = 1, with dphi = (2*pi)/n_divisions

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
        logp_torsions, phis = self._torsion_log_probability_mass_function(growth_context, torsion, positions, r, theta, beta, n_divisions=n_divisions)
        phi_idx = np.argmin(np.abs(phi-phis)) # WARNING: This assumes both phi and phis have domain of [-pi,+pi)
        torsion_logp = logp_torsions[phi_idx] - np.log(2*np.pi / n_divisions) # convert from probability mass function to probability density function so that sum(dphi*p) = 1, with dphi = (2*pi)/n_divisions.
        return torsion_logp

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

class GeometrySystemGenerator(object):
    """
    This is an internal utility class that generates OpenMM systems
    with only valence terms and special parameters to assist in
    geometry proposals.
    """
    _HarmonicBondForceEnergy = "select(step({}+0.1 - growth_idx), (K/2)*(r-r0)^2, 0);"
    _HarmonicAngleForceEnergy = "select(step({}+0.1 - growth_idx), (K/2)*(theta-theta0)^2, 0);"
    _PeriodicTorsionForceEnergy = "select(step({}+0.1 - growth_idx), k*(1+cos(periodicity*theta-phase)), 0);"

    def __init__(self, reference_system, growth_indices, parameter_name, add_extra_torsions=True, add_extra_angles=True, reference_topology=None, use_sterics=False, force_names=None, force_parameters=None, verbose=True):
        """
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
        verbose : bool, optional, default=False
            If True, will print verbose output.

        """
        ONE_4PI_EPS0 = 138.935456 # OpenMM constant for Coulomb interactions (openmm/platforms/reference/include/SimTKOpenMMRealType.h) in OpenMM units
                                  # TODO: Replace this with an import from simtk.openmm.constants once these constants are available there

        default_growth_index = len(growth_indices) # default value of growth index to use in System that is returned
        self.current_growth_index = default_growth_index

        # Nonbonded sterics and electrostatics.
        # TODO: Allow user to select whether electrostatics or sterics components are included in the nonbonded interaction energy.
        self._nonbondedEnergy = "select(step({}+0.1 - growth_idx), U_sterics + U_electrostatics, 0);"
        self._nonbondedEnergy += "growth_idx = max(growth_idx1, growth_idx2);"
        # Sterics
        self._nonbondedEnergy += "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/r)^6;"
        self._nonbondedEnergy += "epsilon = sqrt(epsilon1*epsilon2); sigma = 0.5*(sigma1 + sigma2);"
        # Electrostatics
        self._nonbondedEnergy += "U_electrostatics = ONE_4PI_EPS0*charge1*charge2/r;"
        self._nonbondedEnergy += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0

        # Exceptions (always included)
        self._nonbondedExceptionEnergy = "select(step({}+0.1 - growth_idx), U_exception, 0);"
        self._nonbondedExceptionEnergy += "U_exception = ONE_4PI_EPS0*chargeprod/r + 4*epsilon*x*(x-1.0); x = (sigma/r)^6;"
        self._nonbondedExceptionEnergy += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0

        self.sterics_cutoff_distance = 9.0 * units.angstroms # cutoff for sterics

        self.verbose = verbose

        # Get list of particle indices for new and old atoms.
        new_particle_indices = [ atom.idx for atom in growth_indices ]
        old_particle_indices = [idx for idx in range(reference_system.getNumParticles()) if idx not in new_particle_indices]

        reference_forces = {reference_system.getForce(index).__class__.__name__ : reference_system.getForce(index) for index in range(reference_system.getNumForces())}
        growth_system = openmm.System()
        #create the forces:
        modified_bond_force = openmm.CustomBondForce(self._HarmonicBondForceEnergy.format(parameter_name))
        modified_bond_force.addPerBondParameter("r0")
        modified_bond_force.addPerBondParameter("K")
        modified_bond_force.addPerBondParameter("growth_idx")
        modified_bond_force.addGlobalParameter(parameter_name, default_growth_index)

        modified_angle_force = openmm.CustomAngleForce(self._HarmonicAngleForceEnergy.format(parameter_name))
        modified_angle_force.addPerAngleParameter("theta0")
        modified_angle_force.addPerAngleParameter("K")
        modified_angle_force.addPerAngleParameter("growth_idx")
        modified_angle_force.addGlobalParameter(parameter_name, default_growth_index)

        modified_torsion_force = openmm.CustomTorsionForce(self._PeriodicTorsionForceEnergy.format(parameter_name))
        modified_torsion_force.addPerTorsionParameter("periodicity")
        modified_torsion_force.addPerTorsionParameter("phase")
        modified_torsion_force.addPerTorsionParameter("k")
        modified_torsion_force.addPerTorsionParameter("growth_idx")
        modified_torsion_force.addGlobalParameter(parameter_name, default_growth_index)

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

        # Add (1,4) exceptions, regardless of whether 'use_sterics' is specified, because these are part of the valence forces.
        if 'NonbondedForce' in reference_forces.keys():
            custom_bond_force = openmm.CustomBondForce(self._nonbondedExceptionEnergy.format(parameter_name))
            custom_bond_force.addPerBondParameter("chargeprod")
            custom_bond_force.addPerBondParameter("sigma")
            custom_bond_force.addPerBondParameter("epsilon")
            custom_bond_force.addPerBondParameter("growth_idx")
            custom_bond_force.addGlobalParameter(parameter_name, default_growth_index)
            growth_system.addForce(custom_bond_force)
            # Add exclusions, which are active at all times.
            # (1,4) exceptions are always included, since they are part of the valence terms.
            #print('growth_indices:', growth_indices)
            reference_nonbonded_force = reference_forces['NonbondedForce']
            for exception_index in range(reference_nonbonded_force.getNumExceptions()):
                [particle_index_1, particle_index_2, chargeprod, sigma, epsilon] = reference_nonbonded_force.getExceptionParameters(exception_index)
                growth_idx_1 = new_particle_indices.index(particle_index_1) + 1 if particle_index_1 in new_particle_indices else 0
                growth_idx_2 = new_particle_indices.index(particle_index_2) + 1 if particle_index_2 in new_particle_indices else 0
                growth_idx = max(growth_idx_1, growth_idx_2)
                # Only need to add terms that are nonzero and involve newly added atoms.
                if (growth_idx > 0) and ((chargeprod.value_in_unit_system(units.md_unit_system) != 0.0) or (epsilon.value_in_unit_system(units.md_unit_system) != 0.0)):
                    if self.verbose: _logger.info('Adding CustomBondForce: %5d %5d : chargeprod %8.3f e^2, sigma %8.3f A, epsilon %8.3f kcal/mol, growth_idx %5d' % (particle_index_1, particle_index_2, chargeprod/units.elementary_charge**2, sigma/units.angstrom, epsilon/units.kilocalorie_per_mole, growth_idx))
                    custom_bond_force.addBond(particle_index_1, particle_index_2, [chargeprod, sigma, epsilon, growth_idx])

        #copy parameters for sterics parameters in nonbonded force
        if 'NonbondedForce' in reference_forces.keys() and use_sterics:
            modified_sterics_force = openmm.CustomNonbondedForce(self._nonbondedEnergy.format(parameter_name))
            modified_sterics_force.addPerParticleParameter("charge")
            modified_sterics_force.addPerParticleParameter("sigma")
            modified_sterics_force.addPerParticleParameter("epsilon")
            modified_sterics_force.addPerParticleParameter("growth_idx")
            modified_sterics_force.addGlobalParameter(parameter_name, default_growth_index)
            growth_system.addForce(modified_sterics_force)
            # Translate nonbonded method to cutoff methods.
            reference_nonbonded_force = reference_forces['NonbondedForce']
            if reference_nonbonded_force in [openmm.NonbondedForce.NoCutoff, openmm.NonbondedForce.CutoffNonPeriodic]:
                modified_sterics_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
            elif reference_nonbonded_force in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
                modified_sterics_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            modified_sterics_force.setCutoffDistance(self.sterics_cutoff_distance)
            # Add particle parameters.
            for particle_index in range(reference_nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = reference_nonbonded_force.getParticleParameters(particle_index)
                growth_idx = new_particle_indices.index(particle_index) + 1 if particle_index in new_particle_indices else 0
                modified_sterics_force.addParticle([charge, sigma, epsilon, growth_idx])
                if self.verbose and (growth_idx > 0):
                    _logger.info('Adding NonbondedForce particle %5d : charge %8.3f |e|, sigma %8.3f A, epsilon %8.3f kcal/mol, growth_idx %5d' % (particle_index, charge/units.elementary_charge, sigma/units.angstrom, epsilon/units.kilocalorie_per_mole, growth_idx))
            # Add exclusions, which are active at all times.
            # (1,4) exceptions are always included, since they are part of the valence terms.
            for exception_index in range(reference_nonbonded_force.getNumExceptions()):
                [particle_index_1, particle_index_2, chargeprod, sigma, epsilon] = reference_nonbonded_force.getExceptionParameters(exception_index)
                modified_sterics_force.addExclusion(particle_index_1, particle_index_2)
            # Only compute interactions of new particles with all other particles
            # TODO: Allow inteactions to be resticted to only the residue being grown.
            modified_sterics_force.addInteractionGroup(set(new_particle_indices), set(old_particle_indices))
            modified_sterics_force.addInteractionGroup(set(new_particle_indices), set(new_particle_indices))

        # Add extra ring-closing torsions, if requested.
        if add_extra_torsions:
            if reference_topology==None:
                raise ValueError("Need to specify topology in order to add extra torsions.")
            self._determine_extra_torsions(modified_torsion_force, reference_topology, growth_indices)
        if add_extra_angles:
            if reference_topology==None:
                raise ValueError("Need to specify topology in order to add extra angles")
            self._determine_extra_angles(modified_angle_force, reference_topology, growth_indices)

        # Store growth system
        self._growth_parameter_name = parameter_name
        self._growth_system = growth_system

    def set_growth_parameter_index(self, growth_parameter_index, context=None):
        """
        Set the growth parameter index
        """
        # TODO: Set default force global parameters if context is not None.
        if context is not None:
            context.setParameter(self._growth_parameter_name, growth_parameter_index)
        self.current_growth_index = growth_parameter_index

    def get_modified_system(self):
        """
        Create a modified system with parameter_name parameter. When 0, only core atoms are interacting;
        for each integer above 0, an additional atom is made interacting, with order determined by growth_index.

        Returns
        -------
        growth_system : simtk.openmm.System object
            System with the appropriate modifications, with growth parameter set to maximum.
        """
        return self._growth_system

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

        # DEBUG: Write mol2 file.
        debug = False
        if debug:
            if not hasattr(self, 'omega_index'):
                self.omega_index = 0
            filename = 'omega-%05d.mol2' % self.omega_index
            print("Writing %s" % filename)
            self.omega_index += 1
            oemol_copy = oechem.OEMol(oemol)
            ofs = oechem.oemolostream(filename)
            oechem.OETriposAtomTypeNames(oemol_copy)
            oechem.OEWriteMol2File(ofs, oemol_copy) # Preserve atom naming
            ofs.close()

        #get the omega geometry of the molecule:
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(1)
        omega.SetStrictStereo(False) #TODO: fix stereochem
        omega(oemol)

        #get the list of torsions in the molecule that are not about a rotatable bond
        # Note that only torsions involving heavy atoms are enumerated here.
        rotor = oechem.OEIsRotor()
        torsion_predicate = oechem.OENotBond(rotor)
        non_rotor_torsions = list(oechem.OEGetTorsions(oemol, torsion_predicate))
        relevant_torsion_list = self._select_torsions_without_h(non_rotor_torsions)

        #now, for each torsion, extract the set of indices and the angle
        periodicity = 1
        k = 120.0*units.kilocalories_per_mole # stddev of 12 degrees
        #print([atom.name for atom in growth_indices])
        for torsion in relevant_torsion_list:
            #make sure to get the atom index that corresponds to the topology
            atom_indices = [torsion.a.GetData("topology_index"), torsion.b.GetData("topology_index"), torsion.c.GetData("topology_index"), torsion.d.GetData("topology_index")]
            # Determine phase in [-pi,+pi) interval
            #phase = (np.pi)*units.radians+angle
            phase = torsion.radians + np.pi # TODO: Check that this is the correct convention?
            while (phase >= np.pi):
                phase -= 2*np.pi
            while (phase < -np.pi):
                phase += 2*np.pi
            phase *= units.radian
            #print('PHASE>>>> ' + str(phase)) # DEBUG
            growth_idx = self._calculate_growth_idx(atom_indices, growth_indices)
            atom_names = [torsion.a.GetName(), torsion.b.GetName(), torsion.c.GetName(), torsion.d.GetName()]
            #print("Adding torsion with atoms %s and growth index %d" %(str(atom_names), growth_idx))
            #If this is a CustomTorsionForce, we need to pass the parameters as a list, and it will have the growth_idx parameter.
            #If it's a regular PeriodicTorsionForce, there is no growth_index and the parameters are passed separately.
            if isinstance(torsion_force, openmm.CustomTorsionForce):
                torsion_force.addTorsion(atom_indices[0], atom_indices[1], atom_indices[2], atom_indices[3], [periodicity, phase, k, growth_idx])
            elif isinstance(torsion_force, openmm.PeriodicTorsionForce):
                torsion_force.addTorsion(atom_indices[0], atom_indices[1], atom_indices[2], atom_indices[3], periodicity, phase, k)
            else:
                raise ValueError("The force supplied to this method must be either a CustomTorsionForce or a PeriodicTorsionForce")

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

    def _determine_extra_angles(self, angle_force, reference_topology, growth_indices):
        """
        Determine extra angles to be placed on aromatic ring members. Sometimes,
        the native angle force is too weak to efficiently close the ring. As with the
        torsion force, this method assumes that only one residue is changing at a time.

        Parameters
        ----------
        angle_force : simtk.openmm.CustomAngleForce
            the force to which additional terms will be added
        reference_topology : simtk.openmm.app.Topology
            new/old topology if forward/backward
        growth_indices : list of parmed.atom

        Returns
        -------
        angle_force : simtk.openmm.CustomAngleForce
            The modified angle force
        """
        import itertools
        if len(growth_indices)==0:
            return
        angle_force_constant = 400.0*units.kilojoules_per_mole/units.radians**2
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
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(1)
        omega.SetStrictStereo(False) #TODO: fix stereochem
        omega(oemol)

        #we now have the residue as an oemol. Time to find the relevant angles.
        #There's no equivalent to OEGetTorsions, so first find atoms that are relevant
        #TODO: find out if that's really true
        aromatic_pred = oechem.OEIsAromaticAtom()
        heavy_pred = oechem.OEIsHeavy()
        angle_criteria = oechem.OEAndAtom(aromatic_pred, heavy_pred)

        #get all heavy aromatic atoms:
        #TODO: do this more efficiently
        heavy_aromatics = list(oemol.GetAtoms(angle_criteria))
        for atom in heavy_aromatics:
            #bonded_atoms = [bonded_atom for bonded_atom in list(atom.GetAtoms()) if bonded_atom in heavy_aromatics]
            bonded_atoms = list(atom.GetAtoms())
            for angle_atoms in itertools.combinations(bonded_atoms, 2):
                    angle = oechem.OEGetAngle(oemol, angle_atoms[0], atom, angle_atoms[1])
                    atom_indices = [angle_atoms[0].GetData("topology_index"), atom.GetData("topology_index"), angle_atoms[1].GetData("topology_index")]
                    angle_radians = angle*units.radian
                    growth_idx = self._calculate_growth_idx(atom_indices, growth_indices)
                    #If this is a CustomAngleForce, we need to pass the parameters as a list, and it will have the growth_idx parameter.
                    #If it's a regular HarmonicAngleForce, there is no growth_index and the parameters are passed separately.
                    if isinstance(angle_force, openmm.CustomAngleForce):
                        angle_force.addAngle(atom_indices[0], atom_indices[1], atom_indices[2], [angle_radians, angle_force_constant, growth_idx])
                    elif isinstance(angle_force, openmm.HarmonicAngleForce):
                        angle_force.addAngle(atom_indices[0], atom_indices[1], atom_indices[2], angle_radians, angle_force_constant)
                    else:
                        raise ValueError("Angle force must be either CustomAngleForce or HarmonicAngleForce")
        return angle_force


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

class GeometrySystemGeneratorFast(GeometrySystemGenerator):
    """
    Use updateParametersInContext to make energy evaluation fast.
    """

    def __init__(self, reference_system, growth_indices, parameter_name, add_extra_torsions=True, add_extra_angles=True, reference_topology=None, use_sterics=True, force_names=None, force_parameters=None, verbose=True):
        """
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
        verbose : bool, optional, default=False
            If True, will print verbose output.

        NB: We assume `reference_system` remains unmodified

        """
        self.sterics_cutoff_distance = 9.0 * units.angstroms # cutoff for sterics

        self.verbose = verbose

        # Get list of particle indices for new and old atoms.
        self._new_particle_indices = [ atom.idx for atom in growth_indices ]
        self._old_particle_indices = [idx for idx in range(reference_system.getNumParticles()) if idx not in self._new_particle_indices]
        self._growth_indices = growth_indices

        # Determine forces to keep
        forces_to_keep = ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']
        if use_sterics:
            forces_to_keep += ['NonbondedForce']

        # Create reference system, removing forces we won't use
        self._reference_system = copy.deepcopy(reference_system)
        force_indices_to_remove = list()
        for force_index in range(self._reference_system.getNumForces()):
            force = self._reference_system.getForce(force_index)
            force_name = force.__class__.__name__
            if force_name not in forces_to_keep:
                force_indices_to_remove.append(force_index)
        for force_index in force_indices_to_remove[::-1]:
            self._reference_system.removeForce(force_index)

        # Create new system, copying forces we will keep.
        self._growth_system = copy.deepcopy(self._reference_system)

        #Extract the forces from the system to use for adding auxiliary angles and torsions
        reference_forces = {reference_system.getForce(index).__class__.__name__ : reference_system.getForce(index) for index in range(reference_system.getNumForces())}

        # Ensure 'canonical form' of System has all parameters turned on, or else we'll run into nonbonded exceptions
        self.current_growth_index = -1
        self.set_growth_parameter_index(len(self._growth_indices))

        # Add extra ring-closing torsions, if requested.
        if add_extra_torsions:
            if reference_topology==None:
                raise ValueError("Need to specify topology in order to add extra torsions.")
            self._determine_extra_torsions(reference_forces['PeriodicTorsionForce'], reference_topology, growth_indices)
        if add_extra_angles:
            if reference_topology==None:
                raise ValueError("Need to specify topology in order to add extra angles")
            self._determine_extra_angles(reference_forces['HarmonicAngleForce'], reference_topology, growth_indices)
        # TODO: Precompute growth indices for force terms for speed

    def set_growth_parameter_index(self, growth_index, context=None):
        """
        Set the growth parameter index
        """
        self.current_growth_index = growth_index
        for (growth_force, reference_force) in zip(self._growth_system.getForces(), self._reference_system.getForces()):
            force_name = growth_force.__class__.__name__
            if (force_name == 'HarmonicBondForce'):
                for bond in range(reference_force.getNumBonds()):
                    parameters = reference_force.getBondParameters(bond)
                    this_growth_index = self._calculate_growth_idx(parameters[:2], self._growth_indices)
                    if (growth_index < this_growth_index):
                        parameters[3] *= 0.0
                    growth_force.setBondParameters(bond, *parameters)
            elif (force_name == 'HarmonicAngleForce'):
                for angle in range(reference_force.getNumAngles()):
                    parameters = reference_force.getAngleParameters(angle)
                    this_growth_index = self._calculate_growth_idx(parameters[:3], self._growth_indices)
                    if (growth_index < this_growth_index):
                        parameters[4] *= 0.0
                    growth_force.setAngleParameters(angle, *parameters)
            elif (force_name == 'PeriodicTorsionForce'):
                for torsion in range(reference_force.getNumTorsions()):
                    parameters = reference_force.getTorsionParameters(torsion)
                    this_growth_index = self._calculate_growth_idx(parameters[:4], self._growth_indices)
                    if (growth_index < this_growth_index):
                        parameters[6] *= 0.0
                    growth_force.setTorsionParameters(torsion, *parameters)
            elif (force_name == 'NonbondedForce'):
                for particle_index in range(reference_force.getNumParticles()):
                    parameters = reference_force.getParticleParameters(particle_index)
                    this_growth_index = self._calculate_growth_idx([particle_index], self._growth_indices)
                    if (growth_index < this_growth_index):
                        parameters[0] *= 0.0
                        parameters[2] *= 0.0
                    growth_force.setParticleParameters(particle_index, *parameters)
                for exception_index in range(reference_force.getNumExceptions()):
                    parameters = reference_force.getExceptionParameters(exception_index)
                    this_growth_index = self._calculate_growth_idx(parameters[:2], self._growth_indices)
                    if (growth_index < this_growth_index):
                        parameters[2] *= 1.0e-6 # WORKAROUND // TODO: Change to zero when OpenMM issue is fixed
                        parameters[4] *= 1.0e-6 # WORKAROUND // TODO: Change to zero when OpenMM issue is fixed
                    growth_force.setExceptionParameters(exception_index, *parameters)

            # Update parameters in context
            if context is not None:
                growth_force.updateParametersInContext(context)

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

    Hydrogens are added last in growth order.

    Parameters
    ----------
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        The topology proposal containing the relevant move.
    """

    def __init__(self, topology_proposal, verbose=True):
        self._topology_proposal = topology_proposal
        self.verbose = True # DEBUG

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
        if direction=='forward':
            topology = self._topology_proposal.new_topology
            system = self._topology_proposal.new_system
            structure = parmed.openmm.load_topology(self._topology_proposal.new_topology, self._topology_proposal.new_system)
            unique_atoms = self._topology_proposal.unique_new_atoms
            #atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in range(self._topology_proposal.n_atoms_new) if atom_idx not in self._topology_proposal.unique_new_atoms]
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in self._topology_proposal.new_to_old_atom_map.keys()]
        elif direction=='reverse':
            topology = self._topology_proposal.old_topology
            system = self._topology_proposal.old_system
            structure = parmed.openmm.load_topology(self._topology_proposal.old_topology, self._topology_proposal.old_system)
            unique_atoms = self._topology_proposal.unique_old_atoms
            atoms_with_positions = [structure.atoms[atom_idx] for atom_idx in self._topology_proposal.old_to_new_atom_map.keys()]
        else:
            raise ValueError("direction parameter must be either forward or reverse.")

        # Determine list of atoms to be added.
        new_hydrogen_atoms = [ structure.atoms[idx] for idx in unique_atoms if structure.atoms[idx].atomic_number == 1 ]
        new_heavy_atoms    = [ structure.atoms[idx] for idx in unique_atoms if structure.atoms[idx].atomic_number != 1 ]

        # DEBUG
        #print('STRUCTURE')
        #print(structure)
        #for atom in structure.atoms:
        #    print(atom, atom.bonds, atom.angles, atom.dihedrals)
        #print('')

        def add_atoms(new_atoms, atoms_torsions):
            """
            Add the specified atoms to the ordered list of torsions to be drawn.

            Parameters
            ----------
            new_atoms : list
                List of atoms to be added.
            atoms_torsions : OrderedDict
                List of torsions to be added.

            Returns
            -------
            logp_torsion_choice : float
                The log torsion cchoice probability associated with these added torsions.

            """
            from scipy import special
            logp_torsion_choice = 0.0
            while (len(new_atoms)) > 0:
                #print("atoms left: ", len(new_atoms)) ## IVY
                #print("atom order: ", atoms_with_positions) ## IVY
                eligible_atoms = self._atoms_eligible_for_proposal(new_atoms, atoms_with_positions)
                #print("eligible atoms: ", eligible_atoms) ## IVY

                #randomize positions
                eligible_atoms_in_order = np.random.choice(eligible_atoms, size=len(eligible_atoms), replace=False)

                #the logp of this choice is log(1/n!)
                #gamma is (n-1)!, log-gamma is more numerically stable.
                logp_torsion_choice += -special.gammaln(len(eligible_atoms)+1)

                if (len(new_atoms) > 0) and (len(eligible_atoms) == 0):
                    raise Exception('new_atoms (%s) has remaining atoms to place, but eligible_atoms is empty.' % str(new_atoms))

                #choose the torsions
                for atom in eligible_atoms_in_order:
                    chosen_torsion, logp_choice = self._choose_torsion(atoms_with_positions, atom)
                    atoms_torsions[atom] = chosen_torsion
                    logp_torsion_choice += logp_choice
                    new_atoms.remove(atom)
                    atoms_with_positions.append(atom)

            return logp_torsion_choice

        # Handle heavy atoms before hydrogen atoms
        logp_torsion_choice = 0.0
        atoms_torsions = collections.OrderedDict()
        #print("adding heavy atoms") ##IVY
        logp_torsion_choice += add_atoms(new_heavy_atoms, atoms_torsions)
        #print("adding hydrogen atoms") ## IVY
        logp_torsion_choice += add_atoms(new_hydrogen_atoms, atoms_torsions)

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
            #get all topological torsions for the appropriate atom
            torsions = self._get_topological_torsions(atoms_with_positions, atom)

            #go through the topological torsions (atom1 is always the new atom), and if one of them has
            #atoms 2, 3, 4 in atoms_with_positions, the atom is eligible.
            for torsion in torsions:
                if torsion.atom2 in atoms_with_positions and torsion.atom3 in atoms_with_positions and torsion.atom4 in atoms_with_positions:
                    eligible_atoms.append(atom)

        return list(set(eligible_atoms))

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
        # Compute topological torsions beginning with atom `new_atom` in which all other atoms have positions
        topological_torsions = list()
        atom1 = new_atom
        for bond12 in atom1.bonds:
            atom2 = bond12.atom2 if bond12.atom1==atom1 else bond12.atom1
            if atom2 not in atoms_with_positions:
                continue
            for bond23 in atom2.bonds:
                atom3 = bond23.atom2 if bond23.atom1==atom2 else bond23.atom1
                if (atom3 not in atoms_with_positions) or (atom3 in set([atom1, atom2])):
                    continue
                for bond34 in atom3.bonds:
                    atom4 = bond34.atom2 if bond34.atom1==atom3 else bond34.atom1
                    if (atom4 not in atoms_with_positions) or (atom4 in set([atom1, atom2, atom3])):
                        continue
                    topological_torsions.append((atom1, atom2, atom3, atom4))

        if len(topological_torsions) == 0:
            # Print debug information
            _logger.debug('No topological torsions found!')
            _logger.debug('atoms_with_positions: %s' % str(atoms_with_positions))
            _logger.debug('new_atom: %s' % new_atom)
            _logger.debug('bonds involving new atom:')
            _logger.debug(new_atom.bonds)
            _logger.debug('angles involving new atom:')
            _logger.debug(new_atom.angles)
            _logger.debug('dihedrals involving new atom:')
            _logger.debug(new_atom.dihedrals)

        # Recode topological torsions as parmed Dihedral objects
        topological_torsions = [ parmed.Dihedral(atoms[0], atoms[1], atoms[2], atoms[3]) for atoms in topological_torsions ]
        return topological_torsions


class NoTorsionError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(NoTorsionError, self).__init__(message)

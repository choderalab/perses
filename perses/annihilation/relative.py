import simtk.openmm as openmm
import simtk.unit as unit
import mdtraj as md
import numpy as np
import copy
import enum

InteractionGroup = enum.Enum("InteractionGroup", ['unique_old', 'unique_new', 'core', 'environment'])

#######LOGGING#############################
import logging
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("relative")
_logger.setLevel(logging.INFO)
###########################################

class HybridTopologyFactory(object):
    """
    This class generates a hybrid topology based on a perses topology proposal. This class treats atoms
    in the resulting hybrid system as being from one of four classes:

    unique_old_atom : these atoms are not mapped and only present in the old system. Their interactions will be on for
        lambda=0, off for lambda=1
    unique_new_atom : these atoms are not mapped and only present in the new system. Their interactions will be off
        for lambda=0, on for lambda=1
    core_atom : these atoms are mapped, and are part of a residue that is changing. Their interactions will be those
        corresponding to the old system at lambda=0, and those corresponding to the new system at lambda=1
    environment_atom : these atoms are mapped, and are not part of a changing residue. Their interactions are always
        on and are alchemically unmodified.

    Properties
    ----------
    hybrid_system : openmm.System
        The hybrid system for simulation
    new_to_hybrid_atom_map : dict of int : int
        The mapping of new system atoms to hybrid atoms
    old_to_hybrid_atom_map : dict of int : int
        The mapping of old system atoms to hybrid atoms
    hybrid_positions : [n, 3] np.ndarray
        The positions of the hybrid system
    hybrid_topology : mdtraj.Topology
        The topology of the hybrid system
    omm_hybrid_topology : openmm.app.Topology
        The OpenMM topology object corresponding to the hybrid system

    .. warning :: This API is experimental and subject to change.

    """

    _known_forces = {'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce', 'MonteCarloBarostat'}

    def __init__(self,
                 topology_proposal,
                 current_positions,
                 new_positions,
                 use_dispersion_correction=False,
                 functions=None,
                 softcore_alpha=None,
                 bond_softening_constant=1.0,
                 angle_softening_constant=1.0,
                 soften_only_new=False,
                 neglected_new_angle_terms = [],
                 neglected_old_angle_terms = [],
                 softcore_LJ_v2 = True,
                 softcore_electrostatics = True,
                 softcore_LJ_v2_alpha = 0.85,
                 softcore_electrostatics_alpha = 0.3,
                 softcore_sigma_Q = 1.0,
                 interpolate_old_and_new_14s = False,
                 omitted_terms = None):
        """
        Initialize the Hybrid topology factory.

        Parameters
        ----------
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal object
            TopologyProposal object rendered by the ProposalEngine
        current_positions : [n,3] np.ndarray of float
            The positions of the "old system"
        new_positions : [m,3] np.ndarray of float
            The positions of the "new system"
        use_dispersion_correction : bool, default False
            Whether to use the long range correction in the custom sterics force. This is very expensive for NCMC.
        functions : dict, default None
            Alchemical functions that determine how each force is scaled with lambda. The keys must be strings with
            names beginning with ``lambda_`` and ending with each of bonds, angles, torsions, sterics, electrostatics.
            If functions is none, then the integrator will need to set each of these and parameter derivatives will be unavailable.
            If functions is not None, all lambdas must be specified.
        softcore_alpha: float, default None
            "alpha" parameter of softcore sterics. If None is provided, value will be set to 0.5
        bond_softening_constant : float
            For bonds between unique atoms and unique-core atoms, soften the force constant at the "dummy" endpoint by this factor.
            If 1.0, do not soften
        angle_softening_constant : float
            For bonds between unique atoms and unique-core atoms, soften the force constant at the "dummy" endpoint by this factor.
            If 1.0, do not soften
        neglected_new_angle_terms : list
            list of indices from the HarmonicAngleForce of the new_system for which the geometry engine neglected.
            Hence, these angles must be alchemically grown in for the unique new atoms (forward lambda protocol)
        neglected_old_angle_terms : list
            list of indices from the HarmonicAngleForce of the old_system for which the geometry engine neglected.
            Hence, these angles must be alchemically deleted for the unique old atoms (reverse lambda protocol)
        softcore_LJ_v2 : bool, default True
            implement a new softcore LJ: citation below.
            Gapsys, Vytautas, Daniel Seeliger, and Bert L. de Groot. "New soft-core potential function for molecular dynamics based alchemical free energy calculations." Journal of chemical theory and computation 8.7 (2012): 2373-2382.
        softcore_electrostatics : bool, default True
            softcore electrostatics: citation below.
            Gapsys, Vytautas, Daniel Seeliger, and Bert L. de Groot. "New soft-core potential function for molecular dynamics based alchemical free energy calculations." Journal of chemical theory and computation 8.7 (2012): 2373-2382.
        softcore_LJ_v2_alpha : float, default 0.85
            softcore alpha parameter for LJ v2
        softcore_electrostatics_alpha : float, default 0.3
            softcore alpha parameter for softcore electrostatics.
        softcore_sigma_Q : float, default 1.0
            softcore sigma parameter for softcore electrostatics.
        interpolate_old_and_new_14s : bool, default False
            whether to turn on new 1,4 interactions and turn off old 1,4 interactions; if False, they are present in the nonbonded force
        omitted_terms : dict
            dictionary of terms (by new topology index) that must be annealed in over a lambda protocol

        TODO: Document how positions for hybrid system are constructed
        TODO: allow support for annealing in omitted terms

        """
        _logger.info("Beginning nonbonded method, total particle, barostat, and exceptions retrieval...")
        self._topology_proposal = topology_proposal
        self._old_system = copy.deepcopy(topology_proposal.old_system)
        self._new_system = copy.deepcopy(topology_proposal.new_system)
        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}
        self._hybrid_system_forces = dict()
        self._old_positions = current_positions
        self._new_positions = new_positions
        self._soften_only_new = soften_only_new
        self._interpolate_14s = interpolate_old_and_new_14s
        self.omitted_terms = omitted_terms

        if omitted_terms is not None:
            raise Exception(f"annealing of omitted terms is not currently supported.  Aborting!")

        # New attributes from the modified geometry engine
        if neglected_old_angle_terms:
            self.neglected_old_angle_terms = neglected_old_angle_terms
        else:
            self.neglected_old_angle_terms = []

        if neglected_new_angle_terms:
            self.neglected_new_angle_terms = neglected_new_angle_terms
        else:
            self.neglected_new_angle_terms = []

        if bond_softening_constant != 1.0:
            self._bond_softening_constant = bond_softening_constant
            self._soften_bonds = True
        else:
            self._soften_bonds = False

        if angle_softening_constant != 1.0:
            self._angle_softening_constant = angle_softening_constant
            self._soften_angles = True
        else:
            self._soften_angles = False

        self._use_dispersion_correction = use_dispersion_correction

        self._softcore_LJ_v2 = softcore_LJ_v2
        if self._softcore_LJ_v2:
            self._softcore_LJ_v2_alpha = softcore_LJ_v2_alpha
            assert self._softcore_LJ_v2_alpha >= 0.0 and self._softcore_LJ_v2_alpha <= 1.0, f"softcore_LJ_v2_alpha: ({self._softcore_LJ_v2_alpha}) is not in [0,1]"

        self._softcore_electrostatics = softcore_electrostatics
        if self._softcore_electrostatics:
            self._softcore_electrostatics_alpha = softcore_electrostatics_alpha
            self._softcore_sigma_Q = softcore_sigma_Q
            assert self._softcore_electrostatics_alpha >= 0.0 and self._softcore_electrostatics_alpha <= 1.0, f"softcore_electrostatics_alpha: ({self._softcore_electrostatics_alpha}) is not in [0,1]"
            assert self._softcore_sigma_Q >= 0.0 and self._softcore_sigma_Q <= 1.0, f"softcore_sigma_Q : {self._softcore_sigma_Q} is not in [0, 1]"


        if softcore_alpha is None:
            self.softcore_alpha = 0.5
        else:
            # TODO: Check that softcore_alpha is in a valid range
            self.softcore_alpha = softcore_alpha


        if functions:
            self._functions = functions
            self._has_functions = True
        else:
            self._has_functions = False

        # Prepare dicts of forces, which will be useful later
        # TODO: Store this as self._system_forces[name], name in ('old', 'new', 'hybrid') for compactness
        self._old_system_forces = {type(force).__name__ : force for force in self._old_system.getForces()}
        self._new_system_forces = {type(force).__name__ : force for force in self._new_system.getForces()}
        _logger.info(f"Old system forces: {self._old_system_forces.keys()}")
        _logger.info(f"New system forces: {self._new_system_forces.keys()}")

        # Check that there are no unknown forces in the new and old systems:
        for system_name in ('old', 'new'):
            force_names = getattr(self, '_{}_system_forces'.format(system_name)).keys()
            unknown_forces = set(force_names) - set(self._known_forces)
            if len(unknown_forces) > 0:
                raise ValueError("Unkown forces {} encountered in {} system" % (unknown_forces, system_name))
        _logger.info("No unknown forces.")

        # Get and store the nonbonded method from the system:
        self._nonbonded_method = self._old_system_forces['NonbondedForce'].getNonbondedMethod()
        _logger.info(f"Nonbonded method to be used (i.e. from old system): {self._nonbonded_method}")

        # Start by creating an empty system. This will become the hybrid system.
        self._hybrid_system = openmm.System()

        # Begin by copying all particles in the old system to the hybrid system. Note that this does not copy the
        # interactions. It does, however, copy the particle masses. In general, hybrid index and old index should be
        # the same.
        # TODO: Refactor this into self._add_particles()
        _logger.info("Adding and mapping old atoms to hybrid system...")
        for particle_idx in range(self._topology_proposal.n_atoms_old):
            particle_mass = self._old_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._old_to_hybrid_map[particle_idx] = hybrid_idx

            # If the particle index in question is mapped, make sure to add it to the new to hybrid map as well.
            if particle_idx in self._topology_proposal.old_to_new_atom_map.keys():
                particle_index_in_new_system = self._topology_proposal.old_to_new_atom_map[particle_idx]
                self._new_to_hybrid_map[particle_index_in_new_system] = hybrid_idx

        # Next, add the remaining unique atoms from the new system to the hybrid system and map accordingly.
        # As before, this does not copy interactions, only particle indices and masses.
        _logger.info("Adding and mapping new atoms to hybrid system...")
        for particle_idx in self._topology_proposal.unique_new_atoms:
            particle_mass = self._new_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._new_to_hybrid_map[particle_idx] = hybrid_idx

        # Check that if there is a barostat in the original system, it is added to the hybrid.
        # We copy the barostat from the old system.
        if "MonteCarloBarostat" in self._old_system_forces.keys():
            barostat = copy.deepcopy(self._old_system_forces["MonteCarloBarostat"])
            self._hybrid_system.addForce(barostat)
            _logger.info("Added MonteCarloBarostat.")
        else:
            _logger.info("No MonteCarloBarostat added.")

        # Copy over the box vectors:
        box_vectors = self._old_system.getDefaultPeriodicBoxVectors()
        self._hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)
        _logger.info(f"getDefaultPeriodicBoxVectors added to hybrid: {box_vectors}")

        # Create the opposite atom maps for use in nonbonded force processing; let's omit this from logger
        self._hybrid_to_old_map = {value : key for key, value in self._old_to_hybrid_map.items()}
        self._hybrid_to_new_map = {value : key for key, value in self._new_to_hybrid_map.items()}

        # Assign atoms to one of the classes described in the class docstring
        self._atom_classes = self._determine_atom_classes()
        _logger.info("Determined atom classes.")

        # Construct dictionary of exceptions in old and new systems
        _logger.info("Generating old system exceptions dict...")
        self._old_system_exceptions = self._generate_dict_from_exceptions(self._old_system_forces['NonbondedForce'])
        _logger.info("Generating new system exceptions dict...")
        self._new_system_exceptions = self._generate_dict_from_exceptions(self._new_system_forces['NonbondedForce'])

        self._validate_disjoint_sets()

        # Copy constraints, checking to make sure they are not changing
        _logger.info("Handling constraints...")
        self._handle_constraints()

        # Copy over relevant virtual sites
        _logger.info("Handling virtual sites...")
        self._handle_virtual_sites()

        # Call each of the methods to add the corresponding force terms and prepare the forces:
        _logger.info("Adding bond force terms...")
        self._add_bond_force_terms()

        _logger.info("Adding angle force terms...")
        self._add_angle_force_terms()

        _logger.info("Adding torsion force terms...")
        self._add_torsion_force_terms()

        if 'NonbondedForce' in self._old_system_forces or 'NonbondedForce' in self._new_system_forces:
            _logger.info("Adding nonbonded force terms...")
            self._add_nonbonded_force_terms()

        # Call each force preparation method to generate the actual interactions that we need:
        _logger.info("Handling harmonic bonds...")
        self.handle_harmonic_bonds()

        _logger.info("Handling harmonic angles...")
        self.handle_harmonic_angles()

        _logger.info("Handling torsion forces...")
        self.handle_periodic_torsion_force()

        if 'NonbondedForce' in self._old_system_forces or 'NonbondedForce' in self._new_system_forces:
            _logger.info("Handling nonbonded forces...")
            self.handle_nonbonded()

        if 'NonbondedForce' in self._old_system_forces or 'NonbondedForce' in self._new_system_forces:
            _logger.info("Handling unique_new/old interaction exceptions...")
            if len(self._old_system_exceptions.keys()) == 0 and len(self._new_system_exceptions.keys()) == 0:
                _logger.info("There are no old/new system exceptions.")
            else:
                _logger.info("There are old or new system exceptions...proceeding.")
                self.handle_old_new_exceptions()


        # Get positions for the hybrid
        self._hybrid_positions = self._compute_hybrid_positions()

        # Generate the topology representation
        self._hybrid_topology = self._create_topology()

    def _validate_disjoint_sets(self):
        """
        Conduct a sanity check to make sure that the hybrid maps of the old and new system exception dict keys do not contain both environment and unique_old/new atoms

        """
        for old_indices in self._old_system_exceptions.keys():
            hybrid_indices = (self._old_to_hybrid_map[old_indices[0]], self._old_to_hybrid_map[old_indices[1]])
            if set(old_indices).intersection(self._atom_classes['environment_atoms']) != set():
                assert set(old_indices).intersection(self._atom_classes['unique_old_atoms']) == set(), f"old index exceptions {old_indices} include unique old and environment atoms, which is disallowed"

        for new_indices in self._new_system_exceptions.keys():
            hybrid_indices = (self._new_to_hybrid_map[new_indices[0]], self._new_to_hybrid_map[new_indices[1]])
            if set(hybrid_indices).intersection(self._atom_classes['environment_atoms']) != set():
                assert set(hybrid_indices).intersection(self._atom_classes['unique_new_atoms']) == set(), f"new index exceptions {new_indices} include unique new and environment atoms, which is disallowed"

    def _handle_virtual_sites(self):
        """
        Ensure that all virtual sites in old and new system are copied over to the hybrid system. Note that we do not
        support virtual sites in the changing region.

        """
        for system_name in ('old', 'new'):
            system = getattr(self._topology_proposal, '{}_system'.format(system_name))
            hybrid_atom_map = getattr(self, '_{}_to_hybrid_map'.format(system_name))

            # Loop through virtual sites
            numVirtualSites = 0
            for particle_idx in range(system.getNumParticles()):
                if system.isVirtualSite(particle_idx):
                    numVirtualSites += 1
                    # If it's a virtual site, make sure it is not in the unique or core atoms, since this is currently unsupported
                    hybrid_idx = hybrid_atom_map[particle_idx]
                    if hybrid_idx not in self._atom_classes['environment_atoms']:
                        raise Exception("Virtual sites in changing residue are unsupported.")
                    else:
                        virtual_site = system.getVirtualSite(particle_idx)
                        self._hybrid_system.setVirtualSite(hybrid_idx, virtual_site)
        _logger.info(f"\t_handle_virtual_sites: numVirtualSites: {numVirtualSites}")


    def _determine_core_atoms_in_topology(self, topology, unique_atoms, mapped_atoms, hybrid_map, residue_to_switch):
        """
        Given a topology and its corresponding unique and mapped atoms, return the set of atom indices in the
        hybrid system which would belong to the "core" atom class

        Parameters
        ----------
        topology : simtk.openmm.app.Topology
            An OpenMM topology representing a system of interest
        unique_atoms : set of int
            A set of atoms that are unique to this topology
        mapped_atoms : set of int
            A set of atoms that are mapped to another topology
        residue_to_switch : str
            string name of a residue that is being mutated

        Returns
        -------
        core_atoms : set of int
            set of core atom indices in hybrid topology
        """
        core_atoms = set()

        # Loop through the residues to look for ones with unique atoms
        for residue in topology.residues():
            atom_indices_old_system = {atom.index for atom in residue.atoms()}

            # If the residue contains an atom index that is unique, then the residue is changing.
            # We determine this by checking if the atom indices of the residue have any intersection with the unique atoms
            # likewise, if the name of the residue matches the residue_to_match, then we look for mapped atoms
            if len(atom_indices_old_system.intersection(unique_atoms)) > 0 or residue_to_switch == residue.name:
                # We can add the atoms in this residue which are mapped to the core_atoms set:
                for atom_index in atom_indices_old_system:
                    if atom_index in mapped_atoms:
                        # We specifically want to add the hybrid atom.
                        hybrid_index = hybrid_map[atom_index]
                        core_atoms.add(hybrid_index)

        assert len(core_atoms) >= 3, 'Cannot run a simulation with fewer than 3 core atoms. System has {len(core_atoms)}'

        return core_atoms

    def _determine_atom_classes(self):
        """
        This method determines whether each atom belongs to unique old, unique new, core, or environment, as defined above.
        All the information required is contained in the TopologyProposal passed to the constructor. All indices are
        indices in the hybrid system.

        Returns
        -------
        atom_classes : dict of list
            A dictionary of the form {'core' :core_list} etc.
        """
        atom_classes = {'unique_old_atoms' : set(), 'unique_new_atoms' : set(), 'core_atoms' : set(), 'environment_atoms' : set()}

        # First, find the unique old atoms, as this is the most straightforward:
        for atom_idx in self._topology_proposal.unique_old_atoms:
            hybrid_idx = self._old_to_hybrid_map[atom_idx]
            atom_classes['unique_old_atoms'].add(hybrid_idx)

        # Then the unique new atoms (this is substantially the same as above)
        for atom_idx in self._topology_proposal.unique_new_atoms:
            hybrid_idx = self._new_to_hybrid_map[atom_idx]
            atom_classes['unique_new_atoms'].add(hybrid_idx)

        # The core atoms:
        core_atoms = []
        for new_idx, old_idx in self._topology_proposal._core_new_to_old_atom_map.items():
            new_to_hybrid_idx, old_to_hybrid_index = self._new_to_hybrid_map[new_idx], self._old_to_hybrid_map[old_idx]
            assert new_to_hybrid_idx == old_to_hybrid_index, f"there is a -to_hybrid naming collision in topology proposal core atom map: {self._topology_proposal._core_new_to_old_atom_map}"
            core_atoms.append(new_to_hybrid_idx)


        new_to_hybrid_environment_atoms = set([self._new_to_hybrid_map[idx] for idx in self._topology_proposal._new_environment_atoms])
        old_to_hybrid_environment_atoms = set([self._old_to_hybrid_map[idx] for idx in self._topology_proposal._old_environment_atoms])
        assert new_to_hybrid_environment_atoms == old_to_hybrid_environment_atoms, f"there is a -to_hybrid naming collisions in topology proposal environment atom map: new_to_hybrid: {new_to_hybrid_environment_atoms}; old_to_hybrid: {old_to_hybrid_environment_atoms}"


        atom_classes['core_atoms'] = set(core_atoms)
        atom_classes['environment_atoms'] = new_to_hybrid_environment_atoms # since we asserted this is identical to old_to_hybrid_environment_atoms

        return atom_classes

    def _translate_nonbonded_method_to_custom(self, standard_nonbonded_method):
        """
        Utility function to translate the nonbonded method enum from the standard nonbonded force to the custom version
        `CutoffPeriodic`, `PME`, and `Ewald` all become `CutoffPeriodic`; `NoCutoff` becomes `NoCutoff`; `CutoffNonPeriodic` becomes `CutoffNonPeriodic`

        Parameters
        ----------
        standard_nonbonded_method : openmm.NonbondedForce.NonbondedMethod
            the nonbonded method of the standard force

        Returns
        -------
        custom_nonbonded_method : openmm.CustomNonbondedForce.NonbondedMethod
            the nonbonded method for the equivalent customnonbonded force
        """
        if standard_nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            return openmm.CustomNonbondedForce.CutoffPeriodic
        elif standard_nonbonded_method == openmm.NonbondedForce.NoCutoff:
            return openmm.CustomNonbondedForce.NoCutoff
        elif standard_nonbonded_method == openmm.NonbondedForce.CutoffNonPeriodic:
            return openmm.CustomNonbondedForce.CutoffNonPeriodic
        else:
            raise NotImplementedError("This nonbonded method is not supported.")

    def _handle_constraints(self):
        """
        This method adds relevant constraints from the old and new systems.

        First, all constraints from the old systenm are added.
        Then, constraints to atoms unique to the new system are added.

        """
        constraint_lengths = dict() # lengths of constraints already added
        for system_name in ('old', 'new'):
            system = getattr(self._topology_proposal, '{}_system'.format(system_name))
            hybrid_map = getattr(self, '_{}_to_hybrid_map'.format(system_name))
            for constraint_idx in range(system.getNumConstraints()):
                atom1, atom2, length = system.getConstraintParameters(constraint_idx)
                hybrid_atoms = tuple(sorted([hybrid_map[atom1], hybrid_map[atom2]]))
                if hybrid_atoms not in constraint_lengths.keys():
                    self._hybrid_system.addConstraint(hybrid_atoms[0], hybrid_atoms[1], length)
                    constraint_lengths[hybrid_atoms] = length
                else:
                    # TODO: We can skip this if we have already checked for constraints changing lengths
                    if constraint_lengths[hybrid_atoms] != length:
                        raise Exception('Constraint length is changing for atoms {} in hybrid system: old {} new {}'.format(hybrid_atoms, constraint_lengths[hybrid_atoms], length))
        _logger.debug(f"\t_handle_constraints: constraint_lengths dict: {constraint_lengths}")

    def _determine_interaction_group(self, atoms_in_interaction):
        """
        This method determines which interaction group the interaction should fall under. There are four groups:

        Those involving unique old atoms: any interaction involving unique old atoms should be completely on at lambda=0
            and completely off at lambda=1

        Those involving unique new atoms: any interaction involving unique new atoms should be completely off at lambda=0
            and completely on at lambda=1

        Those involving core atoms and/or environment atoms: These interactions change their type, and should be the old
            character at lambda=0, and the new character at lambda=1

        Those involving only environment atoms: These interactions are unmodified.

        Parameters
        ----------
        atoms_in_interaction : list of int
            List of (hybrid) indices of the atoms in this interaction

        Returns
        -------
        interaction_group : InteractionGroup enum
            The group to which this interaction should be assigned
        """
        # Make the interaction list a set to facilitate operations
        atom_interaction_set = set(atoms_in_interaction)

        # Check if the interaction contains unique old atoms
        if len(atom_interaction_set.intersection(self._atom_classes['unique_old_atoms'])) > 0:
            return InteractionGroup.unique_old

        # Do the same for new atoms
        elif len(atom_interaction_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
            return InteractionGroup.unique_new

        # If the interaction set is a strict subset of the environment atoms, then it is in the environment group
        # and should not be alchemically modified at all.
        elif atom_interaction_set.issubset(self._atom_classes['environment_atoms']):
            return InteractionGroup.environment

        # Having covered the cases of all-environment, unique old-containing, and unique-new-containing, anything else
        # should belong to the last class--contains core atoms but not any unique atoms.
        else:
            return InteractionGroup.core

    def _add_bond_force_terms(self):
        """
        This function adds the appropriate bond forces to the system (according to groups defined above). Note that it
        does _not_ add the particles to the force. It only adds the force to facilitate another method adding the
        particles to the force.

        """
        core_energy_expression = '(K/2)*(r-length)^2;'
        core_energy_expression += 'K = (1-lambda_bonds)*K1 + lambda_bonds*K2;' # linearly interpolate spring constant
        core_energy_expression += 'length = (1-lambda_bonds)*length1 + lambda_bonds*length2;' # linearly interpolate bond length
        if self._has_functions:
            try:
                core_energy_expression += 'lambda_bonds = ' + self._functions['lambda_bonds']
            except KeyError as e:
                print("Functions were provided, but no term was provided for the bonds")
                raise e

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomBondForce(core_energy_expression)
        custom_core_force.addPerBondParameter('length1') # old bond length
        custom_core_force.addPerBondParameter('K1') # old spring constant
        custom_core_force.addPerBondParameter('length2') # new bond length
        custom_core_force.addPerBondParameter('K2') # new spring constant

        if self._has_functions:
            custom_core_force.addGlobalParameter('lambda', 0.0)
            custom_core_force.addEnergyParameterDerivative('lambda')
        else:
            custom_core_force.addGlobalParameter('lambda_bonds', 0.0)

        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['core_bond_force'] = custom_core_force

        # Add a bond force for environment and unique atoms (bonds are never scaled for these):
        standard_bond_force = openmm.HarmonicBondForce()
        self._hybrid_system.addForce(standard_bond_force)
        self._hybrid_system_forces['standard_bond_force'] = standard_bond_force

    def _add_angle_force_terms(self):
        """
        This function adds the appropriate angle force terms to the hybrid system. It does not add particles
        or parameters to the force; this is done elsewhere.
        """
        energy_expression  = '(K/2)*(theta-theta0)^2;'
        energy_expression += 'K = (1.0-lambda_angles)*K_1 + lambda_angles*K_2;' # linearly interpolate spring constant
        energy_expression += 'theta0 = (1.0-lambda_angles)*theta0_1 + lambda_angles*theta0_2;' # linearly interpolate equilibrium angle
        if self._has_functions:
            try:
                energy_expression += 'lambda_angles = ' + self._functions['lambda_angles']
            except KeyError as e:
                print("Functions were provided, but no term was provided for the angles")
                raise e

        # Create the force and add relevant parameters
        custom_core_force = openmm.CustomAngleForce(energy_expression)
        custom_core_force.addPerAngleParameter('theta0_1') # molecule1 equilibrium angle
        custom_core_force.addPerAngleParameter('K_1') # molecule1 spring constant
        custom_core_force.addPerAngleParameter('theta0_2') # molecule2 equilibrium angle
        custom_core_force.addPerAngleParameter('K_2') # molecule2 spring constant

        # Create the force for neglected angles and relevant parameters; the K_1 term will be set to 0
        if len(self.neglected_new_angle_terms) > 0: # if there is at least one neglected angle term from the geometry engine
            _logger.info("\t_add_angle_force_terms: there are > 0 neglected new angles: adding CustomAngleForce")
            custom_neglected_new_force = openmm.CustomAngleForce(energy_expression)
            custom_neglected_new_force.addPerAngleParameter('theta0_1') # molecule1 equilibrium angle
            custom_neglected_new_force.addPerAngleParameter('K_1') # molecule1 spring constant
            custom_neglected_new_force.addPerAngleParameter('theta0_2') # molecule2 equilibrium angle
            custom_neglected_new_force.addPerAngleParameter('K_2') # molecule2 spring constant
        if len(self.neglected_old_angle_terms) > 0: # if there is at least one neglected angle term from the geometry engine
            _logger.info("\t_add_angle_force_terms: there are > 0 neglected old angles: adding CustomAngleForce")
            custom_neglected_old_force = openmm.CustomAngleForce(energy_expression)
            custom_neglected_old_force.addPerAngleParameter('theta0_1') # molecule1 equilibrium angle
            custom_neglected_old_force.addPerAngleParameter('K_1') # molecule1 spring constant
            custom_neglected_old_force.addPerAngleParameter('theta0_2') # molecule2 equilibrium angle
            custom_neglected_old_force.addPerAngleParameter('K_2') # molecule2 spring constant

        if self._has_functions:
            custom_core_force.addGlobalParameter('lambda', 0.0)
            custom_core_force.addEnergyParameterDerivative('lambda')
            if len(self.neglected_new_angle_terms) > 0:
                custom_neglected_new_force.addGlobalParameter('lambda', 0.0)
                custom_neglected_new_force.addEnergyParameterDerivative('lambda')
            if len(self.neglected_old_angle_terms) > 0:
                custom_neglected_old_force.addGlobalParameter('lambda', 0.0)
                custom_neglected_old_force.addEnergyParameterDerivative('lambda')
        else:
            custom_core_force.addGlobalParameter('lambda_angles', 0.0)
            if len(self.neglected_new_angle_terms) > 0:
                custom_neglected_new_force.addGlobalParameter('lambda_angles', 0.0)
            if len(self.neglected_old_angle_terms) > 0:
                custom_neglected_old_force.addGlobalParameter('lambda_angles', 0.0)


        # Add the force to the system and the force dict.
        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['core_angle_force'] = custom_core_force

        if len(self.neglected_new_angle_terms) > 0:
            self._hybrid_system.addForce(custom_neglected_new_force)
            self._hybrid_system_forces['custom_neglected_new_angle_force'] = custom_neglected_new_force
        if len(self.neglected_old_angle_terms) > 0:
            self._hybrid_system.addForce(custom_neglected_old_force)
            self._hybrid_system_forces['custom_neglected_old_angle_force'] = custom_neglected_old_force


        # Add an angle term for environment/unique interactions--these are never scaled
        standard_angle_force = openmm.HarmonicAngleForce()
        self._hybrid_system.addForce(standard_angle_force)
        self._hybrid_system_forces['standard_angle_force'] = standard_angle_force

    def _add_torsion_force_terms(self):
        """
        This function adds the appropriate PeriodicTorsionForce terms to the system. Core torsions are interpolated,
        while environment and unique torsions are always on.
        """
        energy_expression  = '(1-lambda_torsions)*U1 + lambda_torsions*U2;'
        energy_expression += 'U1 = K1*(1+cos(periodicity1*theta-phase1));'
        energy_expression += 'U2 = K2*(1+cos(periodicity2*theta-phase2));'

        if self._has_functions:
            try:
                energy_expression += 'lambda_torsions = ' + self._functions['lambda_torsions']
            except KeyError as e:
                print("Functions were provided, but no term was provided for torsions")
                raise e


        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomTorsionForce(energy_expression)
        custom_core_force.addPerTorsionParameter('periodicity1') # molecule1 periodicity
        custom_core_force.addPerTorsionParameter('phase1') # molecule1 phase
        custom_core_force.addPerTorsionParameter('K1') # molecule1 spring constant
        custom_core_force.addPerTorsionParameter('periodicity2') # molecule2 periodicity
        custom_core_force.addPerTorsionParameter('phase2') # molecule2 phase
        custom_core_force.addPerTorsionParameter('K2') # molecule2 spring constant

        if self._has_functions:
            custom_core_force.addGlobalParameter('lambda', 0.0)
            custom_core_force.addEnergyParameterDerivative('lambda')
        else:
            custom_core_force.addGlobalParameter('lambda_torsions', 0.0)

        # Add the force to the system
        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['custom_torsion_force'] = custom_core_force

        # Create and add the torsion term for unique/environment atoms
        unique_atom_torsion_force = openmm.PeriodicTorsionForce()
        self._hybrid_system.addForce(unique_atom_torsion_force)
        self._hybrid_system_forces['unique_atom_torsion_force'] = unique_atom_torsion_force

    def _add_nonbonded_force_terms(self):
        """
        Add the nonbonded force terms to the hybrid system. Note that as with the other forces,
        this method does not add any interactions. It only sets up the forces.

        Parameters
        ----------
        nonbonded_method : int
            One of the openmm.NonbondedForce nonbonded methods.
        """

        # Add a regular nonbonded force for all interactions that are not changing.
        standard_nonbonded_force = openmm.NonbondedForce()
        self._hybrid_system.addForce(standard_nonbonded_force)
        _logger.info(f"\t_add_nonbonded_force_terms: {standard_nonbonded_force} added to hybrid system")
        self._hybrid_system_forces['standard_nonbonded_force'] = standard_nonbonded_force

        # Create a CustomNonbondedForce to handle alchemically interpolated nonbonded parameters.
        # Select functional form based on nonbonded method.
        # TODO: check _nonbonded_custom_ewald and _nonbonded_custom_cutoff since they take arguments that are never used...
        if self._nonbonded_method in [openmm.NonbondedForce.NoCutoff]:
            _logger.info("\t_add_nonbonded_force_terms: nonbonded_method is NoCutoff")
            sterics_energy_expression = self._nonbonded_custom(self._softcore_LJ_v2)
        elif self._nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
            _logger.info("\t_add_nonbonded_force_terms: nonbonded_method is Cutoff(Periodic or NonPeriodic)")
            epsilon_solvent = self._old_system_forces['NonbondedForce'].getReactionFieldDielectric()
            r_cutoff = self._old_system_forces['NonbondedForce'].getCutoffDistance()
            sterics_energy_expression = self._nonbonded_custom(self._softcore_LJ_v2)
            standard_nonbonded_force.setReactionFieldDielectric(epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        elif self._nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            _logger.info("\t_add_nonbonded_force_terms: nonbonded_method is PME or Ewald")
            [alpha_ewald, nx, ny, nz] = self._old_system_forces['NonbondedForce'].getPMEParameters()
            delta = self._old_system_forces['NonbondedForce'].getEwaldErrorTolerance()
            r_cutoff = self._old_system_forces['NonbondedForce'].getCutoffDistance()
            sterics_energy_expression = self._nonbonded_custom(self._softcore_LJ_v2)
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        else:
            raise Exception("Nonbonded method %s not supported yet." % str(self._nonbonded_method))

        standard_nonbonded_force.setNonbondedMethod(self._nonbonded_method)
        _logger.info(f"\t_add_nonbonded_force_terms: {self._nonbonded_method} added to standard nonbonded force")

        sterics_energy_expression += self._nonbonded_custom_sterics_common()

        sterics_mixing_rules = self._nonbonded_custom_mixing_rules()

        custom_nonbonded_method = self._translate_nonbonded_method_to_custom(self._nonbonded_method)

        total_sterics_energy = "U_sterics;" + sterics_energy_expression + sterics_mixing_rules
        if self._has_functions:
            try:
                total_sterics_energy += 'lambda_sterics  = ' + self._functions['lambda_sterics']
            except KeyError as e:
                print("Functions were provided, but there is no entry for sterics")
                raise e

        sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(total_sterics_energy)
        if self._softcore_LJ_v2:
            sterics_custom_nonbonded_force.addGlobalParameter("softcore_alpha", self._softcore_LJ_v2_alpha)
        else:
            sterics_custom_nonbonded_force.addGlobalParameter("softcore_alpha", self.softcore_alpha)

        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaA") # Lennard-Jones sigma initial
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonA") # Lennard-Jones epsilon initial
        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaB") # Lennard-Jones sigma final
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonB") # Lennard-Jones epsilon final
        sterics_custom_nonbonded_force.addPerParticleParameter("unique_old") # 1 = hybrid old atom, 0 otherwise
        sterics_custom_nonbonded_force.addPerParticleParameter("unique_new") # 1 = hybrid new atom, 0 otherwise

        if self._has_functions:
            sterics_custom_nonbonded_force.addGlobalParameter('lambda', 0.0)
            sterics_custom_nonbonded_force.addEnergyParameterDerivative('lambda')
        else:
            sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics_core", 0.0)
            sterics_custom_nonbonded_force.addGlobalParameter("lambda_electrostatics_core", 0.0)
            sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics_insert", 0.0)
            sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics_delete", 0.0)


        sterics_custom_nonbonded_force.setNonbondedMethod(custom_nonbonded_method)
        _logger.info(f"\t_add_nonbonded_force_terms: {custom_nonbonded_method} added to sterics_custom_nonbonded force")


        self._hybrid_system.addForce(sterics_custom_nonbonded_force)
        self._hybrid_system_forces['core_sterics_force'] = sterics_custom_nonbonded_force
        _logger.info(f"\t_add_nonbonded_force_terms: {sterics_custom_nonbonded_force} added to hybrid system")


        # Set the use of dispersion correction to be the same between the new nonbonded force and the old one:
        # These will be ignored from the _logger for the time being
        if self._old_system_forces['NonbondedForce'].getUseDispersionCorrection():
            self._hybrid_system_forces['standard_nonbonded_force'].setUseDispersionCorrection(True)
            if self._use_dispersion_correction:
                sterics_custom_nonbonded_force.setUseLongRangeCorrection(True)
        else:
            self._hybrid_system_forces['standard_nonbonded_force'].setUseDispersionCorrection(False)

        if self._old_system_forces['NonbondedForce'].getUseSwitchingFunction():
            switching_distance = self._old_system_forces['NonbondedForce'].getSwitchingDistance()
            standard_nonbonded_force.setUseSwitchingFunction(True)
            standard_nonbonded_force.setSwitchingDistance(switching_distance)
            sterics_custom_nonbonded_force.setUseSwitchingFunction(True)
            sterics_custom_nonbonded_force.setSwitchingDistance(switching_distance)
        else:
            standard_nonbonded_force.setUseSwitchingFunction(False)
            sterics_custom_nonbonded_force.setUseSwitchingFunction(False)

    def _nonbonded_custom_sterics_common(self):
        """
        Get a custom sterics expression using amber softcore expression

        Returns
        -------
        sterics_addition : str
            The common softcore sterics energy expression
        """
        sterics_addition = "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;" # interpolation
        sterics_addition += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
        sterics_addition += "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"


        sterics_addition += "lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"
        sterics_addition += "lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
        sterics_addition += "core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);new_interaction = max(unique_new1, unique_new2);old_interaction = max(unique_old1, unique_old2);"

        return sterics_addition

    def _nonbonded_custom(self, v2):
        """
        Get a part of the nonbonded energy expression when there is no cutoff.

        Returns
        -------
        sterics_energy_expression : str
            The energy expression for U_sterics
        electrostatics_energy_expression : str
            The energy expression for electrostatics
        """
        # Soft-core Lennard-Jones
        if v2:
            sterics_energy_expression = "U_sterics = select(step(r - r_LJ), 4*epsilon*x*(x-1.0), U_sterics_quad);"
            sterics_energy_expression += f"U_sterics_quad = Force*(((r - r_LJ)^2)/2 - (r - r_LJ)) + U_sterics_cut;"
            sterics_energy_expression += f"U_sterics_cut = 4*epsilon*((sigma/r_LJ)^6)*(((sigma/r_LJ)^6) - 1.0);"
            sterics_energy_expression += f"Force = -4*epsilon*((-12*sigma^12)/(r_LJ^13) + (6*sigma^6)/(r_LJ^7));"
            sterics_energy_expression += f"x = (sigma/r)^6;"
            sterics_energy_expression += f"r_LJ = softcore_alpha*((26/7)*(sigma^6)*lambda_sterics_deprecated)^(1/6);"
            sterics_energy_expression += f"lambda_sterics_deprecated = new_interaction*(1.0 - lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"
        else:
            sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"

        return sterics_energy_expression

    def _nonbonded_custom_mixing_rules(self):
        """
        Mixing rules for the custom nonbonded force.

        Returns
        -------
        sterics_mixing_rules : str
            The mixing expression for sterics
        electrostatics_mixing_rules : str
            The mixiing rules for electrostatics
        """
        # Define mixing rules.
        sterics_mixing_rules = "epsilonA = sqrt(epsilonA1*epsilonA2);" # mixing rule for epsilon
        sterics_mixing_rules += "epsilonB = sqrt(epsilonB1*epsilonB2);" # mixing rule for epsilon
        sterics_mixing_rules += "sigmaA = 0.5*(sigmaA1 + sigmaA2);" # mixing rule for sigma
        sterics_mixing_rules += "sigmaB = 0.5*(sigmaB1 + sigmaB2);" # mixing rule for sigma
        return sterics_mixing_rules

    def _find_bond_parameters(self, bond_force, index1, index2):
        """
        This is a convenience function to find bond parameters in another system given the two indices.

        Parameters
        ----------
        bond_force : openmm.HarmonicBondForce
            The bond force where the parameters should be found
        index1 : int
           Index1 (order does not matter) of the bond atoms
        index2 : int
           Index2 (order does not matter) of the bond atoms

        Returns
        -------
        bond_parameters : list
            List of relevant bond parameters
        """
        index_set = {index1, index2}
        # Loop through all the bonds:
        for bond_index in range(bond_force.getNumBonds()):
            parms = bond_force.getBondParameters(bond_index)
            if index_set=={parms[0], parms[1]}:
                return parms

        return []

    def handle_harmonic_bonds(self):
        """
        This method adds the appropriate interaction for all bonds in the hybrid system. The scheme used is:

        1) If the two atoms are both in the core, then we add to the CustomBondForce and interpolate between the two
            parameters
        2) If one of the atoms is in core and the other is environment, we have to assert that the bond parameters do not change between
           the old and the new system; then, the parameters are added to the regular bond force
        3) Otherwise, we add the bond to a regular bond force.
        """
        old_system_bond_force = self._old_system_forces['HarmonicBondForce']
        new_system_bond_force = self._new_system_forces['HarmonicBondForce']

        # Make a dict to check the environment-core bonds for consistency between the old and new systems
        # key: hybrid_index_set, value: [(r0_old, k_old)]
        old_core_env_indices = {}

        # First, loop through the old system bond forces and add relevant terms
        _logger.info("\thandle_harmonic_bonds: looping through old_system to add relevant terms...")
        for bond_index in range(old_system_bond_force.getNumBonds()):
            # Get each set of bond parameters
            [index1_old, index2_old, r0_old, k_old] = old_system_bond_force.getBondParameters(bond_index)
            _logger.debug(f"\t\thandle_harmonic_bonds: old bond_index {bond_index} with old indices {index1_old, index2_old}")

            # Map the indices to the hybrid system, for which our atom classes are defined.
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}

            # Now check if it is a subset of the core atoms (that is, both atoms are in the core)
            # If it is, we need to find the parameters in the old system so that we can interpolate
            if index_set.issubset(self._atom_classes['core_atoms']):
                _logger.debug(f"\t\thandle_harmonic_bonds: bond_index {bond_index} is a core (to custom bond force).")
                index1_new = self._topology_proposal.old_to_new_atom_map[index1_old]
                index2_new = self._topology_proposal.old_to_new_atom_map[index2_old]
                new_bond_parameters = self._find_bond_parameters(new_system_bond_force, index1_new, index2_new)
                if not new_bond_parameters:
                    r0_new = r0_old
                    k_new = 0.0*unit.kilojoule_per_mole/unit.angstrom**2
                else:
                    [index1, index2, r0_new, k_new] = self._find_bond_parameters(new_system_bond_force, index1_new, index2_new)
                self._hybrid_system_forces['core_bond_force'].addBond(index1_hybrid, index2_hybrid,[r0_old, k_old, r0_new, k_new])

            # Check if the index set is a subset of anything besides environemnt (in the case of environment, we just add the bond to the regular bond force)
            # that would mean that this bond is core-unique_old or unique_old-unique_old
            elif index_set.issubset(self._atom_classes['unique_old_atoms']) or (len(index_set.intersection(self._atom_classes['unique_old_atoms'])) == 1 and len(index_set.intersection(self._atom_classes['core_atoms'])) == 1):
                _logger.debug(f"\t\thandle_harmonic_bonds: bond_index {bond_index} is a core-unique_old or unique_old-unique old...")

                # If we're not softening bonds, we can just add it to the regular bond force. Likewise if we are only softening new bonds
                if not self._soften_bonds or self._soften_only_new:
                    _logger.debug(f"\t\t\thandle_harmonic_bonds: no softening (to standard bond force)")
                    self._hybrid_system_forces['standard_bond_force'].addBond(index1_hybrid, index2_hybrid, r0_old,
                                                                              k_old)
                # Otherwise, we will need to soften one of the endpoints. For unique old atoms, the softening endpoint is at lambda =1
                else:

                    r0_new = r0_old # The bond length won't change
                    k_new = self._bond_softening_constant * k_old # We multiply the endpoint by the bond softening constant

                    # Now we add to the core bond force, since that is an alchemically-modified force.
                    self._hybrid_system_forces['core_bond_force'].addBond(index1_hybrid, index2_hybrid,
                                                                          [r0_old, k_old, r0_new, k_new])
            elif len(index_set.intersection(self._atom_classes['environment_atoms'])) == 1 and len(index_set.intersection(self._atom_classes['core_atoms'])) == 1:
                _logger.debug(f"\t\thandle_harmonic_bonds: bond_index {bond_index} is an environment-core...")
                self._hybrid_system_forces['standard_bond_force'].addBond(index1_hybrid, index2_hybrid, r0_old, k_old)

            # Otherwise, we just add the same parameters as those in the old system (these are environment atoms, and the parameters are the same)
            elif index_set.issubset(self._atom_classes['environment_atoms']):
                _logger.debug(f"\t\thandle_harmonic_bonds: bond_index {bond_index} is an environment (to standard bond force).")
                self._hybrid_system_forces['standard_bond_force'].addBond(index1_hybrid, index2_hybrid, r0_old, k_old)
            else:
                raise Exception(f"\t\thybrid index set {index_set} does not fit into a canonical atom type")


        # Now loop through the new system to get the interactions that are unique to it.
        _logger.info("\thandle_harmonic_bonds: looping through new_system to add relevant terms...")
        for bond_index in range(new_system_bond_force.getNumBonds()):
            # Get each set of bond parameters
            [index1_new, index2_new, r0_new, k_new] = new_system_bond_force.getBondParameters(bond_index)
            _logger.debug(f"\t\thandle_harmonic_bonds: new bond_index {bond_index} with new indices {index1_new, index2_new}")

            # Convert indices to hybrid, since that is how we represent atom classes:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]
            index_set = {index1_hybrid, index2_hybrid}

            # If the intersection of this set and unique new atoms contains anything, the bond is unique to the new system and must be added
            # all other bonds in the new system have been accounted for already.
            if len(index_set.intersection(self._atom_classes['unique_new_atoms'])) == 2 or (len(index_set.intersection(self._atom_classes['unique_new_atoms'])) == 1 and len(index_set.intersection(self._atom_classes['core_atoms'])) == 1):
                _logger.debug(f"\t\thandle_harmonic_bonds: bond_index {bond_index} is a core-unique_new or unique_new-unique_new...")

                # If we are softening bonds, we have to use the core bond force, and scale the force constant at lambda = 0:
                if self._soften_bonds:
                    _logger.debug(f"\t\t\thandle_harmonic_bonds: softening (to custom bond force)")
                    r0_old = r0_new # Do not change the length
                    k_old = k_new * self._bond_softening_constant # Scale the force constant by the requested parameter

                    # Now we add to the core bond force, since that is an alchemically-modified force.
                    self._hybrid_system_forces['core_bond_force'].addBond(index1_hybrid, index2_hybrid,
                                                                          [r0_old, k_old, r0_new, k_new])

                # If we aren't softening bonds, then just add it to the standard bond force
                else:
                    _logger.debug(f"\t\t\thandle_harmonic_bonds: no softening (to standard bond force)")
                    self._hybrid_system_forces['standard_bond_force'].addBond(index1_hybrid, index2_hybrid, r0_new, k_new)

            # If the bond is in the core, it has probably already been added in the above loop. However, there are some circumstances
            # where it was not (closing a ring). In that case, the bond has not been added and should be added here.
            # This has some peculiarities to be discussed...
            elif index_set.issubset(self._atom_classes['core_atoms']):
                if not self._find_bond_parameters(self._hybrid_system_forces['core_bond_force'], index1_hybrid, index2_hybrid):
                     _logger.debug(f"\t\thandle_harmonic_bonds: bond_index {bond_index} is a SPECIAL core-core (to custom bond force).")
                     r0_old = r0_new
                     k_old = 0.0*unit.kilojoule_per_mole/unit.angstrom**2
                     self._hybrid_system_forces['core_bond_force'].addBond(index1_hybrid, index2_hybrid,
                                                                           [r0_old, k_old, r0_new, k_new])
            elif index_set.issubset(self._atom_classes['environment_atoms']):
                # Already been added
                pass

            elif len(index_set.intersection(self._atom_classes['environment_atoms'])) == 1 and len(index_set.intersection(self._atom_classes['core_atoms'])) == 1:
                _logger.debug(f"\t\thandle_harmonic_bonds: bond_index {bond_index} is an environemnt-core; this has been previously added")
                pass

            else:
                raise Exception(f"\t\thybrid index set {index_set} does not fit into a canonical atom type")


    def _find_angle_parameters(self, angle_force, indices):
        """
        Convenience function to find the angle parameters corresponding to a particular set of indices

        Parameters
        ----------
        angle_force : openmm.HarmonicAngleForce
            The force where the angle of interest may be found.
        indices : list of int
            The indices (any order) of the angle atoms
        Returns
        -------
        angle_parameters : list
            list of angle parameters
        """
        #index_set = set(indices)
        indices_reversed = indices[::-1]

        # Now loop through and try to find the angle:
        for angle_index in range(angle_force.getNumAngles()):
            angle_parameters = angle_force.getAngleParameters(angle_index)

            # Get a set representing the angle indices
            angle_parameter_indices = angle_parameters[:3]

            if indices == angle_parameter_indices or indices_reversed == angle_parameter_indices:
                return angle_parameters
        return []  # Return empty if no matching angle found

    def _find_torsion_parameters(self, torsion_force, indices):
        """
        Convenience function to find the torsion parameters corresponding to a particular set of indices.

        Parameters
        ----------
        torsion_force : openmm.PeriodicTorsionForce
            torsion force where the torsion of interest may be found
        indices : list of int
            The indices of the atoms of the torsion

        Returns
        -------
        torsion_parameters : list
            torsion parameters
        """
        #index_set = set(indices)
        indices_reversed = indices[::-1]

        torsion_parameters_list = list()

        # Now loop through and try to find the torsion:
        for torsion_index in range(torsion_force.getNumTorsions()):
            torsion_parameters = torsion_force.getTorsionParameters(torsion_index)

            # Get a set representing the torsion indices:
            torsion_parameter_indices = torsion_parameters[:4]

            if indices == torsion_parameter_indices or indices_reversed == torsion_parameter_indices:
                torsion_parameters_list.append(torsion_parameters)

        return torsion_parameters_list

    def handle_harmonic_angles(self):
        """
        This method adds the appropriate interaction for all angles in the hybrid system. The scheme used, as with bonds, is:

        1) If the three atoms are all in the core, then we add to the CustomAngleForce and interpolate between the two
            parameters
        2) If the three atoms contain at least one unique new, check if the angle is in the neglected new list, and if so, interpolate from K_1 = 0;
            else, if the three atoms contain at least one unique old, check if the angle is in the neglected old list, and if so, interpolate from K_2 = 0.
        3) If the angle contains at least one environment and at least one core atom, assert there are no unique new atoms and that the angle terms
           are preserved between the new and the old system.  Then add to the standard angle force
        4) Otherwise, we add the angle to a regular angle force since it is environment.
        """
        old_system_angle_force = self._old_system_forces['HarmonicAngleForce']
        new_system_angle_force = self._new_system_forces['HarmonicAngleForce']

        # Make a dict to check the angles involving environment-core bonds for consistency between the old and new systems
        # key: hybrid_index_set, value: [(theta0, k0)]

        # First, loop through all the angles in the old system to determine what to do with them. We will only use the
        # custom angle force if all atoms are part of "core." Otherwise, they are either unique to one system or never
        # change.
        _logger.info("\thandle_harmonic_angles: looping through old_system to add relevant terms...")
        for angle_index in range(old_system_angle_force.getNumAngles()):

            old_angle_parameters = old_system_angle_force.getAngleParameters(angle_index)
            _logger.debug(f"\t\thandle_harmonic_angles: old angle_index {angle_index} with old indices {old_angle_parameters[:3]}")

            # Get the indices in the hybrid system
            hybrid_index_list = [self._old_to_hybrid_map[old_atomid] for old_atomid in old_angle_parameters[:3]]
            hybrid_index_set = set(hybrid_index_list)

            # If all atoms are in the core, we'll need to find the corresponding parameters in the old system and
            # interpolate
            if hybrid_index_set.issubset(self._atom_classes['core_atoms']):
                _logger.debug(f"\t\thandle_harmonic_angles: angle_index {angle_index} is a core (to custom angle force).")
                # Get the new indices so we can get the new angle parameters
                new_indices = [self._topology_proposal.old_to_new_atom_map[old_atomid] for old_atomid in old_angle_parameters[:3]]
                new_angle_parameters = self._find_angle_parameters(new_system_angle_force, new_indices)
                if not new_angle_parameters:
                    new_angle_parameters = [0, 0, 0, old_angle_parameters[3], 0.0*unit.kilojoule_per_mole/unit.radian**2]

                # Add to the hybrid force:
                # the parameters at indices 3 and 4 represent theta0 and k, respectively.
                hybrid_force_parameters = [old_angle_parameters[3], old_angle_parameters[4], new_angle_parameters[3], new_angle_parameters[4]]
                self._hybrid_system_forces['core_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_force_parameters)

            # Check if the atoms are neither all core nor all environment, which would mean they involve unique old interactions
            elif not hybrid_index_set.issubset(self._atom_classes['environment_atoms']):
                if hybrid_index_set.intersection(self._atom_classes['environment_atoms']) != set(): #if there is an environment atom
                    _logger.debug(f"\t\thandle_harmonic_angles: angle_index {angle_index} contains an environment atom")
                    assert hybrid_index_set.intersection(self._atom_classes['unique_old_atoms']) == set(), f"we disallow unique-environment terms"
                    self._hybrid_system_forces['standard_angle_force'].addAngle(hybrid_index_list[0],
                                                                                hybrid_index_list[1],
                                                                                hybrid_index_list[2],
                                                                                old_angle_parameters[3],
                                                                                old_angle_parameters[4])
                else:
                    _logger.debug(f"\t\thandle_harmonic_angles: angle_index {angle_index} is a core with unique_old...")
                    # There are no env atoms, so we can treat this term appropriately

                    # Check if we are softening angles, and not softening only new angles:
                    if self._soften_angles and not self._soften_only_new:
                        _logger.debug(f"\t\t\thandle_harmonic_angles: softening (to custom angle force)")


                        # If we are, then we need to generate the softened parameters (at lambda=1 for old atoms)
                        # We do this by using the same equilibrium angle, and scaling the force constant at the non-interacting
                        # endpoint:
                        if angle_index in self.neglected_old_angle_terms:
                            _logger.debug("\t\t\tsoften angles on but angle is in neglected old, so softening constant is set to zero.")
                            hybrid_force_parameters = [old_angle_parameters[3], old_angle_parameters[4], old_angle_parameters[3], 0.0 * old_angle_parameters[4]]
                            self._hybrid_system_forces['custom_neglected_old_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_force_parameters)
                        else:
                            _logger.debug(f"\t\t\thandle_harmonic_angles: softening (to custom angle force)")
                            hybrid_force_parameters = [old_angle_parameters[3], old_angle_parameters[4], old_angle_parameters[3], self._angle_softening_constant * old_angle_parameters[4]]
                            self._hybrid_system_forces['core_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_force_parameters)


                    # If not, we can just add this to the standard angle force
                    else:
                        if angle_index in self.neglected_old_angle_terms:
                            _logger.debug(f"\t\t\tangle in neglected_old_angle_terms; K_2 is set to zero")
                            hybrid_force_parameters = [old_angle_parameters[3], old_angle_parameters[4], old_angle_parameters[3], 0.0 * old_angle_parameters[4]]
                            self._hybrid_system_forces['custom_neglected_old_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_force_parameters)

                        else:
                            _logger.debug(f"\t\t\thandle_harmonic_bonds: no softening (to standard angle force)")
                            self._hybrid_system_forces['standard_angle_force'].addAngle(hybrid_index_list[0],
                                                                                        hybrid_index_list[1],
                                                                                        hybrid_index_list[2],
                                                                                        old_angle_parameters[3],
                                                                                        old_angle_parameters[4])

            # Otherwise, only environment atoms are in this interaction, so add it to the standard angle force
            elif hybrid_index_set.issubset(self._atom_classes['environment_atoms']):
                _logger.debug(f"\t\thandle_harmonic_angles: angle_index {angle_index} is an environment (to standard angle force)")
                self._hybrid_system_forces['standard_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1],
                                                                            hybrid_index_list[2], old_angle_parameters[3],
                                                                            old_angle_parameters[4])
            else:
                raise Exception(f"\t\thandle_harmonic_angles: angle_index {angle_index} does not fit a canonical form.")

        # Finally, loop through the new system force to add any unique new angles
        _logger.info("\thandle_harmonic_angles: looping through new_system to add relevant terms...")
        for angle_index in range(new_system_angle_force.getNumAngles()):

            new_angle_parameters = new_system_angle_force.getAngleParameters(angle_index)
            _logger.debug(f"\t\thandle_harmonic_angles: new angle_index {angle_index} with new terms {new_angle_parameters[:3]}")

            # Get the indices in the hybrid system
            hybrid_index_list = [self._new_to_hybrid_map[new_atomid] for new_atomid in new_angle_parameters[:3]]
            hybrid_index_set = set(hybrid_index_list)

            # If the intersection of this hybrid set with the unique new atoms is nonempty, it must be added:
            if len(hybrid_index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
                assert hybrid_index_set.intersection(self._atom_classes['environment_atoms']) == set(), f"we disallow angle terms with unique new and environment atoms"
                _logger.debug(f"\t\thandle_harmonic_bonds: angle_index {angle_index} is a core-unique_new or unique_new-unique_new...")

                # Check to see if we are softening angles:
                if self._soften_angles:
                    _logger.info(f"\t\t\thandle_harmonic_bonds: softening (to custom angle force)")

                    if angle_index in self.neglected_new_angle_terms:
                        _logger.debug("\t\t\tsoften angles on but angle is in neglected new, so softening constant is set to zero.")
                        hybrid_force_parameters = [new_angle_parameters[3], new_angle_parameters[4] * 0.0, new_angle_parameters[3], new_angle_parameters[4]]
                        self._hybrid_system_forces['custom_neglected_new_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_force_parameters)
                    else:
                        _logger.debug(f"\t\t\thandle_harmonic_angles: softening (to custom angle force)")
                        hybrid_force_parameters = [new_angle_parameters[3], new_angle_parameters[4] * self._angle_softening_constant, new_angle_parameters[3], new_angle_parameters[4]]
                        self._hybrid_system_forces['core_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1],
                                                                                hybrid_index_list[2],
                                                                                hybrid_force_parameters)
                # Otherwise, just add to the nonalchemical force
                else:
                    if angle_index in self.neglected_new_angle_terms:
                        _logger.debug(f"\t\t\tangle in neglected_new_angle_terms; K_1 is set to zero")
                        hybrid_force_parameters = [new_angle_parameters[3], 0.0 * new_angle_parameters[4], new_angle_parameters[3], new_angle_parameters[4]]
                        self._hybrid_system_forces['custom_neglected_new_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_force_parameters)
                    else:
                        _logger.debug(f"\t\t\thandle_harmonic_bonds: no softening (to standard angle force)")
                        self._hybrid_system_forces['standard_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1],
                                                                                hybrid_index_list[2], new_angle_parameters[3],
                                                                                new_angle_parameters[4])

            elif hybrid_index_set.issubset(self._atom_classes['core_atoms']):
                _logger.debug(f"\t\thandle_harmonic_angles: angle_index {angle_index} is a core (to custom angle force).")
                if not self._find_angle_parameters(self._hybrid_system_forces['core_angle_force'], hybrid_index_list):
                    _logger.debug(f"\t\t\thandle_harmonic_angles: angle_index {angle_index} NOT previously added...adding now...THERE IS A CONSIDERATION NOT BEING MADE!")
                    hybrid_force_parameters = [new_angle_parameters[3], 0.0*unit.kilojoule_per_mole/unit.radian**2, new_angle_parameters[3], new_angle_parameters[4]]
                    self._hybrid_system_forces['core_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1],
                                                                            hybrid_index_list[2],
                                                                            hybrid_force_parameters)
            elif hybrid_index_set.issubset(self._atom_classes['environment_atoms']):
                # We have already added the appropriate environmental atom terms
                pass
            elif hybrid_index_set.intersection(self._atom_classes['environment_atoms']) != set():
                _logger.debug(f"\t\thandle_harmonic_angles: angle_index {angle_index} contains an environment atom; this as already been added")
                assert hybrid_index_set.intersection(self._atom_classes['unique_new_atoms']) == set(), f"we disallow angle terms with unique new and environment atoms"
            else:
                raise Exception(f"\t\thybrid index list {hybrid_index_list} does not fit into a canonical atom set")

    def handle_periodic_torsion_force(self):
        """
        Handle the torsions defined in the new and old systems as such:

        1. old system torsions will enter the ``custom_torsion_force`` if they do not contain ``unique_old_atoms`` and will interpolate from ``on`` to ``off`` from ``lambda_torsions`` = 0 to 1, respectively
        2. new system torsions will enter the ``custom_torsion_force`` if they do not contain ``unique_new_atoms`` and will interpolate from ``off`` to ``on`` from ``lambda_torsions`` = 0 to 1, respectively
        3. old *and* new system torsions will enter the ``unique_atom_torsion_force`` (``standard_torsion_force``) and will *not* be interpolated
        """
        old_system_torsion_force = self._old_system_forces['PeriodicTorsionForce']
        new_system_torsion_force = self._new_system_forces['PeriodicTorsionForce']

        # auxiliary_custom_torsion_force = copy.deepcopy(self._hybrid_system_forces['custom_torsion_force'])
        auxiliary_custom_torsion_force = []
        old_custom_torsions_to_standard = []

        # We need to keep track of what torsions we added so that we do not double count.
        added_torsions = []
        _logger.info("\thandle_periodic_torsion_forces: looping through old_system to add relevant terms...")
        for torsion_index in range(old_system_torsion_force.getNumTorsions()):

            torsion_parameters = old_system_torsion_force.getTorsionParameters(torsion_index)
            _logger.debug(f"\t\thandle_harmonic_torsion_forces: old torsion_index {torsion_index} with old indices {torsion_parameters[:4]}")
            #_logger.debug(f"\t\thandle_harmonic_torsion_forces: old_torsion parameters: {torsion_parameters}")


            # Get the indices in the hybrid system
            hybrid_index_list = [self._old_to_hybrid_map[old_index] for old_index in torsion_parameters[:4]]
            hybrid_index_set = set(hybrid_index_list)

            # If all atoms are in the core, we'll need to find the corresponding parameters in the old system and
            # interpolate
            if hybrid_index_set.intersection(self._atom_classes['unique_old_atoms']) != set():
                # Then it goes to a standard force...
                self._hybrid_system_forces['unique_atom_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1],
                                                                        hybrid_index_list[2], hybrid_index_list[3], torsion_parameters[4],
                                                                        torsion_parameters[5], torsion_parameters[6])
            else:
                # It is a core-only term, an environment-only term, or a core/env term;
                # in any case, it goes to the core torsion_force
                torsion_indices = torsion_parameters[:4]
                hybrid_force_parameters = [torsion_parameters[4], torsion_parameters[5], torsion_parameters[6], 0.0, 0.0, 0.0]
                # self._hybrid_system_forces['custom_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters)
                auxiliary_custom_torsion_force.append([hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters[:3]])

        _logger.info("\thandle_periodic_torsion_forces: looping through new_system to add relevant terms...")
        for torsion_index in range(new_system_torsion_force.getNumTorsions()):
            torsion_parameters = new_system_torsion_force.getTorsionParameters(torsion_index)
            _logger.debug(f"\t\thandle_harmonic_torsions: new torsion_index {torsion_index} with new indices {torsion_parameters[:4]}")

            # Get the indices in the hybrid system:
            hybrid_index_list = [self._new_to_hybrid_map[new_index] for new_index in torsion_parameters[:4]]
            hybrid_index_set = set(hybrid_index_list)

            if hybrid_index_set.intersection(self._atom_classes['unique_new_atoms']) != set():
                # Then it goes to a standard force...
                self._hybrid_system_forces['unique_atom_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1],
                                                                        hybrid_index_list[2], hybrid_index_list[3], torsion_parameters[4],
                                                                        torsion_parameters[5], torsion_parameters[6])
            else:

                torsion_indices = torsion_parameters[:4]

                hybrid_force_parameters = [0.0, 0.0, 0.0, torsion_parameters[4], torsion_parameters[5], torsion_parameters[6]]

                # Check to see if this term is in the olds...
                if [hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters[3:]] in auxiliary_custom_torsion_force:
                    # print('hooray!')
                    # Then this terms has to go to standard and be deleted...
                    old_index = auxiliary_custom_torsion_force.index([hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters[3:]])
                    old_custom_torsions_to_standard.append(old_index)
                    self._hybrid_system_forces['unique_atom_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1],
                                                                            hybrid_index_list[2], hybrid_index_list[3], torsion_parameters[4],
                                                                            torsion_parameters[5], torsion_parameters[6])
                else:
                    # Then this term has to go to the core force...
                    self._hybrid_system_forces['custom_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters)
                    # auxiliary_custom_torsion_force.addTorsion(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters[3:])

        # Now we have to loop through the aux custom torsion force
        # print(f"old_custom_torsions_to_standard: {old_custom_torsions_to_standard}")
        for index in [q for q in range(len(auxiliary_custom_torsion_force)) if q not in old_custom_torsions_to_standard]:
            terms = auxiliary_custom_torsion_force[index]
            hybrid_index_list = terms[:4]
            hybrid_force_parameters = terms[4] + [0., 0., 0.]
            self._hybrid_system_forces['custom_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters)

    def handle_nonbonded(self):
        """

        """
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = self._hybrid_to_old_map
        hybrid_to_new_map = self._hybrid_to_new_map

        # Define new global parameters for NonbondedForce
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter('lambda_electrostatics_core', 0.0)
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter('lambda_sterics_core', 0.0)
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter("lambda_electrostatics_delete", 0.0)
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter("lambda_electrostatics_insert", 0.0)

        # We have to loop through the particles in the system, because nonbonded force does not accept index
        _logger.info("\thandle_nonbonded: looping through all particles in hybrid...")
        for particle_index in range(self._hybrid_system.getNumParticles()):

            if particle_index in self._atom_classes['unique_old_atoms']:
                _logger.debug(f"\t\thandle_nonbonded: particle {particle_index} is a unique_old")
                # Get the parameters in the old system
                old_index = hybrid_to_old_map[particle_index]
                [charge, sigma, epsilon] = old_system_nonbonded_force.getParticleParameters(old_index)

                # Add the particle to the hybrid custom sterics and electrostatics.
                check_index = self._hybrid_system_forces['core_sterics_force'].addParticle([sigma, epsilon, sigma, 0.0*epsilon, 1, 0]) # turning off sterics in forward direction
                assert (particle_index == check_index ), "Attempting to add incorrect particle to hybrid system"

                # Add particle to the regular nonbonded force, but Lennard-Jones will be handled by CustomNonbondedForce
                check_index = self._hybrid_system_forces['standard_nonbonded_force'].addParticle(charge, sigma, 0.0*epsilon) # add charge to standard_nonbonded force
                assert (particle_index == check_index ), "Attempting to add incorrect particle to hybrid system"

                # Charge will be turned off at lambda_electrostatics_delete = 0, on at lambda_electrostatics_delete = 1; kill charge with lambda_electrostatics_delete = 0 --> 1
                self._hybrid_system_forces['standard_nonbonded_force'].addParticleParameterOffset('lambda_electrostatics_delete', particle_index, -charge, 0*sigma, 0*epsilon)

            elif particle_index in self._atom_classes['unique_new_atoms']:
                _logger.debug(f"\t\thandle_nonbonded: particle {particle_index} is a unique_new")
                # Get the parameters in the new system
                new_index = hybrid_to_new_map[particle_index]
                [charge, sigma, epsilon] = new_system_nonbonded_force.getParticleParameters(new_index)

                # Add the particle to the hybrid custom sterics and electrostatics
                check_index = self._hybrid_system_forces['core_sterics_force'].addParticle([sigma, 0.0*epsilon, sigma, epsilon, 0, 1]) # turning on sterics in forward direction
                assert (particle_index == check_index ), "Attempting to add incorrect particle to hybrid system"

                # Add particle to the regular nonbonded force, but Lennard-Jones will be handled by CustomNonbondedForce
                check_index = self._hybrid_system_forces['standard_nonbonded_force'].addParticle(0.0, sigma, 0.0) # charge starts at zero
                assert (particle_index == check_index ), "Attempting to add incorrect particle to hybrid system"

                # Charge will be turned off at lambda_electrostatics_insert = 0, on at lambda_electrostatics_insert = 1; add charge with lambda_electrostatics_insert = 0 --> 1
                self._hybrid_system_forces['standard_nonbonded_force'].addParticleParameterOffset('lambda_electrostatics_insert', particle_index, +charge, 0, 0)

            elif particle_index in self._atom_classes['core_atoms']:
                _logger.debug(f"\t\thandle_nonbonded: particle {particle_index} is a core")
                # Get the parameters in the new and old systems:
                old_index = hybrid_to_old_map[particle_index]
                [charge_old, sigma_old, epsilon_old] = old_system_nonbonded_force.getParticleParameters(old_index)
                new_index = hybrid_to_new_map[particle_index]
                [charge_new, sigma_new, epsilon_new] = new_system_nonbonded_force.getParticleParameters(new_index)

                # Add the particle to the custom forces, interpolating between the two parameters; add steric params and zero electrostatics to core_sterics per usual
                check_index = self._hybrid_system_forces['core_sterics_force'].addParticle([sigma_old, epsilon_old, sigma_new, epsilon_new, 0, 0])
                assert (particle_index == check_index ), "Attempting to add incorrect particle to hybrid system"

                # Still add the particle to the regular nonbonded force, but with zeroed out parameters; add old charge to standard_nonbonded and zero sterics
                check_index = self._hybrid_system_forces['standard_nonbonded_force'].addParticle(charge_old, 0.5*(sigma_old+sigma_new), 0.0)
                assert (particle_index == check_index ), "Attempting to add incorrect particle to hybrid system"

                # Charge is charge_old at lambda_electrostatics = 0, charge_new at lambda_electrostatics = 1
                # TODO: We could also interpolate the Lennard-Jones here instead of core_sterics force so that core_sterics_force could just be softcore
                # interpolate between old and new charge with lambda_electrostatics core; make sure to keep sterics off
                self._hybrid_system_forces['standard_nonbonded_force'].addParticleParameterOffset('lambda_electrostatics_core', particle_index, (charge_new - charge_old), 0, 0)

            # Otherwise, the particle is in the environment
            else:
                _logger.debug(f"\t\thandle_nonbonded: particle {particle_index} is an envronment")
                # The parameters will be the same in new and old system, so just take the old parameters
                old_index = hybrid_to_old_map[particle_index]
                [charge, sigma, epsilon] = old_system_nonbonded_force.getParticleParameters(old_index)

                # Add the particle to the hybrid custom sterics, but they dont change; electrostatics are ignored
                self._hybrid_system_forces['core_sterics_force'].addParticle([sigma, epsilon, sigma, epsilon, 0, 0])

                # Add the environment atoms to the regular nonbonded force as well: should we be adding steric terms here, too?
                self._hybrid_system_forces['standard_nonbonded_force'].addParticle(charge, sigma, epsilon)



        # Now loop pairwise through (unique_old, unique_new) and add exceptions so that they never interact electrostatically (place into Nonbonded Force)
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']

        for old in unique_old_atoms:
            for new in unique_new_atoms:
                self._hybrid_system_forces['standard_nonbonded_force'].addException(old, new, 0.0*unit.elementary_charge**2, 1.0*unit.nanometers, 0.0*unit.kilojoules_per_mole)
                self._hybrid_system_forces['core_sterics_force'].addExclusion(old, new) # This is only necessary to avoid the 'All forces must have identical exclusions' rule

        _logger.info("\thandle_nonbonded: Handling Interaction Groups...")
        self._handle_interaction_groups()

        _logger.info("\thandle_nonbonded: Handling Hybrid Exceptions...")
        self._handle_hybrid_exceptions()

        _logger.info("\thandle_nonbonded: Handling Original Exceptions...")
        self._handle_original_exceptions()

    def _generate_dict_from_exceptions(self, force):
        """
        This is a utility function to generate a dictionary of the form
        (particle1_idx, particle2_idx) : [exception parameters]. This will facilitate access and search of exceptions

        Parameters
        ----------
        force : openmm.NonbondedForce object
            a force containing exceptions

        Returns
        -------
        exceptions_dict : dict
            Dictionary of exceptions
        """
        exceptions_dict = {}

        for exception_index in range(force.getNumExceptions()):
            [index1, index2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
            exceptions_dict[(index1, index2)] = [chargeProd, sigma, epsilon]
        #_logger.debug(f"\t_generate_dict_from_exceptions: Exceptions Dict: {exceptions_dict}" )

        return exceptions_dict

    def _handle_interaction_groups(self):
        """
        Create the appropriate interaction groups for the custom nonbonded forces. The groups are:

        1) Unique-old - core
        2) Unique-old - environment
        3) Unique-new - core
        4) Unique-new - environment
        5) Core - environment
        6) Core - core

        Unique-old and Unique new are prevented from interacting this way, and intra-unique interactions occur in an
        unmodified nonbonded force.

        Must be called after particles are added to the Nonbonded forces
        TODO: we should also be adding the following interaction groups...
        7) Unique-new - Unique-new
        8) Unique-old - Unique-old
        """
        # Get the force objects for convenience:
        sterics_custom_force = self._hybrid_system_forces['core_sterics_force']

        # Also prepare the atom classes
        core_atoms = self._atom_classes['core_atoms']
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']
        environment_atoms = self._atom_classes['environment_atoms']


        sterics_custom_force.addInteractionGroup(unique_old_atoms, core_atoms)

        sterics_custom_force.addInteractionGroup(unique_old_atoms, environment_atoms)

        sterics_custom_force.addInteractionGroup(unique_new_atoms, core_atoms)

        sterics_custom_force.addInteractionGroup(unique_new_atoms, environment_atoms)

        sterics_custom_force.addInteractionGroup(core_atoms, environment_atoms)

        sterics_custom_force.addInteractionGroup(core_atoms, core_atoms)

        sterics_custom_force.addInteractionGroup(unique_new_atoms, unique_new_atoms)

        sterics_custom_force.addInteractionGroup(unique_old_atoms, unique_old_atoms)

    def _handle_hybrid_exceptions(self):
        """
        Instead of excluding interactions that shouldn't occur, we provide exceptions for interactions that were zeroed
        out but should occur.
        """
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']

        import itertools
        # Prepare the atom classes
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']

        # Get the list of interaction pairs for which we need to set exceptions:
        unique_old_pairs = list(itertools.combinations(unique_old_atoms, 2))
        unique_new_pairs = list(itertools.combinations(unique_new_atoms, 2))

        # Add back the interactions of the old unique atoms, unless there are exceptions
        for atom_pair in unique_old_pairs:
            # Since the pairs are indexed in the dictionary by the old system indices, we need to convert
            old_index_atom_pair = (self._hybrid_to_old_map[atom_pair[0]], self._hybrid_to_old_map[atom_pair[1]])

            # Now we check if the pair is in the exception dictionary
            if old_index_atom_pair in self._old_system_exceptions:
                _logger.debug(f"\t\thandle_nonbonded: _handle_hybrid_exceptions: {old_index_atom_pair} is an old system exception")
                [chargeProd, sigma, epsilon] = self._old_system_exceptions[old_index_atom_pair]
                if self._interpolate_14s: #if we are interpolating 1,4 exceptions then we have to
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd*0.0, sigma, epsilon*0.0)
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

                self._hybrid_system_forces['core_sterics_force'].addExclusion(atom_pair[0], atom_pair[1]) # Add exclusion to ensure exceptions are consistent


            # Check if the pair is in the reverse order and use that if so
            elif old_index_atom_pair[::-1] in self._old_system_exceptions:
                _logger.debug(f"\t\thandle_nonbonded: _handle_hybrid_exceptions: {old_index_atom_pair[::-1]} is an old system exception")
                [chargeProd, sigma, epsilon] = self._old_system_exceptions[old_index_atom_pair[::-1]]
                if self._interpolate_14s: # If we are interpolating 1,4 exceptions then we have to
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd*0.0, sigma, epsilon*0.0)
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

                self._hybrid_system_forces['core_sterics_force'].addExclusion(atom_pair[0], atom_pair[1]) # Add exclusion to ensure exceptions are consistent

            # If it's not handled by an exception in the original system, we just add the regular parameters as an exception
            # TODO: this implies that the old-old nonbonded interactions (those which are not exceptions) are always self-interacting throughout lambda protocol...
            # else:
            #     _logger.info(f"\t\thandle_nonbonded: _handle_hybrid_exceptions: {old_index_atom_pair} is NOT an old exception...perhaps this is a problem!")
            #     [charge0, sigma0, epsilon0] = self._old_system_forces['NonbondedForce'].getParticleParameters(old_index_atom_pair[0])
            #     [charge1, sigma1, epsilon1] = self._old_system_forces['NonbondedForce'].getParticleParameters(old_index_atom_pair[1])
            #     chargeProd = charge0*charge1
            #     epsilon = unit.sqrt(epsilon0*epsilon1)
            #     sigma = 0.5*(sigma0+sigma1)
            #     self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)
            #     self._hybrid_system_forces['core_sterics_force'].addExclusion(atom_pair[0], atom_pair[1]) # add exclusion to ensure exceptions are consistent

        # Add back the interactions of the new unique atoms, unless there are exceptions
        for atom_pair in unique_new_pairs:
            # Since the pairs are indexed in the dictionary by the new system indices, we need to convert
            new_index_atom_pair = (self._hybrid_to_new_map[atom_pair[0]], self._hybrid_to_new_map[atom_pair[1]])

            # Now we check if the pair is in the exception dictionary
            if new_index_atom_pair in self._new_system_exceptions:
                _logger.debug(f"\t\thandle_nonbonded: _handle_hybrid_exceptions: {new_index_atom_pair} is a new system exception")
                [chargeProd, sigma, epsilon] = self._new_system_exceptions[new_index_atom_pair]
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd*0.0, sigma, epsilon*0.0)
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

                self._hybrid_system_forces['core_sterics_force'].addExclusion(atom_pair[0], atom_pair[1])



            # Check if the pair is present in the reverse order and use that if so
            elif new_index_atom_pair[::-1] in self._new_system_exceptions:
                _logger.debug(f"\t\thandle_nonbonded: _handle_hybrid_exceptions: {new_index_atom_pair[::-1]} is a new system exception")
                [chargeProd, sigma, epsilon] = self._new_system_exceptions[new_index_atom_pair[::-1]]
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd*0.0, sigma, epsilon*0.0)
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

                self._hybrid_system_forces['core_sterics_force'].addExclusion(atom_pair[0], atom_pair[1])


            # If it's not handled by an exception in the original system, we just add the regular parameters as an exception
            # else:
            #     _logger.info(f"\t\thandle_nonbonded: _handle_hybrid_exceptions: {new_index_atom_pair} is NOT a new exception...perhaps this is a problem!")
            #     [charge0, sigma0, epsilon0] = self._new_system_forces['NonbondedForce'].getParticleParameters(new_index_atom_pair[0])
            #     [charge1, sigma1, epsilon1] = self._new_system_forces['NonbondedForce'].getParticleParameters(new_index_atom_pair[1])
            #     chargeProd = charge0*charge1
            #     epsilon = unit.sqrt(epsilon0*epsilon1)
            #     sigma = 0.5*(sigma0+sigma1)
            #     self._hybrid_system_forces['standard_nonbonded_force'].addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)
            #     self._hybrid_system_forces['core_sterics_force'].addExclusion(atom_pair[0], atom_pair[1]) # add exclusion to ensure exceptions are consistent

    def _handle_original_exceptions(self):
        """
        This method ensures that exceptions present in the original systems are present in the hybrid appropriately.
        """
        # Get what we need to find the exceptions from the new and old systems:
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = {value: key for key, value in self._old_to_hybrid_map.items()}
        hybrid_to_new_map = {value: key for key, value in self._new_to_hybrid_map.items()}

        # First, loop through the old system's exceptions and add them to the hybrid appropriately:
        _logger.debug(f"\tlooping over old system exceptions...")
        for exception_pair, exception_parameters in self._old_system_exceptions.items():

            [index1_old, index2_old] = exception_pair

            [chargeProd_old, sigma_old, epsilon_old] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}


            # In this case, the interaction is only covered by the regular nonbonded force, and as such will be copied to that force
            # In the unique-old case, it is handled elsewhere due to internal peculiarities regarding exceptions
            if index_set.issubset(self._atom_classes['environment_atoms']):
                _logger.debug(f"\t\thandle_nonbonded: _handle_original_exceptions: {exception_pair} is an environment exception pair")
                self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid, index2_hybrid, chargeProd_old, sigma_old, epsilon_old)
                self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)

            # We have already handled unique old - unique old exceptions
            elif len(index_set.intersection(self._atom_classes['unique_old_atoms'])) == 2:
                _logger.debug(f"\t\thandle_nonbonded: _handle_original_exceptions: {exception_pair} is a unique_old-unique_old exception pair (already handled).")
                continue

            # Otherwise, check if one of the atoms in the set is in the unique_old_group and the other is not:
            elif len(index_set.intersection(self._atom_classes['unique_old_atoms'])) == 1:
                _logger.debug(f"\t\thandle_nonbonded: _handle_original_exceptions: {exception_pair} is a unique_old-core or unique_old-environment exception pair.")
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid, index2_hybrid, chargeProd_old*0.0, sigma_old, epsilon_old*0.0)
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid, index2_hybrid, chargeProd_old, sigma_old, epsilon_old)

                self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)

            # If the exception particles are neither solely old unique, solely environment, nor contain any unique old atoms, they are either core/environment or core/core
            # In this case, we need to get the parameters from the exception in the other (new) system, and interpolate between the two
            else:
                _logger.debug(f"\t\thandle_nonbonded: _handle_original_exceptions: {exception_pair} is a core-core or core-environment exception pair.")
                # First get the new indices.
                index1_new = hybrid_to_new_map[index1_hybrid]
                index2_new = hybrid_to_new_map[index2_hybrid]

                # Get the exception parameters:
                new_exception_parms= self._find_exception(new_system_nonbonded_force, index1_new, index2_new)

                # If there's no new exception, then we should just set the exception parameters to be the nonbonded parameters
                if not new_exception_parms:
                    [charge1_new, sigma1_new, epsilon1_new] = new_system_nonbonded_force.getParticleParameters(index1_new)
                    [charge2_new, sigma2_new, epsilon2_new] = new_system_nonbonded_force.getParticleParameters(index2_new)

                    chargeProd_new = charge1_new * charge2_new
                    sigma_new = 0.5 * (sigma1_new + sigma2_new)
                    epsilon_new = unit.sqrt(epsilon1_new*epsilon2_new)
                else:
                    [index1_new, index2_new, chargeProd_new, sigma_new, epsilon_new] = new_exception_parms

                # Interpolate between old and new
                exception_index = self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid, index2_hybrid, chargeProd_old, sigma_old, epsilon_old)
                self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset('lambda_electrostatics_core', exception_index, (chargeProd_new - chargeProd_old), 0, 0)
                self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset('lambda_sterics_core', exception_index, 0, (sigma_new - sigma_old), (epsilon_new - epsilon_old))
                self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)

        # Now, loop through the new system to collect remaining interactions. The only that remain here are
        # uniquenew-uniquenew, uniquenew-core, and uniquenew-environment. There might also be core-core, since not all
        # core-core exceptions exist in both
        _logger.debug(f"\tlooping over new system exceptions...")
        for exception_pair, exception_parameters in self._new_system_exceptions.items():
            [index1_new, index2_new] = exception_pair
            [chargeProd_new, sigma_new, epsilon_new] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]

            index_set = {index1_hybrid, index2_hybrid}

            # If it's a subset of unique_new_atoms, then this is an intra-unique interaction and should have its exceptions
            # specified in the regular nonbonded force. However, this is handled elsewhere as above due to pecularities with exception handling
            if index_set.issubset(self._atom_classes['unique_new_atoms']):
                _logger.debug(f"\t\thandle_nonbonded: _handle_original_exceptions: {exception_pair} is a unique_new-unique_new exception pair (already handled).")
                continue

            # Look for the final class- interactions between uniquenew-core and uniquenew-environment. They are treated
            # similarly: they are simply on and constant the entire time (as a valence term)
            elif len(index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
                _logger.debug(f"\t\thandle_nonbonded: _handle_original_exceptions: {exception_pair} is a unique_new-core or unique_new-environment exception pair.")
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid, index2_hybrid, chargeProd_new*0.0, sigma_new, epsilon_new*0.0)
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid, index2_hybrid, chargeProd_new, sigma_new, epsilon_new)

                self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)

            # However, there may be a core exception that exists in one system but not the other (ring closure)
            elif index_set.issubset(self._atom_classes['core_atoms']):
                _logger.debug(f"\t\thandle_nonbonded: _handle_original_exceptions: {exception_pair} is a core-core exception pair.")

                # Get the old indices
                try:
                    index1_old = self._topology_proposal.new_to_old_atom_map[index1_new]
                    index2_old = self._topology_proposal.new_to_old_atom_map[index2_new]
                except KeyError:
                    continue

                # See if it's also in the old nonbonded force. if it is, then we don't need to add it.
                # But if it's not, we need to interpolate
                if not self._find_exception(old_system_nonbonded_force, index1_old, index2_old):

                    [charge1_old, sigma1_old, epsilon1_old] = old_system_nonbonded_force.getParticleParameters(index1_old)
                    [charge2_old, sigma2_old, epsilon2_old] = old_system_nonbonded_force.getParticleParameters(index2_old)

                    chargeProd_old = charge1_old*charge2_old
                    sigma_old = 0.5 * (sigma1_old + sigma2_old)
                    epsilon_old = unit.sqrt(epsilon1_old*epsilon2_old)

                    exception_index = self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid,
                                                                                                          index2_hybrid,
                                                                                                          chargeProd_old,
                                                                                                          sigma_old,
                                                                                                          epsilon_old)

                    self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset(
                        'lambda_electrostatics_core', exception_index, (chargeProd_new - chargeProd_old), 0, 0)

                    self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset('lambda_sterics_core',
                                                                                                       exception_index,
                                                                                                       0, (sigma_new - sigma_old),
                                                                                                       (epsilon_new - epsilon_old))

                    self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)

    def handle_old_new_exceptions(self):
        """
        Find the exceptions associated with old-old and old-core interactions, as well as new-new and new-core interactions.  Theses exceptions will be placed in
        CustomBondedForce that will interpolate electrostatics and a softcore potential.
        """
        from openmmtools.constants import ONE_4PI_EPS0 # OpenMM constant for Coulomb interactions (implicitly in md_unit_system units)

        old_new_nonbonded_exceptions = "U_electrostatics + U_sterics;"

        if self._softcore_LJ_v2:
            old_new_nonbonded_exceptions += "U_sterics = select(step(r - r_LJ), 4*epsilon*x*(x-1.0), U_sterics_quad);"
            old_new_nonbonded_exceptions += f"U_sterics_quad = Force*(((r - r_LJ)^2)/2 - (r - r_LJ)) + U_sterics_cut;"
            old_new_nonbonded_exceptions += f"U_sterics_cut = 4*epsilon*((sigma/r_LJ)^6)*(((sigma/r_LJ)^6) - 1.0);"
            old_new_nonbonded_exceptions += f"Force = -4*epsilon*((-12*sigma^12)/(r_LJ^13) + (6*sigma^6)/(r_LJ^7));"
            old_new_nonbonded_exceptions += f"x = (sigma/r)^6;"
            old_new_nonbonded_exceptions += f"r_LJ = softcore_alpha*((26/7)*(sigma^6)*lambda_sterics_deprecated)^(1/6);"
            old_new_nonbonded_exceptions += f"lambda_sterics_deprecated = new_interaction*(1.0 - lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"
        else:
            old_new_nonbonded_exceptions += "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            old_new_nonbonded_exceptions += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            old_new_nonbonded_exceptions += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
            old_new_nonbonded_exceptions += "lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"

        old_new_nonbonded_exceptions += "U_electrostatics = (lambda_electrostatics_insert * unique_new + unique_old * (1 - lambda_electrostatics_delete)) * ONE_4PI_EPS0*chargeProd/r;"
        old_new_nonbonded_exceptions += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0

        old_new_nonbonded_exceptions += "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;" # interpolation
        old_new_nonbonded_exceptions += "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"

        old_new_nonbonded_exceptions += "lambda_sterics = new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
        old_new_nonbonded_exceptions += "new_interaction = delta(1-unique_new); old_interaction = delta(1-unique_old);"


        nonbonded_exceptions_force = openmm.CustomBondForce(old_new_nonbonded_exceptions)
        self._hybrid_system.addForce(nonbonded_exceptions_force)
        _logger.debug(f"\thandle_old_new_exceptions: {nonbonded_exceptions_force} added to hybrid system")

        # For reference, set name in force dict
        self._hybrid_system_forces['old_new_exceptions_force'] = nonbonded_exceptions_force

        if self._softcore_LJ_v2:
            nonbonded_exceptions_force.addGlobalParameter("softcore_alpha", self._softcore_LJ_v2_alpha)
        else:
            nonbonded_exceptions_force.addGlobalParameter("softcore_alpha", self.softcore_alpha)
        nonbonded_exceptions_force.addGlobalParameter("lambda_electrostatics_insert", 0.0) # electrostatics
        nonbonded_exceptions_force.addGlobalParameter("lambda_electrostatics_delete", 0.0) # electrostatics
        nonbonded_exceptions_force.addGlobalParameter("lambda_sterics_insert", 0.0) # sterics insert
        nonbonded_exceptions_force.addGlobalParameter("lambda_sterics_delete", 0.0) # sterics delete

        for parameter in ['chargeProd','sigmaA', 'epsilonA', 'sigmaB', 'epsilonB', 'unique_old', 'unique_new']:
            nonbonded_exceptions_force.addPerBondParameter(parameter)

        # Prepare for exceptions loop by grabbing nonbonded forces, hybrid_to_old/new maps
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = {value: key for key, value in self._old_to_hybrid_map.items()}
        hybrid_to_new_map = {value: key for key, value in self._new_to_hybrid_map.items()}

        # First, loop through the old system's exceptions and add them to the hybrid appropriately:
        for exception_pair, exception_parameters in self._old_system_exceptions.items():

            [index1_old, index2_old] = exception_pair
            [chargeProd_old, sigma_old, epsilon_old] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}

            # Otherwise, check if one of the atoms in the set is in the unique_old_group and the other is not:
            if len(index_set.intersection(self._atom_classes['unique_old_atoms'])) > 0 and (chargeProd_old.value_in_unit_system(unit.md_unit_system) != 0.0 or epsilon_old.value_in_unit_system(unit.md_unit_system) != 0.0):
                _logger.debug(f"\t\thandle_old_new_exceptions: {exception_pair} is a unique_old exception pair.")
                if self._interpolate_14s:
                    # If we are interpolating 1,4s, then we anneal this term off; otherwise, the exception force is constant and already handled in the standard nonbonded force
                    nonbonded_exceptions_force.addBond(index1_hybrid, index2_hybrid, [chargeProd_old, sigma_old, epsilon_old, sigma_old, epsilon_old*0.0, 1, 0])



        # Next, loop through the new system's exceptions and add them to the hybrid appropriately
        for exception_pair, exception_parameters in self._new_system_exceptions.items():
            [index1_new, index2_new] = exception_pair
            [chargeProd_new, sigma_new, epsilon_new] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]

            index_set = {index1_hybrid, index2_hybrid}

            # Look for the final class- interactions between uniquenew-core and uniquenew-environment. They are treated
            # similarly: they are simply on and constant the entire time (as a valence term)
            if len(index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0 and (chargeProd_new.value_in_unit_system(unit.md_unit_system) != 0.0 or epsilon_new.value_in_unit_system(unit.md_unit_system) != 0.0):
                _logger.debug(f"\t\thandle_old_new_exceptions: {exception_pair} is a unique_new exception pair.")
                if self._interpolate_14s:
                    # If we are interpolating 1,4s, then we anneal this term on; otherwise, the exception force is constant and already handled in the standard nonbonded force
                    nonbonded_exceptions_force.addBond(index1_hybrid, index2_hybrid, [chargeProd_new, sigma_new, epsilon_new*0.0, sigma_new, epsilon_new, 0, 1])


    def _find_exception(self, force, index1, index2):
        """
        Find the exception that corresponds to the given indices in the given system

        Parameters
        ----------
        force : openmm.NonbondedForce object
            System containing the exceptions
        index1 : int
            The index of the first atom (order is unimportant)
        index2 : int
            The index of the second atom (order is unimportant)

        Returns
        -------
        exception_parameters : list
            List of exception parameters
        """
        index_set = {index1, index2}

        # Loop through the exceptions and try to find one matching the criteria
        for exception_idx in range(force.getNumExceptions()):
            exception_parameters = force.getExceptionParameters(exception_idx)
            if index_set==set(exception_parameters[:2]):
                return exception_parameters
        return []

    def _compute_hybrid_positions(self):
        """
        The positions of the hybrid system. Dimensionality is (n_environment + n_core + n_old_unique + n_new_unique)
        The positions are assigned by first copying all the mapped positions from the old system in, then copying the
        mapped positions from the new system. This means that there is an assumption that the positions common to old
        and new are the same (which is the case for perses as-is).

        Returns
        -------
        hybrid_positions : np.ndarray [n, 3]
            Positions of the hybrid system, in nm
        """
        # Get unitless positions
        old_positions_without_units = np.array(self._old_positions.value_in_unit(unit.nanometer))
        new_positions_without_units = np.array(self._new_positions.value_in_unit(unit.nanometer))

        # Determine the number of particles in the system
        n_atoms_hybrid = self._hybrid_system.getNumParticles()

        # Initialize an array for hybrid positions
        hybrid_positions_array = np.zeros([n_atoms_hybrid, 3])

        # Loop through the old system indices, and assign positions.
        for old_index, hybrid_index in self._old_to_hybrid_map.items():
            hybrid_positions_array[hybrid_index, :] = old_positions_without_units[old_index, :]

        # Do the same for new indices. Note that this overwrites some coordinates, but as stated above, the assumption
        # is that these are the same.
        for new_index, hybrid_index in self._new_to_hybrid_map.items():
            hybrid_positions_array[hybrid_index, :] = new_positions_without_units[new_index, :]

        return unit.Quantity(hybrid_positions_array, unit=unit.nanometers)

    def _create_topology(self):
        """
        Create an mdtraj topology corresponding to the hybrid system.
        This is purely for writing out trajectories--it is not expected to be parameterized.

        Returns
        -------
        hybrid_topology : mdtraj.Topology
        """
        # First, make an md.Topology of the old system:
        old_topology = md.Topology.from_openmm(self._topology_proposal.old_topology)

        # Now make a copy for the hybrid:
        hybrid_topology = copy.deepcopy(old_topology)

        # Next, make a topology of the new system:
        new_topology = md.Topology.from_openmm(self._topology_proposal.new_topology)

        added_atoms = dict()

        # Get the core atoms in the new index system (as opposed to the hybrid index system). We will need this later
        core_atoms_new_indices = {self._hybrid_to_new_map[core_atom] for core_atom in self._atom_classes['core_atoms']}

        # Now, add each unique new atom to the topology (this is the same order as the system)
        for particle_idx in self._topology_proposal.unique_new_atoms:
            new_particle_hybrid_idx = self._new_to_hybrid_map[particle_idx]
            new_system_atom = new_topology.atom(particle_idx)

            # First, we get the residue in the new system associated with this atom
            new_system_residue = new_system_atom.residue

            # Next, we have to enumerate the other atoms in that residue to find mapped atoms
            new_system_atom_set = {atom.index for atom in new_system_residue.atoms}

            # Now, we find the subset of atoms that are mapped. These must be in the "core" category, since they are mapped
            # and part of a changing residue
            mapped_new_atom_indices = core_atoms_new_indices.intersection(new_system_atom_set)

            # Now get the old indices of the above atoms so that we can find the appropriate residue in the old system
            # for this we can use the new to old atom map
            mapped_old_atom_indices = [self._topology_proposal.new_to_old_atom_map[atom_idx] for atom_idx in mapped_new_atom_indices]

            # We can just take the first one--they all have the same residue
            first_mapped_old_atom_index = mapped_old_atom_indices[0]

            # Get the atom object corresponding to this index from the hybrid (which is a deepcopy of the old)
            mapped_hybrid_system_atom = hybrid_topology.atom(first_mapped_old_atom_index)

            # Get the residue that is relevant to this atom
            mapped_residue = mapped_hybrid_system_atom.residue

            # Add the atom using the mapped residue
            added_atoms[new_particle_hybrid_idx] = hybrid_topology.add_atom(new_system_atom.name, new_system_atom.element, mapped_residue)

        # Now loop through the bonds in the new system, and if the bond contains a unique new atom, then add it to the hybrid topology
        for (atom1, atom2) in new_topology.bonds:
            atom1_index_in_hybrid = self._new_to_hybrid_map[atom1.index]
            atom2_index_in_hybrid = self._new_to_hybrid_map[atom2.index]

            # If at least one atom is in the unique new class, we need to add it to the hybrid system
            if atom1_index_in_hybrid in self._atom_classes['unique_new_atoms'] or atom2_index_in_hybrid in self._atom_classes['unique_new_atoms']:
                if atom1.index in self._atom_classes['unique_new_atoms']:
                    atom1_to_bond = added_atoms[atom1.index]
                else:
                    atom1_to_bond = atom1

                if atom2.index in self._atom_classes['unique_new_atoms']:
                    atom2_to_bond = added_atoms[atom2.index]
                else:
                    atom2_to_bond = atom2

                hybrid_topology.add_bond(atom1_to_bond, atom2_to_bond)

        return hybrid_topology

    def old_positions(self, hybrid_positions):
        """
        Get the positions corresponding to the old system

        Parameters
        ----------
        hybrid_positions : [n, 3] np.ndarray with unit
            The positions of the hybrid system

        Returns
        -------
        old_positions : [m, 3] np.ndarray with unit
            The positions of the old system
        """
        n_atoms_old = self._topology_proposal.n_atoms_old
        old_positions = unit.Quantity(np.zeros([n_atoms_old, 3]), unit=unit.nanometer)
        for idx in range(n_atoms_old):
            old_positions[idx, :] = hybrid_positions[idx, :]
        return old_positions

    def new_positions(self, hybrid_positions):
        """
        Get the positions corresponding to the new system.

        Parameters
        ----------
        hybrid_positions : [n, 3] np.ndarray with unit
            The positions of the hybrid system

        Returns
        -------
        new_positions : [m, 3] np.ndarray with unit
            The positions of the new system
        """
        n_atoms_new = self._topology_proposal.n_atoms_new
        new_positions = unit.Quantity(np.zeros([n_atoms_new, 3]), unit=unit.nanometer)
        for idx in range(n_atoms_new):
            new_positions[idx, :] = hybrid_positions[self._new_to_hybrid_map[idx], :]
        return new_positions

    @property
    def hybrid_system(self):
        """
        The hybrid system.

        Returns
        -------
        hybrid_system : openmm.System
            The system representing a hybrid between old and new topologies
        """
        return self._hybrid_system

    @property
    def new_to_hybrid_atom_map(self):
        """
        Give a dictionary that maps new system atoms to the hybrid system.

        Returns
        -------
        new_to_hybrid_atom_map : dict of {int, int}
            The mapping of atoms from the new system to the hybrid
        """
        return self._new_to_hybrid_map

    @property
    def old_to_hybrid_atom_map(self):
        """
        Give a dictionary that maps old system atoms to the hybrid system.

        Returns
        -------
        old_to_hybrid_atom_map : dict of {int, int}
            The mapping of atoms from the old system to the hybrid
        """
        return self._old_to_hybrid_map

    @property
    def hybrid_positions(self):
        """
        The positions of the hybrid system. Dimensionality is (n_environment + n_core + n_old_unique + n_new_unique)
        The positions are assigned by first copying all the mapped positions from the old system in, then copying the
        mapped positions from the new system.

        Returns
        -------
        hybrid_positions : [n, 3] Quantity nanometers
        """
        return self._hybrid_positions

    @property
    def hybrid_topology(self):
        """
        An MDTraj hybrid topology for the purpose of writing out trajectories. Note that we do not expect this to be
        able to be parameterized by the openmm forcefield class.

        Returns
        -------
        hybrid_topology : mdtraj.Topology
        """
        return self._hybrid_topology

    @property
    def omm_hybrid_topology(self):
        """
        An OpenMM format of the hybrid topology. Also cannot be used to parameterize system, only to write out trajectories.

        Returns
        -------
        hybrid_topology : simtk.openmm.app.Topology
        """
        return md.Topology.to_openmm(self._hybrid_topology)

class RepartitionedHybridTopologyFactory(HybridTopologyFactory):
    """
    subclass of the HybridTopologyFactory to allow for more expansive alchemical regions and controllability
    """
    def __init__(self,
                 topology_proposal,
                 current_positions,
                 new_positions,
                 endstate,
                 alchemical_region=None,
                 **kwargs):
        """
        arguments
            topology_proposal : TopologyProposal
                topology proposal of the region of interest
            current_positions : simtk.unit.Quantity
                positions of old system
            new_positions : simtk.unit.Quantity
                positions of new system
            alchemical_region : list, default None
                list of atoms comprising the alchemical region; if None, core_atoms + unique_new_atoms + unique_old_atoms are alchemical region
            endstate : int
                the lambda endstate to parameterize
        """
        from itertools import chain

        self._topology_proposal = topology_proposal
        self._old_system = copy.deepcopy(topology_proposal.old_system)
        self._new_system = copy.deepcopy(topology_proposal.new_system)
        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}
        self._hybrid_system_forces = dict()
        self._old_positions = current_positions
        self._new_positions = new_positions
        self._endstate = endstate

        # Prepare dicts of forces, which will be useful later
        # TODO: Store this as self._system_forces[name], name in ('old', 'new', 'hybrid') for compactness
        self._old_system_forces = {type(force).__name__ : force for force in self._old_system.getForces()}
        self._new_system_forces = {type(force).__name__ : force for force in self._new_system.getForces()}
        _logger.info(f"Old system forces: {self._old_system_forces.keys()}")
        _logger.info(f"New system forces: {self._new_system_forces.keys()}")

        # Check that there are no unknown forces in the new and old systems:
        for system_name in ('old', 'new'):
            force_names = getattr(self, '_{}_system_forces'.format(system_name)).keys()
            unknown_forces = set(force_names) - set(self._known_forces)
            if len(unknown_forces) > 0:
                raise ValueError("Unkown forces {} encountered in {} system" % (unknown_forces, system_name))
        _logger.info("No unknown forces.")

        # Get and store the nonbonded method from the system:
        self._nonbonded_method = self._old_system_forces['NonbondedForce'].getNonbondedMethod()
        _logger.info(f"Nonbonded method to be used (i.e. from old system): {self._nonbonded_method}")

        # Start by creating an empty system. This will become the hybrid system.
        self._hybrid_system = openmm.System()

        # Begin by copying all particles in the old system to the hybrid system. Note that this does not copy the
        # interactions. It does, however, copy the particle masses. In general, hybrid index and old index should be
        # the same.
        # TODO: Refactor this into self._add_particles()
        _logger.info("Adding and mapping old atoms to hybrid system...")
        for particle_idx in range(self._topology_proposal.n_atoms_old):
            particle_mass = self._old_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._old_to_hybrid_map[particle_idx] = hybrid_idx

            #If the particle index in question is mapped, make sure to add it to the new to hybrid map as well.
            if particle_idx in self._topology_proposal.old_to_new_atom_map.keys():
                particle_index_in_new_system = self._topology_proposal.old_to_new_atom_map[particle_idx]
                self._new_to_hybrid_map[particle_index_in_new_system] = hybrid_idx

        # Next, add the remaining unique atoms from the new system to the hybrid system and map accordingly.
        # As before, this does not copy interactions, only particle indices and masses.
        _logger.info("Adding and mapping new atoms to hybrid system...")
        for particle_idx in self._topology_proposal.unique_new_atoms:
            particle_mass = self._new_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._new_to_hybrid_map[particle_idx] = hybrid_idx

        # Check that if there is a barostat in the original system, it is added to the hybrid.
        # We copy the barostat from the old system.
        if "MonteCarloBarostat" in self._old_system_forces.keys():
            barostat = copy.deepcopy(self._old_system_forces["MonteCarloBarostat"])
            self._hybrid_system.addForce(barostat)
            _logger.info("Added MonteCarloBarostat.")
        else:
            _logger.info("No MonteCarloBarostat added.")

        # Copy over the box vectors:
        box_vectors = self._old_system.getDefaultPeriodicBoxVectors()
        self._hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)
        _logger.info(f"getDefaultPeriodicBoxVectors added to hybrid: {box_vectors}")

        # Create the opposite atom maps for use in nonbonded force processing; let's omit this from logger
        self._hybrid_to_old_map = {value : key for key, value in self._old_to_hybrid_map.items()}
        self._hybrid_to_new_map = {value : key for key, value in self._new_to_hybrid_map.items()}

        # Assign atoms to one of the classes described in the class docstring
        self._atom_classes = self._determine_atom_classes()
        _logger.info("Determined atom classes.")

        # Construct dictionary of exceptions in old and new systems
        _logger.info("Generating old system exceptions dict...")
        self._old_system_exceptions = self._generate_dict_from_exceptions(self._old_system_forces['NonbondedForce'])
        _logger.info("Generating new system exceptions dict...")
        self._new_system_exceptions = self._generate_dict_from_exceptions(self._new_system_forces['NonbondedForce'])

        self._validate_disjoint_sets()

        # Copy constraints, checking to make sure they are not changing
        _logger.info("Handling constraints...")
        self._handle_constraints()

        # Copy over relevant virtual sites
        _logger.info("Handling virtual sites...")
        self._handle_virtual_sites()

        # Combine alchemical regions
        default_alchemical_region = set(chain(self._atom_classes['core_atoms'], self._atom_classes['unique_new_atoms'], self._atom_classes['unique_old_atoms']))
        if alchemical_region is None:
            self._alchemical_region = default_alchemical_region
        else:
            assert default_alchemical_region.issubset(set(alchemical_region)), f"the given alchemical region must include _all_ atoms in the default alchemical region"
            self._alchemical_region = set(alchemical_region).union(default_alchemical_region)

        # First thing to do is to copy over all of the standard valence force objects into the hybrid system
        self._handle_bonds()
        self._handle_angles()
        self._handle_torsions()

        # Then add the nonbonded force (this is _slightly_ trickier)
        self._handle_nonbonded()

        # The last thing to do is call the alchemical factory on the _hybrid_system
        self._alchemify()

    def _handle_bonds(self):
        """
        Copy over the appropriate bonds from the old or new system to the hybrid system;

        If the endstate is old, then we copy all of the old system force terms to the hybrid system and then iterate through
        the new system, copying over all of the force terms that contain a unique new atom;
        Do the opposite if at the new endstate
        """
        # Define the force we are going to write to
        self._hybrid_system_forces['HarmonicBondForce'] = openmm.HarmonicBondForce()
        to_force = self._hybrid_system_forces['HarmonicBondForce']

        # Define the template force and the auxiliary force
        if self._endstate == 0:
            template_force = self._old_system_forces['HarmonicBondForce']
            aux_force = self._new_system_forces['HarmonicBondForce']
            target_index_set = self._atom_classes['unique_new_atoms']
        elif self._endstate == 1:
            template_force = self._new_system_forces['HarmonicBondForce']
            aux_force = self._old_system_forces['HarmonicBondForce']
            target_index_set = self._atom_classes['unique_old_atoms']
        else:
            raise Exception(f"endstate must be 0 or 1")

        # Copy over the template force...
        for idx in range(template_force.getNumBonds()):
            p1, p2, length, k = template_force.getBondParameters(idx)
            to_force.addBond(p1, p2, length, k)

        # Query the auxiliary force to extract and copy over the 'special' terms that don't exist in the template force
        for idx in range(aux_force.getNumBonds()):
            p1, p2, length, k = aux_force.getBondParameters(idx)
            if set([p1, p2]).intersection(target_index_set) != set():
                # If there is a target atom in the auxiliary term, write it to the hybrid force
                to_force.addBond(p1, p2, length, k)

        # Then add the to_force to the hybrid_system
        self._hybrid_system.addForce(to_force)

    def _handle_angles(self):
        """
        Copy over the appropriate angles from the old or new system to the hybrid system;

        If the endstate is old, then we copy all of the old system force terms to the hybrid system and then iterate through
        the new system, copying over all of the force terms that contain a unique new atom;
        Do the opposite if at the new endstate

        """
        # Define the force we are going to write to
        self._hybrid_system_forces['HarmonicAngleForce'] = openmm.HarmonicAngleForce()
        to_force = self._hybrid_system_forces['HarmonicAngleForce']

        # Define the template force and the auxiliary force
        if self._endstate == 0:
            template_force = self._old_system_forces['HarmonicAngleForce']
            aux_force = self._new_system_forces['HarmonicAngleForce']
            target_index_set = self._atom_classes['unique_new_atoms']
        elif self._endstate == 1:
            template_force = self._new_system_forces['HarmonicAngleForce']
            aux_force = self._old_system_forces['HarmonicAngleForce']
            target_index_set = self._atom_classes['unique_old_atoms']
        else:
            raise Exception(f"endstate must be 0 or 1")

        # Copy over the template force...
        for idx in range(template_force.getNumAngles()):
            p1, p2, p3, angle, k = template_force.getAngleParameters(idx)
            to_force.addAngle(p1, p2, p3, angle, k)

        # Query the auxiliary force to extract and copy over the 'special' terms that don't exist in the template force
        for idx in range(aux_force.getNumAngles()):
            p1, p2, p3, angle, k = aux_force.getAngleParameters(idx)
            if set([p1, p2, p3]).intersection(target_index_set) != set():
                # If there is a target atom in the auxiliary term, write it to the hybrid force
                to_force.addAngle(p1, p2, p3, angle, k)

        # Then add the to_force to the hybrid_system
        self._hybrid_system.addForce(to_force)

    def _handle_torsions(self):
        """
        Copy over the appropriate torsions from the old or new system to the hybrid system;

        If the endstate is old, then we copy all of the old system force terms to the hybrid system and then iterate through
        the new system, copying over all of the force terms that contain a unique new atom;
        Do the opposite if at the new endstate

        """
        # Define the force we are going to write to
        self._hybrid_system_forces['PeriodicTorsionForce'] = openmm.PeriodicTorsionForce()
        to_force = self._hybrid_system_forces['PeriodicTorsionForce']

        # Define the template force and the auxiliary force
        if self._endstate == 0:
            template_force = self._old_system_forces['PeriodicTorsionForce']
            aux_force = self._new_system_forces['PeriodicTorsionForce']
            target_index_set = self._atom_classes['unique_new_atoms']
        elif self._endstate == 1:
            template_force = self._new_system_forces['PeriodicTorsionForce']
            aux_force = self._old_system_forces['PeriodicTorsionForce']
            target_index_set = self._atom_classes['unique_old_atoms']
        else:
            raise Exception(f"endstate must be 0 or 1")

        # Copy over the template force...
        for idx in range(template_force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = template_force.getTorsionParameters(idx)
            to_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)

        # Query the auxiliary force to extract and copy over the 'special' terms that don't exist in the template force
        for idx in range(aux_force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = aux_force.getTorsionParameters(idx)
            if set([p1, p2, p3, p4]).intersection(target_index_set) != set():
                # If there is a target atom in the auxiliary term, write it to the hybrid force
                to_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)

        # Then add the to_force to the hybrid_system
        self._hybrid_system.addForce(to_force)

    def _handle_nonbonded(self):
        """
        Transcribe nonbonded forces

        """
        # TODO: dominic

    def _alchemify(self):
        """
        Generate an AlchemicalFactory with an appropriate Alchemical region and 'alchemify' the hybrid system

        """
        # TODO: dominic

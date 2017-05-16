import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
import copy


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
    """

    def __init__(self, topology_proposal, current_positions, new_positions):
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
        """
        self._topology_proposal = topology_proposal
        self._old_system = copy.deepcopy(topology_proposal.old_system)
        self._new_system = copy.deepcopy(topology_proposal.new_system)
        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}

        #start by creating an empty system and topology. These will become the hybrid system and topology.
        self._hybrid_system = openmm.System()
        self._hybrid_topology = app.Topology()

        #begin by copying all particles in the old system to the hybrid system. Note that this does not copy the
        #interactions. It does, however, copy the particle masses. In general, hybrid index and old index should be
        #the same.
        for particle_idx in range(self._topology_proposal.natoms_old):
            particle_mass = self._old_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._old_to_hybrid_map[particle_idx] = hybrid_idx

            #If the particle index in question is mapped, make sure to add it to the new to hybrid map as well.
            if particle_idx in self._topology_proposal.old_to_new_atom_map.keys():
                particle_index_in_new_system = self._topology_proposal.old_to_new_atom_map[particle_idx]
                self._new_to_hybrid_map[particle_index_in_new_system] = hybrid_idx

        #Next, add the remaining unique atoms from the new system to the hybrid system and map accordingly.
        #As before, this does not copy interactions, only particle indices and masses.
        for particle_idx in self._topology_proposal.unique_new_atoms:
            particle_mass = self._new_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._new_to_hybrid_map[particle_idx] = hybrid_idx

        #assign atoms to one of the classes described in the class docstring
        self._atom_classes = self._determine_atom_classes()

        #verify that no constraints are changing over the course of the switching.
        self._constraint_check()

        #loop through the forces in the old system and begin to dispatch them to handlers that will add appropriate
        #force terms in the hybrid system. The scheme here is to always add all interactions (with appropriate lambda
        #terms) from the old system, and then only add unique new interactions from the new system.
        old_system_forces = self._topology_proposal.old_system.getForces()

        for force in old_system_forces:
            force_class_name = type(force).__name__
            if force_class_name=="HarmonicBondForce":
                #dispatch to bond force handler
                pass
            elif force_class_name=="HarmonicAngleForce":
                pass
                #dispatch to angle force handler
            elif force_class_name=="PeriodicTorsionForce":
                pass
                #dispatch to torsion force handler
            elif force_class_name=="NonbondedForce":
                pass
                #dispatch to nonbonded force handler
            else:
                raise ValueError("An unknown force class is present.")


    def _get_core_atoms(self):
        """
        Determine which atoms in the old system are part of the "core" class. All necessary information is contained in
        the topology proposal passed to the constructor.

        Returns
        -------
        core_atoms : set of int
            The set of atoms (hybrid topology indexed) that are core atoms.
        environment_atoms : set of int
            The set of atoms (hybrid topology indexed) that are environment atoms.
        """

        core_atoms = set()

        #In order to be either a core or environment atom, the atom must be mapped.
        mapped_old_atoms_set = set(self._topology_proposal.old_to_new_atom_map.keys())
        mapped_new_atoms_set = set(self._topology_proposal.old_to_new_atom_map.values())
        mapped_hybrid_atoms_set = {self._old_to_hybrid_map[atom_idx] for atom_idx in mapped_old_atoms_set}

        #create sets for set arithmetic
        unique_old_set = set(self._topology_proposal.unique_old_atoms)
        unique_new_set = set(self._topology_proposal.unique_new_atoms)

        #we derive core atoms from the old topology:
        core_atoms_from_old = self._determine_core_atoms_in_topology(self._topology_proposal.old_topology,
                                                                     unique_old_set, mapped_old_atoms_set,
                                                                     self._old_to_hybrid_map)

        #we also derive core atoms from the new topology:
        core_atoms_from_new = self._determine_core_atoms_in_topology(self._topology_proposal.new_topology,
                                                                     unique_new_set, mapped_new_atoms_set,
                                                                     self._new_to_hybrid_map)

        #The union of the two will give the core atoms that can result from either new or old topology
        total_core_atoms = core_atoms_from_old.union(core_atoms_from_new)

        #as a side effect, we can now compute the environment atom indices too, by subtracting the core indices
        #from the mapped atom set (since any atom that is mapped but not core is environment)
        environment_atoms = mapped_hybrid_atoms_set.difference(total_core_atoms)

        return total_core_atoms, environment_atoms

    def _determine_core_atoms_in_topology(self, topology, unique_atoms, mapped_atoms, hybrid_map):
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

        Returns
        -------
        core_atoms : set of int
            set of core atom indices in hybrid topology
        """
        core_atoms = set()

        #loop through the residues to look for ones with unique atoms
        for residue in topology.residues():
            atom_indices_old_system = {atom.index for atom in residue.atoms()}

            #if the residue contains an atom index that is unique, then the residue is changing.
            #We determine this by checking if the atom indices of the residue have any intersection with the unique atoms
            if len(atom_indices_old_system.intersection(unique_atoms)) > 0:
                #we can add the atoms in this residue which are mapped to the core_atoms set:
                for atom_index in atom_indices_old_system:
                    if atom_index in mapped_atoms:
                        #we specifically want to add the hybrid atom.
                        hybrid_index = hybrid_map[atom_index]
                        core_atoms.add(hybrid_index)

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

        #first, find the unique old atoms, as this is the most straightforward:
        for atom_idx in self._topology_proposal.unique_old_atoms:
            hybrid_idx = self._old_to_hybrid_map[atom_idx]
            atom_classes['unique_old_atoms'].add(hybrid_idx)

        #Then the unique new atoms (this is substantially the same as above)
        for atom_idx in self._topology_proposal.unique_new_atoms:
            hybrid_idx = self._new_to_hybrid_map[atom_idx]
            atom_classes['unique_new_atoms'].add(hybrid_idx)

        core_atoms, environment_atoms = self._get_core_atoms()

        atom_classes['core_atoms'] = core_atoms
        atom_classes['environment_atoms'] = environment_atoms

        return atom_classes

    def _constraint_check(self):
        """
        This is a check to make sure that constraint lengths do not change over the course of the switching.
        In the future, we will determine a method to deal with this. Raises exception if a constraint length changes.
        """

        #this dict will be of the form {(atom1, atom2) : constraint_value}, with hybrid indices.
        constrained_atoms_dict = {}

        #first, loop through constraints in the old system and add them to the dict, with hybrid indices:
        for constraint_idx in range(self._topology_proposal.old_system.getNumConstraints()):
            atom1, atom2, constraint = self._topology_proposal.old_system.getConstraintParameters(constraint_idx)
            atom1_hybrid = self._old_to_hybrid_map[atom1]
            atom2_hybrid = self._old_to_hybrid_map[atom2]
            constrained_atoms_dict[(atom1_hybrid, atom2_hybrid)] = constraint

        #now, loop through constraints in the new system, and see if we are going to change a constraint length
        for constraint_idx in range(self._topology_proposal.new_system.getNumConstraints()):
            atom1, atom2, constraint = self._topology_proposal.new_system.getConstraintParameters(constraint_idx)
            atom1_hybrid = self._new_to_hybrid_map[atom1]
            atom2_hybrid = self._new_to_hybrid_map[atom2]

            #check if either permutation is in the keys
            if (atom1_hybrid, atom2_hybrid) in constrained_atoms_dict.keys():
                constraint_from_old_system = constrained_atoms_dict[(atom1_hybrid, atom2_hybrid)]
                if constraint != constraint_from_old_system:
                    raise ValueError("Constraints are changing during switching.")

            if (atom2_hybrid, atom1_hybrid) in constrained_atoms_dict.keys():
                constraint_from_old_system = constrained_atoms_dict[(atom2_hybrid, atom1_hybrid)]
                if constraint != constraint_from_old_system:
                    raise ValueError("Constraints are changing during switching.")

    def _bond_force_handler(self, bond_force):
        """
        This method decides how to add bond forces to the hybrid system, based on the compiled atom maps and
        Parameters
        ----------
        bond_force

        Returns
        -------

        """
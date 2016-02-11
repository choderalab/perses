"""
This file contains the base classes for topology proposals
"""

import simtk.openmm as openmm
import simtk.openmm.app as app
from collections import namedtuple
import copy
import os
import openeye.oechem as oechem
import numpy as np
import openeye.oeomega as oeomega
import tempfile
from openmoltools import forcefield_generators
import openeye.oegraphsim as oegraphsim
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import openmoltools
import logging
try:
    from subprocess import getoutput  # If python 3
except ImportError:
    from commands import getoutput  # If python 2


class TopologyProposal(object):
    """
    This is a container class with convenience methods to access various objects needed
    for a topology proposal

    Arguments
    ---------
    new_topology : simtk.openmm.Topology object
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.Topology object
        openmm Topology of the current system
    old_system : simtk.openmm.System object
        openm System of the current state
    logp_proposal : float
        log probability of the proposal
    new_to_old_atom_map : dict
        {new_atom_idx : old_atom_idx} map for the two systems
    chemical_state_key : str
        The current chemical state (unique)
    metadata : dict
        additional information of interest about the state

    Properties
    ----------
    new_topology : simtk.openmm.Topology object
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.Topology object
        openmm Topology of the current system
    old_system : simtk.openmm.System object
        openm System of the current state
    old_positions : [n, 3] np.array, Quantity
        positions of the old system
    logp_proposal : float
        log probability of the proposal
    new_to_old_atom_map : dict
        {new_atom_idx : old_atom_idx} map for the two systems
    old_to_new_atom_map : dict
        {old_atom_idx : new_atom_idx} map for the two systems
    unique_new_atoms : list of int
        List of indices of the unique new atoms
    unique_old_atoms : list of int
        List of indices of the unique old atoms
    natoms_new : int
        Number of atoms in the new system
    natoms_old : int
        Number of atoms in the old system
    chemical_state_key : str
        The current chemical state
    metadata : dict
        additional information of interest about the state
    """

    def __init__(self, new_topology=None, new_system=None, old_topology=None, old_system=None,
                 logp_proposal=None, new_to_old_atom_map=None,old_chemical_state_key=None, new_chemical_state_key=None, metadata=None):

        if new_chemical_state_key is None or old_chemical_state_key is None:
            raise ValueError("chemical_state_keys must be set.")
        self._new_topology = new_topology
        self._new_system = new_system
        self._old_topology = old_topology
        self._old_system = old_system
        self._logp_proposal = logp_proposal
        self._new_chemical_state_key = new_chemical_state_key
        self._old_chemical_state_key = old_chemical_state_key
        self._new_to_old_atom_map = new_to_old_atom_map
        self._old_to_new_atom_map = {old_atom : new_atom for new_atom, old_atom in new_to_old_atom_map.items()}
        self._unique_new_atoms = [atom for atom in range(self._new_system.getNumParticles()) if atom not in self._new_to_old_atom_map.keys()]
        self._unique_old_atoms = [atom for atom in range(self._old_system.getNumParticles()) if atom not in self._new_to_old_atom_map.values()]
        self._metadata = metadata

    @property
    def new_topology(self):
        return self._new_topology
    @property
    def new_system(self):
        return self._new_system
    @property
    def old_topology(self):
        return self._old_topology
    @property
    def old_system(self):
        return self._old_system
    @property
    def logp_proposal(self):
        return self._logp_proposal
    @property
    def new_to_old_atom_map(self):
        return self._new_to_old_atom_map
    @property
    def old_to_new_atom_map(self):
        return self._old_to_new_atom_map
    @property
    def unique_new_atoms(self):
        return self._unique_new_atoms
    @property
    def unique_old_atoms(self):
        return self._unique_old_atoms
    @property
    def n_atoms_new(self):
        return self._new_system.getNumParticles()
    @property
    def n_atoms_old(self):
        return self._old_system.getNumParticles()
    @property
    def new_chemical_state_key(self):
        return self._new_chemical_state_key
    @property
    def old_chemical_state_key(self):
        return self._old_chemical_state_key
    @property
    def metadata(self):
        return self._metadata

class ProposalEngine(object):
    """
    This defines a type which, given the requisite metadata, can produce Proposals (namedtuple)
    of new topologies.

    Arguments
    --------
    proposal_metadata : dict
        Contains information necessary to initialize proposal engine
    """

    def __init__(self, system_generator, proposal_metadata=None):
        self._system_generator = system_generator

    def propose(self, current_system, current_topology, current_metadata=None):
        """
        Base interface for proposal method.

        Arguments
        ---------
        current_system : simtk.openmm.System object
            The current system object
        current_topology : simtk.openmm.app.Topology object
            The current topology
        current_metadata : dict
            Additional metadata about the state
        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        return TopologyProposal(new_topology=app.Topology(), old_topology=app.Topology(), old_system=current_system, old_chemical_state_key="C", new_chemical_state_key="C", logp_proposal=0.0, new_to_old_atom_map={0 : 0}, metadata={'molecule_smiles' : 'CC'})

    def compute_state_key(self, topology):
        """
        Compute the corresponding state key of a given topology,
        according to this proposal engine's scheme.

        Parameters
        ----------
        topology : app.Topology
            the topology in question

        Returns
        -------
        chemical_state_key : str
            The chemical_state_key
        """
        pass

class PolymerProposalEngine(ProposalEngine):
    def __init__(self, system_generator, chain_id, proposal_metadata=None):
        super(PolymerProposalEngine,self).__init__(system_generator, proposal_metadata=proposal_metadata)
        self._chain_id = chain_id

    def propose(self, current_system, current_topology, current_metadata=None):
        return TopologyProposal(new_topology=app.Topology(), old_topology=app.Topology(), old_system=current_system, old_chemical_state_key="C", new_chemical_state_key="C", logp_proposal=0.0, new_to_old_atom_map={0 : 0}, metadata=current_metadata)

    def compute_state_key(self, topology):
        return ''

class PointMutationEngine(PolymerProposalEngine):
    """

    Arguments
    --------
    system_generator : SystemGenerator
    max_point_mutants : int 
    proposal_metadata : dict -- OPTIONAL
        Contains information necessary to initialize proposal engine
    chain_id : str
        id of the chain to mutate
        (using the first chain with the id, if there are multiple)
    allowed_mutations : list(list(tuple)) -- OPTIONAL
        default = None
        ('residue id to mutate','desired mutant residue name (3-letter code)')
    """

    def __init__(self, system_generator, max_point_mutants, chain_id, proposal_metadata=None, allowed_mutations=None):
        super(PointMutationEngine,self).__init__(system_generator, chain_id, proposal_metadata=proposal_metadata)
        self._max_point_mutants = max_point_mutants
        self._ff = system_generator.forcefield 
        self._templates = self._ff._templates
        self._allowed_mutations = allowed_mutations

    def propose(self, current_system, current_topology, current_metadata=None):
        """

        Arguments
        ---------
        current_system : simtk.openmm.System object
            The current system object
        current_topology : simtk.openmm.app.Topology object
            The current topology
        current_metadata : dict -- OPTIONAL
        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        # old_topology : simtk.openmm.app.Topology
        old_topology = copy.deepcopy(current_topology)
        # atom_map : dict, key : int (index of atom in old topology) , value : int (index of same atom in new topology)
        atom_map = dict()
        # metadata : dict, key = 'chain_id' , value : str
        metadata = current_metadata
        if metadata == None:
            metadata = dict()
        # old_chemical_state_key : str
        old_chemical_state_key = self.compute_state_key(current_topology)

        # chain_id : str
        chain_id = self._chain_id
        # save old indeces for mapping -- could just directly save positions instead
        # modeller : simtk.openmm.app.Modeller
        current_positions = np.zeros((current_topology.getNumAtoms(), 3))
        modeller = app.Modeller(current_topology, current_positions)
        # atom : simtk.openmm.app.topology.Atom
        for atom in modeller.topology.atoms():
            # atom.old_index : int
            atom.old_index = atom.index

        if self._allowed_mutations is not None:
            allowed_mutations = self._allowed_mutations
            index_to_new_residues = self._choose_mutation_from_allowed(modeller, chain_id, allowed_mutations)
        else:
            # index_to_new_residues : dict, key : int (index) , value : str (three letter residue name)
            index_to_new_residues = self._propose_mutations(modeller, chain_id)
        # metadata['mutations'] : list(str (three letter WT residue name - index - three letter MUT residue name) )
        metadata['mutations'] = self._save_mutations(modeller, index_to_new_residues)
        # residue_map : list(tuples : simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue))
        residue_map = self._generate_residue_map(modeller, index_to_new_residues)
        # modeller : simtk.openmm.app.Modeller extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        # missing_atoms : dict, key : simtk.openmm.app.topology.Residue, value : list(simtk.openmm.app.topology._TemplateAtomData)
        modeller, missing_atoms = self._delete_excess_atoms(modeller, residue_map)
        # modeller : simtk.openmm.app.Modeller new residue has all correct atoms for desired mutation
        modeller = self._add_new_atoms(modeller, missing_atoms, residue_map)

        # atoms with an old_index attribute should be mapped
        # k : int
        # atom : simtk.openmm.app.topology.Atom
        for k, atom in enumerate(modeller.topology.atoms()):
            atom.index=k
            try:
                atom_map[atom.index] = atom.old_index
            except AttributeError:
                pass
        new_topology = modeller.topology

        # new_chemical_state_key : str
        new_chemical_state_key = self.compute_state_key(new_topology)
        # new_system : simtk.openmm.System
        new_system = self._system_generator.build_system(new_topology)

        return TopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=old_topology, old_system=current_system, old_chemical_state_key=old_chemical_state_key, new_chemical_state_key=new_chemical_state_key, logp_proposal=0.0, new_to_old_atom_map=atom_map)

    def _choose_mutation_from_allowed(self, modeller, chain_id, allowed_mutations):
        """
        Used when allowed mutations have been specified
        Assume (for now) uniform probability of selecting each specified mutant

        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
        chain_id : str
        allowed_mutations : list(list(tuple))
            list of allowed mutant states; each entry in the list is a list because multiple mutations may be desired
            tuple : (str, str) -- residue id and three-letter amino acid code of desired mutant

        Returns
        -------
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        """
        index_to_new_residues = dict()
        
        # chain : simtk.openmm.app.topology.Chain
        for chain in modeller.topology.chains():
            if chain.id == chain_id:
                break
        residue_id_to_index = [residue.id for residue in chain._residues]
        # location_prob : np.array, probability value for each residue location (uniform)
        location_prob = np.array([1.0/len(allowed_mutations) for i in range(len(allowed_mutations))])
        proposed_location = np.random.choice(range(len(allowed_mutations)), p=location_prob)
        for residue_id, residue_name in allowed_mutations[proposed_location]:
            # original_residue : simtk.openmm.app.topology.Residue
            original_residue = chain._residues[residue_id_to_index.index(residue_id)]
            # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
            index_to_new_residues[residue_id_to_index.index(residue_id)] = residue_name
            if residue_name == 'HIS':
                his_state = ['HIE','HID']
                his_prob = np.array([0.5 for i in range(len(his_state))])
                his_choice = np.random.choice(range(len(his_state)),p=his_prob)
                index_to_new_residues[residue_id_to_index.index(residue_id)] = his_state[his_choice]

        # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
        return index_to_new_residues

    def _propose_mutations(self, modeller, chain_id):
        """
        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
        chain_id : str

        Returns
        -------
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        """
        index_to_new_residues = dict()
        
        # this shouldn't be here
        aminos = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
        # chain : simtk.openmm.app.topology.Chain
        for chain in modeller.topology.chains():
            if chain.id == chain_id:
                # num_residues : int
                num_residues = len(chain._residues)
                break
        # location_prob : np.array, probability value for each residue location (uniform)
        location_prob = np.array([1.0/num_residues for i in range(num_residues)])
        for i in range(self._max_point_mutants):
            # proposed_location : int, index of chosen entry in location_prob
            proposed_location = np.random.choice(range(num_residues), p=location_prob)
            # original_residue : simtk.openmm.app.topology.Residue
            original_residue = chain._residues[proposed_location]
            # amino_prob : np.array, probability value for each amino acid option (uniform, must choose different from current)
            amino_prob = np.array([1.0/(len(aminos)-1) for i in range(len(aminos))])
            amino_prob[aminos.index(original_residue.name)] = 0.0
            # proposed_amino_index : int, index of three letter residue name in aminos list
            proposed_amino_index = np.random.choice(range(len(aminos)), p=amino_prob)
            # index_to_new_residues : dict, key : int (index of residue, 0-indexed), value : str (three letter residue name)
            index_to_new_residues[proposed_location] = aminos[proposed_amino_index]
            if aminos[proposed_amino_index] == 'HIS':
                his_state = ['HIE','HID']
                his_prob = np.array([0.5 for i in range(len(his_state))])
                his_choice = np.random.choice(range(len(his_state)),p=his_prob)
                index_to_new_residues[proposed_location] = his_state[his_choice]
        return index_to_new_residues

    def _save_mutations(self, modeller, index_to_new_residues):
        """
        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        Returns
        -------
        mutations : list(str)
            XXX-##-XXX
            three letter WT residue name - id - three letter MUT residue name
            id is based on the protein sequence NOT zero-indexed

        """
        return [r.name+'-'+str(r.id)+'-'+index_to_new_residues[r.index] for r in modeller.topology.residues() if r.index in index_to_new_residues]


    def _generate_residue_map(self, modeller, index_to_new_residues):
        """
        generates list to reference residue instance to be edited, because modeller.topology.residues() cannot be referenced directly by index

        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
        index_to_new_residues : dict
            key : int (index, zero-indexed in chain)
            value : str (three letter residue name)
        Returns
        -------
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue)
        """
        # residue_map : list(tuples : simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue))
        # r : simtk.openmm.app.topology.Residue, r.index : int, 0-indexed
        residue_map = [(r, index_to_new_residues[r.index]) for r in modeller.topology.residues() if r.index in index_to_new_residues]
        return residue_map

    def _delete_excess_atoms(self, modeller, residue_map):
        """
        delete excess atoms from old residues and identify new atoms for new residues

        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue)
        Returns
        -------
        modeller : simtk.openmm.app.Modeller
            extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        missing_atoms : dict
            key : simtk.openmm.app.topology.Residue
            value : list(simtk.openmm.app.topology._TemplateAtomData)
        """
        # delete excess atoms from old residues and identify new atoms for new residues
        # delete_atoms : list(simtk.openmm.app.topology.Atom) atoms from existing residue not in new residue
        delete_atoms = list()
        # missing_atoms : dict, key : simtk.openmm.app.topology.Residue, value : list(simtk.openmm.app.topology._TemplateAtomData)
        missing_atoms = dict()
        # residue : simtk.openmm.app.topology.Residue (existing residue)
        # replace_with : str (three letter residue name of proposed residue)
        for k, (residue, replace_with) in enumerate(residue_map):
            # chain_residues : list(simtk.openmm.app.topology.Residue) all residues in chain ==> why
            chain_residues = list(residue.chain.residues())
            if residue == chain_residues[0]:
                replace_with = 'N'+replace_with
                residue_map[k] = (residue, replace_with)
            if residue == chain_residues[-1]:
                replace_with = 'C'+replace_with
                residue_map[k] = (residue, replace_with)
            # template : simtk.openmm.app.topology._TemplateData
            template = self._templates[replace_with]
            # standard_atoms : set of unique atom names within new residue : str
            standard_atoms = set(atom.name for atom in template.atoms)
            # template_atoms : list(simtk.openmm.app.topology._TemplateAtomData) atoms in new residue
            template_atoms = list(template.atoms)
            # atom_names : set of unique atom names within existing residue : str
            atom_names = set(atom.name for atom in residue.atoms())
            # atom : simtk.openmm.app.topology.Atom in existing residue
            for atom in residue.atoms(): # shouldn't remove hydrogen
                if atom.name not in standard_atoms:
                    delete_atoms.append(atom)
#            if residue == chain_residues[0]: # this doesn't apply?
#                template_atoms = [atom for atom in template_atoms if atom.name not in ('P', 'OP1', 'OP2')]
            # missing : list(simtk.openmm.app.topology._TemplateAtomData) atoms in new residue not found in existing residue
            missing = list()
            # atom : simtk.openmm.app.topology._TemplateAtomData atoms in new residue
            for atom in template_atoms:
                if atom.name not in atom_names:
                    missing.append(atom)
            # BUG : error if missing = 0?
            if len(missing) > 0:
                missing_atoms[residue] = missing
        # modeller : simtk.openmm.app.Modeller extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        modeller = self._to_delete(modeller, delete_atoms)
        modeller = self._to_delete_bonds(modeller, residue_map)
        modeller.topology._numAtoms = len(list(modeller.topology.atoms()))

        return(modeller, missing_atoms)

    def _to_delete(self, modeller, delete_atoms):
        """
        remove instances of atoms and corresponding bonds from modeller

        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
        delete_atoms : list(simtk.openmm.app.topology.Atom)
            atoms from existing residue not in new residue
        Returns
        -------
        modeller : simtk.openmm.app.Modeller
            extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        """
        # delete_atoms : list(simtk.openmm.app.topology.Atom) atoms from existing residue not in new residue
        # atom : simtk.openmm.app.topology.Atom
        for atom in delete_atoms:
            atom.residue._atoms.remove(atom)
            for bond in modeller.topology._bonds:
                if atom in bond:
                    modeller.topology._bonds = list(filter(lambda a: a != bond, modeller.topology._bonds))
        # modeller : simtk.openmm.app.Modeller extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        return modeller

    def _to_delete_bonds(self, modeller, residue_map):
        """
        Remove any bonds between atoms in both new and old residue that do not belong in new residue
        (e.g. breaking the ring in PRO)
        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue (existing residue), str (three letter residue name of proposed residue)
        Returns
        -------
        modeller : simtk.openmm.app.Modeller
            extra atoms and bonds from old residue have been deleted, missing atoms in new residue not yet added
        """

        for residue, replace_with in residue_map:
            # template : simtk.openmm.app.topology._TemplateData
            template = self._templates[replace_with]

            old_res_bonds = list()
            # bond : tuple(simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)
            for atom1, atom2 in modeller.topology._bonds:
                if atom1.residue == residue or atom2.residue == residue:
                    old_res_bonds.append((atom1.name, atom2.name))
            # make a list of bonds that should exist in new residue
            # template_bonds : list(tuple(str (atom name), str (atom name))) bonds in template
            template_bonds = [(template.atoms[bond[0]].name, template.atoms[bond[1]].name) for bond in template.bonds]
            # add any bonds that exist in template but not in new residue
            for bond in old_res_bonds:
                if bond not in template_bonds and (bond[1],bond[0]) not in template_bonds:
                    modeller.topology._bonds = list(filter(lambda a: a != bond, modeller.topology._bonds))
                    modeller.topology._bonds = list(filter(lambda a: a != (bond[1],bond[0]), modeller.topology._bonds))
        return modeller

    def _add_new_atoms(self, modeller, missing_atoms, residue_map):
        """
        add new atoms (and corresponding bonds) to new residues

        Arguments
        ---------
        modeller : simtk.openmm.app.Modeller
            extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        missing_atoms : dict
            key : simtk.openmm.app.topology.Residue
            value : list(simtk.openmm.app.topology._TemplateAtomData)
        residue_map : list(tuples)
            simtk.openmm.app.topology.Residue, str (three letter residue name of new residue)
        Returns
        -------
        modeller : simtk.openmm.app.Modeller
            new residue has all correct atoms for desired mutation
        """
        # add new atoms to new residues
        # modeller : simtk.openmm.app.Modeller extra atoms from old residue have been deleted, missing atoms in new residue not yet added
        # missing_atoms : dict, key : simtk.openmm.app.topology.Residue, value : list(simtk.openmm.app.topology._TemplateAtomData)
        # residue_map : list(tuples : simtk.openmm.app.topology.Residue (old residue), str (three letter residue name of new residue))

        # new_atoms : list(simtk.openmm.app.topology.Atom) atoms that have been added to new residue
        new_atoms = list()
        # k : int
        # residue_ent : tuple(simtk.openmm.app.topology.Residue (old residue), str (three letter residue name of new residue))
        for k, residue_ent in enumerate(residue_map):
            # residue : simtk.openmm.app.topology.Residue (old residue) BUG : wasn't this editing the residue in place what is old and new map
            residue = residue_ent[0]
            # replace_with : str (three letter residue name of new residue)
            replace_with = residue_ent[1]
            # directly edit the simtk.openmm.app.topology.Residue instance
            residue.name = replace_with
            # load template to compare bonds
            # template : simtk.openmm.app.topology._TemplateData
            template = self._templates[replace_with]
            # add each missing atom
            # atom : simtk.openmm.app.topology._TemplateAtomData
            try:
                for atom in missing_atoms[residue]:
                    # new_atom : simtk.openmm.app.topology.Atom
                    new_atom = modeller.topology.addAtom(atom.name, atom.element, residue)
                    # new_atoms : list(simtk.openmm.app.topology.Atom)
                    new_atoms.append(new_atom)
            except KeyError:
                pass
            # make a dictionary to map atom names in new residue to atom object
            # new_res_atoms : dict, key : str (atom name) , value : simtk.openmm.app.topology.Atom
            new_res_atoms = dict()
            # atom : simtk.openmm.app.topology.Atom
            for atom in residue.atoms():
                # atom.name : str
                new_res_atoms[atom.name] = atom
            # make a list of bonds already existing in new residue
            # new_res_bonds : list(tuple(str (atom name), str (atom name))) bonds between atoms both within new residue
            new_res_bonds = list()
            # bond : tuple(simtk.openmm.app.topology.Atom, simtk.openmm.app.topology.Atom)
            for bond in modeller.topology._bonds:
                if bond[0].residue == residue and bond[1].residue == residue:
                    new_res_bonds.append((bond[0].name, bond[1].name))
            # make a list of bonds that should exist in new residue
            # template_bonds : list(tuple(str (atom name), str (atom name))) bonds in template
            template_bonds = [(template.atoms[bond[0]].name, template.atoms[bond[1]].name) for bond in template.bonds]
            # add any bonds that exist in template but not in new residue
            for bond in new_res_bonds:
                if bond not in template_bonds and (bond[1],bond[0]) not in template_bonds:
                    bonded_0 = new_res_atoms[bond[0]]
                    bonded_1 = new_res_atoms[bond[1]]
                    modeller.topology._bonds.remove((bonded_0, bonded_1))
            for bond in template_bonds:
                if bond not in new_res_bonds and (bond[1],bond[0]) not in new_res_bonds:
                    # new_bonded_0 : simtk.openmm.app.topology.Atom
                    new_bonded_0 = new_res_atoms[bond[0]]
                    # new_bonded_1 : simtk.openmm.app.topology.Atom
                    new_bonded_1 = new_res_atoms[bond[1]]
                    modeller.topology.addBond(new_bonded_0, new_bonded_1)
        modeller.topology._numAtoms = len(list(modeller.topology.atoms()))

        # add new bonds to the new residues
        return modeller

    def compute_state_key(self, topology):
        chemical_state_key = ''
        for res in topology.residues():
            chemical_state_key+=res.name

        return chemical_state_key

class SystemGenerator(object):
    """
    This is a utility class to generate OpenMM Systems from
    topology objects.

    Parameters
    ----------
    forcefields_to_use : list of string
        List of the names of ffxml files that will be used in system creation.
    forcefield_kwargs : dict of arguments to createSystem, optional
        Allows specification of various aspects of system creation.
    metadata : dict, optional
        Metadata associated with the SystemGenerator.
    """

    def __init__(self, forcefields_to_use, forcefield_kwargs=None, metadata=None):
        self._forcefield_xmls = forcefields_to_use
        self._forcefield_kwargs = forcefield_kwargs if forcefield_kwargs is not None else {}
        self._forcefield = app.ForceField(*self._forcefield_xmls)
        self._forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    def build_system(self, new_topology):
        """
        Build a system from the new_topology, adding templates
        for the molecules in oemol_list

        Parameters
        ----------
        new_topology : simtk.openmm.app.Topology object
            The topology of the system


        Returns
        -------
        new_system : openmm.System
            A system object generated from the topology
        """
        system = self._forcefield.createSystem(new_topology, **self._forcefield_kwargs)
        return system

    @property
    def ffxmls(self):
        return self._forcefield_xmls
    
    @property
    def forcefield(self):
        return self._forcefield

class SmallMoleculeSetProposalEngine(ProposalEngine):
    """
    This class proposes new small molecules from a prespecified set. It uses
    uniform proposal probabilities, but can be extended.

    Parameters
    ----------
    list_of_smiles : list of string
        list of smiles that will be sampled
    receptor_topology : app.Topology object
        topology of the receptor
    system_generator : SystemGenerator object
        SystemGenerator initialized with the appropriate forcefields
    proposal_metadata : dict
        metadata for the proposal engine
    """

    def __init__(self, list_of_smiles, receptor_topology, system_generator, proposal_metadata=None):
        self._receptor_topology = receptor_topology
        self._smiles_list = list_of_smiles
        self._n_molecules = len(list_of_smiles)
        self._generated_systems = dict()
        self._generated_topologies = dict()
        super(SmallMoleculeSetProposalEngine, self).__init__(system_generator, proposal_metadata=proposal_metadata)

    def propose(self, current_system, current_topology, current_metadata=None):
        """
        Propose the next state, given the current state

        Parameters
        ----------
        current_system : openmm.System object
            the system of the current state
        current_topology : app.Topology object
            the topology of the current state
        current_metadata : dict
            dict containing current smiles as a key

        Returns
        -------
        proposal : TopologyProposal object
           topology proposal object
        """
        current_mol_smiles, current_mol = self._topology_to_smiles(current_topology)

        current_mol_start_index = self._find_mol_start_index(current_topology)

        #choose the next molecule to simulate:
        proposed_mol_smiles, proposed_mol, logp_proposal = self._propose_molecule(current_system, current_topology,
                                                                                current_mol_smiles)

        new_topology, new_mol_start_index = self._build_new_topology(proposed_mol)
        new_system = self._system_generator.build_system(new_topology)
        smiles_new, _ = self._topology_to_smiles(new_topology)
        assert smiles_new == proposed_mol_smiles


        #map the atoms between the new and old molecule only:
        mol_atom_map, alignment_logp = self._get_mol_atom_map(current_mol, proposed_mol)

        #adjust the log proposal for the alignment:
        total_logp = alignment_logp + logp_proposal

        #adjust the atom map for the presence of the receptor:
        adjusted_atom_map = {}
        for (key, value) in mol_atom_map.items():
            adjusted_atom_map[key+new_mol_start_index] = value + current_mol_start_index

        #Create the TopologyProposal and return it
        proposal = TopologyProposal(new_topology=new_topology, new_system=new_system, old_topology=current_topology, old_system=current_system, logp_proposal=total_logp,
                                                 new_to_old_atom_map=adjusted_atom_map, old_chemical_state_key=current_mol_smiles, new_chemical_state_key=proposed_mol_smiles)
        return proposal

    def _topology_to_smiles(self, topology, molecule_name="MOL"):
        """
        Get the smiles string corresponding to a specific residue in an
        OpenMM Topology

        Parameters
        ----------
        topology : app.Topology
            The topology containing the molecule of interest
        molecule_name : string, optional
            The name of the residue. Default MOL

        Returns
        -------
        smiles_string : string
            an isomeric canonicalized SMILES string representing the molecule
        oemol : oechem.OEMol object
            molecule
        """

        matching_molecules = [res for res in topology.residues() if res.name==molecule_name]
        if len(matching_molecules) != 1:
            raise ValueError("More than one residue with the same name!")
        mol_res = matching_molecules[0]
        oemol = forcefield_generators.generateOEMolFromTopologyResidue(mol_res)
        smiles_string = oechem.OECreateIsoSmiString(oemol)
        return smiles_string, oemol

    def compute_state_key(self, topology, molecule_name="MOL"):
        """
        Given a topology, come up with a state key string.
        For this class, the state key is an isomeric canonical SMILES.

        Parameters
        ----------
        topology : app.Topology object
            The topology object in question.
        molecule_name : str, optional
            The name of the molecule residue in the topology, default MOL

        Returns
        -------
        chemical_state_key : str
            isomeric canonical SMILES

        """
        chemical_state_key, _ = self._topology_to_smiles(topology,molecule_name=molecule_name)
        return chemical_state_key

    def _find_mol_start_index(self, topology, resname='MOL'):
        """
        Find the starting index of the molecule in the topology.
        Throws an exception if resname is not present.

        Parameters
        ----------
        topology : app.Topology object
            The topology containing the molecule
        resname : string, optional
            The name of the molecule. Default MOL.

        Returns
        -------
        mol_start_idx : int
            start index of the molecule
        """
        mol_residues = [res for res in topology.residues() if res.name==resname]
        if len(mol_residues)!=1:
            raise ValueError("There can only be one residue with a specific name in the topology.")
        mol_residue = mol_residues[0]
        atoms = list(mol_residue.atoms())
        mol_start_idx = atoms[0].index
        return mol_start_idx

    def _build_new_topology(self, oemol_proposed):
        """
        Construct a new topology
        Parameters
        ----------
        oemol_proposed : oechem.OEMol object
            the proposed OEMol object

        Returns
        -------
        new_topology : app.Topology object
            A topology with the receptor and the proposed oemol
        mol_start_index : int
            The first index of the small molecule
        """
        mol_topology = forcefield_generators.generateTopologyFromOEMol(oemol_proposed)
        new_topology = copy.deepcopy(self._receptor_topology)
        mol_start_index = 0
        newAtoms = {}
        for chain in mol_topology.chains():
            newChain = new_topology.addChain(chain.id)
            for residue in chain.residues():
                newResidue = new_topology.addResidue(residue.name, newChain, residue.id)
                for atom in residue.atoms():
                    newAtom = new_topology.addAtom(atom.name, atom.element, newResidue, atom.id)
                    if atom.index == 0:
                        mol_start_index = newAtom.index
                    newAtoms[atom] = newAtom
        for bond in mol_topology.bonds():
            new_topology.addBond(newAtoms[bond[0]], newAtoms[bond[1]])
        
        return new_topology, mol_start_index


    def _smiles_to_oemol(self, smiles_string):
        """
        Convert the SMILES string into an OEMol

        Returns
        -------
        oemols : np.array of type object
            array of oemols
        """
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles_string)
        mol.SetTitle("MOL")
        oechem.OEAddExplicitHydrogens(mol)
        oechem.OETriposAtomNames(mol)
        oechem.OETriposBondTypeNames(mol)
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(1)
        omega(mol)
        return mol

    def _get_mol_atom_map(self, current_molecule, proposed_molecule):
        """
        Given two molecules, returns the mapping of atoms between them.

        Arguments
        ---------
        current_molecule : openeye.oechem.oemol object
             The current molecule in the sampler
        proposed_molecule : openeye.oechem.oemol object
             The proposed new molecule

        Returns
        -------
        new_to_old_atom_map : dict
            Dictionary of {new_idx : old_idx} format.
        logp_alignment : float
            logp of the molecule alignment if there is more than one
        """
        oegraphmol_current = oechem.OEGraphMol(current_molecule)
        oegraphmol_proposed = oechem.OEGraphMol(proposed_molecule)
        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
        atomexpr = oechem.OEExprOpts_AtomicNumber
        bondexpr = 0
        mcs.Init(oegraphmol_current, atomexpr, bondexpr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        unique = True
        matches = [m for m in mcs.Match(oegraphmol_proposed, unique)]
        match = np.random.choice(matches)
        new_to_old_atom_map = {}
        for matchpair in match.GetAtoms():
            old_index = matchpair.pattern.GetIdx()
            new_index = matchpair.target.GetIdx()
            new_to_old_atom_map[new_index] = old_index
        return new_to_old_atom_map, 1.0/len(matches)

    def _propose_molecule(self, system, topology, molecule_smiles):
        """
        Simple method that randomly chooses a molecule unformly.
        Symmetric proposal, so logp is 0. Override with a mixin.

        Arguments
        ---------
        system : simtk.openmm.System object
            The current system
        topology : simtk.openmm.app.Topology object
            The current topology
        positions : [n, 3] np.ndarray of floats (Quantity nm)
            The current positions of the system
        molecule_smiles : string
            The current molecule smiles

        Returns
        -------
        proposed_idx : int
             The index of the proposed oemol
        mol : oechem.OEMol
            The next molecule to simulate
        logp : float
            The log probability of the choice
        """
        proposed_smiles = np.random.choice(self._smiles_list)
        proposed_mol = self._smiles_to_oemol(proposed_smiles)
        return proposed_smiles, proposed_mol, 0.0


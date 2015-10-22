"""
This file contains the base classes for topology proposals
"""

import simtk.openmm.app as app
from collections import namedtuple
TopologyProposal = namedtuple('TopologyProposal',['old_topology','new_topology','logp', 'new_to_old_atom_map', 'metadata'])
SamplerState = namedtuple('SamplerState',['topology','system','positions', 'metadata'])
import copy
import os

class Transformation(object):
    """
    This defines a type which, given the requisite metadata, can produce Proposals (namedtuple)
    of new topologies.
    
    Arguments
    --------
    proposal_metadata : dict
        Contains information necessary to initialize proposal engine
    """
    
    def __init__(self, proposal_metadata):
        pass
    
    def propose(self, current_system, current_topology, current_positions, current_metadata):
        """
        Base interface for proposal method.
        
        Arguments
	---------
        current_system : simtk.openmm.System object
            The current system object
        current_topology : simtk.openmm.app.Topology object
            The current topology
        current_positions : [n,3] ndarray of floats
            The current positions of the system
        current_metadata : dict
            Additional metadata about the state
        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
	return TopologyProposal(app.Topology(), app.Topology(), 0.0, {0 : 0}, {'molecule_smiles' : 'CC'})

class ProteinTransformation(Transformation):
    """
    This defines a type which, given the requisite metadata, can produce Proposals (namedtuple)
    of new topologies.
    
    Arguments
    --------
    proposal_metadata : dict
        Contains information necessary to initialize proposal engine
    """

    def __init__(self, proposal_metadata):
        # load templates for replacement residues -- can this be done once?
        self.templates = dict()
        templatesPath = os.path.join(os.path.dirname(__file__), 'templates')
        for file in os.listdir(templatesPath):
            templatePdb = app.PDBFile(os.path.join(templatesPath, file))
            name = next(templatePdb.topology.residues()).name
            self.templates[name] = templatePdb      

    def propose(self, current_system, current_topology, current_positions, current_metadata):
        """
        Base interface for proposal method.
        
        Arguments
        ---------
        current_system : simtk.openmm.System object
            The current system object
        current_topology : simtk.openmm.app.Topology object
            The current topology
        current_positions : [n,3] ndarray of floats
            The current positions of the system
        current_metadata : dict
            metadata = {'mutations':[('XXX-##-XXX','X'),('XXX-##-XXX','X')]}
        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        old_topology = copy.deepcopy(current_topology)
        atom_map = dict()
        metadata = current_metadata

        # save old indeces for mapping -- could just directly save positions instead
        modeller = app.Modeller(current_topology, current_positions)
        for atom in modeller.topology.atoms():
            atom.old_index = atom.index

        index_to_new_residues = self._parseMutations(metadata, modeller)
        residue_map = self._generateResidueMap(modeller, index_to_new_residues)
        modeller, missingAtoms = self._deleteExcessAtoms(modeller, residue_map)
        new_residue_map = self._generateResidueMap(modeller, index_to_new_residues)
        modeller = self._addNewAtoms(modeller, missingAtoms, residue_map, new_residue_map)

        # atoms with an old_index attribute should be mapped
        for k, atom in enumerate(modeller.topology.atoms()):
            try:
                atom.index=k
                atom_map[atom.old_index] = atom.index
            except AttributeError:
                pass
        new_topology = modeller.topology

        return modeller, TopologyProposal(old_topology, new_topology, 0.0, atom_map, metadata)

    def _parseMutations(self, metadata, modeller):
        index_to_old_name = dict((r.index, r.name) for r in modeller.topology.residues())
        index_to_new_residues = dict()

        for mutation in metadata['mutations']:
            mut_str = mutation[0]
            chain_id = mutation[1]

            chain_numbers = list() # will need to return this if it's actually doing anything
            resSeq_to_index = dict()
            for chain in modeller.topology.chains():
                if chain.id == chain_id:
                    chain_numbers.append(chain_id)
                    for (residue_number, residue) in enumerate(chain.residues()):
                        resSeq_to_index[int(residue.id)] = residue_number

            # parse mutation to be made
            old_name, resSeq, new_name = mut_str.split("-")
            resSeq = int(resSeq)
            index = resSeq_to_index[resSeq]
            # check that residue id exist in the chain
            if index not in index_to_old_name:
                raise(KeyError("Cannot find index %d in system!" % index))
            if index_to_old_name[index] != old_name:
                raise(ValueError("You asked to mutate %s %d, but that residue is actually %s!" % (old_name, index, index_to_old_name[index])))
            try:
                template = self.templates[new_name]
            except KeyError:
                raise(KeyError("Cannot find residue %s in template library!" % new_name))
            index_to_new_residues[index] = new_name
        return index_to_new_residues

    def _generateResidueMap(self, modeller, index_to_new_residues):
        residue_map = [(r, index_to_new_residues[r.index]) for r in modeller.topology.residues() if r.index in index_to_new_residues]
        return residue_map

    def _deleteExcessAtoms(self, modeller, residue_map):
        # delete excess atoms from old residues and identify new atoms for new residues
        deleteAtoms = list()
        missingAtoms = dict()
        for residue, replaceWith in residue_map:
            # what the fuck is this doing (chain_number is undefined, will be the index of the final chain)
            #if residue.chain.index != chain_number:
                #continue  # Only modify specified chain
            residue.name = replaceWith
            template = self.templates[replaceWith]
            standardAtoms = set(atom.name for atom in template.topology.atoms())
            templateAtoms = list(template.topology.atoms())
            atomNames = set(atom.name for atom in residue.atoms())
            chainResidues = list(residue.chain.residues())
            for atom in residue.atoms(): # shouldn't remove hydrogen
                if atom.name not in standardAtoms:
                    deleteAtoms.append(atom)
            if residue == chainResidues[0]: # this doesn't apply?
                templateAtoms = [atom for atom in templateAtoms if atom.name not in ('P', 'OP1', 'OP2')]
            missing = list()
            for atom in templateAtoms:
                if atom.name not in atomNames:
                    missing.append(atom)
            if len(missing) > 0:
                missingAtoms[residue] = missing
        modeller = self._toDelete(modeller, deleteAtoms)

        return(modeller, missingAtoms)

    def _toDelete(self, modeller, deleteAtoms):
        for atom in deleteAtoms:
            atom.residue._atoms.remove(atom)
            for bond in modeller.topology._bonds:
                if atom in bond:
                    modeller.topology._bonds.remove(bond)
        return modeller

    def _addNewAtoms(self, modeller, missingAtoms, old_residue_map, new_residue_map):
        # add new atoms to new residues
        newAtoms = list()
        for k, residue_ent in enumerate(old_residue_map):
            residue = residue_ent[0]
            replaceWith = residue_ent[1]
            # load template to compare bonds
            template = self.templates[replaceWith]
            # save residue object in current topology
            new_residue = new_residue_map[k][0]
            # add each missing atom
            for atom in missingAtoms[residue]:
                newAtom = modeller.topology.addAtom(atom.name, atom.element, new_residue)
                newAtoms.append(newAtom)
            # make a dictionary to map atom names in new residue to atom object
            new_res_atoms = {}
            for atom in new_residue.atoms():
                new_res_atoms[atom.name] = atom
            # make a list of bonds already existing in new residue
            new_res_bonds = []
            for bond in modeller.topology.bonds():
                if bond[0].residue == new_residue and bond[1].residue == new_residue:
                    new_res_bonds.append((bond[0].name, bond[1].name))
            # make a list of bonds that should exist in new residue
            template_bonds = [(bond[0].name, bond[1].name) for bond in template.topology.bonds()]
            # add any bonds that exist in template but not in new residue
            for bond in template_bonds:
                if bond not in new_res_bonds:
                    new_bonded_0 = new_res_atoms[bond[0]]
                    new_bonded_1 = new_res_atoms[bond[1]]
                    modeller.topology.addBond(new_bonded_0, new_bonded_1)

        # add new bonds to the new residues
        return modeller


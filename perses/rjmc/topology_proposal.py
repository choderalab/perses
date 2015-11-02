"""
This file contains the base classes for topology proposals
"""

import simtk.openmm.app as app
from collections import namedtuple
import openeye.oechem as oechem
import numpy as np
import openeye.oegraphsim as oegraphsim
import openmoltools

TopologyProposal = namedtuple('TopologyProposal',
                              ['old_topology', 'new_topology', 'logp', 'new_to_old_atom_map', 'metadata'])
SamplerState = namedtuple('SamplerState', ['topology', 'system', 'positions', 'metadata'])


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
        return TopologyProposal(current_topology, app.Topology(), 0.0, {0: 0}, {'molecule_smiles': 'CC'})


class SmallSetMoleculeTransformation(Transformation):
    """
    This class implements a proposal based on a finite set of small molecules.
    The proposal probability is based on the tanimoto distance between the molecules.

    Arguments
    ---------
    proposal_metadata : dict
        Must contain entry 'molecule_list' with a list of SMILES of the molecule
    """

    def __init__(self, proposal_metadata):
        self._smiles_list = proposal_metadata['molecule_list']
        self._n_molecules = len(self._smiles_list)
        self._mol_array, self._smiles_dict = self._smiles_to_oemol()
        self._tanimotos = self._compute_tanimoto_distances()
        self._normalize_row_probability()

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
            Additional metadata about the state--in this case, the index of the molecule

        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        molecule_idx = current_metadata['molecule_idx']
        probabilities_forward = self._tanimotos[molecule_idx, :]
        #choose an index:
        proposed_idx = np.random.choice(range(self._n_molecules), p=probabilities_forward)
        probabilities_reverse = self._tanimotos[proposed_idx, :]
        log_forward = np.log(probabilities_forward[proposed_idx])
        log_reverse = np.log(probabilities_reverse[molecule_idx])
        logp = log_reverse - log_forward
        #get the atom map:
        new_to_old_atom_map = self._get_atom_map(self._mol_array[molecule_idx], self._mol_array[proposed_idx])

        #now make a topology out of these things:
        new_topology, positions = self._oemol_to_openmm_system(self._mol_array[proposed_idx], "MOL")

        return TopologyProposal(current_topology, new_topology, logp, new_to_old_atom_map, {'molecule_idx' : proposed_idx})



    def _get_atom_map(self, current_molecule, proposed_molecule):
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
        """
        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
        atomexpr = oechem.OEExprOpts_AtomicNumber
        bondexpr = 0
        mcs.Init(current_molecule, atomexpr, bondexpr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        unique = True
        matches = mcs.Match(proposed_molecule, unique)
        match = matches[0]
        new_to_old_atom_map = {}
        for matchpair in match.GetAtoms():
            old_index = matchpair.pattern.GetIdx()
            new_index = matchpair.target.GetIdx()
            new_to_old_atom_map[new_index] = old_index
        return new_to_old_atom_map

    def _oemol_to_openmm_system(self, oemol, molecule_name):
        """
        Create an openmm system out of an oemol

        Returns
        -------
        system : openmm.System object
            the system from the molecule
        positions : [n,3] np.array of floats
        """
        openmoltools.openeye.enter_temp_directory()
        _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(oemol, tripos_mol2_filename=molecule_name + '.tripos.mol2', conformer=0, residue_name='MOL')
        gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(molecule_name, tripos_mol2_filename)
        prmtop_file, inpcrd_file = openmoltools.utils.run_tleap(molecule_name, gaff_mol2, frcmod)
        prmtop = app.AmberPrmtopFile(prmtop_file)
        crd = app.AmberInpcrdFile(inpcrd_file)
        return prmtop.topology, crd.positions

    def _smiles_to_oemol(self):
        """
        Convert the list of smiles into a list of oemol objects. Explicit hydrogens
        are added, but no geometry is created.

        Returns
        -------
        oemols : np.array of type object
            array of oemols
        """
        list_of_smiles = self._smiles_list
        oemols = np.zeros(self._n_molecules, dtype=object)
        oemol_smile_dict = dict()
        for i, smile in enumerate(list_of_smiles):
            mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(mol, smile)
            oechem.OEAddExplicitHydrogens(mol)
            oemols[i] = mol
            oemol_smile_dict[smile] = i
        return oemols, oemol_smile_dict

    def _compute_tanimoto_distances(self):
        """
        Compute the nxn matrix of tanimoto distance between each molecule
        """
        tanimotos = np.ones([self._n_molecules, self._n_molecules], dtype=np.float64)
        for i in range(self._mol_array):
            for j in range(i):
                fingerprint_i = oegraphsim.OEFingerPrint()
                fingerprint_j = oegraphsim.OEFingerPrint()
                oegraphsim.OEMakeFP(fingerprint_i, self._mol_array[i], oegraphsim.OEFPType_MACCS166)
                oegraphsim.OEMakeFP(fingerprint_j, self._mol_array[j], oegraphsim.OEFPType_MACCS166)
                tanimoto_distance = oegraphsim.OETanimoto(fingerprint_i, fingerprint_j)
                tanimotos[i, j] = tanimoto_distance
                tanimotos[j, i] = tanimoto_distance
        return tanimotos

    def _normalize_row_probability(self):
        """
        Compute the normalizing constant of the proposal probabiltiy
        """
        for i in range(self._n_molecules):
            Z = np.sum(self._tanimotos[i, :])
            self._tanimotos[i, :] = self._tanimotos[i, :] / Z




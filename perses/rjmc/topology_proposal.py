"""
This file contains the base classes for topology proposals
"""

import simtk.openmm.app as app
from collections import namedtuple
import openeye.oechem as oechem
import numpy as np
import openeye.oegraphsim as oegraphsim
import scipy.stats as stats
import openmoltools
import openeye.oeiupac as oeiupac
import openeye.oeomega as oeomega
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
        return TopologyProposal(app.Topology(), app.Topology(), 0.0, {0: 0}, {'molecule_smiles': 'CC'})


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
            Additional metadata about the state--in this case, the current smiles molecule

        Returns
        -------
        proposal : TopologyProposal
            NamedTuple of type TopologyProposal containing forward and reverse
            probabilities, as well as old and new topologies and atom
            mapping
        """
        molecule_idx = self._smiles_dict[current_metadata['smiles']]
        probabilities = self._tanimotos[molecule_idx, :]



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




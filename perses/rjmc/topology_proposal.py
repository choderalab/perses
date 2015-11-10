"""
This file contains the base classes for topology proposals
"""

import simtk.openmm.app as app
from collections import namedtuple
import openeye.oechem as oechem
import numpy as np
import os
import openeye.oeomega as oeomega
import tempfile
import openeye.oegraphsim as oegraphsim
import openmoltools
import logging
try:
    from subprocess import getoutput  # If python 3
except ImportError:
    from commands import getoutput  # If python 2

TopologyProposal = namedtuple('TopologyProposal',
                              ['new_system', 'new_topology', 'logp_proposal', 'new_to_old_atom_map', 'metadata'])
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
    The proposal probability is based on the tanimoto similarity of the MACCS166fp between the molecules.

    Arguments
    ---------
    proposal_metadata : dict
        Must contain entry 'molecule_list' with a list of SMILES of the molecule
    """

    def __init__(self, proposal_metadata):
        self._smiles_list = proposal_metadata['molecule_list']
        self._n_molecules = len(self._smiles_list)
        self._mol_array, self._smiles_dict = self._smiles_to_oemol()
        self._tanimotos = self._compute_tanimoto_similarities()
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
        molecule_smiles : string
            String representing the current ligand
        logp :
        """
        molecule_smiles = current_metadata['molecule_smiles']
        molecule_idx = self._smiles_list.index(molecule_smiles)
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
        new_topology, positions = self._oemol_to_openmm_system(self._mol_array[proposed_idx], "ligand")

        return self._smiles_list[proposed_idx], logp, new_to_old_atom_map


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
        matches = [match for match in mcs.Match(proposed_molecule, unique)]
        match = matches[0]
        new_to_old_atom_map = {}
        for matchpair in match.GetAtoms():
            old_index = matchpair.pattern.GetIdx()
            new_index = matchpair.target.GetIdx()
            new_to_old_atom_map[new_index] = old_index
        return new_to_old_atom_map

    def _oemol_to_openmm_system(self, oegraphmol, molecule_name):
        """
        Create an openmm system out of an oemol

        Returns
        -------
        system : openmm.System object
            the system from the molecule
        positions : [n,3] np.array of floats
        """
        #change into a temporary directory, remembering where we came from
        cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)

        #make the OEGraphMol into a regular OEMol
        oemol = self._oegraphmol_to_oemol(oegraphmol)
        _, tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(oemol,
                                                                        tripos_mol2_filename=molecule_name + '.tripos.mol2',
                                                                        conformer=0, residue_name=molecule_name)
        gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(molecule_name, tripos_mol2_filename)
        prmtop_file, inpcrd_file = openmoltools.utils.run_tleap(molecule_name, gaff_mol2, frcmod)
        prmtop = app.AmberPrmtopFile(prmtop_file)
        crd = app.AmberInpcrdFile(inpcrd_file)

        #return to our origin and remove the temp directory
        os.chdir(cwd)
        #os.unlink(temp_dir)
        return prmtop.topology, crd.positions

    def _oegraphmol_to_oemol(self, oegraphmol):
        """
        This is a utility function to turn the OEGraphMols in this class into
        OEMols, as apparently the OEGraphMols don't inherit coordinates from their parent

        Arguments
        ---------
        oegraphmol : openeye.oechem.OEGraphMol
            The oegraphmol to convert

        Returns
        -------
        oemol : openeye.oechem.OEMol
            A new oemol with the same topology as oegraphmol, but positions as well
        """
        mol = oechem.OEMol(oegraphmol)
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(1)
        omega(mol)
        return mol

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

    def _compute_tanimoto_similarities(self):
        """
        Compute the nxn matrix of tanimoto similarity between each molecule
        """
        tanimotos = np.ones([self._n_molecules, self._n_molecules], dtype=np.float64)
        for i in range(len(self._mol_array)):
            for j in range(i):
                fingerprint_i = oegraphsim.OEFingerPrint()
                fingerprint_j = oegraphsim.OEFingerPrint()
                oegraphsim.OEMakeFP(fingerprint_i, self._mol_array[i], oegraphsim.OEFPType_MACCS166)
                oegraphsim.OEMakeFP(fingerprint_j, self._mol_array[j], oegraphsim.OEFPType_MACCS166)
                tanimoto_similarities = oegraphsim.OETanimoto(fingerprint_i, fingerprint_j)
                tanimotos[i, j] = tanimoto_similarities
                tanimotos[j, i] = tanimoto_similarities
        return tanimotos

    def _normalize_row_probability(self):
        """
        Compute the normalizing constant of the proposal probabiltiy
        """
        for i in range(self._n_molecules):
            Z = np.sum(self._tanimotos[i, :])
            self._tanimotos[i, :] = self._tanimotos[i, :] / Z


class SmallMoleculeTransformation(Transformation):
    """
    This class is a base class for transformations in which a small molecule is the part of the simulation
    that is changed. This class contains some base functionality for that process.
    """

    def __init__(self, proposal_metadata):
        self._smiles_list = proposal_metadata['smiles_list']
        self._n_molecules = len(proposal_metadata['smiles_list'])
        self._oemol_list, self._oemol_smile_dict = self._smiles_to_oemol()
        self._generated_systems = dict()
        self._generated_topologies = dict()

    def propose(self, current_system, current_topology, current_positions, current_metadata):
        """
        Make a proposal for the next small molecule.

        Arguments
        ---------
        system : simtk.openmm.System object
            The current system
        topology : simtk.openmm.app.Topology object
            The current topology
        positions : [n, 3] np.ndarray of floats (Quantity nm)
            The current positions of the system
        current_metadata : dict
            The current metadata

        Returns
        -------
        proposal : TopologyProposal namedtuple
            Contains new system, new topology, new to old atom map, and logp, as well as metadata
        """
        current_mol_smiles = current_metadata['molecule_smiles']
        current_mol = self._oemol_smile_dict[current_mol_smiles]

        #choose the next molecule to simulate:
        proposed_idx, proposed_mol, logp_proposal = self._propose_molecule(current_system, current_topology,
                                                                           current_positions, current_mol_smiles)
        proposed_mol_smiles = self._smiles_list[proposed_idx]

        #map the atoms between the new and old molecule only:
        mol_atom_map = self._get_mol_atom_map(current_mol, proposed_mol)

        #build the topology and system containing the new molecule:
        new_system, new_topology, new_to_old_atom_map = self._build_system(proposed_mol, mol_atom_map)

        #Create the TopologyProposal and return it
        proposal = TopologyProposal(new_system, new_topology, logp_proposal, new_to_old_atom_map,
                                    {'molecule_smiles': proposed_mol_smiles})
        return proposal

    def _build_system(self, proposed_molecule, molecule_smiles, mol_atom_map):
        """
        This is a stub for methods that will build a system for a new proposed molecule.

        Arguments
        ---------
        proposed_molecule : oemol
             The next proposed molecule
        molecule_smiles : string
             The smiles string representing the molecule
        mol_atom_map : dict
             The map of old ligand atoms to new ligand atoms
        Returns
        -------
        new_system : simtk.openmm.System
             A new system object for the molecule-X pair
        new_topology : simtk.openmm.Topology
             A new topology object for the molecule-X pair
        new_to_old_atom_map : dict
             The new to old atom map (using complex system indices)
        """
        raise NotImplementedError

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
            mol = oechem.OEMol()
            oechem.OESmilesToMol(mol, smile)
            oechem.OEAddExplicitHydrogens(mol)
            oemols[i] = mol
            oemol_smile_dict[smile] = i
        return oemols, oemol_smile_dict

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
        """
        oegraphmol_current = oechem.OEGraphMol(current_molecule)
        oegraphmol_proposed = oechem.OEGraphMol(proposed_molecule)
        mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
        atomexpr = oechem.OEExprOpts_AtomicNumber
        bondexpr = 0
        mcs.Init(oegraphmol_current, atomexpr, bondexpr)
        mcs.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())
        unique = True
        matches = [match for match in mcs.Match(oegraphmol_proposed, unique)]
        match = matches[0]
        new_to_old_atom_map = {}
        for matchpair in match.GetAtoms():
            old_index = matchpair.pattern.GetIdx()
            new_index = matchpair.target.GetIdx()
            new_to_old_atom_map[new_index] = old_index
        return new_to_old_atom_map

    def _propose_molecule(self, system, topology, positions, molecule_smiles):
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
        molecule_smiles : dict
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
        proposed_idx = np.choose(range(self._n_molecules))
        return proposed_idx, self._oemol_list[proposed_idx], 0.0


class SingleSmallMolecule(SmallMoleculeTransformation):
    """
    This class is an implementation of a proposal to transform a single small molecule
    in implicit solvent
    """


    def _build_system(self, proposed_molecule, molecule_smiles, mol_atom_map):
        """
        This will build a new system for the proposed molecule in implicit solvent

        Arguments
        ---------
        proposed_molecule : oemol
             The next proposed molecule
        molecule_smiles : string
             The smiles string representing the molecule
        mol_atom_map : dict
             The map of old ligand atoms to new ligand atoms
        Returns
        -------
        new_system : simtk.openmm.System
             A new system object for the molecule-X pair
        new_topology : simtk.openmm.Topology
             A new topology object for the molecule-X pair
        new_to_old_atom_map : dict
             The new to old atom map (same as input here)
        """

        #if we've already made the system, return that and get out
        if molecule_smiles in self._generated_systems.keys():
            return self._generated_systems['mol_smiles'], self._generated_topologies['mol_smiles'], mol_atom_map

        #run antechamber to parameterize, and tleap to create the prmtop
        molecule_name = 'ligand'
        _, tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(proposed_molecule,
                                                                        tripos_mol2_filename=molecule_name + '.tripos.mol2',
                                                                        conformer=0, residue_name=molecule_name)
        gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(molecule_name, tripos_mol2_filename)
        prmtop_file, inpcrd_file = openmoltools.utils.run_tleap(molecule_name, gaff_mol2, frcmod)

        #read in the prmtop
        prmtop = app.AmberPrmtopFile(prmtop_file)

        #add the topology to the generated tops, create the system and do the same for it
        self._generated_topologies['mol_smiles'] = prmtop.topology
        system = prmtop.createSystem(implicitSolvent=app.OBC2)
        self._generated_systems['mol_smiles'] = system

        #return the system and topology, along with the atom map
        return system, prmtop.topology, mol_atom_map


class SmallMoleculeProteinComplex(SmallMoleculeTransformation):
    """
    This class handles cases where small molecule changes are being sampled in the context of a
    protein : ligand complex. This is currently in implicit solvent only.

    Arguments
    ---------
    proposal_metadata : dict
         the metadata for the entire run. Should include a list of SMILES as well as
         the absolute path location of a simulation-ready receptor pdb file.
    """

    def __init__(self, proposal_metadata):
        self._receptor_pdb = proposal_metadata['receptor_pdb']
        self._receptor_topology = app.PDBFile(self._receptor_pdb).getTopology()
        self._natoms_receptor = self._receptor_topology._numAtoms

        super(SmallMoleculeProteinComplex, self).__init__(proposal_metadata)

    def _build_system(self, proposed_molecule, molecule_smiles, mol_atom_map):
        """
        This method builds a new system and topology containing the proposed molecule, as well
        as the corrected atom map (indices shifted by n_atoms_receptor)

        Arguments
        ---------
        proposed_molecule : oemol
             The next proposed molecule
        molecule_smiles : string
             The smiles string representing the molecule
        mol_atom_map : dict
             The map of old ligand atoms to new ligand atoms
        Returns
        -------
        new_system : simtk.openmm.System
             A new system object for the molecule-X pair
        new_topology : simtk.openmm.Topology
             A new topology object for the molecule-X pair
        new_to_old_atom_map : dict
             The new to old atom map
        """
        #adjust the indices of the ligands by n_atoms_receptor
        new_atom_map = {key + self._natoms_receptor : value + self._natoms_receptor for key, value in mol_atom_map.items()}

        #check to see if the system has already been made. if so, return it.
        if molecule_smiles in self._generated_systems.keys():
            return self._generated_systems[molecule_smiles], self._generated_topologies[molecule_smiles], new_atom_map

        prmtop = self._run_tleap(proposed_molecule)
        topology = prmtop.topology
        system = prmtop.createSystem(implicitSolvent=app.OBC2)

        self._generated_systems[molecule_smiles] = system
        self._generated_topologies[molecule_smiles] = topology

        return system, topology, new_atom_map

    def _run_tleap(self, proposed_molecule):
        """
        Utility function to run tleap in a temp directory, generating an AmberPrmtopFile
        object containing the proposed molecule and the receptor associated with this object

        Arguments
        ---------
        proposed_molecule : oechem.oemol
            The proposed new oemol to simulate with the receptor

        Returns
        -------
        prmtop : simtk.openmm.app.AmberPrmtopFile
            AmberPrmtopFile containing the receptor and molecule
        """
        cwd = os.getcwd()
        temp_dir = os.mkdtemp()
        os.chdir(temp_dir)

        #run antechamber to get parameters for molecule
        ligand_name = 'ligand'
        _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(proposed_molecule, tripos_mol2_filename=ligand_name + '.tripos.mol2', conformer=0, residue_name=ligand_name)
        gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(ligand_name, tripos_mol2_filename)

        #now get ready to run tleap to generate prmtop
        tleap_input = self._gen_tleap_input(gaff_mol2, frcmod, "complex")
        tleap_file = open('tleap_commands', 'w')
        tleap_file.writelines(tleap_input)
        tleap_file.close()
        tleap_cmd_str = "tleap -f %s " % tleap_file.name

        #call tleap, log output to logger
        output = getoutput(tleap_cmd_str)
        logging.debug(output)

        #read in the prmtop file
        prmtop = app.AmberPrmtopFile("complex.prmtop")

        #return and clean up
        os.chdir(cwd)
        os.rmdir(temp_dir)

        return prmtop


    def _gen_tleap_input(self, ligand_gaff_mol2, ligand_frcmod, complex_name):
        """
        This is a utility function to generate the input string necessary to run tleap
        """

        tleapstr = """
        # Load AMBER '96 forcefield for protein.
        source oldff/leaprc.ff99SBildn

        # Load GAFF parameters.
        source leaprc.gaff

        # Set GB radii to recommended values for OBC.
        set default PBRadii mbondi2

        # Load in protein.
        receptor = loadPdb {receptor_filename}

        # Load parameters for ligand.
        loadAmberParams {ligand_frcmod}

        # Load ligand.
        ligand = loadMol2 {ligand_gaf_fmol2}

        # Create complex.
        complex = combine {{ receptor ligand }}

        # Check complex.
        check complex

        # Report on net charge.
        charge complex

        # Write parameters.
        saveAmberParm complex {complex_name}.prmtop {complex_name}.inpcrd

        # Exit
        quit
        """

        tleap_input = tleapstr.format(ligand_frcmod=ligand_frcmod, ligand_gaff_mol2=ligand_gaff_mol2,
                                      receptor_filename=self._receptor_pdb, complex_name=complex_name)
        return tleap_input

from __future__ import absolute_import

from perses.dispersed import feptasks
from perses.utils.openeye import *
from perses.utils.data import load_smi
from perses.annihilation.relative import HybridTopologyFactory
from perses.annihilation.lambda_protocol import RelativeAlchemicalState
from perses.rjmc.topology_proposal import TopologyProposal, TwoMoleculeSetProposalEngine, SystemGenerator,SmallMoleculeSetProposalEngine
from perses.rjmc.geometry import FFAllAngleGeometryEngine

from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState

import pymbar
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
from openmoltools import forcefield_generators
import copy
import mdtraj as md
from io import StringIO
from openmmtools.constants import kB
import logging
import os
import dask.distributed as distributed
import parmed as pm

logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("relative_setup")
_logger.setLevel(logging.DEBUG)

#define forward functions for nonequilibrium_switching FEP: canonical lambda protocol
DEFAULT_FORWARD_FUNCTIONS = {
    'lambda_sterics_core': 'lambda',
    'lambda_electrostatics_core': 'lambda',
    'lambda_sterics_insert': 'select(step(0.5 - lambda), 2.0 * lambda, 0.0)',
    'lambda_sterics_delete': 'select(step(lambda - 0.5), 2.0 * (lambda - 0.5), 0.0)',
    'lambda_electrostatics_insert': 'select(step(lambda - 0.5), 2.0 * (lambda - 0.5), 0.0)',
    'lambda_electrostatics_delete': 'select(step(0.5 - lambda), 2.0 * lambda, 0.0)',
    'lambda_bonds': 'lambda',
    'lambda_angles': 'lambda',
    'lambda_torsions': 'lambda'
}

DEFAULT_REVERSE_FUNCTIONS = {
    'lambda_sterics_core': '1.0 - lambda',
    'lambda_electrostatics_core': '1.0 - lambda',
    'lambda_sterics_insert': '1.0 - select(step(lambda - 0.5), 2.0 * (lambda - 0.5), 0.0)',
    'lambda_sterics_delete': '1.0 - select(step(0.5 - lambda), 2.0 * lambda, 0.0)',
    'lambda_electrostatics_insert': '1.0 - select(step(0.5 - lambda), 2.0 * lambda, 0.0)',
    'lambda_electrostatics_delete': '1.0 - select(step(lambda - 0.5), 2.0 * (lambda - 0.5), 0.0)',
    'lambda_bonds': '1.0 - lambda',
    'lambda_angles': '1.0 - lambda',
    'lambda_torsions': '1.0 - lambda'

}

class RelativeFEPSetup(object):
    """
    This class is a helper class for relative FEP calculations. It generates the input objects that are necessary
    legs of a relative FEP calculation. For each leg, that is a TopologyProposal, old_positions, and new_positions.
    Importantly, it ensures that the atom maps in the solvent and complex phases match correctly.
    """
    def __init__(self, ligand_input, old_ligand_index, new_ligand_index, forcefield_files, phases,
                 protein_pdb_filename=None,receptor_mol2_filename=None, pressure=1.0 * unit.atmosphere,
                 temperature=300.0 * unit.kelvin, solvent_padding=9.0 * unit.angstroms, atom_map=None,
                 hmass=4*unit.amus, neglect_angles = False):
        """
        Initialize a NonequilibriumFEPSetup object

        Parameters
        ----------
        ligand_input : str
            the name of the ligand file (any openeye supported format)
            this can either be an .sdf or list of .sdf files, or a list of SMILES strings
        forcefield_files : list of str
            The list of ffxml files that contain the forcefields that will be used
        phases : list of str
            The phases to simulate
        protein_pdb_filename : str, default None
            Protein pdb filename. If none, receptor_mol2_filename must be provided
        receptor_mol2_filename : str, default None
            Receptor mol2 filename. If none, protein_pdb_filename must be provided
        pressure : Quantity, units of pressure
            Pressure to use in the barostat
        temperature : Quantity, units of temperature
            Temperature to use for the Langevin integrator
        solvent_padding : Quantity, units of length
            The amount of padding to use when adding solvent
        neglect_angles : bool
            Whether to neglect certain angle terms for the purpose of minimizing work variance in the RJMC protocol.
        """
        self._pressure = pressure
        self._temperature = temperature
        self._barostat_period = 50
        self._padding = solvent_padding
        self._hmass = hmass
        _logger.info(f"\t\t\t_hmass: {hmass}.\n")
        self._proposal_phase = None

        beta = 1.0 / (kB * temperature)

        mol_list = []

        #all legs need ligands so do this first
        self._ligand_input = ligand_input
        self._old_ligand_index = old_ligand_index
        self._new_ligand_index = new_ligand_index
        _logger.info(f"Handling files for ligands and indices...")
        if type(self._ligand_input) is not list: # the ligand has been provided as a single file
            if self._ligand_input[-3:] == 'smi': #
                _logger.info(f"Detected .smi format.  Proceeding...")
                self._ligand_smiles_old = load_smi(self._ligand_input,self._old_ligand_index)
                self._ligand_smiles_new = load_smi(self._ligand_input,self._new_ligand_index)
                _logger.info(f"\told smiles: {self._ligand_smiles_old}")
                _logger.info(f"\tnew smiles: {self._ligand_smiles_new}")

                all_old_mol = createSystemFromSMILES(self._ligand_smiles_old,title='MOL')
                self._ligand_oemol_old, self._ligand_system_old, self._ligand_positions_old, self._ligand_topology_old = all_old_mol

                all_new_mol = createSystemFromSMILES(self._ligand_smiles_new,title='NEW')
                self._ligand_oemol_new, self._ligand_system_new, self._ligand_positions_new, self._ligand_topology_new = all_new_mol
                _logger.info(f"\tsuccessfully created old and new systems from smiles")

                mol_list.append(self._ligand_oemol_old)
                mol_list.append(self._ligand_oemol_new)

                ffxml = forcefield_generators.generateForceFieldFromMolecules(mol_list)
                _logger.info(f"\tsuccessfully generated ffxml from molecules.")

                # forcefield_generators needs to be able to distinguish between the two ligands
                # while topology_proposal needs them to have the same residue name
                self._ligand_oemol_old.SetTitle("MOL")
                self._ligand_oemol_new.SetTitle("MOL")
                _logger.info(f"\tsetting both molecule oemol titles to 'MOL'.")

                self._ligand_topology_old = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_old)
                self._ligand_topology_new = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_new)
                _logger.info(f"\tsuccessfully generated topologies for both oemols.")

            elif self._ligand_input[-3:] == 'sdf': #
                _logger.info(f"Detected .sdf format.  Proceeding...") #TODO: write checkpoints for sdf format
                self._ligand_oemol_old = createOEMolFromSDF(self._ligand_input, index=self._old_ligand_index)
                self._ligand_oemol_new = createOEMolFromSDF(self._ligand_input, index=self._new_ligand_index)

                mol_list.append(self._ligand_oemol_old)
                mol_list.append(self._ligand_oemol_new)

                self._ligand_positions_old = extractPositionsFromOEMol(self._ligand_oemol_old)
                _logger.info(f"\tsuccessfully extracted positions from OEMOL.")

                ffxml = forcefield_generators.generateForceFieldFromMolecules(mol_list)
                _logger.info(f"\tsuccessfully generated ffxml from molecules.")

                self._ligand_oemol_old.SetTitle("MOL")
                self._ligand_oemol_new.SetTitle("MOL")
                _logger.info(f"\tsetting both molecule oemol titles to 'MOL'.")

                self._ligand_smiles_old = oechem.OECreateSmiString(self._ligand_oemol_old,
                            oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
                self._ligand_smiles_new = oechem.OECreateSmiString(self._ligand_oemol_new,
                            oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
                _logger.info(f"\tsuccessfully created SMILES for both ligand OEMOLs.")

                # replace this with function that will generate the system etc. so that vacuum can be performed
                self._ligand_topology_old = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_old)
                self._ligand_topology_new = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_new)
                _logger.info(f"\tsuccessfully generated topologies for both OEMOLs.")
            else:
                print(f'RelativeFEPSetup can only handle .smi or .sdf files currently')

        else: # the ligand has been provided as a list of .sdf files
            _logger.info(f"Detected list...perhaps this is of sdf format.  Proceeding (but without checkpoints...this may be buggy).") #TODO: write checkpoints and debug for list
            old_ligand = pm.load_file('%s.parm7' % self._ligand_input[0], '%s.rst7' % self._ligand_input[0])
            self._ligand_topology_old = old_ligand.topology
            self._ligand_positions_old = old_ligand.positions
            self._ligand_oemol_old = createOEMolFromSDF('%s.mol2' % self._ligand_input[0])
            self._ligand_smiles_old = oechem.OECreateSmiString(self._ligand_oemol_old,
                                                             oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)

            new_ligand = pm.load_file('%s.parm7' % self._ligand_input[1], '%s.rst7' % self._ligand_input[1])
            self._ligand_topology_new = new_ligand.topology
            self._ligand_positions_new = new_ligand.positions
            self._ligand_oemol_new = createOEMolFromSDF('%s.mol2' % self._ligand_input[1])
            self._ligand_smiles_new = oechem.OECreateSmiString(self._ligand_oemol_new,
                                                             oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)

            mol_list.append(self._ligand_oemol_old)
            mol_list.append(self._ligand_oemol_new)

            old_ligand_parameter_set = pm.openmm.OpenMMParameterSet.from_structure(old_ligand)
            new_ligand_parameter_set = pm.openmm.OpenMMParameterSet.from_structure(new_ligand)
            ffxml = StringIO()
            old_ligand_parameter_set.write(ffxml)
            new_ligand_parameter_set.write(ffxml)
            ffxml = ffxml.getvalue()

        self._ligand_md_topology_old = md.Topology.from_openmm(self._ligand_topology_old)
        self._ligand_md_topology_new = md.Topology.from_openmm(self._ligand_topology_new)
        _logger.info(f"Created mdtraj topologies for both ligands.")

        if 'complex' in phases or 'solvent' in phases:
            self._nonbonded_method = app.PME
            _logger.info(f"Detected complex or solvent phases: setting PME nonbonded method.")
        elif 'vacuum' in phases:
            self._nonbonded_method = app.NoCutoff
            _logger.info(f"Detected vacuum phase: setting noCutoff nonbonded method.")

        if pressure is not None:
            if self._nonbonded_method == app.PME:
                barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
                _logger.info(f"set MonteCarloBarostat.")
            else:
                barostat = None
                _logger.info(f"omitted MonteCarloBarostat.")
            self._system_generator = SystemGenerator(forcefield_files, barostat=barostat,
                                                     forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': self._nonbonded_method,'constraints' : app.HBonds, 'hydrogenMass' : self._hmass})
        else:
            self._system_generator = SystemGenerator(forcefield_files, forcefield_kwargs={'removeCMMotion': False,'nonbondedMethod': self._nonbonded_method,'constraints' : app.HBonds, 'hydrogenMass' : self._hmass})

        _logger.info("successfully called TopologyProposal.SystemGenerator to create ligand systems.")
        self._system_generator._forcefield.loadFile(StringIO(ffxml))

        _logger.info(f"executing SmallMoleculeSetProposalEngine...")
        self._proposal_engine = SmallMoleculeSetProposalEngine([self._ligand_smiles_old, self._ligand_smiles_new], self._system_generator, residue_name='MOL')

        _logger.info(f"instantiating FFAllAngleGeometryEngine...")
        # NOTE: we are conducting the geometry proposal without any neglected angles
        self._geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=100, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = neglect_angles)
        if 'complex' in phases: self._complex_geometry_engine = copy.deepcopy(self._geometry_engine)
        if 'solvent' in phases: self._solvent_geometry_engine = copy.deepcopy(self._geometry_engine)
        if 'vacuum' in phases: self._vacuum_geometry_engine = copy.deepcopy(self._geometry_engine)


        # if we are running multiple phases, we only want to generate one topology proposal, and use the same one for the other legs
        # this is tracked using _proposal_phase
        if 'complex' in phases:
            _logger.info('Generating the topology proposal from the complex leg')
            self._nonbonded_method = app.PME
            _logger.info(f"setting up complex phase...")
            self._setup_complex_phase(protein_pdb_filename,receptor_mol2_filename,mol_list)
            self._complex_topology_old_solvated, self._complex_positions_old_solvated, self._complex_system_old_solvated = self._solvate_system(
            self._complex_topology_old, self._complex_positions_old)
            _logger.info(f"successfully generated complex topology, positions, system")

            self._complex_md_topology_old_solvated = md.Topology.from_openmm(self._complex_topology_old_solvated)

            _logger.info(f"creating TopologyProposal...")
            self._complex_topology_proposal = self._proposal_engine.propose(self._complex_system_old_solvated,
                                                                                self._complex_topology_old_solvated, current_mol=self._ligand_oemol_old,proposed_mol=self._ligand_oemol_new)
            self.non_offset_new_to_old_atom_map = self._proposal_engine.non_offset_new_to_old_atom_map

            self._proposal_phase = 'complex'

            _logger.info(f"conducting geometry proposal...")
            self._complex_positions_new_solvated, self._complex_logp_proposal = self._complex_geometry_engine.propose(self._complex_topology_proposal,
                                                                                self._complex_positions_old_solvated,
                                                                                beta)
            self._complex_logp_reverse = self._complex_geometry_engine.logp_reverse(self._complex_topology_proposal, self._complex_positions_new_solvated, self._complex_positions_old_solvated, beta)
            self._complex_added_valence_energy = self._complex_geometry_engine.forward_final_context_reduced_potential - self._complex_geometry_engine.forward_atoms_with_positions_reduced_potential
            self._complex_subtracted_valence_energy = self._complex_geometry_engine.reverse_final_context_reduced_potential - self._complex_geometry_engine.reverse_atoms_with_positions_reduced_potential
            self._complex_forward_neglected_angles = self._complex_geometry_engine.forward_neglected_angle_terms
            self._complex_reverse_neglected_angles = self._complex_geometry_engine.reverse_neglected_angle_terms


        if 'solvent' in phases:
            _logger.info(f"Detected solvent...")
            if self._proposal_phase is None:
                _logger.info(f"no complex detected in phases...generating unique topology/geometry proposals...")
                self._nonbonded_method = app.PME
                _logger.info(f"solvating ligand...")
                self._ligand_topology_old_solvated, self._ligand_positions_old_solvated, self._ligand_system_old_solvated = self._solvate_system(
                self._ligand_topology_old, self._ligand_positions_old)
                self._ligand_md_topology_old_solvated = md.Topology.from_openmm(self._ligand_topology_old_solvated)

                _logger.info(f"creating TopologyProposal")
                self._solvent_topology_proposal = self._proposal_engine.propose(self._ligand_system_old_solvated,
                                                                                    self._ligand_topology_old_solvated,current_mol=self._ligand_oemol_old,proposed_mol=self._ligand_oemol_new)
                self.non_offset_new_to_old_atom_map = self._proposal_engine.non_offset_new_to_old_atom_map
                self._proposal_phase = 'solvent'
            else:
                _logger.info('Using the topology proposal from the complex leg')
                self._solvent_topology_proposal, self._ligand_positions_old_solvated = self._generate_solvent_topologies(
                    self._complex_topology_proposal, self._complex_positions_old_solvated)

            _logger.info(f"conducting geometry proposal...")
            self._ligand_positions_new_solvated, self._ligand_logp_proposal_solvated = self._solvent_geometry_engine.propose(self._solvent_topology_proposal,
                                                                                    self._ligand_positions_old_solvated, beta)
            self._ligand_logp_reverse_solvated = self._solvent_geometry_engine.logp_reverse(self._solvent_topology_proposal, self._ligand_positions_new_solvated, self._ligand_positions_old_solvated, beta)
            self._solvated_added_valence_energy = self._solvent_geometry_engine.forward_final_context_reduced_potential - self._solvent_geometry_engine.forward_atoms_with_positions_reduced_potential
            self._solvated_subtracted_valence_energy = self._solvent_geometry_engine.reverse_final_context_reduced_potential - self._solvent_geometry_engine.reverse_atoms_with_positions_reduced_potential
            self._solvated_forward_neglected_angles = self._solvent_geometry_engine.forward_neglected_angle_terms
            self._solvated_reverse_neglected_angles = self._solvent_geometry_engine.reverse_neglected_angle_terms

        if 'vacuum' in phases:
            _logger.info(f"Detected solvent...")
            # need to change nonbonded cutoff and remove barostat for vacuum leg
            _logger.info(f"assgning noCutoff to nonbonded_method")
            self._nonbonded_method = app.NoCutoff
            _logger.info(f"calling TopologyProposal.SystemGenerator to create ligand systems.")
            self._system_generator = SystemGenerator(forcefield_files, forcefield_kwargs={'removeCMMotion': False,
                                                    'nonbondedMethod': self._nonbonded_method,'constraints' : app.HBonds})
            self._system_generator._forcefield.loadFile(StringIO(ffxml))
            if self._proposal_phase is None:
                _logger.info('No complex or solvent leg, so performing topology proposal for vacuum leg')
                self._vacuum_topology_old, self._vacuum_positions_old, self._vacuum_system_old = self._solvate_system(self._ligand_topology_old,
                                                                                                         self._ligand_positions_old,vacuum=True)
                self._vacuum_topology_proposal = self._proposal_engine.propose(self._vacuum_system_old,
                                                                                self._vacuum_topology_old,current_mol=self._ligand_oemol_old,proposed_mol=self._ligand_oemol_new)
                self.non_offset_new_to_old_atom_map = self._proposal_engine.non_offset_new_to_old_atom_map
                self._proposal_phase = 'vacuum'
            elif self._proposal_phase == 'complex':
                _logger.info('Using the topology proposal from the complex leg')
                self._vacuum_topology_proposal, self._vacuum_positions_old = self._generate_vacuum_topologies(
                    self._complex_topology_proposal, self._complex_positions_old_solvated)
            elif self._proposal_phase == 'solvent':
                _logger.info('Using the topology proposal from the solvent leg')
                self._vacuum_topology_proposal, self._vacuum_positions_old = self._generate_vacuum_topologies(
                    self._solvent_topology_proposal, self._ligand_positions_old_solvated)

            _logger.info(f"conducting geometry proposal...")
            self._vacuum_positions_new, self._vacuum_logp_proposal = self._vacuum_geometry_engine.propose(self._vacuum_topology_proposal,
                                                                          self._vacuum_positions_old,
                                                                          beta)
            self._vacuum_logp_reverse = self._vacuum_geometry_engine.logp_reverse(self._vacuum_topology_proposal, self._vacuum_positions_new, self._vacuum_positions_old, beta)
            self._vacuum_added_valence_energy = self._vacuum_geometry_engine.forward_final_context_reduced_potential - self._vacuum_geometry_engine.forward_atoms_with_positions_reduced_potential
            self._vacuum_subtracted_valence_energy = self._vacuum_geometry_engine.reverse_final_context_reduced_potential - self._vacuum_geometry_engine.reverse_atoms_with_positions_reduced_potential
            self._vacuum_forward_neglected_angles = self._vacuum_geometry_engine.forward_neglected_angle_terms
            self._vacuum_reverse_neglected_angles = self._vacuum_geometry_engine.reverse_neglected_angle_terms

    def _setup_complex_phase(self,protein_pdb_filename,receptor_mol2_filename,mol_list):
        """
        Runs setup on the protein/receptor file for relative simulations

        Parameters
        ----------
        protein_pdb_filename : str, default None
            Protein pdb filename. If none, receptor_mol2_filename must be provided
        receptor_mol2_filename : str, default None
            Receptor mol2 filename. If none, protein_pdb_filename must be provided
        """
        if protein_pdb_filename:
            self._protein_pdb_filename = protein_pdb_filename
            protein_pdbfile = open(self._protein_pdb_filename, 'r')
            pdb_file = app.PDBFile(protein_pdbfile)
            protein_pdbfile.close()
            self._receptor_positions_old = pdb_file.positions
            self._receptor_topology_old = pdb_file.topology
            self._receptor_md_topology_old = md.Topology.from_openmm(self._receptor_topology_old)

        elif receptor_mol2_filename:
            self._receptor_mol2_filename = receptor_mol2_filename
            self._receptor_mol = createOEMolFromSDF(self._receptor_mol2_filename)
            mol_list.append(self._receptor_mol)
            self._receptor_positions_old = extractPositionsFromOEMol(self._receptor_mol)
            self._receptor_topology_old = forcefield_generators.generateTopologyFromOEMol(self._receptor_mol)
            self._receptor_md_topology_old = md.Topology.from_openmm(self._receptor_topology_old)
        else:
            raise ValueError("You need to provide either a protein pdb or a receptor mol2 to run a complex simulation.")

        self._complex_md_topology_old = self._receptor_md_topology_old.join(self._ligand_md_topology_old)
        self._complex_topology_old = self._complex_md_topology_old.to_openmm()

        n_atoms_complex_old = self._complex_topology_old.getNumAtoms()
        n_atoms_protein_old = self._receptor_topology_old.getNumAtoms()

        self._complex_positions_old = unit.Quantity(np.zeros([n_atoms_complex_old, 3]), unit=unit.nanometers)
        self._complex_positions_old[:n_atoms_protein_old, :] = self._receptor_positions_old
        self._complex_positions_old[n_atoms_protein_old:, :] = self._ligand_positions_old

    def _generate_solvent_topologies(self, topology_proposal, old_positions):
        """
        This method generates ligand-only topologies and positions from a TopologyProposal containing a solvated complex.
        The output of this method is then used when building the solvent-phase simulation with the same atom map.

        Parameters
        ----------
        old_positions : array
            Positions of the fully solvated protein ligand syste

        Returns
        -------
        ligand_topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
            Topology proposal object of the ligand without complex

        old_solvated_positions : array
            positions of the system without complex
        """
        old_complex = md.Topology.from_openmm(topology_proposal.old_topology)
        new_complex = md.Topology.from_openmm(topology_proposal.new_topology)

        atom_map = topology_proposal.old_to_new_atom_map

        old_mol_start_index, old_mol_len = self._proposal_engine._find_mol_start_index(old_complex.to_openmm())
        new_mol_start_index, new_mol_len = self._proposal_engine._find_mol_start_index(new_complex.to_openmm())

        old_pos = unit.Quantity(np.zeros([len(old_positions), 3]), unit=unit.nanometers)
        old_pos[:, :] = old_positions
        old_ligand_positions = old_pos[old_mol_start_index:(old_mol_start_index + old_mol_len), :]

        # subset the topologies:
        old_ligand_topology = old_complex.subset(old_complex.select("resname == 'MOL' "))
        new_ligand_topology = new_complex.subset(new_complex.select("resname == 'MOL' "))

        # solvate the old ligand topology:
        old_solvated_topology, old_solvated_positions, old_solvated_system = self._solvate_system(
            old_ligand_topology.to_openmm(), old_ligand_positions)

        old_solvated_md_topology = md.Topology.from_openmm(old_solvated_topology)

        # now remove the old ligand, leaving only the solvent
        solvent_only_topology = old_solvated_md_topology.subset(old_solvated_md_topology.select("not resname MOL"))
        # append the solvent to the new ligand-only topology:
        new_solvated_ligand_md_topology = new_ligand_topology.join(solvent_only_topology)
        nsl, b = new_solvated_ligand_md_topology.to_dataframe()

        # dirty hack because new_solvated_ligand_md_topology.to_openmm() was throwing bond topology error
        new_solvated_ligand_md_topology = md.Topology.from_dataframe(nsl, b)

        new_solvated_ligand_omm_topology = new_solvated_ligand_md_topology.to_openmm()
        new_solvated_ligand_omm_topology.setPeriodicBoxVectors(old_solvated_topology.getPeriodicBoxVectors())

        # create the new ligand system:
        new_solvated_system = self._system_generator.build_system(new_solvated_ligand_omm_topology)

        new_to_old_atom_map = {atom_map[x] - new_mol_start_index: x - old_mol_start_index for x in
                               old_complex.select("resname == 'MOL' ") if x in atom_map.keys()}

        # adjust the atom map to account for the presence of solvent degrees of freedom:
        # By design, all atoms after the ligands are water, and should be mapped.
        n_water_atoms = solvent_only_topology.to_openmm().getNumAtoms()
        for i in range(n_water_atoms):
            new_to_old_atom_map[new_mol_len + i] = old_mol_len + i

        # make a TopologyProposal
        ligand_topology_proposal = TopologyProposal(new_topology=new_solvated_ligand_omm_topology,
                                                    new_system=new_solvated_system,
                                                    old_topology=old_solvated_topology, old_system=old_solvated_system,
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='MOL',
                                                    new_chemical_state_key='MOL')

        return ligand_topology_proposal, old_solvated_positions

    def _generate_vacuum_topologies(self, topology_proposal, old_positions):
        """
        This method generates ligand-only topologies and positions from a TopologyProposal containing a solvated complex.
        The output of this method is then used when building the solvent-phase simulation with the same atom map.

        Parameters
        ----------
        old_positions : array
            Positions of the fully solvated protein ligand syste

        Returns
        -------
        ligand_topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
            Topology proposal object of the ligand without complex

        old_solvated_positions : array
            positions of the system without complex
        """
        old_complex = md.Topology.from_openmm(topology_proposal.old_topology)
        new_complex = md.Topology.from_openmm(topology_proposal.new_topology)

        atom_map = topology_proposal.old_to_new_atom_map

        old_mol_start_index, old_mol_len = self._proposal_engine._find_mol_start_index(old_complex.to_openmm())
        new_mol_start_index, new_mol_len = self._proposal_engine._find_mol_start_index(new_complex.to_openmm())

        old_pos = unit.Quantity(np.zeros([len(old_positions), 3]), unit=unit.nanometers)
        old_pos[:, :] = old_positions
        old_ligand_positions = old_pos[old_mol_start_index:(old_mol_start_index + old_mol_len), :]

        # subset the topologies:
        old_ligand_topology = old_complex.subset(old_complex.select("resname == 'MOL' "))
        new_ligand_topology = new_complex.subset(new_complex.select("resname == 'MOL' "))

        # convert to openmm topology object
        old_ligand_topology = old_ligand_topology.to_openmm()
        new_ligand_topology = new_ligand_topology.to_openmm()

        # create the new ligand system:
        old_ligand_system = self._system_generator.build_system(old_ligand_topology)
        new_ligand_system = self._system_generator.build_system(new_ligand_topology)

        new_to_old_atom_map = {atom_map[x] - new_mol_start_index: x - old_mol_start_index for x in
                               old_complex.select("resname == 'MOL' ") if x in atom_map.keys()}


        # make a TopologyProposal
        ligand_topology_proposal = TopologyProposal(new_topology=new_ligand_topology,
                                                    new_system=new_ligand_system,
                                                    old_topology=old_ligand_topology, old_system=old_ligand_system,
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='MOL',
                                                    new_chemical_state_key='MOL')

        return ligand_topology_proposal, old_ligand_positions

    def _solvate_system(self, topology, positions, model='tip3p',vacuum=False):
        """
        Generate a solvated topology, positions, and system for a given input topology and positions.
        For generating the system, the forcefield files provided in the constructor will be used.

        Parameters
        ----------
        topology : app.Topology
            Topology of the system to solvate
        positions : [n, 3] ndarray of Quantity nm
            the positions of the unsolvated system
        forcefield : SystemGenerator.forcefield
            forcefield file of solvent to add
        model : str, default 'tip3p'
            solvent model to use for solvation

        Returns
        -------
        solvated_topology : app.Topology
            Topology of the system with added waters
        solvated_positions : [n + 3(n_waters), 3] ndarray of Quantity nm
            Solvated positions
        solvated_system : openmm.System
            The parameterized system, containing a barostat if one was specified.
        """
        modeller = app.Modeller(topology, positions)
        hs = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H'] and atom.residue.name not in ['MOL','OLD','NEW']]
        modeller.delete(hs)
        modeller.addHydrogens(forcefield=self._system_generator._forcefield)
        if not vacuum:
            _logger.info(f"\tpreparing to add solvent")
            modeller.addSolvent(self._system_generator._forcefield, model=model, padding=self._padding)
        else:
            _logger.info(f"\tSkipping solvation of vacuum perturbation")
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()

        # canonicalize the solvated positions: turn tuples into np.array
        solvated_positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
        _logger.info(f"\tparameterizing...")
        solvated_system = self._system_generator.build_system(solvated_topology)
        _logger.info(f"\tSystem parameterized")
        return solvated_topology, solvated_positions, solvated_system

    @property
    def complex_topology_proposal(self):
        return self._complex_topology_proposal

    @property
    def complex_old_positions(self):
        return self._complex_positions_old_solvated

    @property
    def complex_new_positions(self):
        return self._complex_positions_new_solvated

    @property
    def solvent_topology_proposal(self):
        return self._solvent_topology_proposal

    @property
    def solvent_old_positions(self):
        return self._ligand_positions_old_solvated

    @property
    def solvent_new_positions(self):
        return self._ligand_positions_new_solvated

    @property
    def vacuum_topology_proposal(self):
        return self._vacuum_topology_proposal

    @property
    def vacuum_old_positions(self):
        return self._vacuum_positions_old

    @property
    def vacuum_new_positions(self):
        return self._vacuum_positions_new



class NonequilibriumSwitchingFEP(object):
    """
    This class manages Nonequilibrium switching based relative free energy calculations, carried out on a distributed computing framework.
    """

    def __init__(self, topology_proposal, geometry_engine, pos_old, new_positions, use_dispersion_correction=False,
                 forward_functions=DEFAULT_FORWARD_FUNCTIONS, reverse_functions = DEFAULT_REVERSE_FUNCTIONS, ncmc_nsteps = 100, n_equilibrium_steps_per_iteration = 100, temperature=300.0 * unit.kelvin, trajectory_directory=None, trajectory_prefix=None,
                 atom_selection="not water", scheduler_address=None, eq_splitting_string="V R O R V", neq_splitting_string = 'O { V R H R V } O', measure_shadow_work=False, timestep=1.0*unit.femtoseconds,
                 neglected_new_angle_terms = [], neglected_old_angle_terms = [], ncmc_save_interval = None, write_ncmc_configuration = False):
        """
        Create an instance of the NonequilibriumSwitchingFEP driver class

        Parameters
        ----------
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
            TopologyProposal object containing transformation of interest
        geometry_engine : perses.rjmc.geometry.FFAllAngleGeometryEngine
            geometry engine used to create and compute the RJMCMC; we use this to compute the importance weight from the old/new system to the hybrid system (neglecting added valence terms)
        pos_old : [n, 3] ndarray unit.Quantity
            Positions of the old system.
        new_positions : [m, 3] ndarray unit.Quantity
            Positions of the new system
        use_dispersion_correction : bool, default False
            Whether to use the (expensive) dispersion correction
        forward_functions : dict of str: str, default forward_functions as defined by top of file
            How each force's scaling parameter relates to the main lambda that is switched by the integrator from 0 to 1
        reverse_functions : dict of str: str, default reverse_functions as defined by top of file
            How each force's scaling parameter relates to the main lambda that is switched by the integrator from 1 to 0
        ncmc_nsteps : int, default 100
            Number of steps per NCMC trajectory
        n_equilibrium_steps_per_iteration : int, default 100
            Number of steps to run in an iteration of equilibration to generate an iid sample
        temperature : float unit.Quantity
            Temperature at which to perform the simulation, default 300K
        trajectory_directory : str, default None
            Where to write out trajectories resulting from the calculation. If none, no writing is done.
        trajectory_prefix : str, default None
            What prefix to use for this calculation's trajectory files. If none, no writing is done.
        atom_selection : str, default not water
            MDTraj selection syntax for which atomic coordinates to save in the trajectories. Default strips
            all water.
        scheduler_address : str
            address for distributed computing jobs (not currently functional)
        eq_splitting_string : str, default 'V R O R V'
            The integrator splitting to use for equilibrium simulation
        neq_splitting_string : str, default 'O { V R H R V } O'
            The integrator splitting to use for nonequilibrium switching simulation
        neglected_new_angle_terms : list, default []
            list of indices from the HarmonicAngleForce of the new_system for which the geometry engine neglected.
            Hence, these angles must be alchemically grown in for the unique new atoms (forward lambda protocol)
        neglected_old_angle_terms : list, default []
            list of indices from the HarmonicAngleForce of the old_system for which the geometry engine neglected.
            Hence, these angles must be alchemically deleted for the unique old atoms (reverse lambda protocol)
        ncmc_save_interval : int, default None
            interval with which to write ncmc trajectory.  If None, trajectory will not be saved.
            We will assert that the ncmc_nsteps % ncmc_save_interval = 0; otherwise, the protocol will not be complete
        write_ncmc_configuration : bool, default False
            whether to write ncmc annealing perturbations; if True, will write every ncmc_save_interval iterations
        """
        # construct the hybrid topology factory object
        _logger.info(f"writing HybridTopologyFactory")
        self._factory = HybridTopologyFactory(topology_proposal, pos_old, new_positions, neglected_new_angle_terms, neglected_old_angle_terms)
        self.geometry_engine = geometry_engine
        self._ncmc_save_interval = ncmc_nsteps if not ncmc_save_interval else ncmc_save_interval

        #we have to make sure that there is no remainder from ncmc_nsteps % ncmc_save_interval
        try:
            assert ncmc_nsteps % self._ncmc_save_interval == 0
        except ValueError:
            print(f"The work writing interval must be a factor of the total number of ncmc steps; otherwise, the ncmc protocol is incomplete!")


        self._write_ncmc_configuration = write_ncmc_configuration

        # setup splitting string:
        self._eq_splitting_string = eq_splitting_string
        self._neq_splitting_string = neq_splitting_string
        self._measure_shadow_work = measure_shadow_work

        # set up some class attributes
        self._hybrid_system = self._factory.hybrid_system
        self._initial_hybrid_positions = self._factory.hybrid_positions
        self._ncmc_nsteps = ncmc_nsteps
        self._n_equil_steps = n_equilibrium_steps_per_iteration
        self._trajectory_prefix = trajectory_prefix
        self._trajectory_directory = trajectory_directory
        self._zero_endpoint_n_atoms = topology_proposal.n_atoms_old
        self._one_endpoint_n_atoms = topology_proposal.n_atoms_new
        self._atom_selection = atom_selection
        self._current_iteration = 0
        self._endpoint_growth_thermostates = dict()
        self._timestep = timestep

        self._forward_functions = forward_functions
        self._reverse_functions = reverse_functions

        #instantiate growth system thermodynamic and samplerstates
        assert not self.geometry_engine.use_sterics, f"The geometry engine has use_sterics...this is not currently supported."
        if self.geometry_engine.forward_final_growth_system: # if it exists, we have to create a thermodynamic state and set alchemical parameters
            self._endpoint_growth_thermostates[1] = ThermodynamicState(self.geometry_engine.forward_final_growth_system, temperature=temperature) #sampler state is compatible with new system
        else:
            self._endpoint_growth_thermostates[1] = None

        if self.geometry_engine.reverse_final_growth_system: # if it exists, we have to create a thermodynamic state and set alchemical parameters
            self._endpoint_growth_thermostates[0] = ThermodynamicState(self.geometry_engine.reverse_final_growth_system, temperature=temperature) #sampler state is compatible with old system
        else:
            self._endpoint_growth_thermostates[0] = None


        _logger.info(f"instantiating trajectory filenames")
        if self._trajectory_directory and self._trajectory_prefix:
            self._write_traj = True
            self._trajectory_filename = {lambda_state: os.path.join(os.getcwd(), self._trajectory_directory, f"{trajectory_prefix}.eq.lambda_{lambda_state}.h5") for lambda_state in [0,1]}
            _logger.debug(f"eq_traj_filenames: {self._trajectory_filename}")
            self._neq_traj_filename = {lambda_state: os.path.join(os.getcwd(), self._trajectory_directory, f"{trajectory_prefix}.neq.lambda_{lambda_state}") for lambda_state in [0,1]}
            _logger.debug(f"neq_traj_filenames: {self._neq_traj_filename}")
        else:
            self._write_traj = False
            self._trajectory_filename = {0: None, 1: None}
            self._neq_traj_filename = {0: None, 1: None}

        # create the thermodynamic state
        _logger.info(f"Instantiating thermodynamic states 0 and 1.")
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(self._hybrid_system)
        lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

        lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
        lambda_one_alchemical_state.set_alchemical_parameters(1.0)

        # ensure their states are set appropriately
        self._hybrid_alchemical_states = {0: lambda_zero_alchemical_state, 1: lambda_one_alchemical_state}

        # create the base thermodynamic state with the hybrid system
        self._thermodynamic_state = ThermodynamicState(self._hybrid_system, temperature=temperature)

        # Create thermodynamic states for the nonalchemical endpoints
        self._nonalchemical_thermodynamic_states = {
            0: ThermodynamicState(topology_proposal.old_system, temperature=temperature),
            1: ThermodynamicState(topology_proposal.new_system, temperature=temperature)}

        # Now create the compound states with different alchemical states
        self._hybrid_thermodynamic_states = {0: CompoundThermodynamicState(self._thermodynamic_state,
                                                                           composable_states=[
                                                                               self._hybrid_alchemical_states[0]]),
                                             1: CompoundThermodynamicState(copy.deepcopy(self._thermodynamic_state),
                                                                           composable_states=[
                                                                               self._hybrid_alchemical_states[1]])}

        self._ncmc_nsteps = ncmc_nsteps
        self._temperature = temperature

        # set the SamplerState for the lambda 0 and 1 equilibrium simulations
        _logger.info(f"Instantiating SamplerStates")
        self._lambda_one_sampler_state = SamplerState(self._initial_hybrid_positions,
                                                      box_vectors=self._hybrid_system.getDefaultPeriodicBoxVectors())
        self._lambda_zero_sampler_state = copy.deepcopy(self._lambda_one_sampler_state)

        self._sampler_states = {0: copy.deepcopy(self._lambda_zero_sampler_state), 1: copy.deepcopy(self._lambda_one_sampler_state)}


        # initialize by minimizing
        _logger.info(f"Instantiating equilibrium results by minimization")
        [feptasks.minimize(self._hybrid_thermodynamic_states[key], self._sampler_states[key]) for key in [0,1]]
        self._minimized_sampler_states = {i: copy.deepcopy(self._sampler_states[i]) for i in [0,1]}

        # subset the topology appropriately:
        if atom_selection is not None:
            atom_selection_indices = self._factory.hybrid_topology.select(atom_selection)
            self._atom_selection_indices = atom_selection_indices
        else:
            self._atom_selection_indices = None

        _logger.info(f"constructed")

    def run(self, n_iterations=5):
        """
        Run nonequilibrium switching free energy calculations. This entails:

        - 1 iteration of equilibrium at lambda=0 and lambda=1
        - concurrency (parameter) many nonequilibrium trajectories in both forward and reverse
           (e.g., if concurrency is 5, then 5 forward and 5 reverse protocols will be run)
        - 1 iteration of equilibrium at lambda=0 and lambda=1

        Parameters
        ----------
        n_iterations : int, optional, default 5
            The number of times to run the entire sequence described above (concurrency)
        """
        _logger.debug(f"conducting nonequilibrium_switching with {n_iterations} iterations")
        endpoints = [0, 1]
        alchemical_functions = {0: self._forward_functions, 1: self._reverse_functions} # we define the key as 'from lambda {key}'

        #instantiate equilibrium energy dicts
        nonalchemical_reduced_potentials = {0: np.zeros((n_iterations)), 1: np.zeros((n_iterations))}
        added_valence_reduced_potentials = copy.deepcopy(nonalchemical_reduced_potentials)
        alchemical_reduced_potentials = copy.deepcopy(nonalchemical_reduced_potentials)

        #instantiate equilibrium perturbation_TO energy dicts
        nonalchemical_reduced_potentials_to = copy.deepcopy(nonalchemical_reduced_potentials)
        added_valence_reduced_potentials_to = copy.deepcopy(nonalchemical_reduced_potentials)
        alchemical_reduced_potentials_to = copy.deepcopy(nonalchemical_reduced_potentials)

        #instantiate nonequilibrium work dicts: the keys indicate from which equilibrium thermodynamic state the neq_switching is conducted FROM (as opposed to TO)
        nonequilibrium_cum_work = {0: [], 1: []}
        nonequilibrium_prot_work = {0: [], 1: []}
        nonequilibrium_shadow_work = {0: [], 1: []}

        for i in range(n_iterations): #iterate forward and backward n_iterations times
            _logger.debug(f"\tconducting iteration {self._current_iteration}")

            # first name the equilibrium and nonequilibrium trajectory files
            if self._write_traj:
                _logger.debug(f"\twriting traj: {self._trajectory_filename.values()}")
                equilibrium_trajectory_filenames = self._trajectory_filename
                noneq_trajectory_filenames = {lambda_state: self._neq_traj_filename[lambda_state] + f".iteration_{self._current_iteration}" for lambda_state in endpoints}

            for endpoint in endpoints: #iterate 0 and 1 (starting with zero)
                alternate_endpoint = 1 if endpoint == 0 else 0
                _logger.debug(f"\t\tconducting protocol from lambda = {endpoint} to lambda = {alternate_endpoint}")

                # run a round of equilibrium at lambda_0(1)
                _logger.debug(f"\t\tconducting equilibrium run")
                feptasks.run_equilibrium(self._hybrid_thermodynamic_states[endpoint],
                                                               self._sampler_states[endpoint], self._n_equil_steps,
                                                               self._factory._hybrid_topology, 1,
                                                               self._atom_selection_indices,
                                                               equilibrium_trajectory_filenames[endpoint], self._eq_splitting_string, self._timestep)

                # get the perturbations to nonalchemical states: 0 (1)
                _logger.debug(f"\t\tcollecting corrected energy to the nonalchemical state of lambda = {endpoint}")
                valence_energy, nonalchemical_reduced_potential, hybrid_reduced_potential = feptasks.compute_nonalchemical_perturbation(self._hybrid_thermodynamic_states[endpoint],
                                                                 self._endpoint_growth_thermostates[alternate_endpoint],
                                                                 self._sampler_states[endpoint], self._factory,
                                                                 self._nonalchemical_thermodynamic_states[endpoint],
                                                                 endpoint)
                _logger.debug(f"\t\t\thybrid {endpoint} reduced energy, nonalchemical {endpoint} reduced_energy, valence: ({hybrid_reduced_potential}, {nonalchemical_reduced_potential}, {valence_energy})")
                nonalchemical_reduced_potentials[endpoint][self._current_iteration] = nonalchemical_reduced_potential
                added_valence_reduced_potentials[endpoint][self._current_iteration] = valence_energy
                alchemical_reduced_potentials[endpoint][self._current_iteration] = hybrid_reduced_potential

                _logger.debug(f"\t\tcollecting corrected energy to the alternate nonalchemical state (i.e. lambda = {alternate_endpoint})")
                alt_valence_energy, alt_nonalchemical_reduced_potential, alt_hybrid_reduced_potential = feptasks.compute_nonalchemical_perturbation(self._hybrid_thermodynamic_states[alternate_endpoint],
                                                                 self._endpoint_growth_thermostates[endpoint],
                                                                 self._sampler_states[endpoint], self._factory,
                                                                 self._nonalchemical_thermodynamic_states[alternate_endpoint],
                                                                 alternate_endpoint)
                _logger.debug(f"\t\t\thybrid {alternate_endpoint} reduced energy, nonalchemical {alternate_endpoint} reduced_energy, valence: ({alt_hybrid_reduced_potential}, {alt_nonalchemical_reduced_potential}, {alt_valence_energy})")
                nonalchemical_reduced_potentials_to[alternate_endpoint][self._current_iteration] = alt_nonalchemical_reduced_potential
                added_valence_reduced_potentials_to[alternate_endpoint][self._current_iteration] = alt_valence_energy
                alchemical_reduced_potentials_to[alternate_endpoint][self._current_iteration] = alt_hybrid_reduced_potential

                # run a round of nonequilibrium switching from 0 --> 1 (1 --> 0):
                _logger.debug(f"\t\tconducting annealing from lambda = {endpoint}")

                #first, we have to make a deepcopy of the lambda 0,1 sampler state since we don't want to perturb self._sampler_states equilibria
                anneal_from_state = copy.deepcopy(self._sampler_states[endpoint])
                cum_work, prot_work, shadow_work = feptasks.run_protocol(thermodynamic_state = self._hybrid_thermodynamic_states[endpoint],
                                                                         sampler_state = anneal_from_state,
                                                                         alchemical_functions = alchemical_functions[endpoint],
                                                                         topology = self._factory._hybrid_topology,
                                                                         nstep_neq = self._ncmc_nsteps,
                                                                         work_save_interval = self._ncmc_save_interval,
                                                                         splitting = self._neq_splitting_string,
                                                                         atom_indices_to_save = self._atom_selection_indices,
                                                                         trajectory_filename = self._neq_traj_filename[endpoint],
                                                                         write_configuration = self._write_ncmc_configuration,
                                                                         timestep = self._timestep,
                                                                         measure_shadow_work = self._measure_shadow_work)

                nonequilibrium_cum_work[endpoint].append(cum_work)
                nonequilibrium_prot_work[endpoint].append(prot_work)
                nonequilibrium_shadow_work[endpoint].append(shadow_work)

                _logger.debug(f"\t\tcompleted annealing from lambda = {endpoint}")


            self._current_iteration += 1
            _logger.debug(f"\titeration {self._current_iteration} complete")

        # after all tasks have been requested, retrieve the results:
        _logger.debug(f"all annealing and equilibration done; retrieving results")

        #works
        nonequilibrium_cum_work_arrays = {key: np.array(value) for key, value in nonequilibrium_cum_work.items()}
        nonequilibrium_prot_work_arrays = {key: np.array(value) for key, value in nonequilibrium_prot_work.items()}
        nonequilibrium_shadow_work_arrays = {key: np.array(value) for key, value in nonequilibrium_shadow_work.items()}

        self.work_dict = {'cumulative': nonequilibrium_cum_work_arrays,
                     'protocol': nonequilibrium_prot_work_arrays,
                     'shadow': nonequilibrium_shadow_work}

        #energies (used for timeseries and exp estimations)
        _logger.debug(f"alchemical reduced potentials: {alchemical_reduced_potentials}")
        _logger.debug(f"alchemical--> reduced potentials: {alchemical_reduced_potentials_to}")

        self.equilibrium_energies_dict = {'nonalchemical': nonalchemical_reduced_potentials,
                         'alchemical': alchemical_reduced_potentials,
                         'valence': added_valence_reduced_potentials}

        self.equilibrium_perturbed_energies_dict = {'nonalchemical': nonalchemical_reduced_potentials_to,
                                                    'alchemical': alchemical_reduced_potentials_to,
                                                    'valence': added_valence_reduced_potentials_to}


    def equilibrate(self, n_equilibration_iterations = 1):
        """
        Run the equilibrium simulations a specified number of times at the lambda 0, 1 states. This can be used to equilibrate
        the simulation before beginning the free energy calculation.
        """
        _logger.debug(f"conducting equilibration")
        nsteps_equil = self._n_equil_steps
        hybrid_topology = self._factory.hybrid_topology
        atom_indices_to_save_list = self._atom_selection_indices
        eq_splitting = self._eq_splitting_string
        timestep = self._timestep

        # run a round of equilibrium
        for state in [0,1]:
            _logger.debug(f"equilibrating lambda state {state}")
            if self._write_traj:
                _logger.debug(f"\twriting traj to {self._trajectory_filename[state]}")
                equilibrium_trajectory_filename = self._trajectory_filename[state]
            else:
                _logger.debug(f"\tnot writing traj")
                equilibrium_trajectory_filename = None

            feptasks.run_equilibrium(self._hybrid_thermodynamic_states[state], self._sampler_states[state], nsteps_equil,
                                                  hybrid_topology, n_equilibration_iterations,
                                                  atom_indices_to_save_list, equilibrium_trajectory_filename, eq_splitting, timestep)

    def _adjust_for_correlation(self, timeseries_array: np.array):
        """
        Compute statistical inefficiency for timeseries, returning the timeseries with burn in as well as
        the statistical inefficience and the max number of effective samples

        Parameters
        ----------
        timeseries_array : np.array
            Array of timeseries values

        Returns
        -------
        burned_in_series : np.array
            Array starting after burn in
        statistical_inefficiency : float
            Statistical inefficience of timeseries
        Neff_max : float
            Max number of uncorrelated samples
        """
        [t0, g, Neff_max] = pymbar.timeseries.detectEquilibration(timeseries_array)

        return timeseries_array[t0:], g, Neff_max

    def _endpoint_perturbations(self):
        """
        Compute the correlation-adjusted free energy at the endpoints to the nonalchemical systems,
        corrected nonalchemical systems, and alchemical endpoints with EXP

        Returns
        -------
        df0, ddf0 : list of float
            endpoint pertubation with error for lambda 0, kT
        df1, ddf1 : list of float
            endpoint perturbation for lambda 1, kT
        """
        _logger.debug(f"conducting EXP averaging on hybrid reduced potentials and the valence_adjusted endpoint potentials...")
        free_energies = []
        #defining reduced potential differences of the hybrid systems...
        self._hybrid_reduced_potential_differences = {'0_to_1': -1*(self.equilibrium_energies_dict['alchemical'][0] - self.equilibrium_perturbed_energies_dict['alchemical'][1]),
                                                      '1_to_0': -1*(self.equilibrium_energies_dict['alchemical'][1] - self.equilibrium_perturbed_energies_dict['alchemical'][0])}
        self._nonalchemical_reduced_potential_differences = {'0_to_nonalchemical_0': self.equilibrium_perturbed_energies_dict['nonalchemical'][0] + self.equilibrium_perturbed_energies_dict['valence'][0] - self.equilibrium_energies_dict['alchemical'][0],
                                                             '1_to_nonalchemical_1': self.equilibrium_perturbed_energies_dict['nonalchemical'][1] + self.equilibrium_perturbed_energies_dict['valence'][1] - self.equilibrium_energies_dict['alchemical'][1]}

        #conducting endpoint perturbations: will compute exp averaging from equilibrium at 0-->1, and 1-->0 for the hybrid energies AND the valence corrected nonalchemical energies
        template = {'df': [], 'ddf': [], 'correlation': []}
        self._EXP_alchemical_free_energies = {'0_to_1': copy.deepcopy(template), '1_to_0': copy.deepcopy(template)}
        self._EXP_to_nonalchemical_free_energies = {'0': copy.deepcopy(template), '1': copy.deepcopy(template)}


        for lambda_endpoint in [0, 1]:
            _logger.debug(f"\tconducting EXP calculations from {lambda_endpoint}")
            alternate_endpoint = 1 if lambda_endpoint == 0 else 0

            eq_energy_array = self.equilibrium_energies_dict['alchemical'][lambda_endpoint]
            t0, statistical_inefficiency, Neff_max = pymbar.timeseries.detectEquilibration(eq_energy_array)
            equilibrated_energy_array = eq_energy_array[t0:]
            diff_array = self._hybrid_reduced_potential_differences[f"{lambda_endpoint}_to_{alternate_endpoint}"][t0:]

            uncorrelated_indices = pymbar.timeseries.subsampleCorrelatedData(equilibrated_energy_array, g=statistical_inefficiency)
            uncorrelated_diff_array = diff_array[uncorrelated_indices]
            df, ddf_raw = pymbar.EXP(uncorrelated_diff_array) #input to exp are forward works
            ddf_corrected = ddf_raw * np.sqrt(statistical_inefficiency)
            _logger.debug(f"\t\tt0, inefficiency, Neff: {t0}, {statistical_inefficiency}, {Neff_max}")
            _logger.debug(f"\t\tdf, ddf: ({df}, {ddf_corrected})")

            #now to record to the overly complicated dictionaries...
            self._EXP_alchemical_free_energies[f"{lambda_endpoint}_to_{alternate_endpoint}"]["correlation"] = [t0, statistical_inefficiency, Neff_max]
            self._EXP_alchemical_free_energies[f"{lambda_endpoint}_to_{alternate_endpoint}"]["df"] = df
            self._EXP_alchemical_free_energies[f"{lambda_endpoint}_to_{alternate_endpoint}"]["ddf"] = ddf_corrected

            #now to compute the to_nonalchemical free energies:
            nonalch_diff_array = self._nonalchemical_reduced_potential_differences[f"{lambda_endpoint}_to_nonalchemical_{lambda_endpoint}"][t0:]
            uncorrelated_nonalch_diff_array = nonalch_diff_array[uncorrelated_indices]
            nonalch_df, nonalch_ddf = pymbar.EXP(uncorrelated_nonalch_diff_array)
            nonalch_ddf_corrected = nonalch_ddf * np.sqrt(statistical_inefficiency)

            # and record results
            self._EXP_to_nonalchemical_free_energies[f"{lambda_endpoint}"]["correlation"] = [t0, statistical_inefficiency, Neff_max]
            self._EXP_to_nonalchemical_free_energies[f"{lambda_endpoint}"]["df"] = nonalch_df
            self._EXP_to_nonalchemical_free_energies[f"{lambda_endpoint}"]["ddf"] = nonalch_ddf_corrected


    def _alchemical_free_energy(self):
        """
        Use BAR to compute the free energy between lambda 0 and lambda1

        Returns
        -------
        df : float
            Free energy, kT
        ddf_corrected : float
            Error in free energy, kT
        """
        _logger.debug(f"conducting free energy for W_f and W_r calculation with BAR")
        inefficiencies = []
        forward_works = self.work_dict['cumulative'][0]
        num_forward_runs, num_forward_logs_per_run = forward_works.shape
        reverse_works = self.work_dict['cumulative'][1]
        num_reverse_runs, num_reverse_logs_per_run = reverse_works.shape

        assert num_forward_runs == num_reverse_runs

        self._total_works = {'forward': forward_works[:,-1], 'reverse': reverse_works[:,-1]}
        self._BAR_alchemical_free_energies = {'df': [], 'ddf': [], 'correlation': []}
        self._decorrelated_total_works = {}

        # computing decorrelated timeseries calculation for the forward and reverse total accumulated works;
        # TODO: figure out why some of the works nan...; at present, we are ignoring these data
        for direction in ['forward', 'reverse']:
            _logger.debug(f"conducting timeseries computation for {direction} direction")
            series = np.array([i for i in self._total_works[f"{direction}"] if not np.isnan(i)])
            _logger.debug(f"work array: {series}")
            [t0, g, Neff_max, uncorrelated_data] = feptasks.compute_timeseries(series)
            self._BAR_alchemical_free_energies['correlation'].append([t0, g, Neff_max])
            inefficiencies.append(g)
            self._decorrelated_total_works[direction] = uncorrelated_data

        df, ddf = pymbar.BAR(self._decorrelated_total_works['forward'], self._decorrelated_total_works['reverse'])
        ddf_corrected = ddf * np.sqrt(max(inefficiencies))
        self._BAR_alchemical_free_energies['df'] = df
        self._BAR_alchemical_free_energies['ddf'] = ddf_corrected

        _logger.debug(f"df, ddf: {df}, {ddf}")
        _logger.debug(f"correlations: {self._BAR_alchemical_free_energies['correlation']}")

        return df, ddf_corrected

    @property
    def current_free_energy_estimate(self):
        """
        Estimate the free energy based on currently available values
        """
        # Make sure the task queue is empty (all pending calcs are complete) before computing free energy
        # Make sure the task queue is empty (all pending calcs are complete) before computing free energy
        self._endpoint_perturbations()
        [df, ddf] = self._alchemical_free_energy()

        return df, ddf

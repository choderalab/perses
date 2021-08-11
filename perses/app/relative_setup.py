from __future__ import absolute_import

from perses.dispersed import feptasks
from perses.utils.openeye import createOEMolFromSDF, createSystemFromSMILES, extractPositionsFromOEMol, generate_unique_atom_names
from perses.utils.data import load_smi
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol
from perses.rjmc.topology_proposal import TopologyProposal, SmallMoleculeSetProposalEngine
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
from openmmtools.constants import kB
import logging
import os
import dask.distributed as distributed
from collections import namedtuple
from collections import namedtuple
import random
from scipy.special import logsumexp

logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("relative_setup")
_logger.setLevel(logging.INFO)

 # define NamedTuples from feptasks
# EquilibriumResult = namedtuple('EquilibriumResult', ['sampler_state', 'reduced_potentials', 'files', 'timers', 'nonalchemical_perturbations'])
EquilibriumFEPTask = namedtuple('EquilibriumInput', ['sampler_state', 'inputs', 'outputs'])
NonequilibriumFEPTask = namedtuple('NonequilibriumFEPTask', ['particle', 'inputs'])


class RelativeFEPSetup(object):
    """
    This class is a helper class for relative FEP calculations. It generates the input objects that are necessary
    legs of a relative FEP calculation. For each leg, that is a TopologyProposal, old_positions, and new_positions.
    Importantly, it ensures that the atom maps in the solvent and complex phases match correctly.
    """
    def __init__(self,
                 ligand_input,
                 old_ligand_index,
                 new_ligand_index,
                 forcefield_files,
                 phases,
                 protein_pdb_filename=None,
                 receptor_mol2_filename=None,
                 pressure=1.0 * unit.atmosphere,
                 temperature=300.0 * unit.kelvin,
                 solvent_padding=9.0 * unit.angstroms,
                 ionic_strength=0.15 * unit.molar,
                 hmass=4*unit.amus,
                 neglect_angles=False,
                 map_strength='default',
                 atom_expr=None,
                 bond_expr=None,
                 anneal_14s=False,
                 small_molecule_forcefield='gaff-2.11',
                 small_molecule_parameters_cache=None,
                 trajectory_directory=None,
                 trajectory_prefix=None,
                 spectator_filenames=None,
                 nonbonded_method = 'PME',
                 complex_box_dimensions=None,
                 solvent_box_dimensions=None,
                 remove_constraints=False,
                 use_given_geometries = False,
                 given_geometries_tolerance=0.2*unit.angstroms,
                 ):
        """
        Initialize a NonequilibriumFEPSetup object

        Parameters
        ----------
        ligand_input : str
            the name of the ligand file (any openeye supported format)
            this can either be an .sdf or list of .sdf files, or a list of SMILES strings
        forcefield_files : list of str
            The list of ffxml files that contain the forcefields that will be used (excepting for small molecules)
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

        : Quantity, units of concentration
            Concentration of solvent ions to be used when solvating the system
        neglect_angles : bool
            Whether to neglect certain angle terms for the purpose of minimizing work variance in the RJMC protocol.
        anneal_14s : bool, default False
            Whether to anneal 1,4 interactions over the protocol;
                if True, then geometry_engine takes the argument use_14_nonbondeds = False;
                if False, then geometry_engine takes the argument use_14_nonbondeds = True;
        small_molecule_forcefield : str, optional, default='gaff-2.11'
            Small molecule force field name.
            Anything supported by SystemGenerator is supported, but we recommend one of
            ['gaff-1.81', 'gaff-2.11', 'smirnoff99Frosst-1.1.0', 'openff-1.0.0']
        small_molecule_parameters_cache : str, optional, default=None
            If specified, this filename will be used for a small molecule parameter cache by the SystemGenerator.
        trajectory_directory : str, default None
            Where to write out trajectories resulting from the calculation. If none, no writing is done.
        trajectory_prefix : str, default None
            What prefix to use for this calculation's trajectory files. If none, no writing is done.
        spectator_filenames : list, optional, default=None
            If specified, this list is the filenames of any non-alchemical small molecule to be part of the system.
            These will be treated with the same small molecule forcefield as the alchemical ligands, and will only be present in the complex phase
        nonbonded_method : str, default = 'PME'
            nonbonded method, chose one of ['PME','CutoffNonPeriodic','CutoffPeriodic','NoCutoff']
        complex_box_dimensions: Vec(3), optional, default=None
            box dimensions for the complex phase
        solvent_box_dimensions: Vec(3), optional, default=None
            box dimensions for the solvent phase
        remove_constraints : bool, default=False
            if hydrogen constraints should be constrained in the simulation
            default is False, so no constraints removed, but 'all' or 'not water' can be used to remove constraints.
        use_given_geometries : bool, default False
            whether to extract the positions of ligand B and set the unique_new atom positions deterministically;
            if True, `complex` must be in `phases` and .sdf or .mol2 file of ligand must be provided
        given_geometries_tolerance : simtk.unit.Quantity with units of length, default=0.2*angstrom
            If use_given_geometries=True, use this tolerance for identifying mapped atoms
        """
        from openeye import oechem

        # TODO: Refactor this class into initializer that configures options and factory methods that use them.
        # QUESTION: Why do we add these as private object attributes?
        #   If we want users to be able to modify them, they should not be private.
        #   Also, these are all used immediately within __init__, so it doesn't make sense to store them this way.
        self._pressure = pressure
        self._temperature = temperature
        self._barostat_period = 50
        self._pme_tol = 2.5e-04
        self._padding = solvent_padding
        self._ionic_strength = ionic_strength
        self._hmass = hmass
        _logger.info(f"\t\t\t_hmass: {hmass}.\n")
        self._proposal_phase = None
        self._map_strength = map_strength
        self._atom_expr = atom_expr
        self._bond_expr = bond_expr
        self._anneal_14s = anneal_14s
        self._spectator_filenames = spectator_filenames
        self._complex_box_dimensions = complex_box_dimensions
        self._solvent_box_dimensions = solvent_box_dimensions
        self._use_given_geometries = use_given_geometries
        self._given_geometries_tolerance = given_geometries_tolerance
        self._ligand_input = ligand_input

        if self._use_given_geometries:
            assert self._ligand_input[-3:] == 'sdf' or self._ligand_input[-4:] == 'mol2', f"cannot use deterministic atom placement if the ligand input files do not contain geometry information (e.g. in .sdf or .mol2 format)"
            assert 'complex' in phases, f"cannot use deterministic atom placement if complex is not in the specified phases to generate"
            # TODO: add deterministic geometry proposal to solvent and vacuum

        if remove_constraints is False:
            self._h_constraints = app.HBonds
            self._rigid_water = True
        elif remove_constraints == 'all':
            _logger.info(f'Hydrogens will not be constrained. This may be problematic if using a larger timestep')
            self._h_constraints = None
            self._rigid_water = False
        elif remove_constraints == 'not water':
            _logger.info(f'Hydrogens will not be constrained for non-water molecules. This may be problematic if using a larger timestep')
            self._h_constraints = None
            self._rigid_water = True
        else:
            _logger.warning(f"remove_constraints value of {remove_constraints}. Allowed values are False 'all' or 'not water'")

        try:
            self._nonbonded_method = getattr(app,nonbonded_method)
            _logger.info(f'Setting non bonded method to {nonbonded_method}')
        except AttributeError:
            _logger.warning(f'Nonbonded method {nonbonded_method} not recognised')
            if 'complex' in phases or 'solvent' in phases:
                _logger.warning(f"Detected complex or solvent phases: setting PME nonbonded method.")
                self._nonbonded_method = app.PME
            else:
                _logger.info(f"Detected vacuum phase: setting noCutoff nonbonded method.")
                self._nonbonded_method = app.NoCutoff


        self._trajectory_prefix = trajectory_prefix
        self._trajectory_directory = trajectory_directory

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
                _logger.info('  Note that SMILES does not contain geometry information for use in mapping')
                self._ligand_smiles_old = load_smi(self._ligand_input,self._old_ligand_index)
                self._ligand_smiles_new = load_smi(self._ligand_input,self._new_ligand_index)
                _logger.info(f"\told smiles: {self._ligand_smiles_old}")
                _logger.info(f"\tnew smiles: {self._ligand_smiles_new}")

                all_old_mol = createSystemFromSMILES(self._ligand_smiles_old, title='MOL') # should be stereospecific
                self._ligand_oemol_old, self._ligand_system_old, self._ligand_positions_old, self._ligand_topology_old = all_old_mol
                self._ligand_oemol_old = generate_unique_atom_names(self._ligand_oemol_old)

                all_new_mol = createSystemFromSMILES(self._ligand_smiles_new, title='NEW')
                self._ligand_oemol_new, self._ligand_system_new, self._ligand_positions_new, self._ligand_topology_new = all_new_mol
                self._ligand_oemol_new = generate_unique_atom_names(self._ligand_oemol_new)
                _logger.info(f"\tsuccessfully created old and new systems from smiles")

                mol_list.append(self._ligand_oemol_old)
                mol_list.append(self._ligand_oemol_new)

                # forcefield_generators needs to be able to distinguish between the two ligands
                # while topology_proposal needs them to have the same residue name
                self._ligand_oemol_old.SetTitle("MOL")
                self._ligand_oemol_new.SetTitle("MOL")
                _logger.info(f"\tsetting both molecule oemol titles to 'MOL'.")

                self._ligand_topology_old = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_old)
                self._ligand_topology_new = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_new)
                _logger.info(f"\tsuccessfully generated topologies for both oemols.")

            elif self._ligand_input[-3:] == 'sdf' or self._ligand_input[-4:] == 'mol2': #
                _logger.info(f"Detected .sdf format.  Proceeding...") #TODO: write checkpoints for sdf format
                self._ligand_oemol_old = createOEMolFromSDF(self._ligand_input, index=self._old_ligand_index, allow_undefined_stereo=True)
                self._ligand_oemol_new = createOEMolFromSDF(self._ligand_input, index=self._new_ligand_index, allow_undefined_stereo=True)
                # self._ligand_oemol_old = generate_unique_atom_names(self._ligand_oemol_old)
                # self._ligand_oemol_new = generate_unique_atom_names(self._ligand_oemol_new)

                mol_list.append(self._ligand_oemol_old)
                mol_list.append(self._ligand_oemol_new)

                self._ligand_positions_old = extractPositionsFromOEMol(self._ligand_oemol_old)
                self._ligand_positions_new = extractPositionsFromOEMol(self._ligand_oemol_new)
                _logger.info(f"\tsuccessfully extracted positions from OEMOL.")

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
            self._ligand_oemol_old = createOEMolFromSDF(self._ligand_input[self._old_ligand_index])
            self._ligand_oemol_new = createOEMolFromSDF(self._ligand_input[self._new_ligand_index])
            self._ligand_oemol_old = generate_unique_atom_names(self._ligand_oemol_old)
            self._ligand_oemol_new = generate_unique_atom_names(self._ligand_oemol_new)

            self._ligand_oemol_old.SetTitle("OLD")
            self._ligand_oemol_new.SetTitle("NEW")

            mol_list.append(self._ligand_oemol_old)
            mol_list.append(self._ligand_oemol_new)

            # forcefield_generators needs to be able to distinguish between the two ligands
            # while topology_proposal needs them to have the same residue name
            self._ligand_oemol_old.SetTitle("MOL")
            self._ligand_oemol_new.SetTitle("MOL")

            self._ligand_positions_old = extractPositionsFromOEMol(self._ligand_oemol_old)
            _logger.info(f"\tsuccessfully extracted positions from OEMOL.")

            self._ligand_oemol_old.SetTitle("MOL")
            self._ligand_oemol_new.SetTitle("MOL")
            _logger.info(f"\tsetting both molecule oemol titles to 'MOL'.")

            self._ligand_smiles_old = oechem.OECreateSmiString(self._ligand_oemol_old,
                        oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
            self._ligand_smiles_new = oechem.OECreateSmiString(self._ligand_oemol_new,
                        oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
            _logger.info(f"\tsuccessfully created SMILES for both ligand OEMOLs.")
            self._ligand_topology_old = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_old)
            self._ligand_topology_new = forcefield_generators.generateTopologyFromOEMol(self._ligand_oemol_new)

        self._ligand_md_topology_old = md.Topology.from_openmm(self._ligand_topology_old)
        self._ligand_md_topology_new = md.Topology.from_openmm(self._ligand_topology_new)
        _logger.info(f"Created mdtraj topologies for both ligands.")

        # Select barostat
        NONPERIODIC_NONBONDED_METHODS = [app.NoCutoff, app.CutoffNonPeriodic]
        if pressure is not None:
            if self._nonbonded_method not in NONPERIODIC_NONBONDED_METHODS:
                barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
                _logger.info(f"set MonteCarloBarostat because pressure was specified as {pressure} atmospheres")
            else:
                barostat = None
                _logger.info(f"omitted MonteCarloBarostat because pressure was specified but system was not periodic")
        else:
            barostat = None
            _logger.info(f"omitted MonteCarloBarostat because pressure was not specified")

        # Create openforcefield Molecule objects for old and new molecules
        from openff.toolkit.topology import Molecule
        molecules = [ Molecule.from_openeye(oemol,allow_undefined_stereo=True) for oemol in [self._ligand_oemol_old, self._ligand_oemol_new] ]

        # Handle spectator molecules
        if self._spectator_filenames is not None:
            # we have spectator molecules to handle
            # these are lists incase there are multiple spectators
            self._spectator_molecules = []
            self._spectator_positions = []
            self._spectator_topologies = []
            self._spectator_md_topologies = []
            for spectator_file in self._spectator_filenames:
                assert spectator_file[-3:] == 'sdf', 'Spectator molecules must be provided as a .sdf file, in the correct frame of reference of the protein system'
                _logger.info(f'Setting up spectator {spectator_file}')
                spectator_mol = createOEMolFromSDF(spectator_file)
                spectator_mol = generate_unique_atom_names(spectator_mol)
                self._spectator_molecules.append(spectator_mol)
                # add this to a small molecule register
                molecules.append(Molecule.from_openeye(spectator_mol,allow_undefined_stereo=True))
                self._spectator_positions.append(extractPositionsFromOEMol(spectator_mol))
                spectator_topology = forcefield_generators.generateTopologyFromOEMol(spectator_mol)
                self._spectator_md_topologies.append(md.Topology.from_openmm(spectator_topology))
                _logger.info(f"\tsuccessfully generated oemol, positions and topology for spectator {spectator_file}.")
            _logger.info(f'All spectator molecules set up')

        # Create SystemGenerator
        from openmmforcefields.generators import SystemGenerator
        _logger.info(f'PME tolerance: {self._pme_tol}')
        forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': self._pme_tol, 'constraints' : self._h_constraints, 'rigidWater': self._rigid_water, 'hydrogenMass' : self._hmass}
        if small_molecule_forcefield is None or small_molecule_forcefield == 'None':
            self._system_generator = SystemGenerator(forcefields=forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs,
                                      periodic_forcefield_kwargs = {'nonbondedMethod': self._nonbonded_method})
        else:
            self._system_generator = SystemGenerator(forcefields=forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs,
                                                     small_molecule_forcefield=small_molecule_forcefield, molecules=molecules, cache=small_molecule_parameters_cache, periodic_forcefield_kwargs = {'nonbondedMethod': self._nonbonded_method})
        _logger.info("successfully created SystemGenerator to create ligand systems")

        _logger.info(f"executing SmallMoleculeSetProposalEngine...")
        # Create proposal engine
        proposal_engine = SmallMoleculeSetProposalEngine([self._ligand_oemol_old, self._ligand_oemol_new], self._system_generator, residue_name='MOL')
        proposal_engine.map_strength = self._map_strength
        proposal_engine.atom_expr = self._atom_expr
        proposal_engine.bond_expr = self._bond_expr
        proposal_engine.use_given_geometries = self._use_given_geometries
        proposal_engine.given_geometries_tolerance = self._given_geometries_tolerance
        self._proposal_engine = proposal_engine

        _logger.info(f"instantiating FFAllAngleGeometryEngine...")
        if self._use_given_geometries:
            self._geometry_engine = None
        else:
            # NOTE: we are conducting the geometry proposal without any neglected angles
            self._geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=100, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = neglect_angles, use_14_nonbondeds = (not self._anneal_14s))

        # if we are running multiple phases, we only want to generate one topology proposal, and use the same one for the other legs
        # this is tracked using _proposal_phase
        if 'complex' in phases:
            _logger.info('Generating the topology proposal from the complex leg')
            _logger.info(f"setting up complex phase...")
            self._setup_complex_phase(protein_pdb_filename,receptor_mol2_filename,mol_list)
            self._complex_topology_old_solvated, self._complex_positions_old_solvated, self._complex_system_old_solvated = self._solvate_system(
            self._complex_topology_old, self._complex_positions_old,phase='complex',box_dimensions=self._complex_box_dimensions, ionic_strength=self._ionic_strength)
            _logger.info(f"successfully generated complex topology, positions, system")

            self._complex_md_topology_old_solvated = md.Topology.from_openmm(self._complex_topology_old_solvated)

            _logger.info(f"creating TopologyProposal...")
            self._complex_topology_proposal = self._proposal_engine.propose(self._complex_system_old_solvated,
                                          self._complex_topology_old_solvated,
                                          current_mol_id=0, proposed_mol_id=1)

            self.non_offset_new_to_old_atom_map = self._proposal_engine.non_offset_new_to_old_atom_map

            self._proposal_phase = 'complex'

            _logger.info(f"conducting geometry proposal...")
            if self._use_given_geometries:
                self._complex_positions_new_solvated, self._complex_logp_proposal = self._make_new_deterministic_positions('complex'), 0.
            else:
                self._complex_positions_new_solvated, self._complex_logp_proposal = self._geometry_engine.propose(self._complex_topology_proposal,
                                                                                    self._complex_positions_old_solvated,
                                                                                    beta, validate_energy_bookkeeping=False)
            if self._use_given_geometries:
                self._complex_logp_reverse = 0.
                self._complex_added_valence_energy, self._complex_subtracted_valence_energy = None, None
                self._complex_forward_neglected_angles, self._complex_reverse_neglected_angles = None, None
            else:
                self._complex_logp_reverse = self._geometry_engine.logp_reverse(self._complex_topology_proposal, self._complex_positions_new_solvated, self._complex_positions_old_solvated, beta, validate_energy_bookkeeping=False)
                if not self._complex_topology_proposal.unique_new_atoms:
                    assert self._geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
                    assert self._geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
                    self._complex_added_valence_energy = 0.0
                else:
                    self._complex_added_valence_energy = self._geometry_engine.forward_final_context_reduced_potential - self._geometry_engine.forward_atoms_with_positions_reduced_potential

                if not self._complex_topology_proposal.unique_old_atoms:
                    assert self._geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
                    assert self._geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
                    self._complex_subtracted_valence_energy = 0.0
                else:
                    self._complex_subtracted_valence_energy = self._geometry_engine.reverse_final_context_reduced_potential - self._geometry_engine.reverse_atoms_with_positions_reduced_potential

                self._complex_forward_neglected_angles = self._geometry_engine.forward_neglected_angle_terms
                self._complex_reverse_neglected_angles = self._geometry_engine.reverse_neglected_angle_terms
            self._complex_geometry_engine = copy.deepcopy(self._geometry_engine)


        if 'solvent' in phases:
            _logger.info(f"Detected solvent...")
            if self._proposal_phase is None:
                _logger.info(f"no complex detected in phases...generating unique topology/geometry proposals...")
                _logger.info(f"solvating ligand...")
                self._ligand_topology_old_solvated, self._ligand_positions_old_solvated, self._ligand_system_old_solvated = self._solvate_system(
                self._ligand_topology_old, self._ligand_positions_old,phase='solvent',box_dimensions=self._solvent_box_dimensions,ionic_strength=self._ionic_strength)
                self._ligand_md_topology_old_solvated = md.Topology.from_openmm(self._ligand_topology_old_solvated)

                _logger.info(f"creating TopologyProposal")
                self._solvent_topology_proposal = self._proposal_engine.propose(self._ligand_system_old_solvated,
                                                                                self._ligand_topology_old_solvated,
                                                                                current_mol_id=0, proposed_mol_id=1)

                self.non_offset_new_to_old_atom_map = self._proposal_engine.non_offset_new_to_old_atom_map
                self._proposal_phase = 'solvent'
            else:
                _logger.info('Using the topology proposal from the complex leg')
                self._solvent_topology_proposal, self._ligand_positions_old_solvated = self._generate_solvent_topologies(
                    self._complex_topology_proposal, self._complex_positions_old_solvated)

            _logger.info(f"conducting geometry proposal...")
            if self._use_given_geometries:
                self._ligand_positions_new_solvated, self._solvent_logp_proposal = self._make_new_deterministic_positions('solvent'), 0.
            else:
                self._ligand_positions_new_solvated, self._solvent_logp_proposal = self._geometry_engine.propose(self._solvent_topology_proposal,
                                                                                        self._ligand_positions_old_solvated, beta, validate_energy_bookkeeping=False)
            if self._use_given_geometries:
                self._solvent_logp_reverse = 0.
                self._solvent_added_valence_energy, self._solvent_subtracted_valence_energy = None, None
                self._solvent_forward_neglected_angles, self._solvent_reverse_neglected_angles = None, None
            else:
                self._solvent_logp_reverse = self._geometry_engine.logp_reverse(self._solvent_topology_proposal, self._ligand_positions_new_solvated, self._ligand_positions_old_solvated, beta, validate_energy_bookkeeping=False)
                if not self._solvent_topology_proposal.unique_new_atoms:
                    assert self._geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
                    assert self._geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
                    self._solvent_added_valence_energy = 0.0
                else:
                    self._solvent_added_valence_energy = self._geometry_engine.forward_final_context_reduced_potential - self._geometry_engine.forward_atoms_with_positions_reduced_potential

                if not self._solvent_topology_proposal.unique_old_atoms:
                    assert self._geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
                    assert self._geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
                    self._solvent_subtracted_valence_energy = 0.0
                else:
                    self._solvent_subtracted_valence_energy = self._geometry_engine.reverse_final_context_reduced_potential - self._geometry_engine.reverse_atoms_with_positions_reduced_potential

                self._solvent_forward_neglected_angles = self._geometry_engine.forward_neglected_angle_terms
                self._solvent_reverse_neglected_angles = self._geometry_engine.reverse_neglected_angle_terms
            self._solvent_geometry_engine = copy.deepcopy(self._geometry_engine)

        if 'vacuum' in phases:
            _logger.info(f"Detected solvent...")
            # need to change nonbonded cutoff and remove barostat for vacuum leg
            _logger.info(f"assgning noCutoff to nonbonded_method")
            self._nonbonded_method = app.NoCutoff

            self._system_generator.barostat = None
            _logger.info(f'Removing barostat for vacuum phase')
            self._system_generator.forcefield_kwargs['nonbondedMethod'] = self._nonbonded_method
            _logger.info(f'Setting nonbondedMethod to NoCutoff for vacuum phase')

            _logger.info(f"calling SystemGenerator to create ligand systems.")

            if self._proposal_phase is None:
                _logger.info('No complex or solvent leg, so performing topology proposal for vacuum leg')
                self._vacuum_topology_old, self._vacuum_positions_old, self._vacuum_system_old = self._solvate_system(self._ligand_topology_old,
                                                                                                         self._ligand_positions_old,phase='vacuum')
                self._vacuum_topology_proposal = self._proposal_engine.propose(self._vacuum_system_old,
                                                                               self._vacuum_topology_old,
                                                                               current_mol_id=0, proposed_mol_id=1)

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
            if self._use_given_geometries:
                self._vacuum_positions_new, self._vacuum_logp_proposal = self._make_new_deterministic_positions('vacuum'), 0.
            else:
                self._vacuum_positions_new, self._vacuum_logp_proposal = self._geometry_engine.propose(self._vacuum_topology_proposal,
                                                                              self._vacuum_positions_old,
                                                                              beta, validate_energy_bookkeeping=False)
            if self._use_given_geometries:
                if self._use_given_geometries:
                    self._vacuum_logp_reverse = 0.
                    self._vacuum_added_valence_energy, self._vacuum_subtracted_valence_energy = None, None
                    self._vacuum_forward_neglected_angles, self._vacuum_reverse_neglected_angles = None, None
            else:
                self._vacuum_logp_reverse = self._geometry_engine.logp_reverse(self._vacuum_topology_proposal, self._vacuum_positions_new, self._vacuum_positions_old, beta, validate_energy_bookkeeping=False)
                if not self._vacuum_topology_proposal.unique_new_atoms:
                    assert self._geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
                    assert self._geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
                    self._vacuum_added_valence_energy = 0.0
                else:
                    self._vacuum_added_valence_energy = self._geometry_engine.forward_final_context_reduced_potential - self._geometry_engine.forward_atoms_with_positions_reduced_potential

                if not self._vacuum_topology_proposal.unique_old_atoms:
                    assert self._geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
                    assert self._geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
                    self._vacuum_subtracted_valence_energy = 0.0
                else:
                    self._vacuum_subtracted_valence_energy = self._geometry_engine.reverse_final_context_reduced_potential - self._geometry_engine.reverse_atoms_with_positions_reduced_potential

                self._vacuum_forward_neglected_angles = self._geometry_engine.forward_neglected_angle_terms
                self._vacuum_reverse_neglected_angles = self._geometry_engine.reverse_neglected_angle_terms
            self._vacuum_geometry_engine = copy.deepcopy(self._geometry_engine)

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
        # TODO: What if you get both protein pdb and receptor mol2?
        # It might be a better idea to have something to auto-detect the format or a kwarg to specify it.
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

        n_atoms_spectators = 0
        if self._spectator_filenames:
            for i, spectator_topology in enumerate(self._spectator_md_topologies,1):
                _logger.debug(f'Appending spectator number {i} to complex topology')
                self._complex_md_topology_old = self._complex_md_topology_old.join(spectator_topology)
                n_atoms_spectators += spectator_topology.n_atoms
        self._complex_topology_old = self._complex_md_topology_old.to_openmm()

        n_atoms_total_old = self._complex_topology_old.getNumAtoms()
        n_atoms_protein_old = self._receptor_topology_old.getNumAtoms()
        n_atoms_ligand_old = n_atoms_total_old - n_atoms_protein_old - n_atoms_spectators

        self._complex_positions_old = unit.Quantity(np.zeros([n_atoms_total_old, 3]), unit=unit.nanometers)
        self._complex_positions_old[:n_atoms_protein_old, :] = self._receptor_positions_old
        self._complex_positions_old[n_atoms_protein_old:n_atoms_protein_old+n_atoms_ligand_old, :] = self._ligand_positions_old

        if self._spectator_filenames:
            start = n_atoms_protein_old+n_atoms_ligand_old
            for i, spectator_positions in enumerate(self._spectator_positions,1):
                _logger.info(f'Updating positions of spectator number {i} to complex positions')
                n_atoms_spectator, _ = np.shape(spectator_positions)
                _logger.debug(f'Number of spectator atoms: {n_atoms_spectator}')
                self._complex_positions_old[start:start+n_atoms_spectator, :] = spectator_positions
                start += n_atoms_spectator

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
            old_ligand_topology.to_openmm(), old_ligand_positions,phase='solvent', box_dimensions=self._solvent_box_dimensions)

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
        new_solvated_system = self._system_generator.create_system(new_solvated_ligand_omm_topology)

        new_to_old_atom_map = {atom_map[x] - new_mol_start_index: x - old_mol_start_index for x in
                               old_complex.select("resname == 'MOL' ") if x in atom_map.keys()}

        old_alchemical_atoms = [i for i in range(old_mol_len)]
        # adjust the atom map to account for the presence of solvent degrees of freedom:
        # By design, all atoms after the ligands are water, and should be mapped.
        n_water_atoms = solvent_only_topology.to_openmm().getNumAtoms()
        for i in range(n_water_atoms):
            new_to_old_atom_map[new_mol_len + i] = old_mol_len + i

        # make a TopologyProposal
        new_to_old_atom_map = {int(key): int(val) for key, val in new_to_old_atom_map.items()}
        ligand_topology_proposal = TopologyProposal(new_topology=new_solvated_ligand_omm_topology,
                                                    new_system=new_solvated_system,old_alchemical_atoms=old_alchemical_atoms,
                                                    old_topology=old_solvated_topology, old_system=old_solvated_system,
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='A',
                                                    new_chemical_state_key='B')

        _logger.info(f'Adding networkx function to solvent phase')
        from perses.rjmc.topology_proposal import augment_openmm_topology
        # old molecule
        augment_openmm_topology(ligand_topology_proposal.old_topology,
                                self._ligand_oemol_old,
                                [res for res in ligand_topology_proposal.old_topology.residues() if res.name == 'MOL'][0],
                                {i: i for i in range(old_mol_len)})
            # new molecule
        augment_openmm_topology(ligand_topology_proposal.new_topology,
                                self._ligand_oemol_new,
                                [res for res in ligand_topology_proposal.new_topology.residues() if res.name == 'MOL'][0],
                                {i: i for i in range(new_mol_len)})

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
        old_ligand_system = self._system_generator.create_system(old_ligand_topology)
        new_ligand_system = self._system_generator.create_system(new_ligand_topology)

        new_to_old_atom_map = {atom_map[x] - new_mol_start_index: x - old_mol_start_index for x in
                               old_complex.select("resname == 'MOL' ") if x in atom_map.keys()}


        new_to_old_atom_map = {int(key): int(val) for key, val in new_to_old_atom_map.items()}
        # make a TopologyProposal
        ligand_topology_proposal = TopologyProposal(new_topology=new_ligand_topology,
                                                    new_system=new_ligand_system,
                                                    old_topology=old_ligand_topology, old_system=old_ligand_system,
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='A',
                                                    new_chemical_state_key='B')

        _logger.info(f'Adding networkx function to solvent phase')
        from perses.rjmc.topology_proposal import augment_openmm_topology
        # old molecule
        augment_openmm_topology(ligand_topology_proposal.old_topology,
                                self._ligand_oemol_old,
                                [res for res in ligand_topology_proposal.old_topology.residues() if res.name == 'MOL'][0],
                                {i: i for i in range(old_mol_len)})
            # new molecule
        augment_openmm_topology(ligand_topology_proposal.new_topology,
                                self._ligand_oemol_new,
                                [res for res in ligand_topology_proposal.new_topology.residues() if res.name == 'MOL'][0],
                                {i: i for i in range(new_mol_len)})

        return ligand_topology_proposal, old_ligand_positions

    def _make_new_deterministic_positions(self, phase):
        """
        given an old complex topology, positions, the positions of the new ligand, and a topology proposal, generate new complex positions
        """
        top_proposal = getattr(self, f"_{phase}_topology_proposal")
        old_solvated_topology = top_proposal._old_topology
        old_to_new_atom_map = top_proposal._old_to_new_atom_map

        if phase == 'complex':
            old_pos = getattr(self, f"_complex_positions_old_solvated").value_in_unit_system(unit.md_unit_system)
        elif phase == 'solvent':
            old_pos = getattr(self, f"_ligand_positions_old_solvated").value_in_unit_system(unit.md_unit_system)
        elif phase == 'vacuum':
            old_pos = getattr(self, f"_ligand_positions_old").value_in_unit_system(unit.md_unit_system)

        new_positions = np.zeros((top_proposal._new_topology.getNumAtoms(), 3))
        new_positions[list(old_to_new_atom_map.values()),:] = old_pos[list(old_to_new_atom_map.keys()),:]
        new_indices = top_proposal.unique_new_atoms
        old_indices = top_proposal.unique_old_atoms
        if len(new_indices) != 0:
            new_positions[list(top_proposal._new_topology.residue_to_oemol_map.keys())] = self._ligand_positions_new.value_in_unit_system(unit.md_unit_system)[list(top_proposal._new_topology.residue_to_oemol_map.values())]
        else:
            pass
        return new_positions * unit.nanometers

    def _solvate_system(self, topology, positions, model='tip3p',phase='complex', box_dimensions=None,ionic_strength=0.15 * unit.molar):
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
        box_dimensions : tuple of Vec3, default None
            if not None, padding distance will be omitted in favor of a pre-specified set of box dimensions

        Returns
        -------
        solvated_topology : app.Topology
            Topology of the system with added waters
        solvated_positions : [n + 3(n_waters), 3] ndarray of Quantity nm
            Solvated positions
        solvated_system : openmm.System
            The parameterized system, containing a barostat if one was specified.
        """
        # DEBUG: Write PDB file being fed into Modeller to check why MOL isn't being matched
        from simtk.openmm.app import PDBFile
        modeller = app.Modeller(topology, positions)
        # retaining protein protonation from input files
        #hs = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H'] and atom.residue.name not in ['MOL','OLD','NEW']]
        #modeller.delete(hs)
        #modeller.addHydrogens(forcefield=self._system_generator.forcefield)
        _logger.info(f'box_dimensions: {box_dimensions}')
        _logger.info(f'solvent padding: {self._padding._value}')
        run_solvate = True
        if phase == 'solvent':
            self._padding = 9. * unit.angstrom
        if phase == 'vacuum':
            run_solvate = False
            _logger.info(f"\tSkipping solvation of vacuum perturbation")
        if self._padding._value == 0.:
            run_solvate = False
            _logger.info(f"\tSkipping solvation as solvent padding set to zero")
        if run_solvate:
            _logger.info(f"\tpreparing to add solvent")
            if box_dimensions is None:
                modeller.addSolvent(self._system_generator.forcefield, model=model, padding=self._padding, ionicStrength=ionic_strength)
            else:
                modeller.addSolvent(self._system_generator.forcefield, model=model, ionicStrength=ionic_strength, boxSize=box_dimensions)
        solvated_topology = modeller.getTopology()
        if phase == 'complex' and self._padding._value == 0. and box_dimensions is not None:
            _logger.info(f'Complex phase, where padding is set to 0. and box dimensions are provided so setting unit cell dimensions')
            solvated_topology.setUnitCellDimensions(box_dimensions)
        solvated_positions = modeller.getPositions()

        # canonicalize the solvated positions: turn tuples into np.array
        solvated_positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
        _logger.info(f"\tparameterizing...")
        solvated_system = self._system_generator.create_system(solvated_topology)
        _logger.info(f"\tSystem parameterized")

        if self._trajectory_directory is not None and self._trajectory_prefix is not None:
            pdb_filename = f"{self._trajectory_directory}/{self._trajectory_prefix}-{phase}.pdb"
            with open(pdb_filename, 'w') as outfile:
                PDBFile.writeFile(solvated_topology, solvated_positions, outfile)
        else:
            _logger.info('Both trajectory_directory and trajectory_prefix need to be provided to save .pdb')

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

class DaskClient(object):
    """
    This class manages the dask scheduler.

    Parameters
    ----------
    LSF: bool, default False
        whether we are using the LSF dask Client
    num_processes: int, default 2
        number of processes to run.  If not LSF, this argument does nothing
    adapt: bool, default False
        whether to use an adaptive scheduler.  If not LSF, this argument does nothing
    """
    def __init__(self):
        _logger.info(f"Initializing DaskClient")

    def activate_client(self,
                        LSF = True,
                        num_processes = 2,
                        adapt = False):

        if LSF:
            from dask_jobqueue import LSFCluster
            cluster = LSFCluster()
            self._adapt = adapt
            self.num_processes = num_processes

            if self._adapt:
                _logger.debug(f"adapting cluster from 1 to {self.num_processes} processes")
                cluster.adapt(minimum = 2, maximum = self.num_processes, interval = "1s")
            else:
                _logger.debug(f"scaling cluster to {self.num_processes} processes")
                cluster.scale(self.num_processes)

            _logger.debug(f"scheduling cluster with client")
            self.client = distributed.Client(cluster)
        else:
            self.client = None
            self._adapt = False
            self.num_processes = 0

    def deactivate_client(self):
        """
        NonequilibriumSwitchingFEP is not pickleable with the self.client or self.cluster activated.
        This must be called before pickling
        """
        if self.client is not None:
            self.client.close()
            self.client = None

    def scatter(self, df):
        """
        wrapper to scatter the local data df
        """
        if self.client is None:
            #don't actually scatter
            return df
        else:
            return self.client.scatter(df)

    def deploy(self, func, arguments):
        """
        wrapper to map a function and its arguments to the client for scheduling

        Parameters
        ----------
        func : function to map
            arguments: tuple of the arguments that the function will take
        argument : tuple of argument lists

        Returns
        -------
        futures
        """
        if self.client is None:
            if len(arguments) == 1:
                futures = [func(plug) for plug in arguments[0]]
            else:
                futures = [func(*plug) for plug in zip(*arguments)]
        else:
            futures = self.client.map(func, *arguments)
        return futures

    def gather_results(self, futures):
        """
        wrapper to gather a function given its arguments

        Parameters
        ----------
        futures : future pointers

        Returns
        -------
        results
        """
        if self.client is None:
            return futures
        else:
            results = self.client.gather(futures)
            return results

    def wait(self, futures):
        """
        wrapper to wait until futures are complete.
        """
        if self.client is None:
            pass
        else:
            distributed.wait(futures)

class NonequilibriumSwitchingFEP(DaskClient):
    """
    This class manages Nonequilibrium switching based relative free energy calculations, carried out on a distributed computing framework.
    """

    def __init__(self,
                 hybrid_factory,
                 protocol = 'default',
                 n_equilibrium_steps_per_iteration = 100,
                 temperature=300.0 * unit.kelvin,
                 trajectory_directory=None,
                 trajectory_prefix=None,
                 atom_selection="not water",
                 eq_splitting_string="V R O R V",
                 neq_splitting_string = "V R O R V",
                 measure_shadow_work=False,
                 timestep=4.0*unit.femtoseconds,
                 ncmc_save_interval = None,
                 write_ncmc_configuration = False,
                 relative_transform = True):
        """
        Create an instance of the NonequilibriumSwitchingFEP driver class.
        NOTE : defining self.client and self.cluster renders this class non-pickleable; call self.deactivate_client() to close the cluster/client
               objects to render this pickleable.
        Parameters
        ----------
        hybrid_factory : perses.annihilation.relative.HybridTopologyFactory
            hybrid topology factory
        protocol : dict of str: str, default protocol as defined by top of file
            How each force's scaling parameter relates to the main lambda that is switched by the integrator from 0 to 1
            In this case,  the trailblaze threshold must be set explicitly
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
        eq_splitting_string : str, default 'V R O R V'
            The integrator splitting to use for equilibrium simulation
        neq_splitting_string : str, default 'V R O R V'
            The integrator splitting to use for nonequilibrium switching simulation
        ncmc_save_interval : int, default None
            interval with which to write ncmc trajectory.  If None, trajectory will not be saved.
            We will assert that the n_lambdas % ncmc_save_interval = 0; otherwise, the protocol will not be complete
        write_ncmc_configuration : bool, default False
            whether to write ncmc annealing perturbations; if True, will write every ncmc_save_interval iterations
        relative_transform : bool, default True
            whether a relative or absolute alchemical transformation will be conducted.  This extends the utility of the NonequilibriumSwitchingFEP to absolute transforms.
        """
        #Specific to LSF clusters
        # NOTE: assume that the
        _logger.debug(f"instantiating NonequilibriumSwitchingFEP...")

        # construct the hybrid topology factory object
        _logger.info(f"writing HybridTopologyFactory")
        self._factory = hybrid_factory
        topology_proposal = self._factory._topology_proposal

        # use default functions if none specified
        self._protocol = protocol

        self._write_ncmc_configuration = write_ncmc_configuration

        # setup splitting string:
        self._eq_splitting_string = eq_splitting_string
        self._neq_splitting_string = neq_splitting_string
        self._measure_shadow_work = measure_shadow_work

        # set up some class attributes
        self._hybrid_system = self._factory.hybrid_system
        self._initial_hybrid_positions = self._factory.hybrid_positions
        self._n_equil_steps = n_equilibrium_steps_per_iteration
        self._trajectory_prefix = trajectory_prefix
        self._trajectory_directory = trajectory_directory
        self._atom_selection = atom_selection
        self._current_iteration = 0
        self._endpoint_growth_thermostates = dict()
        self._timestep = timestep

        _logger.info(f"instantiating trajectory filenames")
        if self._trajectory_directory and self._trajectory_prefix:
            self._write_traj = True
            self._trajectory_filename = {lambda_state: os.path.join(os.getcwd(), self._trajectory_directory, f"{trajectory_prefix}.eq.lambda_{lambda_state}.h5") for lambda_state in [0,1]}
            _logger.debug(f"eq_traj_filenames: {self._trajectory_filename}")
            self._neq_traj_filename = {direct: os.path.join(os.getcwd(), self._trajectory_directory, f"{trajectory_prefix}.neq.lambda_{direct}") for direct in ['forward', 'reverse']}
            _logger.debug(f"neq_traj_filenames: {self._neq_traj_filename}")
        else:
            self._write_traj = False
            self._trajectory_filename = {0: None, 1: None}
            self._neq_traj_filename = {'forward': None, 'reverse': None}


        # instantiating equilibrium file/rp collection dicts
        self._eq_dict = {0: [], 1: [], '0_decorrelated': None, '1_decorrelated': None, '0_reduced_potentials': [], '1_reduced_potentials': []}
        self._eq_files_dict = {0: [], 1: []}
        self._eq_timers = {0: [], 1: []}

        # instantiating neq_switching collection dicts:
        self._nonalchemical_reduced_potentials = {'from_0': [], 'from_1': [], 'to_0': [], 'to_1': []}
        self._added_valence_reduced_potentials = {'from_0': [], 'from_1': [], 'to_0': [], 'to_1': []}
        self._alchemical_reduced_potentials = {'from_0': [], 'from_1': [], 'to_0': [], 'to_1': []}

        self._alchemical_reduced_potential_differences = {'forward': [], 'reverse': []}
        self._nonalchemical_reduced_potential_differences = {0: [], 1: []} # W_f is work from 0(1)_alch to 0(1)_nonalch
        self._EXP = {'forward': [], 'reverse': [], 0: [], 1: []}
        self._BAR = []


        #instantiate nonequilibrium work dicts: the keys indicate from which equilibrium thermodynamic state the neq_switching is conducted FROM (as opposed to TO)
        self._nonequilibrium_cumulative_work = {'forward': None, 'reverse': None}
        self._nonequilibrium_shadow_work = copy.deepcopy(self._nonequilibrium_cumulative_work)
        self._nonequilibrium_timers = copy.deepcopy(self._nonequilibrium_cumulative_work)
        self.online_timer = []
        self._failures = copy.deepcopy(self._nonequilibrium_cumulative_work)
        self.survival = copy.deepcopy(self._nonequilibrium_cumulative_work)
        self.dg_EXP = copy.deepcopy(self._nonequilibrium_cumulative_work)
        self.dg_BAR = None

        # create the thermodynamic state
        self.relative_transform = relative_transform
        if self.relative_transform:
            _logger.info(f"Instantiating thermodynamic states 0 and 1.")
            lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(self._hybrid_system)
            lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

            lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
            lambda_one_alchemical_state.set_alchemical_parameters(1.0)

            # ensure their states are set appropriately
            self._hybrid_alchemical_states = {0: lambda_zero_alchemical_state, 1: lambda_one_alchemical_state}

            # create the base thermodynamic state with the hybrid system
            self._thermodynamic_state = ThermodynamicState(self._hybrid_system, temperature=temperature)

            # Now create the compound states with different alchemical states
            self._hybrid_thermodynamic_states = {0: CompoundThermodynamicState(self._thermodynamic_state,
                                                                               composable_states=[
                                                                                   self._hybrid_alchemical_states[0]]),
                                                 1: CompoundThermodynamicState(copy.deepcopy(self._thermodynamic_state),
                                                                               composable_states=[
                                                                                   self._hybrid_alchemical_states[1]])}

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


        # create an empty dict of starting and ending sampler_states
        self.start_sampler_states = {_direction: [] for _direction in ['forward', 'reverse']}
        self.end_sampler_states = {_direction: [] for _direction in ['forward', 'reverse']}

        #create observable list
        self.iterations = {_direction: [0] for _direction in ['forward', 'reverse']}
        self.observable = {_direction: [1.0] for _direction in ['forward', 'reverse']}
        self.allowable_resampling_methods = {'multinomial': NonequilibriumSwitchingFEP.multinomial_resample}
        self.allowable_observables = {'ESS': NonequilibriumSwitchingFEP.ESS, 'CESS': NonequilibriumSwitchingFEP.CESS}
        _logger.info(f"constructed")

    def compute_nonalchemical_perturbations(self, start, end, sampler_state, geometry_engine):
        """
        In order to correct for the artifacts associated with CustomNonbondedForces, we need to compute the potential difference between the alchemical endstates and the
        valence energy-corrected nonalchemical endstates.

        Parameters
        ----------
        start : int (0 or 1)
            the starting lambda value which the equilibrium sampler state corresponds to
        end : int (0 or 1)
            the alternate lambda value which the equilibrium sampler state will be annealed to
        sampler_state : openmmtools.states.SamplerState
            the equilibrium sampler state of the current lambda (start)
        geometry_engine : perses.rjmc.geometry.FFAllAngleGeometryEngine
            geometry engine used to create and compute the RJMCMC; we use this to compute the importance weight from the old/new system to the hybrid system (neglecting added valence terms)
        """
        if not hasattr(NonequilibriumSwitchingFEP, 'geometry_engine'):
            self.geometry_engine = geometry_engine
            # Create thermodynamic states for the nonalchemical endpoints
            topology_proposal = self._factory.topology_proposal
            self._nonalchemical_thermodynamic_states = {
                0: ThermodynamicState(topology_proposal.old_system, temperature=self._temperature),
                1: ThermodynamicState(topology_proposal.new_system, temperature=self._temperature)}

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

        #define the nonalchemical_perturbation_args for endstate perturbations before running nonequilibrium switching
        _lambda, _lambda_rev = start, end
        nonalchemical_perturbation_args = {'hybrid_thermodynamic_states': [self._hybrid_thermodynamic_states[_lambda], self._hybrid_thermodynamic_states[_lambda_rev]],
                                            '_endpoint_growth_thermostates': [self._endpoint_growth_thermostates[_lambda_rev], self._endpoint_growth_thermostates[_lambda]],
                                            'factory': self._factory,
                                            'nonalchemical_thermostates': [self._nonalchemical_thermodynamic_states[_lambda], self._nonalchemical_thermodynamic_states[_lambda_rev]],
                                            'lambdas': [_lambda, _lambda_rev]}
        _logger.debug(f"\tnonalchemical_perturbation_args for lambda_start = {_lambda}, lambda_end = {_lambda_rev}: {nonalchemical_perturbation_args}")

        #then we will conduct a perturbation on the given sampler state with the appropriate arguments
        valence_energy, nonalchemical_reduced_potential, hybrid_reduced_potential = feptasks.compute_nonalchemical_perturbation(nonalchemical_perturbation_args['hybrid_thermodynamic_states'][0],
                                                                                                                       nonalchemical_perturbation_args['_endpoint_growth_thermostates'][0],
                                                                                                                       sampler_state,
                                                                                                                       nonalchemical_perturbation_args['factory'],
                                                                                                                       nonalchemical_perturbation_args['nonalchemical_thermostates'][0],
                                                                                                                       nonalchemical_perturbation_args['lambdas'][0])
        alt_valence_energy, alt_nonalchemical_reduced_potential, alt_hybrid_reduced_potential = feptasks.compute_nonalchemical_perturbation(nonalchemical_perturbation_args['hybrid_thermodynamic_states'][1],
                                                                                                                       nonalchemical_perturbation_args['_endpoint_growth_thermostates'][1],
                                                                                                                       sampler_state,
                                                                                                                       nonalchemical_perturbation_args['factory'],
                                                                                                                       nonalchemical_perturbation_args['nonalchemical_thermostates'][1],
                                                                                                                       nonalchemical_perturbation_args['lambdas'][1])
        nonalch_perturbations = {'valence_energies': (valence_energy, alt_valence_energy),
                                 'nonalchemical_reduced_potentials': (nonalchemical_reduced_potential, alt_nonalchemical_reduced_potential),
                                 'hybrid_reduced_potentials': (hybrid_reduced_potential, alt_hybrid_reduced_potential)}

        #now to write the results to the logger attributes
        nonalchemical_reduced_potentials = nonalchemical_perturbation_dict['nonalchemical_reduced_potentials']
        self._nonalchemical_reduced_potentials[f"from_{_lambda}"].append(nonalchemical_reduced_potentials[0])
        self._nonalchemical_reduced_potentials[f"to_{_lambda_rev}"].append(nonalchemical_reduced_potentials[1])

        valence_energies = nonalchemical_perturbation_dict['valence_energies']
        self._added_valence_reduced_potentials[f"from_{_lambda}"].append(valence_energies[0])
        self._added_valence_reduced_potentials[f"to_{_lambda_rev}"].append(valence_energies[1])

        hybrid_reduced_potentials = nonalchemical_perturbation_dict['hybrid_reduced_potentials']
        self._alchemical_reduced_potentials[f"from_{_lambda}"].append(hybrid_reduced_potentials[0])
        self._alchemical_reduced_potentials[f"to_{_lambda_rev}"].append(hybrid_reduced_potentials[1])


    def instantiate_particles(self,
                              n_lambdas = None,
                              n_particles = 5,
                              direction = None,
                              ncmc_save_interval = None,
                              collision_rate = np.inf/unit.picoseconds,
                              LSF = False,
                              num_processes = 2,
                              adapt = False):
        """
        Instantiate sMC particles. This entails loading n_iterations snapshots from disk (from each endstate of specified)
        and distributing Particle classes.

        Parameters
        ----------
        n_lambdas : int, default None
            number of lambdas values.
            if None, then we must give an online protocol in algorithm 4 or allow for trailblaze
        n_particles : int, optional, default 5
            The number of times to run the entire sequence described above (concurrency)
        direction : str, default None
            which direction to conduct the simulation, 'forward', 'reverse', or None; None will run both
        ncmc_save_interval : int, default None
            the interval with which to save configurations of the nonequilibrium trajectory.
            If None, the iterval is set to ncmc_save_interval, so only one configuration is saved.
            If ncmc_save_interval does not evenly divide into n_lambdas, an error is thrown.
        collision_rate : float*openmm.simtk.picoseconds**(-1), default np.inf/unit.picoseconds
            collision rate for integrator /unit.picoseconds

        LSF: bool, default False
            whether we are using the LSF dask Client
        num_processes : int, default 2
            number of processes to run.  This argument does nothing if not LSF
        adapt : bool, default False
            whether to use an adaptive scheduler.

        """
        _logger.debug(f"conducting nonequilibrium_switching with {n_particles} iterations")

        #check n_lambdas
        if n_lambdas is None:
            _logger.info(f"n_lambdas is set to None; presuming trailblazed protocol")
            self._ncmc_save_interval = ncmc_save_interval
        else:
            self._ncmc_save_interval = ncmc_save_interval
            try:
                if self._ncmc_save_interval is not None:
                    assert n_lambdas % self._ncmc_save_interval == 0
            except ValueError:
                print(f"The work writing interval must be a factor of the total number of ncmc steps; otherwise, the ncmc protocol is incomplete!")

        _logger.debug(f"ncmc save interval set as {self._ncmc_save_interval}")
        self._n_lambdas = n_lambdas
        self._collision_rate = collision_rate

        # Now we have to pull the files
        if direction == None:
            directions = ['forward', 'reverse']
            start = [0,1]
            end = [1,0]
        else:
            directions = [direction]
            if direction != 'forward' and direction != 'reverse':
                raise Exception(f"direction must be 'forward' or 'reverse'; argument was given as {direction}.")
            start = [0] if direction == 'forward' else [1]
            end = [1] if direction == 'forward' else [0]

        NonequilibriumFEPSetup_dict = {_direction: [] for _direction in directions}
        for i in range(n_particles):
            for start_lambda, end_lambda, _direction in zip(start, end, directions):
                #create a unique thermodynamic state
                thermodynamic_state = copy.deepcopy(self._hybrid_thermodynamic_states[start_lambda])
                inputs_dict = {'thermodynamic_state': thermodynamic_state,
                               'sampler_state': None,
                               'direction': _direction,
                               'topology': self._factory._hybrid_topology,
                               'n_lambdas': self._n_lambdas,
                               'work_save_interval': self._ncmc_save_interval,
                               'splitting': self._neq_splitting_string,
                               'atom_indices_to_save': self._atom_selection_indices,
                               'trajectory_filename': None,
                               'write_configuration': self._write_ncmc_configuration,
                               'timestep': self._timestep,
                               'collision_rate': self._collision_rate,
                               'measure_shadow_work': self._measure_shadow_work,
                               'label': self._current_iteration,
                               'lambda_protocol': self._protocol
                               }

                #pull the sampler_state
                sampler_state = self.pull_trajectory_snapshot(start_lambda)
                inputs_dict['sampler_state'] = copy.deepcopy(sampler_state)

                #log sampler states
                self.start_sampler_states[_direction].append(copy.deepcopy(sampler_state))
                # self.compute_nonalchemical_perturbations(start_lambda, end_lambda, sampler_state)

                if self._write_traj: #check if we should make 'trajectory_filename' not None
                    _logger.debug(f"\twriting eq traj: {self._trajectory_filename.values()}")
                    _logger.debug(f"\twriting neq traj: {self._neq_traj_filename.values()}")
                    noneq_trajectory_filename = self._neq_traj_filename[_direction] + f".iteration_{self._current_iteration:04}.h5"
                    inputs_dict['trajectory_filename'] = noneq_trajectory_filename

                task = NonequilibriumFEPTask(particle = None, inputs = inputs_dict)
                NonequilibriumFEPSetup_dict[_direction].append(task)
                self._current_iteration += 1
                _logger.debug(f"\titeration {self._current_iteration} of direction {_direction} complete")

        #activate the client
        self.activate_client(LSF = LSF,
                            num_processes = num_processes,
                            adapt = adapt)



        #now to scatter the jobs and map
        self.particle_futures = {_direction: None for _direction in directions}
        for _direction in directions:
            futures = self.deploy(feptasks.Particle.launch_particle, (NonequilibriumFEPSetup_dict[_direction],))
            #futures = [feptasks.Particle.launch_particle(i) for i in NonequilibriumFEPSetup_dict[_direction]]
            self.particle_futures[_direction] = futures
            self.wait(futures)

    def algorithm_4(self,
                    observable = 'ESS',
                    trailblaze_observable_threshold = None,
                    resample_observable_threshold = None,
                    num_integration_steps = 1,
                    resampling_method = 'multinomial',
                    online_protocol = None):
        """
        conduct algorithm 4 according to https://arxiv.org/abs/1303.3123

        Parameters
        ----------
        observable : str
            an observable measure in self.allowable_observables
        trailblaze_observable_threshold : float, default None
            the normalized value used to determine how far to trailblaze.
            If set to None, the online_protocol must be specified.
        resample_observable_threshold : float, default None
            the normalized value used to determine when to resample
            if set to None, resampling will be skipped without computing observables
        num_integration_steps : int, default 1
            number of neq integration steps to perform during annealing
        resampling_method : str, default 'multinomial'
            the method used to resample
        online_protocol : dict of np.array
            dict of the form {_direction: np.array() for _direction in self.particle_futures.keys()}
            the constraints: the np arrays must start and end at 0, 1 (1, 0) for 'forward' ('reverse').
            if the online protocol is None and the trailblaze_observable_threshold is also None, an error will be thrown.
            if n_lambdas is specified in self.instantiate_particles, then len(np.array()) for each direction must equal n_lambdas.
            online_protocol is not None, it will override trailblaze

        """
        import time
        _logger.info(f"Conducting resampling adaptive sMC...")

        if resampling_method not in self.allowable_resampling_methods: #check resampler
            raise Exception(f"{resampling_method} is not a currently supported resampling method.")
        elif observable not in self.allowable_observables.keys(): #check observable
            raise Exception(f"{observable} is not a currently supported observable")

        if online_protocol is None:
            _logger.info(f"online protocol is none...")
            if self._n_lambdas is None:
                _logger.debug(f"_n_lambdas was not specified.  trailblazing")
                self._trailblaze = True
                if trailblaze_observable_threshold is None:
                    raise Exception(f"self._trailblaze = True, but the observable threshold is set to None.  Aborting.")
                self.online_protocol = {'forward': [0.0], 'reverse': [1.0]}
            else:
                _logger.debug(f"_n_lambdas was specified but online protocol was not; creating linearly-spaced lambda protocol; not trailblazing")
                self.online_protocol = {'forward': np.linspace(0, 1, self._n_lambdas), 'reverse': np.linspace(1, 0, self._n_lambdas)}
                self._trailblaze = False
        else:
            _logger.debug(f"online protocol was specified; not trailblazing")
            self._trailblaze = False
            self.online_protocol = online_protocol
            if self._n_lambdas is not None:
                assert len(self.online_protocol) == self._n_lambdas, f"online_protocol was specified, but its length is not the specified number of lambdas.  Aborting."
            for _direction in self.particle_futures.keys():
                if _direction == 'forward':
                    assert 'forward' in self.online_protocol.keys(), f"'forward' not in online_protocol.keys()"
                    assert self.online_protocol['forward'][0] == 0.0 and self.online_protocol['forward'][-1] == 1.0, f"the forward endstates should be 0.0 and 1.0"
                elif _direction == 'reverse':
                    assert 'reverse' in self.online_protocol.keys(), f"'reverse' not in online_protocol.keys()"
                    assert self.online_protocol['reverse'][0] == 1.0 and self.online_protocol['reverse'][-1] == 0.0, f"the reverse endstates should be 1.0 and 0.0"


        #define a thermodynamic state
        thermodynamic_state = copy.deepcopy(self._hybrid_thermodynamic_states[0])

        #define termination lambdas
        self.end_lambdas = {'forward': 1.0, 'reverse': 0.0}

        #define a step counter
        self.step_counters = {_direction: 0 for _direction in self.particle_futures.keys()}

        #define a pass
        self._pass = {_direction: False for _direction in self.particle_futures.keys()}

        _logger.info(f"All setups are complete; proceeding with annealing and resampling with threshold {resample_observable_threshold} until the final lambda is weighed.")
        while True: #we must break the while loop at some point
            start = time.time()
            normalized_observable_values = {_direction: None for _direction in self.particle_futures.keys()}
            #we attempt to trailblaze
            if self._trailblaze:
                for _direction, futures in self.particle_futures.items():
                    _logger.debug(f"\ttrailblazing from lambda = {self.online_protocol[_direction][self.step_counters[_direction]]}")
                    if self.online_protocol[_direction][self.step_counters[_direction]] == self.end_lambdas[_direction]:
                        _logger.debug(f"the current lambda is the last lambda; skipping trailblaze")
                        self._pass[_direction] = True
                        continue
                    self.step_counters[_direction] += 1
                    if len(self.online_protocol[_direction]) > 1:
                        initial_guess = min([2 * self.online_protocol[_direction][-1] - self.online_protocol[_direction][-2], 1.0]) if _direction == 'forward' else max([2 * self.online_protocol[_direction][-1] - self.online_protocol[_direction][-2], 0.0])
                    else:
                        initial_guess = None
                    new_lambda, normalized_observable_values[_direction] = self.binary_search(futures = futures,
                                                                                              start_val = self.online_protocol[_direction][-1],
                                                                                              end_val = self.end_lambdas[_direction],
                                                                                              observable = self.allowable_observables[observable],
                                                                                              observable_threshold = self.observable[_direction][-1] * trailblaze_observable_threshold,
                                                                                              thermodynamic_state = thermodynamic_state,
                                                                                              max_iterations=20,
                                                                                              initial_guess = initial_guess,
                                                                                              precision_threshold = 1e-6)
                    _logger.debug(f"\ttrailblazing from lambda = {self.online_protocol[_direction][-1]} to lambda = {new_lambda}")
                    self.online_protocol[_direction].append(new_lambda)
                    AIS_futures = self.deploy(feptasks.Particle.distribute_anneal, (futures, [new_lambda]*len(futures), [num_integration_steps]*len(futures)))
                    self.particle_futures.update({_direction: AIS_futures})

            else: #we increment by 1 index
                for _direction, futures in self.particle_futures.items():
                    if self.step_counters[_direction] == len(self.online_protocol[_direction]) - 1:
                        self._pass[_direction] = True
                        continue #if the step counter is at the last lambda index, we continue
                    self.step_counters[_direction] += 1
                    new_lambda = self.online_protocol[_direction][self.step_counters[_direction]]
                    AIS_futures = self.deploy(feptasks.Particle.distribute_anneal, (futures, [new_lambda]*len(futures), [num_integration_steps]*len(futures)))
                    #AIS_futures = [feptasks.Particle.distribute_anneal(future, new_lambda, num_integration_steps) for future in futures]
                    self.particle_futures.update({_direction: AIS_futures})

            #we wait for the futures
            [self.wait(self.particle_futures[_direction]) for _direction in self.particle_futures.keys()]


            #attempt to resample all directions
            if resample_observable_threshold is None:
                pass
            else:
                normalized_observable_values = self.attempt_resample(observable = observable,
                                                                 resampling_method = resampling_method,
                                                                 resample_observable_threshold = resample_observable_threshold)

            [self.observable[_direction].append(i) for _direction, i in normalized_observable_values.items() if i is not None]
            [self.iterations[_direction].append(self.step_counters[_direction]) for _direction in self.particle_futures.keys() if not self._pass[_direction]]
            self.online_timer.append(time.time() - start)

            #attempt to break from while loop:
            if all(self.online_protocol[_direction][self.step_counters[_direction]] == self.end_lambdas[_direction] for _direction in self.particle_futures.keys()):
                _logger.info(f"all particle future directions have reached the last iteration")
                break
            else:
                _logger.info(f"\tnot all particle future directions are complete: {[(_direction, self.online_protocol[_direction][self.step_counters[_direction]]) for _direction in self.particle_futures.keys()]}")

        #check to ensure that all of the remote timers are equal to the online timer
        for _direction, futures in self.particle_futures.items():
            #we have reached the max number of steps; check this
            _lambda_futures = self.deploy(feptasks.Particle.pull_current_lambda, (futures,))
            _lambdas = self.gather_results(_lambda_futures)
            #_lambdas = [feptasks.Particle.pull_current_lambda(future) for future in futures]
            if all(_lambda == self.end_lambdas[_direction] for _lambda in _lambdas):
                pass
            else:
                raise Exception(f"online protocol is complete, but for direction {_direction}, the Particle lambdas are {_lambdas}!")

        #now we can parse the outputs
        self.gather_neq_results()

        #and then we can compute the free energy
        self.compute_sMC_free_energy()
        self.deactivate_client()
        _logger.info(f"Complete!")


    def gather_neq_results(self):
        """
        Compact function for gathering results from the Scheduler
        """
        _logger.debug(f"Gathering results...")

        #now gather
        for _direction, futures in self.particle_futures.items():
            _logger.debug(f"parsing {_direction} direction...")

            successes = self.gather_results(self.deploy(feptasks.Particle.pull_success, (futures,)))
            #successes = [feptasks.Particle.pull_success(future) for future in futures]
            self._nonequilibrium_cumulative_work[_direction] = np.array(self.gather_results(self.deploy(feptasks.Particle.pull_cumulative_work_profile, (futures,))))
            #self._nonequilibrium_cumulative_work[_direction] = np.array([feptasks.Particle.pull_cumulative_work_profile(future) for future, success in zip(futures, successes) if success])
            self._nonequilibrium_shadow_work[_direction] = np.array(self.gather_results(self.deploy(feptasks.Particle.pull_shadow_work, (futures,))))
            #self._nonequilibrium_shadow_work[_direction] = np.array([feptasks.Particle.pull_shadow_work(future) for future, success in zip(futures, successes) if success])
            self._nonequilibrium_timers[_direction] = self.gather_results(self.deploy(feptasks.Particle.pull_timers, (futures,)))
            #self._nonequilibrium_timers[_direction] = [feptasks.Particle.pull_timers(future) for future, success in zip(futures, successes) if success]
            self.end_sampler_states[_direction] = self.gather_results(self.deploy(feptasks.Particle.pull_sampler_state, (futures,)))
            #self.end_sampler_states[_direction] = [feptasks.Particle.pull_sampler_state(future) for future, success in zip(futures, successes) if success]
            labels = np.array(self.gather_results(self.deploy(feptasks.Particle.pull_labels, (futures,))))
            #labels = np.array([feptasks.Particle.pull_labels(future) for future, success in zip(futures, successes) if success])
            self._failures[_direction] = [label[0] for label, success in zip(labels, successes) if not success]
            self.particle_futures.update({_direction: None}) #make object pickleable
            try:
                max_step = max(self.iterations[_direction])
                iterations = [i/max_step for i in self.iterations[_direction]]
                first_one = iterations.index(1.)
                shrunken_iterations = np.array(iterations[:first_one + 1])
                self.iterations.update({_direction: shrunken_iterations})

                labels = np.array(self.gather_results(self.deploy(feptasks.Particle.pull_labels, (futures,))))
                #labels = np.array([feptasks.Particle.pull_labels(future) for future, success in zip(futures, successes) if success])
                num_particles, num_switches = labels.shape
                survival = np.array([len(set(list(labels[:, i]))) / num_particles for i in range(num_switches)])
                _logger.debug(f"\tlabels: {labels}")
                self.survival[_direction] = survival
            except Exception as e:
                _logger.info(f"{e}")

    def compute_sMC_free_energy(self):
        """
        Given self._nonequilibrium_cumulative_work, compute the free energy profile
        """
        for _direction, works in self._nonequilibrium_cumulative_work.items():
            if works is not None:
                self.dg_EXP[_direction] = pymbar.EXP(works[:,-1])

        if all(work is not None for work in self._nonequilibrium_cumulative_work.values()):
            #then we can compute a BAR estimate
            self.dg_BAR = pymbar.BAR(self._nonequilibrium_cumulative_work['forward'][:,-1], self._nonequilibrium_cumulative_work['reverse'][:,-1])


    def attempt_resample(self, observable = 'ESS', resampling_method = 'multinomial', resample_observable_threshold = 0.5):
        """
        Attempt to resample particles given an observable diagnostic and a resampling method.

        Parameters
        ----------
        observable : str, default 'ESS'
            the observable used as a resampling diagnostic; this calls a key in self.allowable_observables
        resampling_method: str, default multinomial
            method used to resample, this calls a key in self.allowable_resampling_methods
        resample_observable_threshold : float, default 0.5
            the threshold to diagnose a resampling event.
            If None, will automatically return without observables

        Returns
        -------
        observable_value: dict
            the value of the observable for each direction
        """
        _logger.debug(f"\tAttempting to resample...")
        normalized_observable_values = {_direction: None for _direction in self.particle_futures.keys()}

        for _direction, futures in self.particle_futures.items():
            if self._pass[_direction]:
                continue

            #first, we have to pull the information for the diagnostic
            _logger.debug(f"\t\tdirection: {_direction}")
            work_tuple_futures = self.deploy(feptasks.Particle.pull_work_increments, (futures,))
            work_tuples = self.gather_results(work_tuple_futures)
            #work_tuples = [feptasks.Particle.pull_work_increments(future) for future in futures]
            works_prev = np.array([tup[0] for tup in work_tuples])
            works_incremental = np.array([tup[1] - tup[0] for tup in work_tuples])
            cumulative_works = np.array([tup[1] for tup in work_tuples])
            _logger.debug(f"\tincremental works: {works_incremental}")
            normalized_observable_value = self.allowable_observables[observable](works_prev, works_incremental) / len(works_incremental)

            #decide whether to resample
            _logger.debug(f"\tnormalized observable value: {normalized_observable_value}")
            if normalized_observable_value <= resample_observable_threshold: #then we resample
                _logger.debug(f"\tnormalized observable value ({normalized_observable_value}) <= {resample_observable_threshold}.  Resampling")

                #pull the sampler states and cumulative works
                #sampler_states = [feptasks.Particle.pull_sampler_state(future) for future in futures]
                sampler_states = self.gather_results(self.deploy(feptasks.Particle.pull_sampler_state, (futures,)))

                #pull previous labels
                #previous_labels = [feptasks.Particle.pull_last_label(future) for future in futures]
                previous_labels = self.gather_results(self.deploy(feptasks.Particle.pull_last_label, (futures,)))

                #resample
                resampled_works, resampled_sampler_states, resampled_labels = self.allowable_resampling_methods[resampling_method](cumulative_works,
                                                                                                                sampler_states,
                                                                                                                num_resamples = len(sampler_states),
                                                                                                                previous_labels = previous_labels)

                #push resamples
                resample_futures = self.deploy(feptasks.Particle.push_resamples, (futures, resampled_sampler_states, resampled_labels, resampled_works))
                #resample_futures = [feptasks.Particle.push_resamples(future, sampler_state, label, work) for future, sampler_state, label, work in zip(futures, resampled_sampler_states, resampled_labels, resampled_works)]
                self.particle_futures.update({_direction: resample_futures})
                normalized_observable_value = 1.0
            else:
                _logger.debug(f"\tnormalized observable value ({normalized_observable_value}) > {resample_observable_threshold}.  Skipping resampling.")

            #wait for resamples
            [self.wait(self.particle_futures[_direction]) for _direction in self.particle_futures.keys()]

            normalized_observable_values[_direction] = normalized_observable_value

        return normalized_observable_values

    @staticmethod
    def ESS(works_prev, works_incremental):
        """
        compute the effective sample size (ESS) as given in Eq 3.15 in https://arxiv.org/abs/1303.3123.

        Parameters
        ----------
        works_prev: np.array
            np.array of floats representing the accumulated works at t-1 (unnormalized)
        works_incremental: np.array
            np.array of floats representing the incremental works at t (unnormalized)

        Returns
        -------
        ESS: float
            effective sample size
        """
        prev_weights_normalized = np.exp(-works_prev - logsumexp(-works_prev))
        #_logger.debug(f"\t\tnormalized weights: {prev_weights_normalized}")
        incremental_weights_unnormalized = np.exp(-works_incremental)
        #_logger.debug(f"\t\tincremental weights (unnormalized): {incremental_weights_unnormalized}")
        ESS = np.dot(prev_weights_normalized, incremental_weights_unnormalized)**2 / np.dot(np.power(prev_weights_normalized, 2), np.power(incremental_weights_unnormalized, 2))
        #_logger.debug(f"\t\tESS: {ESS}")
        return ESS

    @staticmethod
    def CESS(works_prev, works_incremental):
        """
        compute the conditional effective sample size (CESS) as given in Eq 3.16 in https://arxiv.org/abs/1303.3123.

        Parameters
        ----------
        works_prev: np.array
            np.array of floats representing the accumulated works at t-1 (unnormalized)
        works_incremental: np.array
            np.array of floats representing the incremental works at t (unnormalized)

        Returns
        -------
        CESS: float
            conditional effective sample size
        """
        prev_weights_normalization = np.exp(logsumexp(-works_prev))
        prev_weights_normalized = np.exp(-works_prev) / prev_weights_normalization
        #_logger.debug(f"\t\tnormalized weights: {prev_weights_normalized}")
        incremental_weights_unnormalized = np.exp(-works_incremental)
        #_logger.debug(f"\t\tincremental weights (unnormalized): {incremental_weights_unnormalized}")
        N = len(prev_weights_normalized)
        CESS = N * np.dot(prev_weights_normalized, incremental_weights_unnormalized)**2 / np.dot(prev_weights_normalized, np.power(incremental_weights_unnormalized, 2))
        #_logger.debug(f"\t\tCESS: {CESS}")
        return CESS

    @staticmethod
    def multinomial_resample(cumulative_works, sampler_states, num_resamples, previous_labels):
        r"""
        from a list of cumulative works and sampler states, resample the sampler states N times with replacement
        from a multinomial distribution conditioned on the weights w_i \propto e^{-cumulative_works_i}

        Parameters
        ----------
        cumulative_works : np.array
            generalized accumulated works at time t for all particles
        sampler_states : list of (openmmtools.states.SamplerState)
            list of sampler states at time t for all particles
        num_resamples : int, default len(sampler_states)
            number of resamples to conduct; default doesn't change the number of particles
        previous_labels : list of int
            previous labels of the particles

        Returns
        -------
        resampled_works : np.array([1.0/num_resamples]*num_resamples)
            resampled works (uniform)
        resampled_sampler_states : list of (openmmtools.states.SamplerState)
            resampled sampler states of size num_resamples
        corrected resampled_labels : list of ints
            resampled labels for tracking particle duplicates
        """
        normalized_weights = np.exp(-cumulative_works - logsumexp(-cumulative_works))
        resampled_labels = np.random.choice(len(normalized_weights), num_resamples, p=normalized_weights, replace = True)
        resampled_sampler_states = [sampler_states[i] for i in resampled_labels]
        resampled_works = np.array([np.average(cumulative_works)] * num_resamples)
        corrected_resampled_labels = np.array([previous_labels[i] for i in resampled_labels])

        return resampled_works, resampled_sampler_states, corrected_resampled_labels

    def binary_search(self,
                      futures,
                      start_val,
                      end_val,
                      observable,
                      observable_threshold,
                      thermodynamic_state,
                      max_iterations=20,
                      initial_guess = None,
                      precision_threshold = None):
        """
        Given corresponding start_val and end_val of observables, conduct a binary search to find min value for which the observable threshold
        is exceeded.

        Parameters
        ----------
        futures:
            list of dask.Future objects that point to futures
        start_val: float
            start value of binary search
        end_val: float
            end value of binary search
        observable : function
            function to compute an observable
        observable_threshold : float
            the threshold of the observable used to satisfy the binary search criterion
        thermodynamic_state:
            thermodynamic state with which to compute importance weights
        max_iterations: int, default 20
            maximum number of interations to conduct
        initial_guess: float, default None
            guess where the threshold is achieved
        precision_threshold: float, default None
            precision threshold below which, the max iteration will break

        Returns
        -------
        midpoint: float
            maximum value that doesn't exceed threshold
        _observable : float
            observed value of observable
        """
        _base_end_val = end_val
        _logger.debug(f"\t\t\tmin, max values: {start_val}, {end_val}. ")
        cumulative_work_futures = self.deploy(feptasks.Particle.pull_cumulative_work, (futures,))
        sampler_state_futures = self.deploy(feptasks.Particle.pull_sampler_state, (futures,))
        cumulative_works = np.array(self.gather_results(cumulative_work_futures))
        sampler_states = self.gather_results(sampler_state_futures)
        thermodynamic_state = self.modify_thermodynamic_state(thermodynamic_state, current_lambda = start_val)
        current_rps = np.array([feptasks.compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])

        if initial_guess is not None:
            midpoint = initial_guess
        else:
            midpoint = (start_val + end_val) * 0.5
        _logger.debug(f"\t\t\tinitial midpoint is: {midpoint}")

        for _ in range(max_iterations):
            _logger.debug(f"\t\t\titeration {_}: current lambda: {midpoint}")
            thermodynamic_state = self.modify_thermodynamic_state(thermodynamic_state, current_lambda = midpoint)
            new_rps = np.array([feptasks.compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])
            _observable = observable(cumulative_works, new_rps - current_rps) / len(current_rps)
            _logger.debug(f"\t\t\tobservable: {_observable}")
            if _observable <= observable_threshold:
                _logger.debug(f"\t\t\tobservable {_observable} <= observable_threshold {observable_threshold}")
                end_val = midpoint
            else:
                _logger.debug(f"\t\t\tobservable {_observable} > observable_threshold {observable_threshold}")
                start_val = midpoint
            midpoint = (start_val + end_val) * 0.5
            if precision_threshold is not None:
                if abs(_base_end_val - midpoint) <= precision_threshold:
                    _logger.debug(f"\t\t\tthe difference between the original max val ({_base_end_val}) and the midpoint is less than the precision_threshold ({precision_threshold}).  Breaking with original max val.")
                    midpoint = _base_end_val
                    thermodynamic_state = self.modify_thermodynamic_state(thermodynamic_state, current_lambda = midpoint)
                    new_rps = np.array([feptasks.compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])
                    _observable = observable(cumulative_works, new_rps - current_rps) / len(current_rps)
                    break
                elif abs(end_val - start_val) <= precision_threshold:
                    _logger.debug(f"\t\t\tprecision_threshold: {precision_threshold} is exceeded.  Breaking")
                    midpoint = end_val
                    thermodynamic_state = self.modify_thermodynamic_state(thermodynamic_state, current_lambda = midpoint)
                    new_rps = np.array([feptasks.compute_reduced_potential(thermodynamic_state, sampler_state) for sampler_state in sampler_states])
                    _observable = observable(cumulative_works, new_rps - current_rps) / len(current_rps)
                    break

        return midpoint, _observable

    def equilibrate(self,
                    n_equilibration_iterations = 1,
                    endstates = [0,1],
                    max_size = 1024*1e3,
                    decorrelate=False,
                    timer = False,
                    minimize = False,
                    LSF = False,
                    num_processes = 2,
                    adapt = False):
        """
        Run the equilibrium simulations a specified number of times at the lambda 0, 1 states. This can be used to equilibrate
        the simulation before beginning the free energy calculation.

        Parameters
        ----------
        n_equilibration_iterations : int; default 1
            number of equilibrium simulations to run, each for lambda = 0, 1.
        endstates : list, default [0,1]
            at which endstate(s) to conduct n_equilibration_iterations (either [0] ,[1], or [0,1])
        max_size : float, default 1.024e6 (bytes)
            number of bytes allotted to the current writing-to file before it is finished and a new equilibrium file is initiated.
        decorrelate : bool, default False
            whether to parse all written files serially and remove correlated snapshots; this returns an ensemble of iid samples in theory.
        timer : bool, default False
            whether to trigger the timing in the equilibration; this adds an item to the EquilibriumResult, which is a list of times for various
            processes in the feptask equilibration scheme.
        minimize : bool, default False
            Whether to minimize the sampler state before conducting equilibration. This is passed directly to feptasks.run_equilibration

        LSF: bool, default False
            whether we are using the LSF dask Client
        num_processes : int, default 2
            number of processes to run.  This argument does nothing if not LSF
        adapt : bool, default False
            whether to use an adaptive scheduler.

        Returns
        -------
        equilibrium_result : perses.dispersed.feptasks.EquilibriumResult
            equilibrium result namedtuple
        """

        _logger.debug(f"conducting equilibration")

        # run a round of equilibrium
        _logger.debug(f"iterating through endstates to submit equilibrium jobs")
        EquilibriumFEPTask_list = []
        for state in endstates: #iterate through the specified endstates (0 or 1) to create appropriate EquilibriumFEPTask inputs
            _logger.debug(f"\tcreating lambda state {state} EquilibriumFEPTask")
            input_dict = {'thermodynamic_state': self._hybrid_thermodynamic_states[state],
                          'nsteps_equil': self._n_equil_steps,
                          'topology': self._factory.hybrid_topology,
                          'n_iterations': n_equilibration_iterations,
                          'splitting': self._eq_splitting_string,
                          'atom_indices_to_save': None,
                          'trajectory_filename': None,
                          'max_size': max_size,
                          'timer': timer,
                          '_minimize': minimize,
                          'file_iterator': 0,
                          'timestep': self._timestep}

            if self._write_traj:
                _logger.debug(f"\twriting traj to {self._trajectory_filename[state]}")
                equilibrium_trajectory_filename = self._trajectory_filename[state]
                input_dict['trajectory_filename'] = equilibrium_trajectory_filename
            else:
                _logger.debug(f"\tnot writing traj")

            if self._eq_dict[state] == []:
                _logger.debug(f"\tself._eq_dict[{state}] is empty; initializing file_iterator at 0 ")
            else:
                last_file_num = int(self._eq_dict[state][-1][0][-7:-3])
                _logger.debug(f"\tlast file number: {last_file_num}; initiating file iterator as {last_file_num + 1}")
                file_iterator = last_file_num + 1
                input_dict['file_iterator'] = file_iterator
            task = EquilibriumFEPTask(sampler_state = self._sampler_states[state], inputs = input_dict, outputs = None)
            EquilibriumFEPTask_list.append(task)

        _logger.debug(f"scattering and mapping run_equilibrium task")
        #remote_EquilibriumFEPTask_list = self.client.scatter(EquilibriumFEPTask_list)
        #distributed.progress(remote_EquilibriumFEPTask_list, notebook = False)

        #futures_EquilibriumFEPTask_list = self.client.map(feptasks.run_equilibrium, remote_EquilibriumFEPTask_list)
        self.activate_client(LSF = LSF,
                            num_processes = num_processes,
                            adapt = adapt)
        futures = self.deploy(feptasks.run_equilibrium, (EquilibriumFEPTask_list,))
        #distributed.progress(futures, notebook = False)
        eq_results = self.gather_results(futures)
        self.deactivate_client()


        _logger.debug(f"finished submitting tasks; gathering...")
        #eq_results = self.client.gather(futures_EquilibriumFEPTask_list)
        for state, eq_result in zip(endstates, eq_results):
            _logger.debug(f"\tcomputing equilibrium task future for state = {state}")
            self._eq_dict[state].extend(eq_result.outputs['files'])
            self._eq_dict[f"{state}_reduced_potentials"].extend(eq_result.outputs['reduced_potentials'])
            self._sampler_states[state] = eq_result.sampler_state
            self._eq_timers[state].append(eq_result.outputs['timers'])

        _logger.debug(f"collections complete.")
        if decorrelate: # if we want to decorrelate all sample
            _logger.debug(f"decorrelating data")
            for state in endstates:
                _logger.debug(f"\tdecorrelating lambda = {state} data.")
                traj_filename = self._trajectory_filename[state]
                if os.path.exists(traj_filename[:-2] + f'0000' + '.h5'):
                    _logger.debug(f"\tfound traj filename: {traj_filename[:-2] + f'0000' + '.h5'}; proceeding...")
                    [t0, g, Neff_max, A_t, uncorrelated_indices] = feptasks.compute_timeseries(np.array(self._eq_dict[f"{state}_reduced_potentials"]))
                    _logger.debug(f"\tt0: {t0}; Neff_max: {Neff_max}; uncorrelated_indices: {uncorrelated_indices}")
                    self._eq_dict[f"{state}_decorrelated"] = uncorrelated_indices

                    #now we just have to turn the file tuples into an array
                    _logger.debug(f"\treorganizing decorrelated data; files w/ num_snapshots are: {self._eq_dict[state]}")
                    iterator, corrected_dict = 0, {}
                    for tupl in self._eq_dict[state]:
                        new_list = [i + iterator for i in range(tupl[1])]
                        iterator += len(new_list)
                        decorrelated_list = [i for i in new_list if i in uncorrelated_indices]
                        corrected_dict[tupl[0]] = decorrelated_list
                    self._eq_files_dict[state] = corrected_dict
                    _logger.debug(f"\t corrected_dict for state {state}: {corrected_dict}")

    def modify_thermodynamic_state(self, thermodynamic_state, current_lambda):
        """
        modify a thermodynamic state in place
        """
        if self.relative_transform:
            thermodynamic_state.set_alchemical_parameters(current_lambda, LambdaProtocol(functions = self._protocol))
            return thermodynamic_state
        else:
            raise Exception(f"modifying a local thermodynamic state when self.relative_transform = False is not supported.  Aborting!")
    def pull_trajectory_snapshot(self, endstate):
        """
        Draw randomly a single snapshot from self._eq_files_dict
        Parameters
        ----------
        endstate: int
            lambda endstate from which to extract an equilibrated snapshot, either 0 or 1
        Returns
        -------
        sampler_state: openmmtools.SamplerState
            sampler state with positions and box vectors if applicable
        """
        #pull a random index
        _logger.debug(f"\tpulling a decorrelated trajectory snapshot...")
        index = random.choice(self._eq_dict[f"{endstate}_decorrelated"])
        _logger.debug(f"\t\tpulled decorrelated index label {index}")
        files = [key for key in self._eq_files_dict[endstate].keys() if index in self._eq_files_dict[endstate][key]]
        _logger.debug(f"\t\t files corresponding to index {index}: {files}")
        assert len(files) == 1, f"files: {files} doesn't have one entry; index: {index}, eq_files_dict: {self._eq_files_dict[endstate]}"
        file = files[0]
        file_index = self._eq_files_dict[endstate][file].index(index)
        _logger.debug(f"\t\tfile_index: {file_index}")

        #now we load file as a traj and create a sampler state with it
        traj = md.load_frame(file, file_index)
        positions = traj.openmm_positions(0)
        box_vectors = traj.openmm_boxes(0)
        sampler_state = SamplerState(positions, box_vectors = box_vectors)

        return sampler_state

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

    def _endpoint_perturbations(self, direction = None, num_subsamples = 100):
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
        if direction == None:
            directions = ['forward', 'reverse']
        else:
            if direction != 'forward' and direction != 'reverse':
                raise Exception(f"direction must be 'forward' or 'reverse'; direction argument was given as {direction}")
        for _direction in directions:
            if _direction == 'forward':
                start_lambda, end_lambda = 0, 1
            elif _direction == 'reverse':
                start_lambda, end_lambda = 1, 0
            else:
                raise Exception(f"direction may only be 'forward' or 'reverse'; the indicated direction was {_direction}")

            if self._alchemical_reduced_potentials[f"from_{start_lambda}"] == [] or self._alchemical_reduced_potentials[f"to_{end_lambda}"] == []:
                raise Exception(f"direction of perturbation calculation was {_direction} but alchemical reduced potentials returned an empty list")
            if self._nonalchemical_reduced_potentials[f"from_{start_lambda}"] == []:
                raise Exception(f"direction of perturbation calculation was {_direction} but nonalchemical reduced potentials returned an empty list")

            alchemical_reduced_potential_differences = [i-j for i, j in zip(self._alchemical_reduced_potentials[f"to_{end_lambda}"], self._alchemical_reduced_potentials[f"from_{start_lambda}"])]
            nonalchemical_reduced_potential_differences = [(i + j) - k for i, j, k in zip(self._nonalchemical_reduced_potentials[f"from_{start_lambda}"], self._added_valence_reduced_potentials[f"from_{start_lambda}"], self._alchemical_reduced_potentials[f"from_{start_lambda}"])]

            #now to decorrelate the differences:
            [alch_t0, alch_g, alch_Neff_max, alch_A_t, alch_full_uncorrelated_indices] = feptasks.compute_timeseries(np.array(alchemical_reduced_potential_differences))
            [nonalch_t0, nonalch_g, nonalch_Neff_max, nonalch_A_t, nonalch_full_uncorrelated_indices] = feptasks.compute_timeseries(np.array(nonalchemical_reduced_potential_differences))

            _logger.debug(f"alchemical decorrelation_results for {_direction}: (t0: {alch_t0}, g: {alch_g}, Neff_max: {alch_Neff_max})")
            _logger.debug(f"nonalchemical decorrelation_results for {start_lambda}: (t0: {nonalch_t0}, g: {nonalch_g}, Neff_max: {nonalch_Neff_max})")

            self._alchemical_reduced_potential_differences[_direction] = np.array([alchemical_reduced_potential_differences[i] for i in alch_full_uncorrelated_indices])
            self._nonalchemical_reduced_potential_differences[start_lambda] = np.array([nonalchemical_reduced_potential_differences[i] for i in nonalch_full_uncorrelated_indices])

            #now to bootstrap results
            alchemical_exp_results = np.array([pymbar.EXP(np.random.choice(self._alchemical_reduced_potential_differences[_direction], size = (len(self._alchemical_reduced_potential_differences[_direction]))), compute_uncertainty=False) for _ in range(num_subsamples)])
            self._EXP[_direction] = (np.average(alchemical_exp_results), np.std(alchemical_exp_results)/np.sqrt(num_subsamples))
            _logger.debug(f"alchemical exp result for {_direction}: {self._EXP[_direction]}")

            nonalchemical_exp_results = np.array([pymbar.EXP(np.random.choice(self._nonalchemical_reduced_potential_differences[start_lambda], size = (len(self._nonalchemical_reduced_potential_differences[start_lambda]))), compute_uncertainty=False) for _ in range(num_subsamples)])
            self._EXP[start_lambda] = (np.average(nonalchemical_exp_results), np.std(nonalchemical_exp_results)/np.sqrt(num_subsamples))
            _logger.debug(f"nonalchemical exp result for {start_lambda}: {self._EXP[start_lambda]}")

    def _alchemical_free_energy(self, num_subsamples = 100):
        """
        Compute (by bootstrapping) the BAR estimate for forward and reverse protocols.
        """

        for _direction in ['forward', 'reverse']:
            if self._nonequilibrium_cum_work[_direction] == []:
                raise Exception(f"Attempt to compute BAR estimate failed because self._nonequilibrium_cum_work[{_direction}] has no work values")

        work_subsamples = {'forward': [np.random.choice(self._nonequilibrium_cum_work['forward'], size = (len(self._nonequilibrium_cum_work['forward']))) for _ in range(num_subsamples)],
                           'reverse': [np.random.choice(self._nonequilibrium_cum_work['reverse'], size = (len(self._nonequilibrium_cum_work['reverse']))) for _ in range(num_subsamples)]}

        bar_estimates = np.array([pymbar.BAR(forward_sample, reverse_sample, compute_uncertainty=False) for forward_sample, reverse_sample in zip(work_subsamples['forward'], work_subsamples['reverse'])])
        df, ddf = np.average(bar_estimates), np.std(bar_estimates) / np.sqrt(num_subsamples)
        self._BAR = [df, ddf]
        return (df, ddf)





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

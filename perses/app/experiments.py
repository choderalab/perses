import yaml
import numpy as np
import pickle
import os
import sys
import simtk.unit as unit
import logging
from perses.utils.data import load_smi

from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
from perses.annihilation.relative import HybridTopologyFactory
from perses.app.relative_setup import NonequilibriumSwitchingFEP, RelativeFEPSetup
from perses.annihilation.lambda_protocol import LambdaProtocol

from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter, sams, replicaexchange
from perses.utils.smallmolecules import render_atom_mapping
from perses.tests.utils import validate_endstate_energies
from openmoltools import forcefield_generators
from perses.utils.openeye import *

#import perses dask Client
from perses.app.relative_setup import DaskClient

logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("setup_relative_calculation")
_logger.setLevel(logging.INFO)

ENERGY_THRESHOLD = 1e-4
from openmmtools.constants import kB

class Experiments(DaskClient):
    """
    Create a NetworkX graph representing the set of chemical states to be sampled.
    Vertices represent nonalchemical states (i.e. ligands/protein mutants).
    Edges represent the alchemical transformations between nonalchemical states.
    """
    default_arguments = {'pressure': 1.0 * unit.atmosphere,
                         'temperature': 300.0 * unit.kelvin,
                         'solvent_padding': 9.0 * unit.angstroms,
                         'hmass': 4 * unit.amus,
                         'map_strength': 'default',
                         'phases': ['vacuum', 'solvent', 'complex'],
                         'forcefield_files': ['gaff.xml', 'amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                         'neglect_angles': False,
                         'anneal_14s': False,
                         'water_model': 'tip3p'}

    known_phases = ['vacuum', 'solvent', 'complex'] # we omit complex phase in the known_phases if a receptor_filename is None

    def __init__(self,
                 ligand_input,
                 receptor_filename = None,
                 graph_connectivity = 'fully_connected',
                 cost = None,
                 resources = None,
                 **kwargs):
        """
        Initialize NetworkX graph and build connectivity with a `graph_connectivity` input.

        Parameters
        ----------
        ligand_input : str
            the name of the ligand file (any openeye supported format)
            this can either be an .sdf or list of .sdf files, or a list of SMILES strings
        receptor_filename : str, default None
            Receptor mol2 or pdb filename. If None, complex leg will be omitted.
        graph_connectivity : str or np.matrix or list of tuples, default 'fully_connected'
            The graph connectivity information for the experiment.  This accepts one of several allowable
            strings (corresponding to connectivity defaults), a 2d np.array specifying the explicit connectivity
            matrix for indexed ligands, or a list of tuples corresponding to pairwise ligand index connections.
            The default is 'fully_connected', which specified a fully connected (i.e. complete or single-clique) graph.
        cost : str, default None
            this is currently a placeholder variable for the specification of the sampling method to be used on the graph.
        resources : dict?, default None
            this is yet another placeholder variable for the allocation of resources


        Parseable **kwargs:
        The following kwargs are parseable from a setup yaml, but have defaults given if no setup yaml is given.
        The kwargs mostly consist of the thermodynamic state of the graph, several potential modifications to the
        hybrid factory (i.e. the alchemical system), the single-topology mapping criteria, and the sampler-specific
        parameters (i.e. parameters specific to Replica Exchange, SAMS, and sMC.)

        forcefield_files : list of str
            The list of ffxml files that contain the forcefields that will be used
        pressure : Quantity, units of pressure
            Pressure to use in the barostat
        temperature : Quantity, units of temperature
            Temperature to use for the Langevin integrator
        solvent_padding : Quantity, units of length
            The amount of padding to use when adding solvent
        neglect_angles : bool
            Whether to neglect certain angle terms for the purpose of minimizing work variance in the RJMC protocol.
        anneal_14s : bool, default False
            Whether to anneal 1,4 interactions over the protocol;
                if True, then geometry_engine takes the argument use_14_nonbondeds = False;
                if False, then geometry_engine takes the argument use_14_nonbondeds = True;



        TODO:
        1. change the name of 'default_arguments' to something more appropriate.
        2. allow custom atom mapping for all edges in graph.  currently, we can only specify one of three mapping schemes for all molecules
        3.
        """
        _logger.info(f"Parsing ligand input file...")
        self.ligand_input = ligand_input
        self.receptor_filename = receptor_filename
        self.graph_connectivity = self._create_connectivity_matrix(graph_connectivity)
        self.cost = cost
        self.resources = resources
        self._parse_ligand_input()

        #Now we must create some defaults for thermodynamic states
        self._create_default_arguments(kwargs)
        self.beta = 1.0 / (kB * default_arguments['temperature'])

        #Now we can create a system generator for each phase.
        self._create_system_generator()

        #Now create the proposal engine
        self.proposal_engine = SmallMoleculeSetProposalEngine(self.smiles_list,
                                                              self.system_generator,
                                                              map_strength = default_arguments['map_strength'],
                                                              residue_name='MOL')
        #create a geometry engine
        self.geometry_engine = FFAllAngleGeometryEngine(metadata=None,
                                                        use_sterics=False,
                                                        n_bond_divisions=100,
                                                        n_angle_divisions=180,
                                                        n_torsion_divisions=360,
                                                        verbose=True,
                                                        storage=None,
                                                        bond_softening_constant=1.0,
                                                        angle_softening_constant=1.0,
                                                        neglect_angles = default_arguments['neglect_angles'],
                                                        use_14_nonbondeds = not default_arguments['anneal_14s'])

    def create_network(self):
        """
        This is the main function of the class.  It builds a networkx graph on all of the transformations.
        """
        import networkx as nx
        graph = nx.DiGraph()

        #first, let's add the nodes
        for index, smiles in self.smiles_list:
            node_attribs = {'smiles': smiles}
            graph.add_node(index, **node_attribs)

        #then, let's add the edges
        nonzero_edge_rows, nonzero_edge_cols = np.nonzero(self.graph_connectivity)
        for i, j in zip(nonzero_edge_rows, nonzero_edge_cols):
            current_oemol, current_positions, current_topology = self.ligand_oemol_pos_top[i]
            proposed_oemol, proposed_positions, proposed_topology = self.ligand_oemol_pos_top[j]
            proposals =  self._generate_proposals(current_oemol = current_oemol,
                                                  proposed_oemol = proposed_oemol, 
                                                  current_positions = current_positions,
                                                  current_topology = current_topology)
            graph.add_edge(i,j, **proposals)
            graph.edges[i,j]['weight'] = self.graph_connectivity[i,j]

        return graph


    def _parse_ligand_input(self):
        """
        Parse the ligand input.
        Creates the following attributes:
        1. self.ligand_oemol_pos_top : list of tuple(oemol, position, topology)
        2. self.ligand_ffxml : xml for ligands
        3. self.smiles_list : list of smiles
        4. self.ligand_md_topologies : list of mdtraj.Topology objects
                                       corresponding to self.ligand_oemol_pos_top topologies.
        """
        self.ligand_oemol_pos_top = []
        if type(self.ligand_input) is str: # the ligand has been provided as a single file
            _logger.debug(f"ligand input is a str; checking for .smi and .sdf file.")
            if self._ligand_input[-3:] == 'smi':
                _logger.info(f"Detected .smi format.  Proceeding...")
                smiles_list = load_smi(self.ligand_input)

                #create a ligand data list to hold all ligand oemols, systems, positions, topologies
                for smiles in self.smiles_list:
                    _logger.debug(f"creating oemol, system, positions, and openmm.Topology for smiles: {smiles}...")
                    oemol, system, positions, topology = createSystemFromSMILES(smiles, title='MOL')
                    self.ligand_oemol_pos_top.append((oemol, positions))

                #pull all of the oemols (in order) to make an appropriate ffxml
                mol_list = [_tuple[0] for _tuple in self.ligand_oemol_pos_top]
                self.ligand_ffxml = forcefield_generators.generateForceFieldFromMolecules(mol_list)

                #now make all of the oemol titles 'MOL'
                [self.ligand_oemol_pos_top[i][0].SetTitle("MOL") for i in range(len(self.ligand_oemol_pos_top))]

                #the last thing to do is to make ligand topologies
                ligand_topologies = [forcefield_generators.generateTopologyFromOEMol(data[0]) for data in self.ligand_oemol_pos_top]
                [self.ligand_oemol_pos_top[i] + topology for i, topology in enumerate(ligand_topologies)]

            elif self._ligand_input[-3:] == 'sdf': #
                _logger.info(f"Detected .sdf format.  Proceeding...") #TODO: write checkpoints for sdf format
                oemols = createOEMolFromSDF(self._ligand_input, index = None)
                positions = [extractPositionsFromOEMol(oemol) for oemol in oemols]
                self.ligand_ffxml = forcefield_generators.generateForceFieldFromMolecules(oemols)
                [oemol.SetTitle("MOL") for oemol in oemols]
                self.smiles_list = [ oechem.OECreateSmiString(oemol, oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens) for oemol in oemols]
                self.ligand_oemol_pos_top = [(oemol, position, forcefield_generators.generateTopologyFromOEMol(oemol)) for oemol, position in zip(oemols, positions)]

        else:
            raise Exception(f"the ligand input can only be a string pointing to an .sdf or .smi file.  Aborting!")

        self.ligand_md_topologies = [md.Topology.from_openmm(item[2]) for item in self.ligand_oemol_pos_top]
    def _create_default_arguments(self):
        """
        Define kwargs that will replace default_arguments. note we ar just updating the class attributes
        """
        #first update the __dict__ with kwargs
        #assert that every keyword is in the set of default_arguments:
        assert set(kwargs.keys()).issubset(set(default_arguments.keys())), f"kwargs keys: {kwargs.keys()} is not a subset of default argument keys: {default_arguments.keys()}"
        for keyword in kwargs.keys():
            #assert keyword in default_arguments.keys(), f"kwarg keyword {keyword} is not in default argument keys: {default_arguments.keys()}"
            assert type(kwargs[keyword]) == type(default_arguments[keyword]), f"kwarg {keyword}: {kwargs[keyword]} type ({type(kwargs[keyword])}) is not the appropriate type ({type(default_arguments[keyword])})"

        #specal phasese argument:
        if 'phases' in kwargs.keys():
            assert set(kwargs['phases']).issubset(set(known_phases)), f"{set(kwargs['phases'])} is not a subset of known phases: {set(known_phases)}.  Aborting!"

        args_left_as_default = set(default_arguments.keys()).difference(set(kwargs.keys()))

        for arg in args_left_as_default:
            _logger.info(f"{arg} was left as default of {default_arguments[arg]}")

        default_arguments.update(kwargs)

        #update the nonbonded method from the default arguments...
        if 'complex' in default_arguments['phases'] or 'solvent' in default_arguments['phases']:
            self.nonbonded_method = app.PME
            _logger.info(f"Detected complex or solvent phases: setting PME nonbonded method.")
        elif 'vacuum' in default_arguments['phases']:
            self.nonbonded_method = app.NoCutoff
            _logger.info(f"Detected vacuum phase: setting noCutoff nonbonded method.")

    def _create_system_generator(self):
        """
        Wrap the process for generating a dict of system generators for each phase.
        """
        if default_arguments['pressure'] is not None:
            if self.nonbonded_method == app.PME:
                barostat = openmm.MonteCarloBarostat(default_arguments['pressure'],
                                                     default_arguments['temperature'],
                                                     50)
            else:
                barostat = None
            self.system_generator = SystemGenerator(default_arguments['forcefield_files'],
                                                    barostat=barostat,
                                                     forcefield_kwargs={'removeCMMotion': False,
                                                                        'nonbondedMethod': self.nonbonded_method,
                                                                        'constraints' : app.HBonds,
                                                                        'hydrogenMass' : default_arguments['hmass']})
        else:
            self.system_generator = SystemGenerator(forcefield_files,
                                                    forcefield_kwargs={'removeCMMotion': False,
                                                                       'nonbondedMethod': self.nonbonded_method,
                                                                       'constraints' : app.HBonds,
                                                                       'hydrogenMass' : default_arguments['hmass']})

        self._system_generator._forcefield.loadFile(StringIO(self.ligand_ffxml))

    def _setup_complex_phase(self, ligand_oemol, ligand_positions, ligand_topology):
        """
        Creates complex positions and topology given ligand positions and topology.

        Arguments
        ---------
        ligand_oemol : oechem.oemol object
            oemol of ligand (this is only necessary for .sdf-type receptor files)
        ligand_positions : unit.Quantity(np.ndarray(), units = units.nanometers)
            positions of the ligand
        ligand_topology : mdtraj.Topology
            md topology of the ligand

        Returns
        -------
        complex_md_topology : mdtraj.Topology
            complex mdtraj topology
        complex_topology : simtk.openmm.Topology
            complex openmm topology
        complex_positions : unit.Quantity(np.ndarray(), units = units.nanometers)
            positions of the complex
        """
        if self.receptor_filename[-3:] == 'pdb':
            with open(self.receptor_filename, 'r') as pdbfile:
                receptor_pdb = app.PDBFile(pdbfile)
            receptor_positions = receptor_pdb.positions
            receptor_topology = receptor_pdb.topology
            receptor_mdtraj_topology = md.Topology.from_openmm(receptor_topology)

        elif self.receptor_filename[:-4] == 'mol2':
            receptor_mol = createOEMolFromSDF(self.receptor_filename)
            receptor_positions = extractPositionsFromOEMol(receptor_mol)
            receptor_topology = self._receptor_topology_old = forcefield_generators.generateTopologyFromOEMol(receptor_mol)
            receptor_mdtraj_topology = md.Topology.from_openmm(receptor_topology)

        complex_md_topology = receptor_mdtraj_topology.join(ligand_topology)
        complex_topology = self.complex_md_topology.to_openmm()
        n_atoms_complex = self.complex_topology.getNumAtoms()
        n_atoms_receptor = receptor_topology.getNumAtoms()

        complex_positions = unit.Quantity(np.zeros([n_atoms_complex, 3]), unit=unit.nanometers)
        complex_positions[:n_atoms_receptor, :] = receptor_positions
        complex_positions[n_atoms_receptor:, :] = ligand_positions

        return complex_md_topology, complex_topology, complex_positions

    def _solvate(self, topology, positions, model = 'tip3p', vacuum = False):
        """
        solvate a topology, position and return a topology, position, and system

        Argumnts
        --------
        topology : simtk.openmm.Topology
            topology of the object to be solvated
        positions : unit.Quantity(np.ndarray(), units = units.nanometers)
            positions of the complex
        model : str, default 'tip3p'
            solvent model to use for solvation
        vacuum : bool, default False
            whether to prepare system in vacuum

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
            modeller.addSolvent(self._system_generator._forcefield, model=model, padding=self._padding, ionicStrength=0.15*unit.molar)
        else:
            _logger.info(f"\tSkipping solvation of vacuum perturbation")
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()

        # canonicalize the solvated positions: turn tuples into np.array
        solvated_positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
        solvated_system = self._system_generator.build_system(solvated_topology)
        return solvated_topology, solvated_positions, solvated_system

    def _generate_solvent_topologies(self, topology_proposal, old_positions):
        """
        This method generates ligand-only topologies and positions from a TopologyProposal containing a solvated complex.
        The output of this method is then used when building the solvent-phase simulation with the same atom map.

        Parameters
        ----------
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
            Topology proposal object of the ligand in complex
        old_positions : array
            Positions of the fully solvated complex

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

        old_mol_start_index, old_mol_len = self.proposal_engine._find_mol_start_index(old_complex.to_openmm())
        new_mol_start_index, new_mol_len = self.proposal_engine._find_mol_start_index(new_complex.to_openmm())

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
        new_solvated_system = self.system_generator.build_system(new_solvated_ligand_omm_topology)

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
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='A',
                                                    new_chemical_state_key='B')

        return ligand_topology_proposal, old_solvated_positions

    def _generate_vacuum_topologies(self, topology_proposal, old_positions):
        """
        This method generates ligand-only topologies and positions from a TopologyProposal containing a solvated complex.
        The output of this method is then used when building the solvent-phase simulation with the same atom map.

        Parameters
        ----------
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
            topology proposal to parse
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

        old_mol_start_index, old_mol_len = self.proposal_engine._find_mol_start_index(old_complex.to_openmm())
        new_mol_start_index, new_mol_len = self.proposal_engine._find_mol_start_index(new_complex.to_openmm())

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
        old_ligand_system = self.system_generator.build_system(old_ligand_topology)
        new_ligand_system = self.system_generator.build_system(new_ligand_topology)

        new_to_old_atom_map = {atom_map[x] - new_mol_start_index: x - old_mol_start_index for x in
                               old_complex.select("resname == 'MOL' ") if x in atom_map.keys()}


        # make a TopologyProposal
        ligand_topology_proposal = TopologyProposal(new_topology=new_ligand_topology,
                                                    new_system=new_ligand_system,
                                                    old_topology=old_ligand_topology, old_system=old_ligand_system,
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='A',
                                                    new_chemical_state_key='B')

        return ligand_topology_proposal, old_ligand_positions

    def _handle_valence_energies(topology_proposal):
        """
        simple wrapper function to return forward and reverse valence energies from the complex proposal

        Arguments
        ---------
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
            topology proposal to parse

        Returns
        -------
        added_valence_energy : float
            the reduced valence energy pulled from geometry_engine forward
        subtracted_valence_energy : float
            the reduced valence energy pulled from geometry_engine reverse
        """
        if not topology_proposal.unique_new_atoms:
            assert self.geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self.geometry_engine.forward_final_context_reduced_potential})"
            assert self.geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
            added_valence_energy = 0.0
        else:
            added_valence_energy = self.geometry_engine.forward_final_context_reduced_potential - self.geometry_engine.forward_atoms_with_positions_reduced_potential

        if not topology_proposal.unique_old_atoms:
            assert self.geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self.geometry_engine.reverse_final_context_reduced_potential})"
            assert self.geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self.geometry_engine.reverse_atoms_with_positions_reduced_potential})"
            subtracted_valence_energy = 0.0
        else:
            subtracted_valence_energy = self.geometry_engine.reverse_final_context_reduced_potential - self.geometry_engine.reverse_atoms_with_positions_reduced_potential

        return added_valence_energy, subtracted_valence_energy


    def _generate_proposals(self, current_oemol, proposed_oemol, current_positions, current_topology):
        """
        Create topology and geometry proposals for a ligand in every specified phase.
        If complex is specified, the topology proposal is recycled for all phases; else, the proposal is conducted
        in solvent and recycled for vacuum.  If the only phase is vacuum, then we generate a single proposal without recycling.

        Arguments
        ---------

        """
        proposals = {}
        if 'complex' in default_arguments['phases']:
            assert self.nonbonded_method == app.PME, f"Complex phase is specified, but the nonbonded method is not {app.PME} (is currently {self.nonbonded_method})."
            complex_md_topology, complex_topology, complex_positions = self._setup_complex_phase(current_oemol, current_positions, current_topology)
            solvated_complex_topology, solvated_complex_positions, solvated_complex_system = self._solvate(complex_topology,
                                                                                                           complex_positions,
                                                                                                           model = default_arguments['water_model'],
                                                                                                           vacuum = False)
            solvated_complex_md_topology = md.Topology.from_openmm(solvated_complex_topology)
            complex_topology_proposal = self.proposal_engine.propose(current_system = solvated_complex_system,
                                                                     current_topology = solvated_complex_topology,
                                                                     current_mol = current_oemol,
                                                                     proposed_mol = proposed_oemol)

            proposed_solvated_complex_positions, complex_logp_proposal = self.geometry_engine.propose(complex_topology_proposal,
                                                                                                      solvated_complex_positions,
                                                                                                      self.beta)
            complex_logp_reverse = self.geometry_engine.logp_reverse(complex_topology_proposal,
                                                             proposed_solvated_complex_positions,
                                                             solvated_complex_positions, self.beta)

            complex_added_valence_energy, complex_subtracted_valence_energy = self._handle_valence_energies(complex_topology_proposal)
            complex_forward_neglected_angles = self.geometry_engine.forward_neglected_angle_terms
            complex_reverse_neglected_angles = self._geometry_engine.reverse_neglected_angle_terms

            #now to add it to phases
            proposals.update({'complex': {'topology_proposal': copy.deepcopy(complex_topology_proposal),
                                          'proposed_positions': proposed_solvated_complex_positions,
                                          'logp_proposal': complex_logp_proposal,
                                          'logp_reverse': complex_logp_reverse,
                                          'added_valence_energy': complex_added_valence_energy,
                                          'subtracted_valence_energy': complex_subtracted_valence_energy,
                                          'forward_neglected_angles': complex_forward_neglected_angles,
                                          'reverse_neglected_angles': complex_reverse_neglected_angles}
                                          })

        if 'solvent' in default_arguments['phases']:
            if 'complex' in default_arguments['phases']:
                assert 'complex' in proposals.keys(), f"'complex' is a phase that should have been handled, but it is not in proposals."
                assert self.nonbonded_method == app.PME, f"solvent phase is specified, but the nonbonded method is not {app.PME} (is currently {self.nonbonded_method})."
                solvated_ligand_topology_proposal, solvated_ligand_positions = self._generate_solvent_topologies(topology_proposal = complex_topology_proposal,
                                                                                                        old_positions = solvated_complex_positions)
            else:
                solvated_ligand_topology, solvated_ligand_positions, solvated_ligand_system = self._solvate(current_topology,
                                                                                                            current_positions,
                                                                                                            model = default_arguments['water_model'],
                                                                                                            vacuum = False)
                solvated_ligand_md_topology = md.Topology.from_openmm(solvated_ligand_topology)
                solvated_ligand_topology_proposal = self.proposal_engine.propose(current_system = solvated_ligand_system,
                                                                                 current_topology = solvated_ligand_topology,
                                                                                 current_mol = current_oemol,
                                                                                 proposed_mol = proposed_oemol)
            proposed_solvated_ligand_positions, solvent_logp_proposal = self.geometry_engine.propose(solvated_ligand_topology_proposal,
                                                                                             solvated_ligand_positions,
                                                                                             self.beta)
            solvent_logp_reverse = self.geometry_engine.logp_reverse(solvated_ligand_topology_proposal,
                                                                     proposed_solvated_ligand_positions,
                                                                     solvated_ligand_positions,
                                                                     self.beta)

            solvated_added_valence_energy, solvated_subtracted_valence_energy = self._handle_valence_energies(solvated_ligand_topology_proposal)
            solvated_forward_neglected_angles = self.geometry_engine.forward_neglected_angle_terms
            solvated_reverse_neglected_angles = self.geometry_engine.reverse_neglected_angle_terms

            #now to add it to phases
            proposals.update({'solvent': {'topology_proposal': copy.deepcopy(solvated_ligand_topology_proposal),
                                          'proposed_positions': proposed_solvated_ligand_positions,
                                          'logp_proposal': solvent_logp_proposal,
                                          'logp_reverse': solvent_logp_reverse,
                                          'added_valence_energy': solvated_added_valence_energy,
                                          'subtracted_valence_energy': solvent_subtracted_valence_energy,
                                          'forward_neglected_angles': solvated_forward_neglected_angles,
                                          'reverse_neglected_angles': solvated_reverse_neglected_angles}
                                          })

        if 'vacuum' in default_arguments['phases']:
            vacuum_system_generator = SystemGenerator(default_arguments['forcefield_files'],
                                                      forcefield_kwargs={'removeCMMotion': False,
                                                                         'nonbondedMethod': app.NoCutoff,
                                                                         'constraints' : app.HBonds})
            vacuum_system_generator._forcefield.loadFile(StringIO(self.ligand_ffxml))
            if 'complex' not in default_arguments['phases'] and 'solvent' not in default_arguments['phases']:
                vacuum_ligand_topology, vacuum_ligand_positions, vacuum_ligand_system = self._solvate(current_topology,
                                                                                                      current_positions,
                                                                                                      vacuum=True)
                vacuum_ligand_topology_proposal = self.proposal_engine.propose(vacuum_ligand_system,
                                                                               vacuum_ligand_topology,
                                                                               current_mol = current_oemol,
                                                                               proposed_mol = proposed_oemol)
            elif 'complex' in default_arguments['phases']:
                vacuum_ligand_topology_proposal, vacuum_ligand_positions = self._generate_vacuum_topologies(complex_topology_proposal,
                                                                                                            solvated_complex_positions)
            elif 'solvent' in default_arguments['phases']:
                vacuum_topology_proposal, vacuum_ligand_positions = self._generate_vacuum_topologies(solvated_ligand_topology_proposal,
                                                                                                  solvated_ligand_positions)
            else:
                raise Exeption(f"There is an unnacounted for error in the topology proposal generation for vacuum phase.  Aborting!")

            proposed_vacuum_ligand_positions, vacuum_logp_proposal = self.geometry_engine.propose(vacuum_ligand_topology_proposal,
                                                                                                  vacuum_ligand_positions,
                                                                                                  self.beta)
            vacuum_logp_reverse = self.geonetry_engine.logp_reverse(vacuum_ligand_topology_proposal,
                                                                    proposed_vacuum_ligand_positions,
                                                                    vacuum_ligand_positions,
                                                                    self.beta)

            vacuum_added_valence_energy, vacuum_subtracted_valence_energy = self._handle_valence_energies(vacuum_ligand_topology_proposal)
            vacuum_forward_neglected_angles = self.geometry_engine.forward_neglected_angle_terms
            vacuum_reverse_neglected_angles = self.geometry_engine.reverse_neglected_angle_terms

            #now to add it to phases
            proposals.update({'vacuum': {'topology_proposal': copy.deepcopy(vacuum_ligand_topology_proposal),
                                          'proposed_positions': proposed_vacuum_ligand_positions,
                                          'logp_proposal': vacuum_logp_proposal,
                                          'logp_reverse': vacuum_logp_reverse,
                                          'added_valence_energy': vacuum_added_valence_energy,
                                          'subtracted_valence_energy': vacuum_subtracted_valence_energy,
                                          'forward_neglected_angles': vacuum_forward_neglected_angles,
                                          'reverse_neglected_angles': vacuum_reverse_neglected_angles}
                                          })

        return proposals

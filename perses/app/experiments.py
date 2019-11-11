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
                         'anneal_14s': False}

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
        self.graph_connectivity = graph_connectivity
        self.cost = cost
        self.resources = resources
        self._parse_ligand_input()

        #Now we must create some defaults for thermodynamic states
        self._create_default_arguments(kwargs)

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

        # now we have to create a Networkx graph.











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

    def _setup_networkx_graph(self):
        """
        Create networkx graph for the set of ligands with connectivity.
        """
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
        complex_positions : unit.Quantity(np.ndarray(), units = units.nanometers)
            positions of the complex
        complex_topology : mdtraj.Topology
            md topology of the complex
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

        self.complex_md_topology = receptor_mdtraj_topology.join(ligand_topology)
        self.complex_topology = self.complex_md_topology.to_openmm()
        n_atoms_complex = self.complex_topology.getNumAtoms()
        n_atoms_receptor = receptor_topology.getNumAtoms()

        self.complex_positions = unit.Quantity(np.zeros([n_atoms_complex, 3]), unit=unit.nanometers)
        self.complex_positions[:n_atoms_receptor, :] = receptor_positions
        self.complex_positions[n_atoms_receptor:, :] = ligand_positions

    def _setup_solvent_phase(self):
        """
        setup the solvent phase of the simulation; if the complex phase is undefined,
        we have to make a separate topology proposal.
        """

from __future__ import absolute_import
from perses.dispersed import feptasks
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator, LangevinIntegrator
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState
import openmmtools.mcmc as mcmc
import threading
import queue
import openmmtools.alchemy as alchemy
import pymbar
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
from perses.tests.utils import createSystemFromIUPAC, get_data_filename, extractPositionsFromOEMOL
from perses.annihilation.new_relative import HybridTopologyFactory
from perses.rjmc.topology_proposal import TopologyProposal, TwoMoleculeSetProposalEngine, SystemGenerator, SmallMoleculeSetProposalEngine
from perses.rjmc.geometry import FFAllAngleGeometryEngine
import openeye.oechem as oechem
import celery
from openmoltools import forcefield_generators
import copy
import mdtraj as md
import mdtraj.utils as mdtrajutils
from io import StringIO
from openmmtools.constants import kB
import logging
import os
import pickle
import dask.distributed as distributed

from perses.dispersed.feptasks import NonequilibriumSwitchingMove
work_lock = threading.Lock()

_logger = logging.getLogger(__name__)

class NonequilibriumFEPSetup(object):
    """
    This class is a helper class for nonequilibrium FEP. It generates the input objects that are necessary for the two
    legs of a relative FEP calculation. For each leg, that is a TopologyProposal, old_positions, and new_positions.
    Importantly, it ensures that the atom maps in the solvent and complex phases match correctly.
    """

    def __init__(self, protein_pdb_filename, ligand_file, old_ligand_index, new_ligand_index, forcefield_files,
                 pressure=1.0 * unit.atmosphere, temperature=300.0 * unit.kelvin, solvent_padding=9.0 * unit.angstroms):
        """
        Initialize a NonequilibriumFEPSetup object

        Parameters
        ----------
        protein_pdb_filename : str
            The name of the protein pdb file
        ligand_file : str
            the name of the ligand file (any openeye supported format)
        ligand_smiles : list of two str
            The SMILES strings representing the two ligands
        forcefield_files : list of str
            The list of ffxml files that contain the forcefields that will be used
        pressure : Quantity, units of pressure
            Pressure to use in the barostat
        temperature : Quantity, units of temperature
            Temperature to use for the Langevin integrator
        solvent_padding : Quantity, units of length
            The amount of padding to use when adding solvent
        """
        self._protein_pdb_filename = protein_pdb_filename
        self._pressure = pressure
        self._temperature = temperature
        self._barostat_period = 50
        self._padding = solvent_padding

        self._ligand_file = ligand_file
        self._old_ligand_index = old_ligand_index
        self._new_ligand_index = new_ligand_index

        self._old_ligand_oemol = self.load_sdf(self._ligand_file, index=self._old_ligand_index)
        self._new_ligand_oemol = self.load_sdf(self._ligand_file, index=self._new_ligand_index)

        self._old_ligand_positions = extractPositionsFromOEMOL(self._old_ligand_oemol)

        ffxml=forcefield_generators.generateForceFieldFromMolecules([self._old_ligand_oemol, self._new_ligand_oemol])

        self._old_ligand_oemol.SetTitle("MOL")
        self._new_ligand_oemol.SetTitle("MOL")

        self._new_ligand_smiles = oechem.OECreateSmiString(self._new_ligand_oemol, oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
        #self._old_ligand_smiles = '[H]c1c(c(c(c(c1N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])nc(n3[H])[H])[H])[H])S(=O)(=O)C([H])([H])[H])[H]'
        self._old_ligand_smiles = oechem.OECreateSmiString(self._old_ligand_oemol, oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)

        print(self._new_ligand_smiles)
        print(self._old_ligand_smiles)

        self._old_ligand_topology = forcefield_generators.generateTopologyFromOEMol(self._old_ligand_oemol)
        self._old_ligand_md_topology = md.Topology.from_openmm(self._old_ligand_topology)
        self._new_ligand_topology = forcefield_generators.generateTopologyFromOEMol(self._new_ligand_oemol)
        self._new_liands_md_topology = md.Topology.from_openmm(self._new_ligand_topology)


        protein_pdbfile = open(self._protein_pdb_filename, 'r')
        pdb_file = app.PDBFile(protein_pdbfile)
        protein_pdbfile.close()

        self._protein_topology_old = pdb_file.topology
        self._protein_md_topology_old = md.Topology.from_openmm(self._protein_topology_old)
        self._protein_positions_old = pdb_file.positions
        self._forcefield = app.ForceField(*forcefield_files)
        self._forcefield.loadFile(StringIO(ffxml))

        print("Generated forcefield")

        self._complex_md_topology_old = self._protein_md_topology_old.join(self._old_ligand_md_topology)
        self._complex_topology_old = self._complex_md_topology_old.to_openmm()

        n_atoms_complex_old = self._complex_topology_old.getNumAtoms()
        n_atoms_protein_old = self._protein_topology_old.getNumAtoms()

        self._complex_positions_old = unit.Quantity(np.zeros([n_atoms_complex_old, 3]), unit=unit.nanometers)
        self._complex_positions_old[:n_atoms_protein_old, :] = self._protein_positions_old
        self._complex_positions_old[n_atoms_protein_old:, :] = self._old_ligand_positions

        if pressure is not None:
            barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
            self._system_generator = SystemGenerator(forcefield_files, barostat=barostat, forcefield_kwargs={'nonbondedMethod' : app.PME})
        else:
            self._system_generator = SystemGenerator(forcefield_files)

        #self._complex_proposal_engine = TwoMoleculeSetProposalEngine(self._old_ligand_smiles, self._new_ligand_smiles, self._system_generator, residue_name="MOL")
        self._complex_proposal_engine = TwoMoleculeSetProposalEngine(self._old_ligand_oemol, self._new_ligand_oemol, self._system_generator, residue_name="MOL")
        self._geometry_engine = FFAllAngleGeometryEngine()

        self._complex_topology_old_solvated, self._complex_positions_old_solvated, self._complex_system_old_solvated = self._solvate_system(self._complex_topology_old, self._complex_positions_old)
        self._complex_md_topology_old_solvated = md.Topology.from_openmm(self._complex_topology_old_solvated)
        print(self._complex_proposal_engine._smiles_list)

        beta = 1.0 / (kB * temperature)

        self._complex_topology_proposal = self._complex_proposal_engine.propose(self._complex_system_old_solvated, self._complex_topology_old_solvated)
        self._complex_positions_new_solvated, _ = self._geometry_engine.propose(self._complex_topology_proposal, self._complex_positions_old_solvated, beta)

        #now generate the equivalent objects for the solvent phase. First, generate the ligand-only topologies and atom map
        self._solvent_topology_proposal, self._old_solvent_positions = self._generate_ligand_only_topologies(self._complex_positions_old_solvated, self._complex_positions_new_solvated)
        self._new_solvent_positions, _ = self._geometry_engine.propose(self._solvent_topology_proposal, self._old_solvent_positions, beta)

    def load_sdf(self, sdf_filename, index=0):
        """
        Load an SDF file into an OEMol. Since SDF files can contain multiple molecules, an index can be provided as well.

        Parameters
        ----------
        sdf_filename : str
            The name of the SDF file
        index : int, default 0
            The index of the molecule in the SDF file

        Returns
        -------
        mol : openeye.oechem.OEMol object
            The loaded oemol object
        """
        ifs = oechem.oemolistream()
        ifs.open(sdf_filename)
        #get the list of molecules
        mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]
        #we'll always take the first for now
        mol_to_return = mol_list[index]
        return mol_to_return

    def _solvate_system(self, topology, positions, model='tip3p'):
        """
        Generate a solvated topology, positions, and system for a given input topology and positions.
        For generating the system, the forcefield files provided in the constructor will be used.

        Parameters
        ----------
        topology : app.Topology
            Topology of the system to solvate
        positions : [n, 3] ndarray of Quantity nm
            the positions of the unsolvated system

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
        hs = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H'] and atom.residue.name != "MOL"]
        modeller.delete(hs)
        modeller.addHydrogens(forcefield=self._forcefield)
        print("preparing to add solvent")
        modeller.addSolvent(self._forcefield, model=model, padding=self._padding)
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()
        print("solvent added, parameterizing")
        solvated_system = self._system_generator.build_system(solvated_topology)
        print("System parameterized")

        return solvated_topology, solvated_positions, solvated_system

    def _generate_ligand_only_topologies(self, old_positions, new_positions):
        """
        This method generates ligand-only topologies and positions from a TopologyProposal containing a solvated complex.
        The output of this method is then used when building the solvent-phase simulation with the same atom map.

        Parameters
        ----------
        topology_proposal : perses.rjmc.TopologyProposal
             TopologyProposal representing the solvated complex transformation

        Returns
        -------
        old_ligand_topology : app.Topology
            The old topology without the receptor or solvent
        new_ligand_topology : app.Topology
            The new topology without the receptor or solvent
        old_ligand_positions : [m, 3] ndarray of Quantity nm
            The positions of the old ligand without receptor or solvent
        new_ligand_positions : [n, 3] ndarray of Quantity nm
            The positions of the new ligand without receptor or solvent
        atom_map : dict of int: it
            The mapping between the two topologies without ligand or solvent.
        """
        old_complex = md.Topology.from_openmm(self._complex_topology_proposal.old_topology)
        new_complex = md.Topology.from_openmm(self._complex_topology_proposal.new_topology)

        complex_atom_map = self._complex_topology_proposal.old_to_new_atom_map

        old_mol_start_index, old_mol_len = self._complex_proposal_engine._find_mol_start_index(old_complex.to_openmm())
        new_mol_start_index, new_mol_len = self._complex_proposal_engine._find_mol_start_index(new_complex.to_openmm())

        old_pos = unit.Quantity(np.zeros([len(old_positions), 3]), unit=unit.nanometers)
        old_pos[:,:] = old_positions
        old_ligand_positions = old_pos[old_mol_start_index:(old_mol_start_index+old_mol_len), :]
        new_ligand_positions = new_positions[new_mol_start_index:(new_mol_start_index+new_mol_len), :]

        #atom_map_adjusted = {}

        #loop through the atoms in the map. If the old index is creater than the old_mol_start_index but less than that
        #plus the old mol length, then it is valid to include its adjusted value in the map.
        #for old_idx, new_idx in complex_atom_map.items():
        #    if old_idx > old_mol_start_index and old_idx < old_mol_len + old_mol_start_index:
        #        atom_map_adjusted[old_idx - old_mol_len] = new_idx - new_mol_start_index

        #subset the topologies:

        old_ligand_topology = old_complex.subset(old_complex.select("resname == 'MOL' "))
        new_ligand_topology = new_complex.subset(new_complex.select("resname == 'MOL' "))

        #solvate the old ligand topology:
        old_solvated_topology, old_solvated_positions, old_solvated_system = self._solvate_system(old_ligand_topology.to_openmm(), old_ligand_positions)

        old_solvated_md_topology = md.Topology.from_openmm(old_solvated_topology)

        #now remove the old ligand, leaving only the solvent
        solvent_only_topology = old_solvated_md_topology.subset(old_solvated_md_topology.select("water"))

        #append the solvent to the new ligand-only topology:
        new_solvated_ligand_md_topology = new_ligand_topology.join(solvent_only_topology)
        nsl,b = new_solvated_ligand_md_topology.to_dataframe()
        #dirty hack because new_solvated_ligand_md_topology.to_openmm() was throwing bond topology error
        new_solvated_ligand_md_topology = md.Topology.from_dataframe(nsl,b)

        new_solvated_ligand_omm_topology = new_solvated_ligand_md_topology.to_openmm()
        new_solvated_ligand_omm_topology.setPeriodicBoxVectors(old_solvated_topology.getPeriodicBoxVectors())

        #create the new ligand system:
        new_solvated_system = self._system_generator.build_system(new_solvated_ligand_omm_topology)

        new_to_old_atom_map = {complex_atom_map[x]-new_mol_start_index:x-old_mol_start_index for x in old_complex.select("resname == 'MOL' ") if x in complex_atom_map.keys()}
        #adjust the atom map to account for the presence of solvent degrees of freedom:
        #By design, all atoms after the ligands are water, and should be mapped.
        n_water_atoms = solvent_only_topology.to_openmm().getNumAtoms()
        for i in range(n_water_atoms):
            new_to_old_atom_map[new_mol_len+i] = old_mol_len + i

        #change the map to accomodate the TP:
        #new_to_old_atom_map = {value : key for key, value in atom_map_adjusted.items()}

        #make a TopologyProposal
        ligand_topology_proposal = TopologyProposal(new_topology=new_solvated_ligand_omm_topology, new_system=new_solvated_system,
                                                    old_topology=old_solvated_topology, old_system=old_solvated_system,
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='A',
                                                    new_chemical_state_key='B')

        return ligand_topology_proposal, old_solvated_positions

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
        return self._old_solvent_positions
    @property
    def solvent_new_positions(self):
        return self._new_solvent_positions

class NonequilibriumSwitchingFEP(object):
    """
    This class manages Nonequilibrium switching based relative free energy calculations, carried out on a distributed computing framework.
    """

    default_forward_functions = {
        'lambda_sterics' : 'lambda',
        'lambda_electrostatics' : 'lambda',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }

    def __init__(self, topology_proposal, pos_old, new_positions, use_dispersion_correction=False,
                 forward_functions=None, n_equil_steps=1000, ncmc_nsteps=100, nsteps_per_iteration=1,
                 temperature=300.0 * unit.kelvin, trajectory_directory=None, trajectory_prefix=None, atom_selection="not water", scheduler_address=None):
        """
        Create an instance of the NonequilibriumSwitchingFEP driver class

        Parameters
        ----------
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
            TopologyProposal object containing transformation of interest
        pos_old : [n, 3] ndarray unit.Quantity
            Positions of the old system.
        new_positions : [m, 3] ndarray unit.Quantity
            Positions of the new system
        use_dispersion_correction : bool, default False
            Whether to use the (expensive) dispersion correction
        forward_functions : dict of str: str, default None
            How each force's scaling parameter relates to the main lambda that is switched by the integrator.
        n_equil_steps : int, default 1000
            Number of equilibrium steps between switching events
        ncmc_nsteps : int, default 100
            Number of steps per NCMC trajectory
        nsteps_per_iteration : int, default one
            Number of steps to take per MCMove; this controls how often configurations are written out.
        temperature : float unit.Quantity
            Temperature at which to perform the simulation, default 300K
        trajectory_directory : str, default None
            Where to write out trajectories resulting from the calculation. If none, no writing is done.
        trajectory_prefix : str, default None
            What prefix to use for this calculation's trajectory files. If none, no writing is done.
        atom_selection : str, default not water
            MDTraj selection syntax for which atomic coordinates to save in the trajectories. Default strips
            all water.
        scheduler_address : str, default None
            The address of the dask scheduler. If None, local will be used.
        """
        if scheduler_address is None:
            self._map = map
            self._gather = lambda mapped_list: list(mapped_list)
        else:
            if scheduler_address=='localhost':
                self._client = distributed.Client()
            else:
                self._client = distributed.Client(scheduler_address)
            self._map = self._client.map
            self._gather = self._client.gather

        #construct the hybrid topology factory object
        self._factory = HybridTopologyFactory(topology_proposal, pos_old, new_positions, use_dispersion_correction=use_dispersion_correction)

        #use default functions if none specified
        if forward_functions == None:
            self._forward_functions = self.default_forward_functions
        else:
            self._forward_functions = forward_functions

        #reverse functions to get a symmetric protocol
        self._reverse_functions = {param : param_formula.replace("lambda", "(1-lambda)") for param, param_formula in self._forward_functions.items()}

        #set up some class attributes
        self._hybrid_system = self._factory.hybrid_system
        self._initial_hybrid_positions = self._factory.hybrid_positions
        self._ncmc_nsteps = ncmc_nsteps
        self._nsteps_per_iteration = nsteps_per_iteration
        self._trajectory_prefix = trajectory_prefix
        self._trajectory_directory = trajectory_directory
        self._zero_endpoint_n_atoms = topology_proposal.n_atoms_old
        self._one_endpoint_n_atoms = topology_proposal.n_atoms_new
        self._atom_selection = atom_selection
        self._current_iteration = 0

        if self._trajectory_directory and self._trajectory_prefix:
            self._write_traj = True
            self._trajectory_filename = {lambda_state: os.path.join(self._trajectory_directory, trajectory_prefix+"lambda%d" % lambda_state + ".h5") for lambda_state in [0,1]}
            self._neq_traj_filename = {lambda_state: os.path.join(self._trajectory_directory, trajectory_prefix + ".{iteration}.neq.lambda%d" % lambda_state + ".h5") for lambda_state in [0,1]}
        else:
            self._write_traj = False
            self._trajectory_filename = {0: None, 1: None}
            self._neq_traj_filename = {0: None, 1: None}

        #initialize lists for results
        self._total_work = {0 : [], 1 : []}
        self._reduced_potential_differences = {0 : [], 1 : []}


        #Set the number of times that the nonequilbrium move will have to be run in order to complete a protocol:
        if self._ncmc_nsteps % self._nsteps_per_iteration != 0:
            logging.warning("The number of ncmc steps is not divisible by the number of steps per iteration. You may not have a full protocol.")
        self._n_iterations_per_call = self._ncmc_nsteps // self._nsteps_per_iteration

        #create the thermodynamic state
        lambda_zero_alchemical_state = alchemy.AlchemicalState.from_system(self._hybrid_system)
        lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

        lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
        lambda_one_alchemical_state.set_alchemical_parameters(1.0)

        #ensure their states are set appropriately
        self._hybrid_alchemical_states = {0 : lambda_zero_alchemical_state, 1: lambda_one_alchemical_state}

        #create the base thermodynamic state with the hybrid system
        self._thermodynamic_state = ThermodynamicState(self._hybrid_system, temperature=temperature)

        #Create thermodynamic states for the nonalchemical endpoints
        self._nonalchemical_thermodynamic_states = {0 : ThermodynamicState(topology_proposal.old_system, temperature=temperature), 1: ThermodynamicState(topology_proposal.new_system, temperature=temperature)}

        #Now create the compound states with different alchemical states
        self._hybrid_thermodynamic_states = {0: CompoundThermodynamicState(self._thermodynamic_state, composable_states=[self._hybrid_alchemical_states[0]]), 1: CompoundThermodynamicState(self._thermodynamic_state, composable_states=[self._hybrid_alchemical_states[1]])}

        #create the forward and reverse integrators
        self._integrators = {0 : AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=self._forward_functions, nsteps_neq=ncmc_nsteps, temperature=temperature),
                             1: AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=self._reverse_functions, nsteps_neq=ncmc_nsteps, temperature=temperature)}

        #create the forward and reverse MCMoves
        self._ne_mc_moves = {n : NonequilibriumSwitchingMove(self._integrators[n], self._nsteps_per_iteration) for n in (0,1)}

        #create the equilibrium MCMove
        self._equilibrium_mc_move = mcmc.LangevinSplittingDynamicsMove(n_steps=n_equil_steps)

        #set the SamplerState for the lambda 0 and 1 equilibrium simulations
        self._lambda_one_sampler_state = SamplerState(self._initial_hybrid_positions, box_vectors=self._hybrid_system.getDefaultPeriodicBoxVectors())
        self._lambda_zero_sampler_state = copy.deepcopy(self._lambda_one_sampler_state)

        self._sampler_states = {0: SamplerState(self._initial_hybrid_positions, box_vectors=self._hybrid_system.getDefaultPeriodicBoxVectors()),
                                1: copy.deepcopy(self._lambda_one_sampler_state)}

        #initialize by minimizing
        self._equilibrium_results = [feptasks.EquilibriumResult(result, 0.0) for result in self.minimize()]

        #subset the topology appropriately:
        if atom_selection is not None:
            atom_selection_indices = self._factory.hybrid_topology.select(atom_selection)
            self._atom_selection_indices = atom_selection_indices
        else:
            self._atom_selection_indices = None

        print("Constructed")

    def minimize(self, max_steps=50):
        """
        Minimize both end states. This method updates the _sampler_state attributes for each lambda

        Parameters
        ----------
        max_steps : int, default 50
            max number of steps for openmm minimizer.
        """
        minimized = self._map(feptasks.minimize, self._hybrid_thermodynamic_states.values(), self._sampler_states.values(), [self._equilibrium_mc_move, self._equilibrium_mc_move])
        _logger.info("Minimizing")
        return self._gather(minimized)

    def run(self, n_iterations=5):
        """
        Run one iteration of the nonequilibrium switching free energy calculations. This entails:

        - 1 iteration of equilibrium at lambda=0 and lambda=1
        - concurrency (parameter) many nonequilibrium trajectories in both forward and reverse
           (e.g., if concurrency is 5, then 5 forward and 5 reverse protocols will be run)
        - 1 iteration of equilibrium at lambda=0 and lambda=1

        Parameters
        ----------
        n_iterations : int, optional, default 5
            The number of times to run the entire sequence described above
        """
        endpoints = [0, 1]
        eq_mc_move_list = [self._equilibrium_mc_move, self._equilibrium_mc_move]
        hybrid_topology_list = [self._factory.hybrid_topology, self._factory.hybrid_topology]
        niterations_per_call_list = [self._n_iterations_per_call, self._n_iterations_per_call]
        atom_indices_to_save_list = [self._atom_selection_indices, self._atom_selection_indices]
        hybrid_factory_list = [self._factory, self._factory]

        endpoint_perturbation_results_list = []
        nonequilibrium_results_list = []
        for i in range(n_iterations):

            if self._write_traj:
                equilibrium_trajectory_filenames = self._trajectory_filename.values()
                noneq_trajectory_filenames = [self._neq_traj_filename[lambda_state].format(iteration=i) for lambda_state in endpoints]
            else:
                equilibrium_trajectory_filenames = [None, None]
                noneq_trajectory_filenames = [None, None]

            #run a round of equilibrium
            self._equilibrium_results = list(self._map(feptasks.run_equilibrium, self._equilibrium_results, self._hybrid_thermodynamic_states.values(), eq_mc_move_list, hybrid_topology_list, niterations_per_call_list, atom_indices_to_save_list, equilibrium_trajectory_filenames))

            #get the perturbations to nonalchemical states:
            endpoint_perturbation_results_list.append(self._map(feptasks.compute_nonalchemical_perturbation, self._equilibrium_results, hybrid_factory_list, self._nonalchemical_thermodynamic_states.values(), endpoints))

            #run a round of nonequilibrium switching:
            nonequilibrium_results_list.append(self._map(feptasks.run_protocol, self._equilibrium_results, self._hybrid_thermodynamic_states.values(), self._ne_mc_moves.values(), hybrid_topology_list, niterations_per_call_list, atom_indices_to_save_list, noneq_trajectory_filenames))

            self._current_iteration +=1
            print(self._current_iteration)

        #after all tasks have been requested, retrieve the results:
        for i in range(n_iterations):
            endpoint_perturbations = self._gather(endpoint_perturbation_results_list[i])
            nonequilibrium_results = self._gather(nonequilibrium_results_list[i])

            for lambda_state in [0,1]:
                self._reduced_potential_differences[lambda_state].append(endpoint_perturbations[lambda_state])

                #for the nonequilibrium results, we have to access the last element of the cumulative work, since that
                #is the total work
                self._total_work[lambda_state].append(nonequilibrium_results[lambda_state].cumulative_work[-1])


    def equilibrate(self, n_iterations=100):
        """
        Run the equilibrium simulations a specified number of times without writing to a file. This can be used to equilibrate
        the simulation before beginning the free energy calculation.

        Parameters
        ----------
        n_iterations : int
            The number of times to apply the equilibrium MCMove
        """
        eq_mc_move_list = [self._equilibrium_mc_move, self._equilibrium_mc_move]
        hybrid_topology_list = [self._factory.hybrid_topology, self._factory.hybrid_topology]
        niterations_per_call_list = [self._n_iterations_per_call, self._n_iterations_per_call]
        atom_indices_to_save_list = [self._atom_selection_indices, self._atom_selection_indices]

        for i in range(n_iterations):

            #don't write out any files
            equilibrium_trajectory_filenames = [None, None]

            #run a round of equilibrium
            self._equilibrium_results = self._map(feptasks.run_equilibrium, self._equilibrium_results, self._hybrid_thermodynamic_states.values(), eq_mc_move_list, hybrid_topology_list, niterations_per_call_list, atom_indices_to_save_list, equilibrium_trajectory_filenames)

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
        Compute the correlation-adjusted free energy at the endpoints to the nonalchemical systems.

        Returns
        -------
        df0, ddf0 : list of float
            endpoint pertubation with error for lambda 0, kT
        df1, ddf1 : list of float
            endpoint perturbation for lambda 1, kT
        """
        free_energies = []
        for lambda_endpoint in [0, 1]:
            work_array = np.array(self._reduced_potential_differences[lambda_endpoint])
            burned_in, statistical_inefficiency, Neff_max = self._adjust_for_correlation(work_array)

            _logger.info("Number of effective samples of endpoint pertubation at lambda %d is %f" % (lambda_endpoint, Neff_max))

            df, ddf_raw = pymbar.EXP(burned_in)

            #correct by multiplying the stddev by the statistical inefficiency
            ddf_corrected = ddf_raw * np.sqrt(statistical_inefficiency)

            free_energies.append([df, ddf_corrected])

        return free_energies[0], free_energies[1]

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
        statistical_inefficiencies = []
        work_arrays = []
        for lambda_endpoint in [0, 1]:
            work_array = np.array(self._total_work[lambda_endpoint])
            work_arrays.append(work_array)

            burned_in, statistical_inefficiency, Neff_max = self._adjust_for_correlation(work_array)

            _logger.info("Number of effective samples of switching at lambda %d is %f" % (lambda_endpoint, Neff_max))

            statistical_inefficiencies.append(statistical_inefficiency)

        #for now we'll take the max of the two to decide how to report the error
        statistical_inefficiency = max(statistical_inefficiencies)

        df, ddf_raw = pymbar.BAR(work_arrays[0], work_arrays[1])

        ddf_corrected = ddf_raw*np.sqrt(statistical_inefficiency)

        return df, ddf_corrected

    @property
    def current_free_energy_estimate(self):
        """
        Estimate the free energy based on currently available values
        """
        #Make sure the task queue is empty (all pending calcs are complete) before computing free energy
        #Make sure the task queue is empty (all pending calcs are complete) before computing free energy
        [[df0, ddf0], [df1, ddf1]] = self._endpoint_perturbations()
        [df, ddf] = self._alchemical_free_energy()

        ddf_overall = np.sqrt(ddf0**2 + ddf1**2 + ddf**2)
        return -df0 + df + df1, ddf_overall

def run_setup(setup_options):
    """
    Run the setup pipeline and return the relevant setup objects based on a yaml input file.

    Parameters
    ----------
    setup_options : dict
        result of loading yaml input file

    Returns
    -------
    fe_setup : NonequilibriumFEPSetup
        The setup class for this calculation
    ne_fep : NonequilibriumSwitchingFEP
        The nonequilibrium driver class
    """
    #We'll need the protein PDB file (without missing atoms)
    protein_pdb_filename = setup_options['protein_pdb']

    #And a ligand file containing the pair of ligands between which we will transform
    ligand_file = setup_options['ligand_file']

    #get the indices of ligands out of the file:
    old_ligand_index = setup_options['old_ligand_index']
    new_ligand_index = setup_options['new_ligand_index']

    forcefield_files = setup_options['forcefield_files']

    #get the simulation parameters
    pressure = setup_options['pressure'] * unit.atmosphere
    temperature = setup_options['temperature'] * unit.kelvin
    solvent_padding_angstroms = setup_options['solvent_padding'] * unit.angstrom

    setup_pickle_file = setup_options['save_setup_pickle_as']

    fe_setup = NonequilibriumFEPSetup(protein_pdb_filename, ligand_file, old_ligand_index, new_ligand_index, forcefield_files, pressure=pressure, temperature=temperature, solvent_padding=solvent_padding_angstroms)

    pickle_outfile = open(setup_pickle_file, 'wb')

    try:
        pickle.dump(fe_setup, pickle_outfile)
    except Exception as e:
        print(e)
        print("Unable to save setup object as a pickle")
    finally:
        pickle_outfile.close()

    print("Setup object has been created.")

    phase = setup_options['phase']

    if phase == "complex":
        topology_proposal = fe_setup.complex_topology_proposal
        old_positions = fe_setup.complex_old_positions
        new_positions = fe_setup.complex_old_positions
    elif phase == "solvent":
        topology_proposal = fe_setup.solvent_topology_proposal
        old_positions = fe_setup.solvent_old_positions
        new_positions = fe_setup.solvent_new_positions
    else:
        raise ValueError("Phase must be either complex or solvent.")

    forward_functions = setup_options['forward_functions']

    n_equilibrium_steps_per_iteration = setup_options['n_equilibrium_steps_per_iteration']
    n_steps_ncmc_protocol = setup_options['n_steps_ncmc_protocol']
    n_steps_per_move_application = setup_options['n_steps_per_move_application']

    trajectory_directory = setup_options['trajectory_directory']
    trajectory_prefix = setup_options['trajectory_prefix']
    atom_selection = setup_options['atom_selection']

    scheduler_address = setup_options['scheduler_address']

    ne_fep = NonequilibriumSwitchingFEP(topology_proposal, old_positions, new_positions,
                                                       forward_functions=forward_functions,
                                                       n_equil_steps=n_equilibrium_steps_per_iteration,
                                                       ncmc_nsteps=n_steps_ncmc_protocol,
                                                       nsteps_per_iteration=n_steps_per_move_application,
                                                       temperature=temperature,
                                                       trajectory_directory=trajectory_directory,
                                                       trajectory_prefix=trajectory_prefix,
                                                       atom_selection=atom_selection,
                                                       scheduler_address=scheduler_address)

    print("Nonequilibrium switching driver class constructed")

    return fe_setup, ne_fep
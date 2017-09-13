from perses.distributed import feptasks
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator, LangevinIntegrator
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState
import openmmtools.mcmc as mcmc
import openmmtools.cache as cache
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
from perses.distributed.feptasks import NonequilibriumSwitchingMove

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
        if self._pressure is not None:
            barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
            solvated_system.addForce(barostat)

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
                 forward_functions=None, ncmc_nsteps=100, nsteps_per_iteration=1, concurrency=4, platform_name="OpenCL",
                 temperature=300.0 * unit.kelvin, trajectory_directory=None, trajectory_prefix=None):

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
        self._concurrency = concurrency
        self._ncmc_nsteps = ncmc_nsteps
        self._nsteps_per_iteration = nsteps_per_iteration
        self._trajectory_prefix = trajectory_prefix
        self._trajectory_directory = trajectory_directory

        #initialize lists for results
        self._forward_nonequilibrium_trajectories = []
        self._reverse_nonequilibrium_trajectories = []
        self._forward_nonequilibrium_cumulative_works = []
        self._reverse_nonequilibrium_cumulative_works = []
        self._forward_nonequilibrium_results = []
        self._reverse_nonequilibrium_results = []
        self._forward_total_work = []
        self._reverse_total_work = []

        #Set the number of times that the nonequilbrium move will have to be run in order to complete a protocol:
        if self._ncmc_nsteps % self._nsteps_per_iteration != 0:
            logging.warning("The number of ncmc steps is not divisible by the number of steps per iteration. You may not have a full protocol.")
        self._n_iterations_per_call = self._ncmc_nsteps // self._nsteps_per_iteration

        #create the thermodynamic state
        lambda_zero_alchemical_state = alchemy.AlchemicalState.from_system(self._hybrid_system)
        lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

        #ensure their states are set appropriately
        lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
        lambda_one_alchemical_state.set_alchemical_parameters(0.0)

        #create the base thermodynamic state with the hybrid system
        self._thermodynamic_state = ThermodynamicState(self._hybrid_system, temperature=temperature)

        #Now create the compound states with different alchemical states
        self._lambda_zero_thermodynamic_state = CompoundThermodynamicState(self._thermodynamic_state, composable_states=[lambda_zero_alchemical_state])
        self._lambda_one_thermodynamic_state = CompoundThermodynamicState(self._thermodynamic_state, composable_states=[lambda_one_alchemical_state])

        #create the forward and reverse integrators
        self._forward_integrator = AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=self._forward_functions, nsteps_neq=ncmc_nsteps, temperature=temperature)
        self._reverse_integrator = AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=self._reverse_functions, nsteps_neq=ncmc_nsteps, temperature=temperature)

        #create the forward and reverse MCMoves
        self._forward_ne_mc_move = NonequilibriumSwitchingMove(self._forward_integrator, self._nsteps_per_iteration)
        self._reverse_ne_mc_move = NonequilibriumSwitchingMove(self._reverse_integrator, self._nsteps_per_iteration)

        #create the equilibrium MCMove
        self._equilibrium_mc_move = mcmc.LangevinSplittingDynamicsMove()

        #set the SamplerState for the lambda 0 and 1 equilibrium simulations
        self._lambda_one_sampler_state = SamplerState(self._initial_hybrid_positions, box_vectors=self._hybrid_system.getDefaultPeriodicBoxVectors())
        self._lambda_zero_sampler_state = copy.deepcopy(self._lambda_one_sampler_state)

        #initialize by minimizing
        self.minimize()

        #initialize the trajectories for the lambda 0 and 1 equilibrium simulations

        a_0, b_0, c_0, alpha_0, beta_0, gamma_0 = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*self._lambda_zero_sampler_state.box_vectors)
        a_1, b_1, c_1, alpha_1, beta_1, gamma_1 = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*self._lambda_one_sampler_state.box_vectors)

        self._lambda_zero_traj = md.Trajectory(np.array(self._lambda_zero_sampler_state.positions), self._factory.hybrid_topology, unitcell_lengths=[a_0, b_0, c_0], unitcell_angles=[alpha_0, beta_0, gamma_0])
        self._lambda_one_traj = md.Trajectory(np.array(self._lambda_one_sampler_state.positions), self._factory.hybrid_topology, unitcell_lengths=[a_1, b_1, c_1], unitcell_angles=[alpha_1, beta_1, gamma_1])

    def minimize(self, max_steps=50):
        """
        Minimize both end states. This method updates the _sampler_state attributes for each lambda

        Parameters
        ----------
        max_steps : int, default 50
            max number of steps for openmm minimizer.
        """
        #Asynchronously invoke the tasks
        minimized_lambda_zero_result = feptasks.minimize.delay(self._lambda_zero_thermodynamic_state, self._lambda_zero_sampler_state, self._equilibrium_mc_move, max_iterations=max_steps)
        minimized_lambda_one_result = feptasks.minimize.delay(self._lambda_one_thermodynamic_state, self._lambda_one_sampler_state, self._equilibrium_mc_move, max_iterations=max_steps)

        #now synchronously retrieve the results and save the sampler states.
        self._lambda_zero_sampler_state = minimized_lambda_zero_result.get()
        self._lambda_one_sampler_state = minimized_lambda_one_result.get()

    def run(self, n_iterations=5, concurrency=1):
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
        concurrency: int, default 1
            The number of concurrent nonequilibrium protocols to run; note that with greater than one,
            error estimation may be more complicated.
        """
        for i in range(n_iterations):
            self._run_equilibrium()
            self._run_nonequilibrium(concurrency=concurrency, n_iterations=self._n_iterations_per_call)
            self._run_equilibrium()

        if self._trajectory_directory:
            self._write_equilibrium_trajectories(self._trajectory_directory, self._trajectory_prefix)

    def _run_equilibrium(self, n_iterations=1):
        """
        Run one iteration of equilibrium at lambda=1 and lambda=0, and replace the current equilibrium sampler states
        with the results of the equilibrium calculation, as well as extend the current equilibrium trajectories.

        Parameters
        ----------
        n_iterations : int, default 1
            How many times to run the n_steps of equilibrium
        """
        #run equilibrium for lambda=0 and lambda=1
        lambda_zero_result = feptasks.run_equilibrium.delay(self._lambda_zero_thermodynamic_state, self._lambda_zero_sampler_state, self._equilibrium_mc_move, self._factory.hybrid_topology, n_iterations)
        lambda_one_result = feptasks.run_equilibrium.delay(self._lambda_one_thermodynamic_state, self._lambda_one_sampler_state, self._equilibrium_mc_move, self._factory.hybrid_topology, n_iterations)

        #retrieve the results of the calculation
        self._lambda_zero_sampler_state, traj_zero_result = lambda_zero_result.get()
        self._lambda_one_sampler_state, traj_one_result = lambda_one_result.get()

        #join the trajectories to the reference trajectories, if the object exists,
        #otherwise, simply create it
        if self._lambda_zero_traj:
            self._lambda_zero_traj = self._lambda_zero_traj.join(traj_zero_result, check_topology=False)
        else:
            self._lambda_zero_traj = traj_zero_result

        if self._lambda_one_traj:
            self._lambda_one_traj = self._lambda_one_traj.join(traj_one_result, check_topology=False)
        else:
            self._lambda_one_traj = traj_one_result

    def _run_nonequilibrium(self, concurrency=1, n_iterations=1):
        """
        Run concurrency-many nonequilibrium protocols in both directions. This method stores the result object, but does
        not retrieve the results. Note that n_iterations is important, since in order to perform an entire switching trajectory
        (from 0 to 1 or vice versa), we require that n_steps*n_iterations = protocol length

        Parameters
        ----------
        concurrency : int, default 1
            The number of protocols to run in each direction simultaneously
        n_iterations : int, default 1
            The number of times to have the NE move applied. Note that as above if n_steps*n_iterations!=ncmc_nsteps,
            the protocol will not be run properly.
        """

        #set up the group object that will be used to compute the nonequilibrium results.
        forward_protocol_group = celery.group(
            feptasks.run_protocol.s(self._lambda_zero_thermodynamic_state, self._lambda_zero_sampler_state,
                                    self._forward_ne_mc_move, self._factory.hybrid_topology, n_iterations) for i in
            range(concurrency))
        reverse_protocol_group = celery.group(
            feptasks.run_protocol.s(self._lambda_one_thermodynamic_state, self._lambda_one_sampler_state,
                                    self._reverse_ne_mc_move, self._factory.hybrid_topology, n_iterations) for i in
            range(concurrency))

        #get the result objects:
        self._forward_nonequilibrium_results.append(forward_protocol_group.apply_async())
        self._reverse_nonequilibrium_results.append(reverse_protocol_group.apply_async())

    def retrieve_nonequilibrium_results(self):
        """
        Retrieve any pending results that were generated by computations from the run() call. Note that this will block
        until all have completed. This method will update the list of trajectories as well as the nonequilibrium work values.
        """
        for result in self._forward_nonequilibrium_results:
            result_group = result.join()
            for result in result_group:
                traj, cum_work = result

                #we can take the final element as the total work
                self._forward_total_work.append(cum_work[-1])

                #we'll append the cumulative work and the trajectory to the appropriate lists
                self._forward_nonequilibrium_cumulative_works.append(cum_work)
                self._forward_nonequilibrium_trajectories.append(traj)

        for result in self._reverse_nonequilibrium_results:
            result_group = result.join()
            for result in result_group:
                traj, cum_work = result

                #we can take the final element as the total work
                self._reverse_total_work.append(cum_work[-1])

                #we'll append the cumulative work and the trajectory to the appropriate lists
                self._reverse_nonequilibrium_cumulative_works.append(cum_work)
                self._reverse_nonequilibrium_trajectories.append(traj)

    def write_nonequilibrium_trajectories(self, directory, file_prefix):
        """
        Write out an MDTraj h5 file for each nonequilibrium trajectory. The files will be placed in
        [directory]/file_prefix-[forward, reverse]-index.h5. This method will ensure that all pending
        results are collected.

        Parameters
        ----------
        directory : str
            The directory in which to place the files
        file_prefix : str
            A prefix for the filenames
        """
        self.retrieve_nonequilibrium_results()

        #loop through the forward trajectories
        for index, forward_trajectory in enumerate(self._forward_nonequilibrium_trajectories):

            #construct the name for this file
            full_filename = os.path.join(directory, file_prefix + "forward" + str(index) + ".h5")

            #save the trajectory
            forward_trajectory.save_hdf5(full_filename)

        #repeat for the reverse trajectories:
        for index, reverse_trajectory in enumerate(self._reverse_nonequilibrium_trajectories):

            #construct the name for this file
            full_filename = os.path.join(directory, file_prefix + "reverse" + str(index) + ".h5")

            #save the trajectory
            reverse_trajectory.save_hdf5(full_filename)

    def _write_equilibrium_trajectories(self, directory, file_prefix):
        """
        Write out an MDTraj h5 file for each nonequilibrium trajectory. The files will be placed in
        [directory]/file_prefix-[lambda0, lambda1].h5.

        Parameters
        ----------
        directory : str
            The directory in which to place the files
        file_prefix : str
            A prefix for the filenames
        """
        lambda_zero_filename = os.path.join(directory, file_prefix + "-" + "lambda0" + ".h5")
        lambda_one_filename = os.path.join(directory, file_prefix + "-" + "lambda1" + ".h5")

        filenames = [lambda_zero_filename, lambda_one_filename]
        trajs = [self._lambda_zero_traj, self._lambda_one_traj]

        #open the existing file if it exists, and append. Otherwise create it
        for filename, traj in zip(filenames, trajs):
            if not os.path.exists(filename):
                traj.save_hdf5(filename)
            else:
                written_traj = md.load(filename)
                concatenated_traj = written_traj.join(traj)
                concatenated_traj.save_hdf5(filename)

        #delete the trajectories.
        self._lambda_one_traj = None
        self._lambda_zero_traj = None


    @property
    def lambda_zero_equilibrium_trajectory(self):
        return self._lambda_zero_traj

    @property
    def lambda_one_equilibrium_trajectory(self):
        return self._lambda_one_traj

    @property
    def forward_nonequilibrium_trajectories(self):
        return self._forward_nonequilibrium_trajectories

    @property
    def reverse_nonequilibrium_trajectories(self):
        return self._reverse_nonequilibrium_trajectories

    @property
    def forward_cumulative_works(self):
        return self._forward_nonequilibrium_cumulative_works

    @property
    def reverse_cumulative_works(self):
        return self._reverse_nonequilibrium_cumulative_works

    @property
    def current_free_energy_estimate(self):
        [df, ddf] = pymbar.BAR(self._forward_total_work, self._reverse_total_work)
        return [df, ddf]

if __name__=="__main__":
    #import os
    #gaff_filename = get_data_filename("data/gaff.xml")
    #forcefield_files = [gaff_filename, 'tip3p.xml', 'amber99sbildn.xml']
    #path_to_schrodinger_inputs = "/Users/grinawap/Downloads"
    #protein_file = os.path.join(path_to_schrodinger_inputs, "/Users/grinawap/Downloads/CDK2_fixed_nohet.pdb")
    #molecule_file = os.path.join(path_to_schrodinger_inputs, "/Users/grinawap/Downloads/Inputs_for_FEP/CDK2_ligands.mol2")
    #fesetup = NonequilibriumFEPSetup(protein_file, molecule_file, 0, 2, forcefield_files)
    import pickle
    infile = open("fesetup.pkl", 'rb')
    fesetup = pickle.load(infile)
    infile.close()
    #pickle.dump(fesetup, outfile)
    #outfile.close()
    ne_fep = NonequilibriumSwitchingFEP(fesetup.solvent_topology_proposal, fesetup.solvent_old_positions, fesetup.solvent_new_positions)
    print("ne-fep initialized")
    ne_fep.run(n_iterations=2)
    ne_fep.retrieve_nonequilibrium_results()
    print("retrieved")

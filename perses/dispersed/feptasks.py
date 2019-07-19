import simtk.openmm as openmm
import openmmtools.cache as cache
from typing import List, Tuple, Union, NamedTuple
import os
import copy

#Add the variables specific to the Alchemical langevin integrator
#only do this if we're not using the DummyContextCache
# if type(cache.global_context_cache) == cache.ContextCache:
#     cache.global_context_cache.COMPATIBLE_INTEGRATOR_ATTRIBUTES.update({
#          "protocol_work" : 0.0,
#          "Eold" : 0.0,
#          "Enew" : 0.0,
#          "lambda" : 0.0,
#          "nsteps" : 0.0,
#          "step" : 0.0,
#          "n_lambda_steps" : 0.0,
#          "lambda_step" : 0.0
#      })


import openmmtools.mcmc as mcmc
import openmmtools.integrators as integrators
import openmmtools.states as states
import numpy as np
import mdtraj as md
from perses.annihilation.relative import HybridTopologyFactory
import mdtraj.utils as mdtrajutils
import pickle
import simtk.unit as unit
import tqdm
from perses.tests.utils import compute_potential_components
from openmmtools.constants import kB
import pdb
import logging
logging.basicConfig(level = logging.DEBUG)
import tqdm

# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("feptasks")
_logger.setLevel(logging.DEBUG)


cache.global_context_cache.platform = openmm.Platform.getPlatformByName('Reference') #this is just a local version
#Make containers for results from tasklets. This allows us to chain tasks together easily.
EquilibriumResult = NamedTuple('EquilibriumResult', [('sampler_state', states.SamplerState), ('reduced_potential', float)])
NonequilibriumResult = NamedTuple('NonequilibriumResult', [('cumulative_work', np.array), ('protocol_work', np.array), ('shadow_work', np.array)])

class NonequilibriumSwitchingMove(mcmc.BaseIntegratorMove):
    """
    This class represents an MCMove that runs a nonequilibrium switching protocol using the AlchemicalNonequilibriumLangevinIntegrator.
    It is simply a wrapper around the aforementioned integrator, which must be provided in the constructor.

    Parameters
    ----------
    alchemical_functions : dict
        Leptop parse-able strings for each parameter in the system...this is specified separately for forward and reverse protocols.
    splitting : str
        Splitting string for integrator
    temperature : unit.Quantity(float, units = unit.kelvin)
        temperature at which to run the simulation
    nsteps_neq : int
        number of steps in the integrator
    timestep : float
        size of timestep (units of time)
    work_save_interval : int, default None
        The frequency with which to record the cumulative total work. If None, only save the total work at the end
    top: md.Topology, default None
        The topology to use to write the positions along the protocol. If None, don't write anything.
    subset_atoms : np.array, default None
        The indices of the subset of atoms to write. If None, write all atoms (if writing is enabled)
    save_configuration : bool, default False
        whether to save the ncmc trajectory
    measure_shadow_work : bool, default False
        whether to measure the shadow work from the integrator
    Attributes
    ----------
    current_total_work : float
        The total work in kT.
    """

    def __init__(self, alchemical_functions: dict, splitting: str, temperature: unit.Quantity, nsteps_neq: int, timestep: unit.Quantity,
        work_save_interval: int=None, top: md.Topology=None, subset_atoms: np.array=None, save_configuration: bool=False, measure_shadow_work: bool=False, **kwargs):

        super(NonequilibriumSwitchingMove, self).__init__(n_steps=nsteps_neq, **kwargs)
        self.context_cache = cache.global_context_cache
        if measure_shadow_work:
            measure_heat = True
        else:
            measure_heat = False

        self._integrator = integrators.AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=alchemical_functions,
                                                                                  nsteps_neq=nsteps_neq,
                                                                                  timestep = timestep,
                                                                                  temperature=temperature,
                                                                                  measure_shadow_work = measure_shadow_work,
                                                                                  splitting = splitting,
                                                                                  measure_heat = measure_heat)
        self._ncmc_nsteps = nsteps_neq
        self._beta = 1.0 / (kB*temperature)
        self._work_save_interval = work_save_interval

        self._save_configuration = save_configuration
        self._measure_shadow_work = measure_shadow_work
        #check that the work write interval is a factor of the number of steps, so we don't accidentally record the
        #work before the end of the protocol as the end
        if self._ncmc_nsteps % self._work_save_interval != 0:
            raise ValueError("The work writing interval must be a factor of the total number of steps")

        self._number_of_step_moves = self._ncmc_nsteps // self._work_save_interval

        #use the number of step moves plus one, since the first is always zero
        self._cumulative_work = np.zeros(self._number_of_step_moves+1)
        self._shadow_work = 0.0
        self._protocol_work = np.zeros(self._number_of_step_moves+1)
        self._heat = 0.0

        self._topology = top
        self._subset_atoms = subset_atoms
        self._trajectory = None

        #if we have a trajectory, set up some ancillary variables:
        if self._topology is not None:
            n_atoms = self._topology.n_atoms
            n_iterations = self._number_of_step_moves
            self._trajectory_positions = np.zeros([n_iterations, n_atoms, 3])
            self._trajectory_box_lengths = np.zeros([n_iterations, 3])
            self._trajectory_box_angles = np.zeros([n_iterations, 3])
        else:
            self._save_configuration = False

        self._current_total_work = 0.0

    def _get_integrator(self):
        """
        Get the integrator associated with this move. In this case, it is simply the integrator passed in to the constructor.

        Returns
        -------
        integrator : openmmtools.integrators.AlchemicalNonequilibriumLangevinIntegrator
            The integrator that is associated with this MCMove
        """
        return self._integrator

    def apply(self, thermodynamic_state, sampler_state):
        """Propagate the state through the integrator.
        This updates the SamplerState after the integration. It will apply the full NCMC protocol.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        """
        """Propagate the state through the integrator.
        This updates the SamplerState after the integration. It also logs
        benchmarking information through the utils.Timer class.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        Returns
        -------
        sampler_state : openmmtools.states.SamplerState
            The updated sampler state from the context after switching is conducted

        See Also
        --------
        openmmtools.utils.Timer
        """
        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        context, integrator = context_cache.get_context(thermodynamic_state, self._integrator)

        sampler_state.apply_to_context(context, ignore_velocities=False)

        # reset the integrator after it is bound to a context and before annealing
        integrator.reset()

        self._cumulative_work[0] = integrator.get_protocol_work(dimensionless=True)

        if self._cumulative_work[0] != 0.0:
            raise RuntimeError("The initial cumulative work after reset was not zero.")

        if self._measure_shadow_work:
            initial_energy = self._beta * (sampler_state.potential_energy + sampler_state.kinetic_energy)

        #loop through the number of times we have to apply in order to collect the requested work and trajectory statistics.

        for iteration in range(self._number_of_step_moves):
            try:
                integrator.step(self._work_save_interval)
            except Exception as e:
                self._trajectory = md.Trajectory(self._trajectory_positions, self._topology, unitcell_lengths=self._trajectory_box_lengths, unitcell_angles=self._trajectory_box_angles)
                raise e
            self._current_protocol_work = integrator.get_protocol_work(dimensionless=True)
            self._cumulative_work[iteration+1] = self._current_protocol_work
            self._protocol_work[iteration+1] = self._cumulative_work[iteration+1] - self._cumulative_work[iteration]

            #if we have a trajectory, we'll also write to it
            if self._save_configuration:
                sampler_state.update_from_context(context)

                #record positions for writing to trajectory
                #we need to check whether the user has requested to subset atoms (excluding water, for instance)

                if self._subset_atoms is None:
                    self._trajectory_positions[iteration, :, :] = sampler_state.positions[:, :].value_in_unit_system(unit.md_unit_system)
                else:
                    self._trajectory_positions[iteration, :, :] = sampler_state.positions[self._subset_atoms, :].value_in_unit_system(unit.md_unit_system)

                #get the box angles and lengths
                a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
                self._trajectory_box_lengths[iteration, :] = [a, b, c]
                self._trajectory_box_angles[iteration, :] = [alpha, beta, gamma]

        sampler_state.update_from_context(context)

        if self._save_configuration:
            self._trajectory = md.Trajectory(self._trajectory_positions, self._topology, unitcell_lengths=self._trajectory_box_lengths, unitcell_angles=self._trajectory_box_angles)

        self._total_work = self._cumulative_work[-1]

        if self._measure_shadow_work:
            total_heat = integrator.get_heat(dimensionless=True)
            final_energy = self._beta * (sampler_state.potential_energy + sampler_state.kinetic_energy)
            total_energy_change = final_energy - initial_energy
            self._shadow_work = total_energy_change - (total_heat + self._cumulative_work[-1])
            self._total_work += self._shadow_work
        else:
            self._shadow_work = 0.0


    def reset(self):
        """
        Reset the work statistics on the associated ContextCache integrator.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            the thermodynamic state for which this integrator is cached.
        """
        self._integrator.reset()
        self._current_protocol_work = 0.0

    @property
    def current_total_work(self):
        """
        Get the current total work in kT

        Returns
        -------
        current_total_work : float
            the current total work performed by this move since the last reset()
        """
        return self._current_total_work

    @property
    def trajectory(self):
        if self._save_configuration is None:
            raise NoTrajectoryException("Tried to access a trajectory without providing a topology.")
        elif self._trajectory is None:
            raise NoTrajectoryException("Tried to access a trajectory on a move that hasn't been used yet.")
        else:
            return self._trajectory

    @property
    def cumulative_work(self):
        return self._cumulative_work

    @property
    def shadow_work(self):
        if not self._measure_shadow_work:
            raise ValueError("Can't return shadow work if it isn't being measured")
        return self._shadow_work

    @property
    def protocol_work(self):
        return self._protocol_work

    def __getstate__(self):
        dictionary = super(NonequilibriumSwitchingMove, self).__getstate__()
        dictionary['integrator'] = pickle.dumps(self._integrator)
        dictionary['current_total_work'] = self.current_total_work
        dictionary['measure_shadow_work'] = self._integrator._measure_shadow_work
        dictionary['measure_heat'] = self._integrator._measure_heat
        dictionary['metropolized_integrator'] = self._integrator._metropolized_integrator
        return dictionary

    def __setstate__(self, serialization):
        super(NonequilibriumSwitchingMove, self).__setstate__(serialization)
        self._current_total_work = serialization['current_total_work']
        self._integrator = pickle.loads(serialization['integrator'])
        integrators.RestorableIntegrator.restore_interface(self._integrator)
        self._integrator._measure_shadow_work = serialization['measure_shadow_work']
        self._integrator._measure_heat = serialization['measure_heat']
        self._integrator._metropolized_integrator = serialization['metropolized_integrator']


def run_protocol(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState,
                 alchemical_functions: dict, topology: md.Topology, nstep_neq: int = 1000, work_save_interval: int = 1, splitting: str="V R O H R V",
                 atom_indices_to_save: List[int] = None, trajectory_filename: str = None, write_configuration: bool = False, timestep: unit.Quantity=1.0*unit.femtoseconds, measure_shadow_work: bool=False) -> tuple:
    """
    Perform a nonequilibrium switching protocol and return the nonequilibrium protocol work. Note that it is expected
    that this will perform an entire protocol, that is, switching lambda completely from 0 to 1, in increments specified
    by the ne_mc_move. The trajectory that results, along with the work values, will contain n_iterations elements.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state at which to run the protocol
    sampler_state : openmmtools.states.SamplerState
        The sampler state from which to run the protocol; this should be an equilibrium sample
    alchemical_functions : dict
        The alchemical functions to use for switching
    topology : mdtraj.Topology
        An MDtraj topology for the system to generate trajectories
    nstep_neq : int
        The number of nonequilibrium steps in the protocol
    work_save_interval : int
        How often to write the work and, if requested, configurations
    splitting : str, default "V R O H R V"
        The splitting string to use for the Langevin integration
    atom_indices_to_save : list of int, default None
        list of indices to save (when excluding waters, for instance). If None, all indices are saved.
    trajectory_filename : str, default None
        Full filepath of output trajectory, if desired. If None, no trajectory file is written.
    write_configuration : bool, default False
        Whether to also write configurations of the trajectory at the requested interval.
    timestep : unit.Quantity, default 1 fs
        The timestep to use in the integrator
    measure_shadow_work : bool, default False
        Whether to compute the shadow work; there is additional overhead in the integrator cost
    Returns
    -------
    cumulative_work : np.array
        the cumulative work as a function of the integration step; len(cumulative_work) = work_save_interval
    protocol_work : np.array
        the work per iteration of annealing; len(protocol_work) = work_save_interval
    shadow_work : float
        the shadow work accumulated by the discrete integrator
    """
    #get the temperature needed for the simulation
    temperature = thermodynamic_state.temperature

    #get the atom indices we need to subset the topology and positions
    if atom_indices_to_save is None:
        atom_indices = list(range(topology.n_atoms))
        subset_topology = topology
    else:
        subset_topology = topology.subset(atom_indices_to_save)
        atom_indices = atom_indices_to_save

    _logger.debug(f"Instantiating NonequilibriumSwitchingMove class")
    ne_mc_move = NonequilibriumSwitchingMove(alchemical_functions = alchemical_functions,
                                             splitting = splitting,
                                             temperature = temperature,
                                             nsteps_neq = nstep_neq,
                                             timestep = timestep,
                                             work_save_interval = work_save_interval,
                                             top = subset_topology,
                                             subset_atoms = atom_indices,
                                             save_configuration = write_configuration,
                                             measure_shadow_work=measure_shadow_work)

    #apply the nonequilibrium move; sampler state gets updated
    _logger.debug(f"applying thermodynamic state and sampler state to the ne_mc_move")
    ne_mc_move.apply(thermodynamic_state, sampler_state)

    #get the cumulative work
    cumulative_work = ne_mc_move._cumulative_work

    #get the protocol work
    protocol_work = ne_mc_move._protocol_work

    #if we're measuring shadow work, get that. Otherwise just fill in zeros:
    if measure_shadow_work:
        shadow_work = ne_mc_move._shadow_work
    else:
        shadow_work = np.zeros_like(protocol_work)


    #if desired, write nonequilibrium trajectories:
    if trajectory_filename is not None:
        #if writing configurations was requested, get the trajectory
        if write_configuration:
            try:
                trajectory = ne_mc_move.trajectory
                write_nonequilibrium_trajectory(nonequilibrium_result, trajectory, trajectory_filename)
                _logger.debug(f"successfully wrote nonequilibrium trajectory to {trajectory_filename}")
            except NoTrajectoryException:
                print(f"there is no trajectory filename to which to write")
                pass

    return (cumulative_work, protocol_work, shadow_work)

def run_equilibrium(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState,
                    nsteps_equil: int, topology: md.Topology, n_iterations : int,
                    atom_indices_to_save: List[int] = None, trajectory_filename: str = None, splitting: str="V R O R V", timestep: unit.Quantity=1.0*unit.femtoseconds) -> EquilibriumResult:
    """
    Run n_iterations*nsteps_equil integration steps (likely at the lambda 0 state).  n_iterations mcmc moves are conducted in the initial equilibration, returning n_iterations
    reduced potentials.  This is the guess as to the burn-in time for a production.  After which, a single mcmc move of nsteps_equil
    will be conducted at a time, including a time-series (pymbar) analysis to determine whether the data are decorrelated.
    The loop will conclude when a single configuration yields an iid sample.  This will be saved.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state (including context parameters) that should be used
    sampler_state : openmmtools.states.SamplerState
        The sampler state (which wraps box vectors and positions) to be equilibrated
    nsteps_equil : int
        The number of equilibrium steps that a move should make when apply is called
    topology : mdtraj.Topology
        an MDTraj topology object used to construct the trajectory
    n_iterations : int
        The minimum number of times to apply the move. Note that this is not the number of steps of dynamics; it is
        n_iterations*n_steps (which is set in the MCMove).
    splitting: str, default "V R O H R V"
        The splitting string for the dynamics
    atom_indices_to_save : list of int, default None
        list of indices to save (when excluding waters, for instance). If None, all indices are saved.
    trajectory_filename : str, optional, default None
        Full filepath of trajectory files. If none, trajectory files are not written.
    splitting: str, default "V R O H R V"
        The splitting string for the dynamics
    """
    _logger.debug(f"running equilibrium")
    #get the atom indices we need to subset the topology and positions
    if atom_indices_to_save is None:
        atom_indices = list(range(topology.n_atoms))
        subset_topology = topology
    else:
        subset_topology = topology.subset(atom_indices_to_save)
        atom_indices = atom_indices_to_save

    n_atoms = subset_topology.n_atoms

    #construct the MCMove:
    mc_move = mcmc.LangevinSplittingDynamicsMove(n_steps=nsteps_equil, splitting=splitting)
    mc_move.n_restart_attempts = 10

    #create a numpy array for the trajectory
    #reduced_potentials = []
    trajectory_positions = np.zeros([n_iterations, n_atoms, 3])
    trajectory_box_lengths = np.zeros([n_iterations, 3])
    trajectory_box_angles = np.zeros([n_iterations, 3])

    #loop through iterations and apply MCMove, then collect positions into numpy array
    _logger.debug(f"conducting {n_iterations} of production")
    for iteration in tqdm.trange(n_iterations):
        _logger.debug(f"\tconducting iteration {iteration}")
        mc_move.apply(thermodynamic_state, sampler_state)
        #reduced_potential = compute_reduced_potential(thermodynamic_state, sampler_state)
        #reduced_potentials.append(reduced_potential)

        trajectory_positions[iteration, :,:] = sampler_state.positions[atom_indices, :].value_in_unit_system(unit.md_unit_system)

        #get the box lengths and angles
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
        trajectory_box_lengths[iteration, :] = [a, b, c]
        trajectory_box_angles[iteration, :] = [alpha, beta, gamma]
    _logger.debug(f"production done")


    #construct trajectory object:
    trajectory = md.Trajectory(trajectory_positions, subset_topology, unitcell_lengths=trajectory_box_lengths, unitcell_angles=trajectory_box_angles)

    #get the reduced potential from the final frame for endpoint perturbations
    reduced_potential_final_frame = thermodynamic_state.reduced_potential(sampler_state)

    #If there is a trajectory filename passed, write out the results here:
    if trajectory_filename is not None:
        write_equilibrium_trajectory(trajectory, trajectory_filename)



def minimize(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState,
             max_iterations: int=100) -> states.SamplerState:
    """
    Minimize the given system and state, up to a maximum number of steps.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The state at which the system could be minimized
    sampler_state : openmmtools.states.SamplerState
        The starting state at which to minimize the system.
    max_iterations : int, optional, default 20
        The maximum number of minimization steps. Default is 20.

    Returns
    -------
    sampler_state : openmmtools.states.SamplerState
        The posititions and accompanying state following minimization
    """
    if type(cache.global_context_cache) == cache.DummyContextCache:
        integrator = openmm.VerletIntegrator(1.0) #we won't take any steps, so use a simple integrator
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state, integrator)
    else:
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
    sampler_state.apply_to_context(context, ignore_velocities = True)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations = max_iterations)
    sampler_state.update_from_context(context)

def compute_reduced_potential(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState) -> float:
    """
    Compute the reduced potential of the given SamplerState under the given ThermodynamicState.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state under which to compute the reduced potential
    sampler_state : openmmtools.states.SamplerState
        The sampler state for which to compute the reduced potential

    Returns
    -------
    reduced_potential : float
        unitless reduced potential (kT)
    """
    if type(cache.global_context_cache) == cache.DummyContextCache:
        integrator = openmm.VerletIntegrator(1.0) #we won't take any steps, so use a simple integrator
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state, integrator)
    else:
        context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
    sampler_state.apply_to_context(context, ignore_velocities=True)
    return thermodynamic_state.reduced_potential(context)

def write_nonequilibrium_trajectory(nonequilibrium_trajectory: md.Trajectory, trajectory_filename: str) -> float:
    """
    Write the results of a nonequilibrium switching trajectory to a file. The trajectory is written to an
    mdtraj hdf5 file, whereas the cumulative work is written to a numpy file.

    Parameters
    ----------
    nonequilibrium_trajectory : md.Trajectory
        The trajectory resulting from a nonequilibrium simulation
    trajectory_filename : str
        The full filepath for where to store the trajectory

    Returns
    -------
    True : bool
    """
    if nonequilibrium_trajectory is not None:
        nonequilibrium_trajectory.save_hdf5(trajectory_filename)

    return True

def write_equilibrium_trajectory(trajectory: md.Trajectory, trajectory_filename: str) -> float:
    """
    Write the results of an equilibrium simulation to disk. This task will append the results to the given filename.
    Parameters
    ----------
    trajectory : md.Trajectory
        the trajectory resulting from an equilibrium simulation
    trajectory_filename : str
        the name of the trajectory file to which we should append

    Returns
    -------
    True
    """
    if not os.path.exists(trajectory_filename):
        trajectory.save_hdf5(trajectory_filename)
        _logger.debug(f"{trajectory_filename} does not exist; instantiating and writing to.")
    else:
        _logger.debug(f"{trajectory_filename} exists; appending.")
        written_traj = md.load_hdf5(trajectory_filename)
        concatenated_traj = written_traj.join(trajectory)
        concatenated_traj.save_hdf5(trajectory_filename)

    return True

def compute_nonalchemical_perturbation(alchemical_thermodynamic_state: states.ThermodynamicState,  growth_thermodynamic_state: states.ThermodynamicState, hybrid_sampler_state: states.SamplerState, hybrid_factory: HybridTopologyFactory, nonalchemical_thermodynamic_state: states.ThermodynamicState, lambda_state: int) -> tuple:
    """
    Compute the perturbation of transforming the given hybrid equilibrium result into the system for the given nonalchemical_thermodynamic_state

    Parameters
    ----------
    alchemical_thermodynamic_state: states.ThermodynamicState
        alchemical thermostate
    growth_thermodynamic_state : states.ThermodynamicState
    hybrid_sampler_state: states.SamplerState
        sampler state for the alchemical thermodynamic_state
    hybrid_factory : HybridTopologyFactory
        Hybrid factory necessary for getting the positions of the nonalchemical system
    nonalchemical_thermodynamic_state : states.ThermodynamicState
        ThermodynamicState of the nonalchemical system
    lambda_state : int
        Whether this is lambda 0 or 1
    Returns
    -------
    work : float
        perturbation in kT from the hybrid system to the nonalchemical one
    """
    #get the objects we need to begin
    hybrid_reduced_potential = compute_reduced_potential(alchemical_thermodynamic_state, hybrid_sampler_state)
    hybrid_positions = hybrid_sampler_state.positions

    #get the positions for the nonalchemical system
    if lambda_state==0:
        nonalchemical_positions = hybrid_factory.old_positions(hybrid_positions)
        nonalchemical_alternate_positions = hybrid_factory.new_positions(hybrid_positions)
    elif lambda_state==1:
        nonalchemical_positions = hybrid_factory.new_positions(hybrid_positions)
        nonalchemical_alternate_positions = hybrid_factory.old_positions(hybrid_positions)
    else:
        raise ValueError("lambda_state must be 0 or 1")

    nonalchemical_sampler_state = states.SamplerState(nonalchemical_positions, box_vectors=hybrid_sampler_state.box_vectors)
    nonalchemical_alternate_sampler_state = states.SamplerState(nonalchemical_alternate_positions, box_vectors=hybrid_sampler_state.box_vectors)

    nonalchemical_reduced_potential = compute_reduced_potential(nonalchemical_thermodynamic_state, nonalchemical_sampler_state)

    #now for the growth system (set at lambda 0 or 1) so we can get the valence energy
    if growth_thermodynamic_state:
        valence_energy = compute_reduced_potential(growth_thermodynamic_state, nonalchemical_alternate_sampler_state)
    else:
        valence_energy = 0.0

    #now, the corrected energy of the system (for dispersion correction) is the nonalchemical_reduced_potential + valence_energy
    return (valence_energy, nonalchemical_reduced_potential, hybrid_reduced_potential)

def compute_timeseries(reduced_potentials: np.array) -> list:
    """
    Use pymbar timeseries to compute the uncorrelated samples in an array of reduced potentials.  Returns the uncorrelated sample indices.
    """
    from pymbar import timeseries
    t0, g, Neff_max = timeseries.detectEquilibration(reduced_potentials) #computing indices of uncorrelated timeseries
    A_t_equil = reduced_potentials[t0:]
    uncorrelated_indices = timeseries.subsampleCorrelatedData(A_t_equil, g=g)
    A_t = A_t_equil[uncorrelated_indices]

    return [t0, g, Neff_max, A_t]

import simtk.openmm as openmm
import openmmtools.cache as cache
from typing import List, Tuple, Union, NamedTuple
import os

#Add the variables specific to the Alchemical langevin integrator
cache.global_context_cache.COMPATIBLE_INTEGRATOR_ATTRIBUTES.update({
    "protocol_work" : 0.0,
    "Eold" : 0.0,
    "Enew" : 0.0,
    "lambda" : 0.0,
    "nsteps" : 0.0,
    "step" : 0.0,
    "n_lambda_steps" : 0.0,
    "lambda_step" : 0.0
})

import openmmtools.mcmc as mcmc
import openmmtools.integrators as integrators
import openmmtools.states as states
import numpy as np
import mdtraj as md
from perses.annihilation.new_relative import HybridTopologyFactory
import mdtraj.utils as mdtrajutils
import pickle
import simtk.unit as unit



#Make containers for results from tasklets. This allows us to chain tasks together easily.
EquilibriumResult = NamedTuple('EquilibriumResult', [('sampler_state', states.SamplerState), ('reduced_potential', float)])
NonequilibriumResult = NamedTuple('NonequilibriumResult', [('cumulative_work', np.array)])

class NonequilibriumSwitchingMove(mcmc.BaseIntegratorMove):
    """
    This class represents an MCMove that runs a nonequilibrium switching protocol using the AlchemicalNonequilibriumLangevinIntegrator.
    It is simply a wrapper around the aforementioned integrator, which must be provided in the constructor.

    Parameters
    ----------
    n_steps : int
        The number of integration steps to take each time the move is applied.
    integrator : openmmtools.integrators.AlchemicalNonequilibriumLangevinIntegrator
        The integrator that will be used for the nonequilibrium switching.
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).
    reassign_velocities : bool, optional
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move (default is False).
    restart_attempts : int, optional
        When greater than 0, if after the integration there are NaNs in energies,
        the move will restart. When the integrator has a random component, this
        may help recovering. An IntegratorMoveError is raised after the given
        number of attempts if there are still NaNs.

    Attributes
    ----------
    n_steps : int
    context_cache : openmmtools.cache.ContextCache
    reassign_velocities : bool
    restart_attempts : int or None
    current_total_work : float
    """

    def __init__(self, integrator: integrators.AlchemicalNonequilibriumLangevinIntegrator, n_steps: int, **kwargs):
        super(NonequilibriumSwitchingMove, self).__init__(n_steps, **kwargs)
        self._integrator = integrator
        self._current_total_work = 0.0

    def _get_integrator(self, thermodynamic_state):
        """
        Get the integrator associated with this move. In this case, it is simply the integrator passed in to the constructor.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            thermodynamic state; unused here.

        Returns
        -------
        integrator : openmmtools.integrators.AlchemicalNonequilibriumLangevinIntegrator
            The integrator that is associated with this MCMove
        """
        return self._integrator

    def reset(self):
        """
        Reset the work statistics on the associated ContextCache integrator.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            the thermodynamic state for which this integrator is cached.
        """
        self._integrator.reset()

    def _after_integration(self, context, thermodynamic_state):
        """
        Accumulate the work after n_steps is performed.

        Parameters
        ----------
        context : openmm.Context
            The OpenMM context which is performing the integration
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The relevant thermodynamic state for this context and integrator
        """
        integrator = context.getIntegrator()
        self._current_total_work = integrator.get_total_work(dimensionless=True)

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


def update_broker_location(broker_location, backend_location=None):
    """
    Update the location of the broker and backend from the default.

    Parameters
    ----------
    broker_location : str
        the url of the broker
    backend_location: str, optional
        the url of the backend. If none, broker_location is used.
    """
    if backend_location is None:
        backend_location = broker_location
    app.conf.update(broker=broker_location, backend=broker_location)

def run_protocol(equilibrium_result: EquilibriumResult, thermodynamic_state: states.ThermodynamicState,
                 ne_mc_move: NonequilibriumSwitchingMove, topology: md.Topology, n_iterations: int,
                 atom_indices_to_save: List[int] = None, trajectory_filename: str = None) -> NonequilibriumResult:
    """
    Perform a nonequilibrium switching protocol and return the nonequilibrium protocol work. Note that it is expected
    that this will perform an entire protocol, that is, switching lambda completely from 0 to 1, in increments specified
    by the ne_mc_move. The trajectory that results, along with the work values, will contain n_iterations elements.

    Parameters
    ----------
    equilibrium_result : EquilibriumResult namedtuple
        The result of an equilibrium simulation
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state at which to run the protocol
    ne_mc_move : perses.distributed.relative_setup.NonequilibriumSwitchingMove
        The move that will be used to perform the switching.
    topology : mdtraj.Topology
        An MDtraj topology for the system to generate trajectories
    n_iterations : int
        The number of times to apply the specified MCMove
    atom_indices_to_save : list of int, default None
        list of indices to save (when excluding waters, for instance). If None, all indices are saved.
    trajectory_filename : str, default None
        Full filepath of output trajectory, if desired. If None, no trajectory file is written.
    Returns
    -------
    nonequilibrium_result : NonequilibriumResult
        result object containing the trajectory of the nonequilibrium calculation, as well as the cumulative work
        for each frame.
    """
    #get the sampler state needed for the simulation
    sampler_state = equilibrium_result.sampler_state

    #get the atom indices we need to subset the topology and positions
    if atom_indices_to_save is None:
        atom_indices = list(range(topology.n_atoms))
        subset_topology = topology
    else:
        subset_topology = topology.subset(atom_indices_to_save)
        atom_indices = atom_indices_to_save

    n_atoms = subset_topology.n_atoms

    #create a numpy array for the trajectory
    trajectory_positions = np.zeros([n_iterations, n_atoms, 3])
    trajectory_box_lengths = np.zeros([n_iterations, 3])
    trajectory_box_angles = np.zeros([n_iterations, 3])

    #create a numpy array for the work values
    cumulative_work = np.zeros(n_iterations)
    #rdb.set_trace()
    #reset the MCMove to ensure that we are starting with zero work.
    ne_mc_move.reset()

    #now loop through the iterations and run the protocol:
    for iteration in range(n_iterations):
        #apply the nonequilibrium move
        ne_mc_move.apply(thermodynamic_state, sampler_state)

        #record the positions as a result
        trajectory_positions[iteration, :, :] = sampler_state.positions[atom_indices, :].value_in_unit_system(unit.md_unit_system)

        #get the box angles and lengths
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
        trajectory_box_lengths[iteration, :] = [a, b, c]
        trajectory_box_angles[iteration, :] = [alpha, beta, gamma]

        #record the cumulative work as a result
        cumulative_work[iteration] = ne_mc_move.current_total_work

    #create an MDTraj trajectory with this data
    trajectory = md.Trajectory(trajectory_positions, subset_topology, unitcell_lengths=trajectory_box_lengths, unitcell_angles=trajectory_box_angles)

    #create a result object and return that
    nonequilibrium_result = NonequilibriumResult(cumulative_work)

    #if desired, write nonequilibrium trajectories:
    if trajectory_filename is not None:
        #to get the filename for cumulative work, replace the extension of the trajectory file with .cw.npy
        filepath_parts = trajectory_filename.split(".")
        filepath_parts[-1] = "cw.npy"
        cum_work_filepath = ".".join(filepath_parts)

        write_nonequilibrium_trajectory(nonequilibrium_result, trajectory, trajectory_filename, cum_work_filepath)

    return nonequilibrium_result

def run_equilibrium(equilibrium_result: EquilibriumResult, thermodynamic_state: states.ThermodynamicState,
                    mc_move: mcmc.MCMCMove, topology: md.Topology,
                    atom_indices_to_save: List[int] = None, trajectory_filename: str = None) -> EquilibriumResult:
    """
    Run nsteps of equilibrium sampling at the specified thermodynamic state and return the final sampler state
    as well as a trajectory of the positions after each application of an MCMove. This means that if the MCMove
    is configured to run 1000 steps of dynamics, and n_iterations is 100, there will be 100 frames in the resulting
    trajectory; these are the result of 100,000 steps (1000*100) of dynamics.

    Parameters
    ----------
    equilibrium_result : EquilibriumResult
       EquilibriumResult namedtuple containing the information necessary to resume
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state (including context parameters) that should be used
    mc_move : openmmtools.mcmc.MCMove
        The move to apply to the system
    topology : mdtraj.Topology
        an MDTraj topology object used to construct the trajectory
    n_iterations : int
        The number of times to apply the move. Note that this is not the number of steps of dynamics; it is
        n_iterations*n_steps (which is set in the MCMove).
    atom_indices_to_save : list of int, default None
        list of indices to save (when excluding waters, for instance). If None, all indices are saved.
    trajectory_filename : str, optional, default None
        Full filepath of trajectory files. If none, trajectory files are not written.
    Returns
    -------
    equilibrium_result : EquilibriumResult
        Container namedtuple that has the SamplerState for resuming, an MDTraj trajectory, and the reduced potential of the
        final frame.
    """
    sampler_state = equilibrium_result.sampler_state
    n_iterations = 10
    #get the atom indices we need to subset the topology and positions
    if atom_indices_to_save is None:
        atom_indices = list(range(topology.n_atoms))
        subset_topology = topology
    else:
        subset_topology = topology.subset(atom_indices_to_save)
        atom_indices = atom_indices_to_save

    n_atoms = subset_topology.n_atoms

    #create a numpy array for the trajectory
    trajectory_positions = np.zeros([n_iterations, n_atoms, 3])
    trajectory_box_lengths = np.zeros([n_iterations, 3])
    trajectory_box_angles = np.zeros([n_iterations, 3])

    #loop through iterations and apply MCMove, then collect positions into numpy array
    for iteration in range(n_iterations):
        mc_move.apply(thermodynamic_state, sampler_state)

        trajectory_positions[iteration, :] = sampler_state.positions[atom_indices, :].value_in_unit_system(unit.md_unit_system)

        #get the box lengths and angles
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*sampler_state.box_vectors)
        trajectory_box_lengths[iteration, :] = [a, b, c]
        trajectory_box_angles[iteration, :] = [alpha, beta, gamma]

    #construct trajectory object:
    trajectory = md.Trajectory(trajectory_positions, subset_topology, unitcell_lengths=trajectory_box_lengths, unitcell_angles=trajectory_box_angles)

    #get the reduced potential from the final frame for endpoint perturbations
    reduced_potential_final_frame = thermodynamic_state.reduced_potential(sampler_state)

    #construct equilibrium result object
    equilibrium_result = EquilibriumResult(sampler_state, reduced_potential_final_frame)

    #If there is a trajectory filename passed, write out the results here:
    if trajectory_filename is not None:
        write_equilibrium_trajectory(equilibrium_result, trajectory, trajectory_filename)

    return equilibrium_result

def minimize(thermodynamic_state: states.ThermodynamicState, sampler_state: states.SamplerState, mc_move: mcmc.MCMCMove,
             max_iterations: int=20) -> states.SamplerState:
    """
    Minimize the given system and state, up to a maximum number of steps.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The state at which the system could be minimized
    sampler_state : openmmtools.states.SamplerState
        The starting state at which to minimize the system.
    mc_move : openmmtools.mcmc.MCMove
        The move type. This is not directly relevant, but it will
        determine whether a context can be reused. It is recommended that
        the same move as the equilibrium protocol is used here.
    max_iterations : int, optional, default 20
        The maximum number of minimization steps. Default is 20.

    Returns
    -------
    sampler_state : openmmtools.states.SamplerState
        The posititions and accompanying state following minimization
    """
    mcmc_sampler = mcmc.MCMCSampler(thermodynamic_state, sampler_state, mc_move)
    mcmc_sampler.minimize(max_iterations=max_iterations)
    return mcmc_sampler.sampler_state

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
    context, integrator = cache.global_context_cache.get_context(thermodynamic_state)
    sampler_state.apply_to_context(context, ignore_velocities=True)
    return thermodynamic_state.reduced_potential(context)

def write_nonequilibrium_trajectory(nonequilibrium_result: NonequilibriumResult, nonequilibrium_trajectory: md.Trajectory, trajectory_filename: str, cum_work_filename: str) -> float:
    """
    Write the results of a nonequilibrium switching trajectory to a file. The trajectory is written to an
    mdtraj hdf5 file, whereas the cumulative work is written to a numpy file.

    Parameters
    ----------
    nonequilibrium_result : NonequilibriumResult namedtuple
        The result of a nonequilibrium switching calculation
    nonequilibrium_trajectory : md.Trajectory
        The trajectory resulting from a nonequilibrium simulation
    trajectory_filename : str
        The full filepath for where to store the trajectory
    cum_work_filename : str
        The full filepath for where to store the work trajectory

    Returns
    -------
    final_work : float
        The final value of the work trajectory
    """
    nonequilibrium_trajectory.save_hdf5(trajectory_filename)
    np.save(cum_work_filename, nonequilibrium_result.cumulative_work)

    return nonequilibrium_result.cumulative_work[-1]

def write_equilibrium_trajectory(equilibrium_result: EquilibriumResult, trajectory: md.Trajectory, trajectory_filename: str) -> float:
    """
    Write the results of an equilibrium simulation to disk. This task will append the results to the given filename.
    Parameters
    ----------
    equilibrium_result : EquilibriumResult namedtuple
        the result of an equilibrium calculation
    trajectory : md.Trajectory
        the trajectory resulting from an equilibrium simulation
    trajectory_filename : str
        the name of the trajectory file to which we should append

    Returns
    -------
    reduced_potential_final_frame : float
        the reduced potential of the final frame
    """
    if not os.path.exists(trajectory_filename):
        trajectory.save_hdf5(trajectory_filename)
    else:
        written_traj = md.load_hdf5(trajectory_filename)
        concatenated_traj = written_traj.join(trajectory)
        concatenated_traj.save_hdf5(trajectory_filename)

    return equilibrium_result.reduced_potential

def compute_nonalchemical_perturbation(equilibrium_result: EquilibriumResult, hybrid_factory: HybridTopologyFactory, nonalchemical_thermodynamic_state: states.ThermodynamicState, lambda_state: int):
    """
    Compute the perturbation of transforming the given hybrid equilibrium result into the system for the given nonalchemical_thermodynamic_state

    Parameters
    ----------
    equilibrium_result : EquilibriumResult
        Result of the equilibrium simulation
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
    hybrid_reduced_potential = equilibrium_result.reduced_potential
    hybrid_sampler_state = equilibrium_result.sampler_state
    hybrid_positions = hybrid_sampler_state.positions

    #get the positions for the nonalchemical system
    if lambda_state==0:
        nonalchemical_positions = hybrid_factory.old_positions(hybrid_positions)
    elif lambda_state==1:
        nonalchemical_positions = hybrid_factory.new_positions(hybrid_positions)
    else:
        raise ValueError("lambda_state must be 0 or 1")

    nonalchemical_sampler_state = states.SamplerState(nonalchemical_positions, box_vectors=hybrid_sampler_state.box_vectors)

    nonalchemical_reduced_potential = compute_reduced_potential(nonalchemical_thermodynamic_state, nonalchemical_sampler_state)

    return hybrid_reduced_potential - nonalchemical_reduced_potential
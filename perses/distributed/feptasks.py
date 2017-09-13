import celery
from celery.contrib import rdb
import simtk.openmm as openmm
import openmmtools.cache as cache

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
import openmmtools.utils as utils
import openmmtools.integrators as integrators
import redis
import numpy as np
import mdtraj as md
import mdtraj.utils as mdtrajutils
import pickle

broker_name_server = "redis://localhost"

def get_broker_name():
    """
    This is a utility function for getting the broker location
    """
    redis_client = redis.Redis(host=broker_name_server)
    broker_location = redis_client.get("broker_location")

    if broker_location is None:
        raise ValueError("The specified broker name server does not contain a record of a broker.")

    return broker_location

broker_location = broker_name_server
app = celery.Celery('perses.distributed.feptasks', broker=broker_location, backend=broker_location)
app.conf.update(accept_content=['pickle', 'application/x-python-serialize'], task_serializer='pickle', result_serializer='pickle')

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

    def __init__(self, integrator, n_steps, **kwargs):
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

    def reset(self, thermodynamic_state):
        """
        Reset the work statistics on the associated ContextCache integrator.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            the thermodynamic state for which this integrator is cached.
        """

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        #Get the integrator from the context cache
        context, integrator = context_cache.get_context(thermodynamic_state, self._integrator)

        #Reset the statistics on the integrator
        integrator.reset()

        #reset the class's own statistics:
        self._current_total_work = 0.0

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
        self._current_total_work += integrator.get_total_work(dimensionless=True)

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

@app.task(serializer="pickle")
def run_protocol(thermodynamic_state, sampler_state, ne_mc_move, topology, n_iterations):
    """
    Perform a nonequilibrium switching protocol and return the nonequilibrium protocol work. Note that it is expected
    that this will perform an entire protocol, that is, switching lambda completely from 0 to 1, in increments specified
    by the ne_mc_move. The trajectory that results, along with the work values, will contain n_iterations elements.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state at which to run the protocol
    sampler_state : openmmtools.states.SamplerState
        The initial sampler state at which to run the protocol, including positions.
    ne_mc_move : perses.distributed.relative_setup.NonequilibriumSwitchingMove
        The move that will be used to perform the switching.
    topology : mdtraj.Topology
        An MDtraj topology for the system to generate trajectories
    n_iterations : int
        The number of times to apply the specified MCMove

    Returns
    -------
    trajectory : mdtraj.Trajectory
        Trajectory containing n_iterations frames

    """
    n_atoms = topology.n_atoms

    #create a numpy array for the trajectory
    trajectory_positions = np.zeros([n_iterations, n_atoms, 3])
    trajectory_box_lengths = np.zeros([n_iterations, 3])
    trajectory_box_angles = np.zeros([n_iterations, 3])

    #create a numpy array for the work values
    cumulative_work = np.zeros(n_iterations)
    #rdb.set_trace()
    #reset the MCMove to ensure that we are starting with zero work.
    ne_mc_move.reset(thermodynamic_state)

    #now loop through the iterations and run the protocol:
    for iteration in range(n_iterations):
        #apply the nonequilibrium move
        ne_mc_move.apply(thermodynamic_state, sampler_state)

        #record the positions as a result
        trajectory_positions[iteration, :, :] = sampler_state.positions

        #get the box angles and lengths
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(sampler_state.box_vectors)
        trajectory_box_lengths[iteration, :] = [a, b, c]
        trajectory_box_angles[iteration, :] = [alpha, beta, gamma]

        #record the cumulative work as a result
        cumulative_work[iteration] = ne_mc_move.current_total_work

    #create an MDTraj trajectory with this data
    trajectory = md.Trajectory(trajectory_positions, topology, unitcell_lengths=trajectory_box_lengths, unitcell_angles=trajectory_box_angles)

    return trajectory, cumulative_work

@app.task(serializer="pickle")
def run_equilibrium(thermodynamic_state, sampler_state, mc_move, topology, n_iterations):
    """
    Run nsteps of equilibrium sampling at the specified thermodynamic state and return the final sampler state
    as well as a trajectory of the positions after each application of an MCMove. This means that if the MCMove
    is configured to run 1000 steps of dynamics, and n_iterations is 100, there will be 100 frames in the resulting
    trajectory; these are the result of 100,000 steps (1000*100) of dynamics.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state (including context parameters) that should be used
    sampler_state : openmmtools.states.SamplerState
        The state of the sampler (such as positions) from which to start
    mc_move : openmmtools.mcmc.MCMove
        The move to apply to the system
    topology : mdtraj.Topology
        an MDTraj topology object used to construct the trajectory
    n_iterations : int
        The number of times to apply the move. Note that this is not the number of steps of dynamics; it is
        n_iterations*n_steps (which is set in the MCMove).

    Returns
    -------
    sampler_state : openmmtools.SamplerState
        The sampler state after equilibrium has been run
    trajectory : mdtraj.Trajectory
        A trajectory consisting of one frame per application of the MCMove
    """
    n_atoms = topology.n_atoms

    #create a numpy array for the trajectory
    trajectory_positions = np.zeros([n_iterations, n_atoms, 3])
    trajectory_box_lengths = np.zeros([n_iterations, 3])
    trajectory_box_angles = np.zeros([n_iterations, 3])

    #loop through iterations and apply MCMove, then collect positions into numpy array
    for iteration in range(n_iterations):
        mc_move.apply(thermodynamic_state, sampler_state)

        trajectory_positions[iteration, :] = sampler_state.positions

        #get the box lengths and angles
        a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(sampler_state.box_vectors)
        trajectory_box_lengths[iteration, :] = [a, b, c]
        trajectory_box_angles[iteration, :] = [alpha, beta, gamma]

    #construct trajectory object:
    trajectory = md.Trajectory(trajectory_positions, topology, unitcell_lengths=trajectory_box_lengths, unitcell_angles=trajectory_box_angles)

    return sampler_state, trajectory

@app.task(serializer="pickle")
def minimize(thermodynamic_state, sampler_state, mc_move, max_iterations=20):
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


import celery
from celery.contrib import rdb
import simtk.openmm as openmm
import openmmtools.cache as cache
import openmmtools.mcmc as mcmc
import redis
import numpy as np
import mdtraj as md

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
    trajectory_positions = np.zeros([n_atoms, 3, n_iterations])

    #create a numpy array for the work values
    cumulative_work = np.zeros(n_iterations)

    #reset the MCMove to ensure that we are starting with zero work.
    ne_mc_move.reset(thermodynamic_state)

    #now loop through the iterations and run the protocol:
    for iteration in range(n_iterations):
        #apply the nonequilibrium move
        ne_mc_move.apply(thermodynamic_state, sampler_state)

        #record the positions as a result
        trajectory_positions[:, :, iteration] = sampler_state.positions

        #record the cumulative work as a result
        cumulative_work[iteration] = ne_mc_move.current_total_work

    #create an MDTraj trajectory with this data
    trajectory = md.Trajectory(trajectory_positions, topology)

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
    trajectory_positions = np.zeros([n_atoms, 3, n_iterations])

    #loop through iterations and apply MCMove, then collect positions into numpy array
    for iteration in range(n_iterations):
        mc_move.apply(thermodynamic_state, sampler_state)
        trajectory_positions[:, :, iteration] = sampler_state.positions

    #construct trajectory object:
    trajectory = md.Trajectory(trajectory_positions, topology)

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


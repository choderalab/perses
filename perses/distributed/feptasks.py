import celery
from celery.contrib import rdb
import simtk.openmm as openmm
import openmmtools.cache as cache
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
app.conf.update(accept_content=['pickle', 'application/x-python-serialize'],task_serializer='pickle', result_serializer='pickle')


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


class NonequilibriumSwitchTask(celery.Task):
    """
    This is a base class for nonequilibrium switching tasks.
    """

    def __init__(self):
        platform = openmm.Platform.getPlatformByName("OpenCL")
        self._cache = cache.ContextCache(platform=platform)

@app.task(bind=True, base=NonequilibriumSwitchTask, serializer="pickle")
def run_protocol(self, thermodynamic_state, sampler_state, mc_move, n_iterations):
    """
    Perform a nonequilibrium switching protocol and return the nonequilibrium protocol work.

    Parameters
    ----------
    sampler_state : openmmtools.states.SamplerState object
        Object containing the positions of the

    Returns
    -------
    work : float
        The dimensionless nonequilibrium protocol work.
    """
    switching_ctx, integrator_neq = self._cache.get_context(thermodynamic_state, integrator)
    switching_ctx.setPositions(starting_positions)
    integrator_neq.reset()
    integrator_neq.step(nsteps)
    work = integrator_neq.get_protocol_work(dimensionless=True)
    return work

@app.task(bind=True, base=NonequilibriumSwitchTask, serializer="pickle")
def run_equilibrium(self, thermodynamic_state, sampler_state, mc_move, topology, n_iterations):
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
        trajectory_positions[:, 3, iteration] = sampler_state.positions

    #construct trajectory object:
    trajectory = md.Trajectory(trajectory_positions, topology)

    return sampler_state, trajectory

@app.task(bind=True, base=NonequilibriumSwitchTask, serializer="pickle")
def minimize(self, starting_positions, nsteps_max, lambda_state, functions, thermodynamic_state, integrator):
    equilibrium_ctx, integrator = self._cache.get_context(thermodynamic_state, integrator)

    equilibrium_ctx.setPositions(starting_positions)

    for parm in functions.keys():
        equilibrium_ctx.setParameter(parm, lambda_state)

    openmm.LocalEnergyMinimizer.minimize(equilibrium_ctx, maxIterations=nsteps_max)

    initial_state = equilibrium_ctx.getState(getPositions=True, getEnergy=True)

    return initial_state.getPositions(asNumpy=True)

@app.task(bind=True, base=NonequilibriumSwitchTask, serializer="pickle")
def dummy_task(self, integrator):
    rdb.set_trace()
    return integrator

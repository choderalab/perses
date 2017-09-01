import celery
from celery.contrib import rdb
import simtk.openmm as openmm
import openmmtools.cache as cache
import redis

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
def run_protocol(self, starting_positions, nsteps, thermodynamic_state, integrator):
    """
    Perform a nonequilibrium switching protocol and return the nonequilibrium protocol work.

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
def run_equilibrium(self, starting_positions, nsteps, lambda_state, functions, thermodynamic_state, integrator):
    """
    Run nsteps of equilibrium sampling at the specified thermodynamic state and return the positions.

    Returns
    -------
    positions : [n, 3] np.ndarray quantity
    """
    equilibrium_ctx, integrator = self._cache.get_context(thermodynamic_state, integrator)
    equilibrium_ctx.setPositions(starting_positions)
    for parm in functions.keys():
        equilibrium_ctx.setParameter(parm, lambda_state)
    integrator.step(nsteps)
    state = equilibrium_ctx.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)
    return positions

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

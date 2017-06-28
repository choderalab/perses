import celery
from celery.contrib import rdb
import simtk.openmm as openmm
import openmmtools.integrators as integrators
import openmmtools.cache as cache
import simtk.unit as unit
import numpy as np
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


class NonequilibriumSwitchTask(celery.Task):
    """
    This is a base class for nonequilibrium switching tasks.
    """

    def __init__(self):
        self._forward_context = None
        self._forward_integrator = None
        self._reverse_context = None
        self._reverse_integrator = None
        self._equilibrium_context = None
        self._equilibrium_integrator = None

@app.task(bind=True, base=NonequilibriumSwitchTask, serializer="pickle")
def run_protocol(self, starting_positions, system, nsteps, direction, functions, temperature=300*unit.kelvin, platform_name="OpenCL"):
    """
    Perform a switching protocol and return the nonequilibrium switching weight

    Returns
    -------
    weight : float64
        The nonequilibrium switching weight
    """
    if direction == 'forward':
        if self._forward_context is None:
            integrator = integrators.AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=functions, nsteps_neq=nsteps, temperature=temperature)
            platform = openmm.Platform.getPlatformByName(platform_name)
            switching_ctx = openmm.Context(system, integrator, platform)
            self._forward_context = switching_ctx
            self._forward_integrator = integrator
        else:
            switching_ctx = self._forward_context
            integrator = self._forward_integrator
    elif direction == 'reverse':
        if self._reverse_context is None:
            integrator = integrators.AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=functions, nsteps_neq=nsteps, temperature=temperature)
            platform = openmm.Platform.getPlatformByName(platform_name)
            switching_ctx = openmm.Context(system, integrator, platform)
            self._reverse_context = switching_ctx
            self._reverse_integrator = integrator
        else:
            switching_ctx = self._reverse_context
            integrator = self._reverse_integrator
    switching_ctx.setPositions(starting_positions)
    integrator.step(nsteps)
    work = integrator.getGlobalVariableByName("protocol_work")
    integrator.reset()
    return work

@app.task(bind=True, base=NonequilibriumSwitchTask, serializer="pickle")
def run_equilibrium(self, starting_positions, system, nsteps, lambda_state, functions, temperature=300.0*unit.kelvin, platform_name="OpenCL"):
    """
    Run nsteps of equilibrium sampling at the specified thermodynamic state and return the positions.

    Returns
    -------
    positions : [n, 3] np.ndarray quantity
    """
    if self._equilibrium_context is None:
        integrator = openmm.LangevinIntegrator(temperature, 5.0 / unit.picosecond, 1.0*unit.femtosecond)
        platform = openmm.Platform.getPlatformByName(platform_name)
        equilibrium_ctx = openmm.Context(system, integrator, platform)
        self._equilibrium_context = equilibrium_ctx
    else:
        equilibrium_ctx = self._equilibrium_context
        integrator = self._equilibrium_context.getIntegrator()
    equilibrium_ctx.setPositions(starting_positions)
    for parm in functions.keys():
        equilibrium_ctx.setParameter(parm, lambda_state)
    integrator.step(nsteps)
    state = equilibrium_ctx.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)
    return positions

@app.task(bind=True, base=NonequilibriumSwitchTask, serializer="pickle")
def minimize(self, starting_positions, system, lambda_state, functions, nsteps_max, temperature=300.0*unit.kelvin, platform_name="OpenCL"):
    if self._equilibrium_context is None:
        integrator = openmm.LangevinIntegrator(temperature, 5.0 / unit.picosecond, 1.0*unit.femtosecond)
        platform = openmm.Platform.getPlatformByName(platform_name)
        equilibrium_ctx = openmm.Context(system, integrator, platform)
        self._equilibrium_context = equilibrium_ctx
    else:
        equilibrium_ctx = self._equilibrium_context

    equilibrium_ctx.setPositions(starting_positions)

    for parm in functions:
        equilibrium_ctx.setParameter(parm, lambda_state)

    openmm.LocalEnergyMinimizer.minimize(equilibrium_ctx, maxIterations=nsteps_max)

    initial_state = equilibrium_ctx.getState(getPositions=True, getEnergy=True)

    return initial_state.getPositions(asNumpy=True)
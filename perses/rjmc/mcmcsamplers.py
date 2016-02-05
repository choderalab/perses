"""
MCMC and SAMS samplers for expanded ensemble simulations
"""

import simtk.openmm as openmm
import simtk.unit as unit
import simtk.openmm.app as app

import numpy as np


class MCMCSampler(object):
    """
    MCMC sampler object

    Parameters
    ----------
    system : simtk.openmm.System object
        The system to simulate
    integrator : simtk.openmm.Integrator object
        OpenMM integrator to use (could be CustomIntegrator)
    positions : [n,3] np.ndarray
        Initial positions of the system
    """

    def __init__(self, system, integrator, positions, platform_name='CPU'):
        self._system = system
        self._integrator = integrator
        self._initial_positions = positions
        platform = openmm.Platform.getPlatformByName(platform_name)
        self._context = openmm.Context(system, integrator, platform)

    def integrate(self, nsteps):
        """
        Perform n steps of integration

        Parameters
        ----------
        nsteps : int
            Number of steps to integrate
        """
        self._integrator.step(nsteps)

    @property
    def potential_energy(self):
        state = self._context.getState(getEnergy=True)
        return state.getPotentialEnergy()
    @property
    def positions(self):
        state = self._context.getState(getPositions=True)
        return state.getPositions(asNumpy=True)
    @property
    def kinetic_energy(self):
        state = self._context.getState(getEnergy=True)
        return state.getKineticEnergy()

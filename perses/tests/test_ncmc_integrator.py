"""
Unit tests for NCMC integrator.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
import math
import numpy as np
from functools import partial
from unittest import skipIf
import os

running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

################################################################################
# CONSTANTS
################################################################################

from openmmtools.constants import kB

################################################################################
# TESTS
################################################################################

def collect_switching_data(system, positions, functions, temperature, collision_rate, timestep, platform, ghmc_nsteps=200, ncmc_nsteps=50, niterations=100, direction='insert', ncmc_integrator=None):
    """
    Collect switching data.

    """

    work_n = np.zeros([niterations], np.float64) # work[iteration] is work (log probability) in kT from iteration `iteration`

    # Create integrators.
    integrator = openmm.CompoundIntegrator()
    # Create GHMC integrator.
    from openmmtools.integrators import GHMCIntegrator
    ghmc_integrator = GHMCIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep)
    integrator.addIntegrator(ghmc_integrator)
    # Create an NCMC switching integrator.
    integrator.addIntegrator(ncmc_integrator)

    # Create Context
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    naccept_n = np.zeros([niterations], np.int32)
    ntrials_n = np.zeros([niterations], np.int32)
    for iteration in range(niterations):
        # Equilibrate
        integrator.setCurrentIntegrator(0)
        if direction == 'insert':
            context.setParameter('x0', 0)
        elif direction == 'delete':
            context.setParameter('x0', 1)
        else:
            raise Exception("direction '%s' unknown; must be 'insert' or 'delete'." % direction)
        integrator.step(ghmc_nsteps)

        # Switch
        integrator.setCurrentIntegrator(1)
        ncmc_integrator.reset()
        if ncmc_nsteps == 0:
            integrator.step(1)
        else:
            integrator.step(ncmc_nsteps)
            #print("The step is %d" % ncmc_integrator.get_step())
        work_n[iteration] = - ncmc_integrator.getLogAcceptanceProbability(context)

        if ncmc_integrator.has_statistics:
            (naccept_n[iteration], ntrials_n[iteration]) = ncmc_integrator.getGHMCStatistics(context)

    if ncmc_integrator.has_statistics:
        print('GHMC: %d / %d accepted' % (naccept_n.sum(), ntrials_n.sum()))

    # Clean up
    del context, integrator
    return work_n

@skipIf(running_on_github_actions, "Skip expensive test on GH Actions")
def test_ncmc_integrator_harmonic_oscillator():
   """
   Check NCMC integrator switching works for 0, 1, and 50 switching steps with a harmonic oscillator.

   """
   for integrator_type in ["VV", "GHMC"]:
       for ncmc_nsteps in [0, 1, 50]:
           f = partial(check_harmonic_oscillator_ncmc, ncmc_nsteps, ncmc_integrator=integrator_type)
           f.description = "Testing %s NCMC switching using harmonic oscillator with %d NCMC steps" % (integrator_type, ncmc_nsteps)
           yield f

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

################################################################################
# CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

################################################################################
# TESTS
################################################################################

def collect_switching_data(system, positions, functions, temperature, collision_rate, timestep, platform, ghmc_nsteps=200, ncmc_nsteps=50, niterations=20, direction='insert'):
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
    from perses.annihilation import NCMCAlchemicalIntegrator
    ncmc_integrator = NCMCAlchemicalIntegrator(temperature, system, functions, direction=direction, nsteps=ncmc_nsteps, timestep=timestep) # 'insert' drags lambda from 0 -> 1
    integrator.addIntegrator(ncmc_integrator)

    # Create Context
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

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
        integrator.step(1)
        work_n[iteration] = ncmc_integrator.getLogAcceptanceProbability()

    # Clean up
    del context, integrator

    return work_n

def check_harmonic_oscillator_ncmc(ncmc_nsteps=50):
    """
    Test NCMC switching of a 3D harmonic oscillator.
    In this test, the oscillator center is dragged in space, and we check the computed free energy difference with BAR, which should be 0.
    """
    # Parameters for 3D harmonic oscillator
    mass = 39.948 * unit.amu # mass of particle (argon)
    sigma = 5.0 * unit.angstrom # standard deviation of harmonic oscillator
    collision_rate = 5.0/unit.picosecond # collision rate
    temperature = 300.0 * unit.kelvin # temperature
    platform_name = 'Reference' # platform anme
    NSIGMA_MAX = 6.0 # number of standard errors away from analytical solution tolerated before Exception is thrown

    # Compute derived quantities.
    kT = kB * temperature # thermal energy
    beta = 1.0 / kT # inverse energy
    K = kT / sigma**2 # spring constant
    tau = 2 * math.pi * unit.sqrt(mass/K) # time constant
    timestep = tau / 20.0
    platform = openmm.Platform.getPlatformByName(platform_name)

    # Create a 3D harmonic oscillator with context parameter controlling center of oscillator.
    system = openmm.System()
    system.addParticle(mass)
    energy_expression = '(K/2.0) * ((x-x0)^2 + y^2 + z^2);'
    force = openmm.CustomExternalForce(energy_expression)
    force.addGlobalParameter('K', K.in_unit_system(unit.md_unit_system))
    force.addGlobalParameter('x0', 0.0)
    force.addParticle(0, [])
    system.addForce(force)

    # Set the positions at the origin.
    positions = unit.Quantity(np.zeros([1, 3], np.float32), unit.angstroms)

    # Run NCMC switching trials where the spring center is switched with lambda: 0 -> 1 over a finite number of steps.
    functions = { 'x0' : 'lambda' } # drag spring center x0

    w_f = collect_switching_data(system, positions, functions, temperature, collision_rate, timestep, platform, ncmc_nsteps=ncmc_nsteps, direction='insert')
    w_r = collect_switching_data(system, positions, functions, temperature, collision_rate, timestep, platform, ncmc_nsteps=ncmc_nsteps, direction='delete')

    from pymbar import BAR
    [df, ddf] = BAR(w_f, w_r, method='self-consistent-iteration')
    if (abs(df) > NSIGMA_MAX * ddf):
        raise Exception('Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0' % (ncmc_nsteps, df, ddf, NSIGMA_MAX))

def test_ncmc_integrator_harmonic_oscillator():
    """
    Check NCMC integrator switching works for 0, 1, and 50 switching steps with a harmonic oscillator.

    """
    for ncmc_nsteps in [0, 1, 50]:
        f = partial(check_harmonic_oscillator_ncmc, ncmc_nsteps)
        f.description = "Testing NCMC switching using harmonic oscillator with %d NCMC steps" % ncmc_nsteps
        yield f

if __name__ == '__main__':
    test_ncmc_harmonic_oscillator()

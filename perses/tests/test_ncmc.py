"""
Unit tests for NCMC switching engine.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
import math
import numpy as np
from perses import annihilation 

################################################################################
# CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

################################################################################
# TESTS
################################################################################

def test_ncmc_harmonic_oscillator():
    """
    Test NCMC switching of a 3D harmonic oscillator.
    In this test, the oscillator center is dragged in space, and we check the computed free energy difference with BAR,
    which should be 0.

    """
    # Parameters for 3D harmonic oscillator
    mass = 39.948 * unit.amu # mass of particle (argon)
    sigma = 5.0 * unit.angstrom # standard deviation of harmonic oscillator
    temperature = 300.0 * unit.kelvin # temperature
    nsteps = 500 # number of switching steps
    platform_name = 'Reference' # platform anme

    # Compute derived quantities.
    kT = kB * temperature # thermal energy
    beta = 1.0 / kT # inverse energy
    K = kT / sigma**2 # spring constant
    tau = 2 * math.pi * unit.sqrt(mass/K) # time constant
    timestep = tau / 20.0

    # Create a 3D harmonic oscillator with context parameter.
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

    # Run NCMC switching trials where the spring center is switched with lambda: 0 -> 1.

    # Create an NCMC switching integrator.
    functions = { 'x0' : 'lambda' } # drag spring center x0
    from annihilation import NCMCAlchemicalIntegrator
    ncmc_integrator = NCMCAlchemicalIntegrator(temperature, alchemical_system, functions, mode='insert', nsteps=nsteps, timestep=timestep) # 'insert' drags lambda from 0 -> 1
    # Create a Context
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(alchemical_system, ncmc_integrator, platform)
    context.setPositions(testsystem.positions)
    # Run the integrator
    ncmc_integrator.step(1)
    # Retrieve the log acceptance probability
    log_ncmc = ncmc_integrator.log_ncmc
    print(log_ncmc)


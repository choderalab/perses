#!/usr/bin/env python
"""
Test of shake with zero bond length.


"""

from simtk import openmm
from simtk import unit
from simtk.openmm import app
import copy
import numpy as np
from openmmtools import testsystems

# Set conditions for simulation.
# Roughly corresponds to conditions from http://www.cstl.nist.gov/srs/LJ_PURE/mc.htm
nparticles = 500
mass = 39.9 * unit.amu
sigma = 3.4 * unit.angstrom
epsilon = 0.238 * unit.kilocalories_per_mole
#reduced_density = 0.860     # reduced_density = density * (sigma**3)
reduced_density = 0.960     # reduced_density = density * (sigma**3)
reduced_temperature = 0.850 # reduced_temperature = kB * temperature / epsilon
reduced_pressure = 1.2660   # reduced_pressure = pressure * (sigma**3) / epsilon

platform_name = 'CPU'    # OpenMM platform name to use for simulation
platform = openmm.Platform.getPlatformByName(platform_name)

r0 = 2.0**(1./6.) * sigma   # minimum potential distance for Lennard-Jones interaction
characteristic_timescale = unit.sqrt((mass * r0**2) / (72 * epsilon)) # characteristic timescale for bound Lennard-Jones interaction
                                                            # http://borisv.lk.net/matsc597c-1997/simulations/Lecture5/node3.html
timestep = 0.01 * characteristic_timescale # integrator timestep

# From http://www.cstl.nist.gov/srs/LJ_PURE/md.htm
#characteristic_timescale = unit.sqrt(mass * sigma**2 / epsilon)
#timestep = 0.05 * characteristic_timescale

print "characteristic timescale = %.3f ps" % (characteristic_timescale / unit.picoseconds)
print "timestep = %.12f ps" % (timestep / unit.picoseconds)

collision_rate = 5.0 / unit.picoseconds # collision rate for Langevin thermostat
barostat_frequency = 25 # number of steps between barostat updates

# Set parameters for number of simulation replicates, number of iterations per simulation, and number of steps per iteration.
nreplicates = 100
niterations = 10000
nsteps_per_iteration = 25

# Compute real units.
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
density = reduced_density / (sigma**3)
temperature = reduced_temperature * epsilon / kB
pressure = reduced_pressure * epsilon / (sigma**3)
kT = kB * temperature

# Create the Lennard-Jones fluid.
testsystem = testsystems.LennardJonesFluid(nparticles=nparticles, mass=mass, sigma=sigma, epsilon=epsilon, reduced_density=reduced_density)

# Move two particles on top of each other.
print "Modifying positions..."
positions = testsystem.positions
positions[1,:] = positions[0,:]
print positions

# Add exclusion
master_system = testsystem.system
print "Adding exclusion..."
forces = { master_system.getForce(index).__class__.__name__ : master_system.getForce(index) for index in range(master_system.getNumForces()) }
force = forces['NonbondedForce']
chargeProd = 0.0 * unit.elementary_charge**2
sigma = 1.0 * unit.angstroms
epsilon = 0.0 * unit.kilocalories_per_mole
force.addException(0, 1, chargeProd, sigma, epsilon)

# Add constraint.
distance = 0.0 * unit.angstroms
master_system.addConstraint(0, 1, distance)

# Construct initial positions by minimization.
print "Minimizing to obtain initial positions..."
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
context = openmm.Context(master_system, integrator)
context.setPositions(positions)
openmm.LocalEnergyMinimizer.minimize(context)
state = context.getState(getPositions=True)
initial_positions = state.getPositions(asNumpy=True)
del context, integrator, state

# Run replicates of the simulation.
for replicate in range(nreplicates):
    # Make a new copy of the system.
    print "Making a deep copy of the system..."
    system = copy.deepcopy(master_system)

    # Add a barostat to the system.
    # NOTE: This is added to a new copy of the system to ensure barostat random number seeds are unique.
    print "Adding barostat..."
    barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency)
    system.addForce(barostat)

    # Create integrator
    print "Creating LangevinIntegrator..."
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)

    # Create context.
    print "Creating Context..."
    context = openmm.Context(system, integrator, platform)

    # Set initial conditions.
    print "Setting initial positions..."
    context.setPositions(initial_positions)
    print "Setting initial velocities appropriate for temperature..."
    context.setVelocitiesToTemperature(temperature)

    # Record initial data.
    state = context.getState(getEnergy=True)
    reduced_volume = state.getPeriodicBoxVolume() / (nparticles * sigma**3)
    reduced_density = 1.0 / reduced_volume
    reduced_potential = state.getPotentialEnergy() / kT
    print "replicate %5d / %5d : initial                 : density %8.3f | potential %8.3f" % (replicate, nreplicates, reduced_density, reduced_potential)

    # Run simulation.
    for iteration in range(niterations):
        # Integrate the simulation.
        integrator.step(nsteps_per_iteration)

        # Record data.
        state = context.getState(getEnergy=True)
        reduced_volume = state.getPeriodicBoxVolume() / (nparticles * sigma**3)
        reduced_density = 1.0 / reduced_volume
        reduced_potential = state.getPotentialEnergy() / kT

        if ((iteration + 1) % 100) == 0:
            print "replicate %5d / %5d : iteration %5d / %5d : density %8.3f | potential %8.3f" % (replicate, nreplicates, iteration+1, niterations, reduced_density, reduced_potential)

    # Clean up.
    del context, integrator

    print ""



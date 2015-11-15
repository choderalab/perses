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
from functools import partial

################################################################################
# CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

################################################################################
# TESTS
################################################################################

def simulate(system, positions, nsteps=500, timestep=1.0*unit.femtoseconds, temperature=300.0*unit.kelvin, collision_rate=20.0/unit.picoseconds):
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(nsteps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return positions

def check_alchemical_elimination(ncmc_nsteps=50):
    """
    Test alchemical elimination engine on alanine dipeptide null transformation.

    """

    NSIGMA_MAX = 6.0 # number of standard errors away from analytical solution tolerated before Exception is thrown

    # Create an alanine dipeptide null transformation, where N-methyl group is deleted and then inserted.
    from openmmtools import testsystems
    testsystem = testsystems.AlanineDipeptideVacuum()
    from perses.rjmc.topology_proposal import TopologyProposal
    new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    topology_proposal = TopologyProposal(old_system=testsystem.system, old_topology=testsystem.topology, old_positions=testsystem.positions, new_system=testsystem.system, new_topology=testsystem.topology, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())

    # Initialize engine
    from perses.annihilation.ncmc_switching import NCMCEngine
    ncmc_engine = NCMCEngine(topology_proposal, nsteps=ncmc_nsteps)

    niterations = 20 # number of round-trip switching trials
    positions = testsystem.positions
    logP_insert_n = np.zeros([niterations], np.float64)
    logP_delete_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        # Equilibrate
        positions = simulate(testsystem.system, positions)

        # Delete atoms
        [positions, logP_delete] = ncmc_engine.integrate(positions, direction='delete')

        # Insert atoms
        [positions, logP_insert] = ncmc_engine.integrate(positions, direction='insert')

        # Compute total probability
        logP_delete_n[iteration] = logP_delete
        logP_insert_n[iteration] = logP_insert


    # Check free energy difference is withing NSIGMA_MAX standard errors of zero.
    logP_n = logP_delete_n + logP_insert_n
    from pymbar import EXP
    [df, ddf] = EXP(logP_n)
    if (abs(df) > NSIGMA_MAX * ddf):
        msg = 'Delta F (%d steps switching) = %f +- %f kT; should be within %f sigma of 0' % (ncmc_nsteps, df, ddf, NSIGMA_MAX)
        msg += 'delete logP:\n'
        msg += str(logP_delete_n) + '\n'
        msg += 'insert logP:\n'
        msg += str(logP_insert_n) + '\n'
        msg += 'logP:\n'
        msg += str(logP_n) + '\n'
        raise Exception(msg)

def test_alchemical_elimination():
    """
    Check alchemical elimination for alanine dipeptide in vacuum with 0, 1, and 50 switching steps.

    """
    for ncmc_nsteps in [0, 1, 50]:
        f = partial(check_alchemical_elimination, ncmc_nsteps)
        f.description = "Testing alchemical elimination using alanine dipeptide with %d NCMC steps" % ncmc_nsteps
        yield f

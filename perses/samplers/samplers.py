"""
Samplers for perses automated molecular design.

TODO
----
* Determine where `System` object should be stored: In `SamplerState` or in `Thermodynamic State`, or both, or neither?
* Can we create a generalized, extensible `SamplerState` that also stores chemical/thermodynamic state information?
* Can we create a generalized log biasing weight container class that gracefully handles new chemical states that have yet to be explored?

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from openmmtools import testsystems
import copy
import time

from perses.storage import NetCDFStorageView
from perses.samplers import thermodynamics
from perses.tests.utils import quantity_is_finite

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# CONSTANTS
################################################################################

from perses.samplers.thermodynamics import kB

################################################################################
# THERMODYNAMIC STATE
################################################################################

from perses.samplers.thermodynamics import ThermodynamicState

################################################################################
# UTILITY FUNCTIONS
################################################################################

def log_sum_exp(a_n):
    """
    Compute log(sum(exp(a_n)))

    Parameters
    ----------
    a_n : dict of objects : floats

    """
    a_n = np.array(list(a_n.values()))
    return np.log( np.sum( np.exp(a_n - a_n.max() ) ) )

################################################################################
# MCMC sampler state
################################################################################

class SamplerState(object):
    """
    Sampler state for MCMC move representing everything that may be allowed to change during
    the simulation.

    Parameters
    ----------
    system : simtk.openmm.System
       Current system specifying force calculations.
    positions : array of simtk.unit.Quantity compatible with nanometers
       Particle positions.
    velocities : optional, array of simtk.unit.Quantity compatible with nanometers/picoseconds
       Particle velocities.
    box_vectors : optional, 3x3 array of simtk.unit.Quantity compatible with nanometers
       Current box vectors.

    Fields
    ------
    system : simtk.openmm.System
       Current system specifying force calculations.
    positions : array of simtk.unit.Quantity compatible with nanometers
       Particle positions.
    velocities : optional, array of simtk.unit.Quantity compatible with nanometers/picoseconds
       Particle velocities.
    box_vectors : optional, 3x3 array of simtk.unit.Quantity compatible with nanometers
       Current box vectors.
    potential_energy : optional, simtk.unit.Quantity compatible with kilocalories_per_mole
       Current potential energy.
    kinetic_energy : optional, simtk.unit.Quantity compatible with kilocalories_per_mole
       Current kinetic energy.
    total_energy : optional, simtk.unit.Quantity compatible with kilocalories_per_mole
       Current total energy.
    platform : optional, simtk.openmm.Platform
       Platform to use for Context creation to initialize sampler state.

    Examples
    --------

    Create a sampler state for a system with box vectors.

    >>> # Create a test system
    >>> test = testsystems.LennardJonesFluid()
    >>> # Create a sampler state manually.
    >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions, box_vectors=box_vectors)

    Create a sampler state for a system without box vectors.

    >>> # Create a test system
    >>> test = testsystems.LennardJonesCluster()
    >>> # Create a sampler state manually.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)

    Notes
    -----
    CMMotionRemover forces are automatically removed from the system.

    TODO
    ----
    * Can we remove the need to create a Context in initializing the sampler state by using the Reference platform and skipping energy calculations?

    """
    def __init__(self, system, positions, velocities=None, box_vectors=None, platform=None):
        assert quantity_is_finite(positions)
        if velocities is not None:
            assert quantity_is_finite(self.velocities)

        self.system = copy.deepcopy(system)

        # Remove CMMotionRemover, since it can cause problems with GHMC and NCMC.
        forces_to_remove = [ force_index for force_index in range(self.system.getNumForces()) if (self.system.getForce(force_index).__class__.__name__ == 'CMMotionRemover') ]
        for force_index in reversed(forces_to_remove):
            system.removeForce(force_index)

        self.positions = positions
        self.velocities = velocities
        self.box_vectors = box_vectors

        # Create Context.
        context = self.createContext(platform=platform)

        # Get state.
        openmm_state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)

        # Populate context.
        self.positions = openmm_state.getPositions(asNumpy=True)
        self.velocities = openmm_state.getVelocities(asNumpy=True)
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=False)
        self.potential_energy = openmm_state.getPotentialEnergy()
        self.kinetic_energy = openmm_state.getKineticEnergy()
        self.total_energy = self.potential_energy + self.kinetic_energy
        self.volume = thermodynamics.volume(self.box_vectors)

        # Clean up.
        del context

    @classmethod
    def createFromContext(cls, context):
        """
        Create an SamplerState object from the information in a current OpenMM Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
           The Context object from which to create a sampler state.

        Returns
        -------
        sampler_state : SamplerState
           The sampler state containing positions, velocities, and box vectors.

        Examples
        --------

        >>> # Create a test system
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(test.system, integrator, platform)
        >>> # Set positions and velocities.
        >>> context.setPositions(test.positions)
        >>> context.setVelocitiesToTemperature(298 * unit.kelvin)
        >>> # Create a sampler state from the Context.
        >>> sampler_state = SamplerState.createFromContext(context)
        >>> # Clean up.
        >>> del context, integrator

        """
        # Get state.
        openmm_state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)

        # Create new object, bypassing init.
        self = SamplerState.__new__(cls)

        # Populate context.
        self.system = copy.deepcopy(context.getSystem())
        self.positions = openmm_state.getPositions(asNumpy=True)
        self.velocities = openmm_state.getVelocities(asNumpy=True)
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)
        self.potential_energy = openmm_state.getPotentialEnergy()
        self.kinetic_energy = openmm_state.getKineticEnergy()
        self.total_energy = self.potential_energy + self.kinetic_energy
        self.volume = thermodynamics.volume(self.box_vectors)

        assert quantity_is_finite(self.positions)
        assert quantity_is_finite(self.velocities)

        return self

    def createContext(self, integrator=None, platform=None, thermodynamic_state=None):
        """
        Create an OpenMM Context object from the current sampler state.

        Parameters
        ----------
        integrator : simtk.openmm.Integrator, optional, default=None
           The integrator to use for Context creation.
           If not specified, a VerletIntegrator with 1 fs timestep is created.
        platform : simtk.openmm.Platform, optional, default=None
           If specified, the Platform to use for context creation.
        thermodynamic_state : ThermodynamicState, optional, default=None
            If a pressure is specified in the thermodynamic state, a barostat will be added
            to periodic systems.

        Returns
        -------
        context : simtk.openmm.Context
           The created OpenMM Context object

        Notes
        -----
        If the selected or default platform fails, the CPU and Reference platforms will be tried, in that order.
        If the system is periodic and has a pressure defined, a MonteCarloBarostat is added.

        Examples
        --------

        Create a context for a system with periodic box vectors.

        >>> # Create a test system
        >>> test = testsystems.LennardJonesFluid()
        >>> # Create a sampler state manually.
        >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
        >>> sampler_state = SamplerState(positions=test.positions, box_vectors=box_vectors, system=test.system)
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
        >>> context = sampler_state.createContext(integrator)
        >>> # Clean up.
        >>> del context

        Create a context for a system without periodic box vectors.

        >>> # Create a test system
        >>> test = testsystems.LennardJonesCluster()
        >>> # Create a sampler state manually.
        >>> sampler_state = SamplerState(positions=test.positions, system=test.system)
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
        >>> context = sampler_state.createContext(integrator)
        >>> # Clean up.
        >>> del context

        TODO
        ----
        * Generalize fallback platform order to [CUDA, OpenCL, CPU, Reference] ordering.

        """

        if not self.system:
            raise Exception("SamplerState must have a 'system' object specified to create a Context")

        # Use a Verlet integrator if none is specified.
        if integrator is None:
            integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)

        # Create a copy of the system
        system = copy.deepcopy(self.system)

        # If thermodynamic state is specified with a pressure, add a barostat.
        if (thermodynamic_state is not None) and (thermodynamic_state.pressure is not None):
            if not system.usesPeriodicBoundaryConditions():
                raise Exception('Specified a pressure but system does not have periodic boundary conditions')
            barostat = openmm.MonteCarloBarostat(thermodynamic_state.pressure, thermodynamic_state.temperature)
            system.addForce(barostat)

        # Create a Context.
        if platform:
            context = openmm.Context(system, integrator, platform)
        else:
            context = openmm.Context(system, integrator)

        # Set box vectors, if specified.
        if (self.box_vectors is not None):
            try:
                # try tuple of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0], self.box_vectors[1], self.box_vectors[2])
            except:
                # try numpy 3x3 matrix of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0,:], self.box_vectors[1,:], self.box_vectors[2,:])

        # Set positions.
        try:
            context.setPositions(self.positions)
        except Exception as e:
            msg = str(e) + '\n'
            msg += "System has %d particles\n" % self.system.getNumParticles()
            msg += "Positions has %d particles\n" % len(self.positions)
            raise Exception(msg)

        # Set velocities, if specified.
        if (self.velocities is not None):
            context.setVelocities(self.velocities)

        return context

    def minimize(self, tolerance=None, maxIterations=None, platform=None):
        """
        Minimize the current configuration.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity compatible with kilocalories_per_mole/anstroms, optional, default = 1*kilocalories_per_mole/anstrom
           Tolerance to use for minimization termination criterion.

        maxIterations : int, optional, default = 100
           Maximum number of iterations to use for minimization.

        platform : simtk.openmm.Platform, optional
           Platform to use for minimization.

        Examples
        --------

        >>> # Create a test system
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Minimize
        >>> sampler_state.minimize()

        """

        if (tolerance is None):
            tolerance = 1.0 * unit.kilocalories_per_mole / unit.angstroms

        if (maxIterations is None):
            maxIterations = 100

        # Use LocalEnergyMinimizer
        from simtk.openmm import LocalEnergyMinimizer
        context = self.createContext(platform=platform)
        logger.debug("LocalEnergyMinimizer: platform is %s" % context.getPlatform().getName())
        logger.debug("Minimizing with tolerance %s and %d max. iterations." % (tolerance, maxIterations))
        LocalEnergyMinimizer.minimize(context, tolerance, maxIterations)

        # Retrieve data.
        sampler_state = SamplerState.createFromContext(context)
        sampler_state.velocities = None # erase velocities since we may change dimensionality
        self.positions = sampler_state.positions
        self.potential_energy = sampler_state.potential_energy
        self.total_energy = sampler_state.total_energy

        del context

        return

    def has_nan(self):
        """Return True if any of the generalized coordinates are nan.

        Notes
        -----

        Currently checks only the positions.
        """
        x = self.positions / unit.nanometers

        if np.any(np.isnan(x)):
            return True
        else:
            return False

################################################################################
# GHMC INTEGRATOR (TEMPORARY UNTIL MOVED TO OPENMMTOOLS)
################################################################################

class GHMCIntegrator(openmm.CustomIntegrator):

    """
    Generalized hybrid Monte Carlo (GHMC) integrator.

    """

    def __init__(self, temperature=298.0 * unit.kelvin, collision_rate=91.0 / unit.picoseconds, timestep=1.0 * unit.femtoseconds):
        """
        Create a generalized hybrid Monte Carlo (GHMC) integrator.

        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           The collision rate.
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           The integration timestep.

        Notes
        -----
        This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
        Metrpolization step to ensure sampling from the appropriate distribution.

        Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
        accepted, respectively.

        TODO
        ----
        * Move initialization of 'sigma' to setting the per-particle variables.
        * Generalize to use MTS inner integrator.

        Examples
        --------

        Create a GHMC integrator.

        >>> temperature = 298.0 * unit.kelvin
        >>> collision_rate = 91.0 / unit.picoseconds
        >>> timestep = 1.0 * unit.femtoseconds
        >>> integrator = GHMCIntegrator(temperature, collision_rate, timestep)

        References
        ----------
        Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
        http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

        """

        # Initialize constants.
        kT = kB * temperature
        gamma = collision_rate

        # Create a new custom integrator.
        super(GHMCIntegrator, self).__init__(timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("kT", kT)  # thermal energy
        self.addGlobalVariable("b", np.exp(-gamma * timestep))  # velocity mixing parameter
        self.addPerDofVariable("sigma", 0) # velocity standard deviation
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("vold", 0)  # old velocities
        self.addPerDofVariable("xold", 0)  # old positions
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("accept", 0)  # accept or reject
        self.addGlobalVariable("naccept", 0)  # number accepted
        self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials
        self.addPerDofVariable("x1", 0)  # position before application of constraints

        #
        # Pre-computation.
        # This only needs to be done once.
        # TODO: Change this to setPerDofVariableByName("sigma", unit.sqrt(kT / mass).value_in_unit_system(unit.md_unit_system))
        #
        self.addComputePerDof("sigma", "sqrt(kT/m)")

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

        #
        # Metropolized symplectic step.
        #
        self.addConstrainPositions()
        self.addConstrainVelocities()

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + v*dt")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        self.addConstrainVelocities()
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        # TODO: Check if accept/reject logic correctly handles nans
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.beginIfBlock("accept != 1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.endBlock()

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

        #
        # Accumulate statistics.
        #
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

################################################################################
# MCMC SAMPLER
################################################################################

class MCMCSampler(object):
    """
    Markov chain Monte Carlo (MCMC) sampler.

    This is a minimal functional implementation placeholder until we can replace this with MCMCSampler from `openmmmcmc`.

    Properties
    ----------
    positions : simtk.unit.Quantity of size [nparticles,3] with units compatible with nanometers
        The current positions.
    iteration : int
        Iterations completed.
    verbose : bool
        If True, verbose output is printed

    References
    ----------
    [1]

    Examples
    --------
    >>> # Create a test system
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298.0*unit.kelvin)
    >>> # Create an MCMC sampler
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state)
    >>> # Turn off verbosity
    >>> sampler.verbose = False
    >>> # Run the sampler
    >>> sampler.run()

    """
    def __init__(self, thermodynamic_state, sampler_state, topology=None, storage=None, integrator_name='GHMC'):
        """
        Create an MCMC sampler.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to simulate
        sampler_state : SamplerState
            The initial sampler state to simulate from.
        topology : simtk.openmm.app.Topology, optional, default=None
            Topology object corresponding to system being simulated (for writing)
        storage : NetCDFStorage, optional, default=None
            Storage layer to use for writing.
        integrator_name : str, optional, default='GHMC'
            Name of the integrator to use for propagation.

        """
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        self.sampler_state = copy.deepcopy(sampler_state)
        self.topology = topology
        self.integrator_name = integrator_name

        self.storage = None
        if storage is not None:
            self.storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # Initialize
        self.iteration = 0
        # For GHMC integrator
        self.collision_rate = 5.0 / unit.picoseconds
        self.timestep = 1.0 * unit.femtoseconds
        self.nsteps = 500 # number of steps per update
        self.verbose = True

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if self.verbose:
            print("." * 80)
            print("MCMC sampler iteration %d" % self.iteration)

        # Create an integrator
        if self.integrator_name == 'GHMC':
            # TODO: Migrate GHMCIntegrator back to openmmtools
            #from openmmtools.integrators import GHMCIntegrator
            integrator = GHMCIntegrator(temperature=self.thermodynamic_state.temperature, collision_rate=self.collision_rate, timestep=self.timestep)
            if self.verbose: print("Taking %d steps of GHMC..." % self.nsteps)
        elif self.integrator_name == 'Langevin':
            from simtk.openmm import LangevinIntegrator
            integrator = LangevinIntegrator(self.thermodynamic_state.temperature, self.collision_rate, self.timestep)
            if self.verbose: print("Taking %d steps of Langevin dynamics..." % self.nsteps)
        else:
            raise Exception("integrator_name '%s' not valid." % (self.integrator_name))

        start_time = time.time()

        # Create a Context
        context = self.sampler_state.createContext(integrator=integrator, thermodynamic_state=self.thermodynamic_state)
        context.setVelocitiesToTemperature(self.thermodynamic_state.temperature)

        if self.verbose:
            # Print platform
            print("Using platform '%s'" % context.getPlatform().getName())

            # DEBUG ENERGIES
            state = context.getState(getEnergy=True,getForces=True)
            kT = kB * self.thermodynamic_state.temperature
            print("potential  = %.3f kT" % (state.getPotentialEnergy() / kT))
            print("kinetic    = %.3f kT" % (state.getKineticEnergy() / kT))
            force_unit = (kT / unit.angstrom)
            force_norm = np.sqrt(np.mean( (state.getForces(asNumpy=True) / force_unit)**2 ))
            print("force norm = %.3f kT/A/dof" % force_norm)

        # Integrate to update sample
        integrator.step(self.nsteps)

        # Recover sampler state from Context
        self.sampler_state = SamplerState.createFromContext(context)
        self.sampler_state.velocities = None # erase velocities since we may change dimensionality next

        # Write positions and box vectors
        if self.storage:
            kT = kB * self.thermodynamic_state.temperature
            self.storage.write_configuration('positions', self.sampler_state.positions, self.topology, iteration=self.iteration)
            self.storage.write_quantity('kinetic_energy', self.sampler_state.kinetic_energy / kT, iteration=self.iteration)
            self.storage.write_quantity('potential_energy', self.sampler_state.potential_energy / kT, iteration=self.iteration)
            self.storage.write_quantity('volume', self.sampler_state.volume / unit.angstroms**3, iteration=self.iteration)

        # Report statistics.
        if self.integrator_name == 'GHMC':
            naccept = integrator.getGlobalVariableByName('naccept')
            fraction_accepted = float(naccept) / float(self.nsteps)
            if self.verbose: print("Accepted %d / %d GHMC steps (%.2f%%)." % (naccept, self.nsteps, fraction_accepted * 100))
            if self.storage: self.storage.write_quantity('fraction_accepted', fraction_accepted, iteration=self.iteration)

        if self.verbose:
            print('Finished integration in %.3f s' % (time.time() - start_time))
            final_energy = context.getState(getEnergy=True).getPotentialEnergy() * self.thermodynamic_state.beta
            print('Final energy is %12.3f kT' % (final_energy))

        del context, integrator

        # TODO: We currently are forced to update the default box vectors in System because we don't propagate them elsewhere in the code
        # so if they change during simulation, we're in trouble.  We should instead have the code use SamplerState throughout, and likely
        # should generalize SamplerState to include additional dynamical variables (like chemical state key?)
        if self.sampler_state.box_vectors is not None:
            self.thermodynamic_state.system.setDefaultPeriodicBoxVectors(*self.sampler_state.box_vectors)
            self.sampler_state.system.setDefaultPeriodicBoxVectors(*self.sampler_state.box_vectors)

        if self.verbose:
            print("." * 80)

        if self.storage: self.storage.sync()

        # Increment iteration count
        self.iteration += 1

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()

################################################################################
# EXPANDED ENSEMBLE SAMPLER
################################################################################

class ExpandedEnsembleSampler(object):
    """
    Method of expanded ensembles sampling engine.

    Properties
    ----------
    sampler : MCMCSampler
        The MCMC sampler used for updating positions.
    proposal_engine : ProposalEngine
        The ProposalEngine to use for proposing new sampler states and topologies.
    system_generator : SystemGenerator
        The SystemGenerator to use for creating System objects following proposals.
    state : hashable object
        The current sampler state. Can be any hashable object.
    states : set of hashable object
        All known states.
    iteration : int
        Iterations completed.
    naccepted : int
        Number of accepted thermodynamic/chemical state changes.
    nrejected : int
        Number of rejected thermodynamic/chemical state changes.
    number_of_state_visits : dict of state_key
        Cumulative counts of visited states.
    verbose : bool
        If True, verbose output is printed.

    References
    ----------
    [1] Lyubartsev AP, Martsinovski AA, Shevkunov SV, and Vorontsov-Velyaminov PN. New approach to Monte Carlo calculation of the free energy: Method of expanded ensembles. JCP 96:1776, 1992
    http://dx.doi.org/10.1063/1.462133

    Examples
    --------
    >>> # Create a test system
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a SystemGenerator and rebuild the System.
    >>> from perses.rjmc.topology_proposal import SystemGenerator
    >>> system_generator = SystemGenerator(['amber99sbildn.xml'], forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None })
    >>> test.system = system_generator.build_system(test.topology)
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298.0*unit.kelvin)
    >>> # Create an MCMC sampler
    >>> mcmc_sampler = MCMCSampler(thermodynamic_state, sampler_state)
    >>> # Turn off verbosity
    >>> mcmc_sampler.verbose = False
    >>> # Create an Expanded Ensemble sampler
    >>> from perses.rjmc.topology_proposal import PointMutationEngine
    >>> from perses.rjmc.geometry import FFAllAngleGeometryEngine
    >>> geometry_engine = FFAllAngleGeometryEngine(metadata={})
    >>> allowed_mutations = [[('2','ALA')],[('2','VAL'),('2','LEU')]]
    >>> proposal_engine = PointMutationEngine(test.topology, system_generator, max_point_mutants=1, chain_id='1', proposal_metadata=None, allowed_mutations=allowed_mutations)
    >>> exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, test.topology, 'ACE-ALA-NME', proposal_engine, geometry_engine)
    >>> # Run the sampler
    >>> exen_sampler.run()

    """
    def __init__(self, sampler, topology, state_key, proposal_engine, geometry_engine, log_weights=None, scheme='ncmc-geometry-ncmc', options=None, platform=None, envname=None, storage=None):
        """
        Create an expanded ensemble sampler.

        p(x,k) \propto \exp[-u_k(x) + g_k]

        where g_k is the log weight.

        Parameters
        ----------
        sampler : MCMCSampler
            MCMCSampler initialized with current SamplerState
        topology : simtk.openmm.app.Topology
            Current topology
        state : hashable object
            Current chemical state
        proposal_engine : ProposalEngine
            ProposalEngine to use for proposing new chemical states
        geometry_engine : GeometryEngine
            GeometryEngine to use for dimension matching
        log_weights : dict of object : float
            Log weights to use for expanded ensemble biases.
        scheme : str, optional, default='ncmc-geometry-ncmc'
            Update scheme. One of ['ncmc-geometry-ncmc', 'geometry-ncmc-geometry', 'geometry-ncmc']
        options : dict, optional, default=dict()
            Options for initializing switching scheme, such as 'timestep', 'nsteps', 'functions' for NCMC
        platform : simtk.openmm.Platform, optional, default=None
            Platform to use for NCMC switching.  If `None`, default (fastest) platform is used.
        storage : NetCDFStorageView, optional, default=None
            If specified, use this storage layer.

        """
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.sampler = sampler
        self.topology = topology
        self.state_key = state_key
        self.proposal_engine = proposal_engine
        self.log_weights = log_weights
        self.scheme = scheme
        if self.log_weights is None: self.log_weights = dict()

        self.storage = None
        if storage is not None:
            self.storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # Initialize
        self.iteration = 0
        option_names = ['timestep', 'nsteps', 'functions']
        if options is None:
            options = dict()
        for option_name in option_names:
            if option_name not in options:
                options[option_name] = None
        if options['nsteps']:
            self._switching_nsteps = options['nsteps']
        else:
            self._switching_nsteps = 0
        if scheme in ['ncmc-geometry-ncmc','geometry-ncmc']:
            from perses.annihilation.ncmc_switching import NCMCEngine
            self.ncmc_engine = NCMCEngine(temperature=self.sampler.thermodynamic_state.temperature, timestep=options['timestep'], nsteps=options['nsteps'], functions=options['functions'], platform=platform, storage=self.storage)
        elif scheme=='geometry-ncmc-geometry':
            from perses.annihilation.ncmc_switching import NCMCHybridEngine
            self.ncmc_engine = NCMCHybridEngine(temperature=self.sampler.thermodynamic_state.temperature, timestep=options['timestep'], nsteps=options['nsteps'], functions=options['functions'], platform=platform)
        else:
            raise Exception("Expanded ensemble state proposal scheme '%s' unsupported" % self.scheme)
        self.geometry_engine = geometry_engine
        self.naccepted = 0
        self.nrejected = 0
        self.number_of_state_visits = dict()
        self.verbose = False
        self.pdbfile = None # if not None, write PDB file
        self.geometry_pdbfile = None # if not None, write PDB file of geometry proposals
        self.accept_everything = False # if True, will accept anything that doesn't lead to NaNs

    @property
    def state_keys(self):
        return self.log_weights.keys()

    def get_log_weight(self, state_key):
        """
        Get the log weight of the specified state.

        Parameters
        ----------
        state_key : hashable object
            The state key (e.g. chemical state key) to look up.

        Returns
        -------
        log_weight : float
            The log weight of the provided state key.

        Note
        ----
        This adds the key to the self.log_weights dict.

        """
        if state_key not in self.log_weights:
            self.log_weights[state_key] = 0.0
        return self.log_weights[state_key]

    def update_positions(self):
        """
        Sample new positions.
        """
        self.sampler.update()

    def update_state(self):
        """
        Sample the thermodynamic state.
        """

        initial_time = time.time()

        # Check that system and topology have same number of atoms.
        old_system = self.sampler.sampler_state.system
        old_topology = self.topology
        old_topology_natoms = sum([1 for atom in old_topology.atoms()]) # number of topology atoms
        old_system_natoms = old_system.getNumParticles()
        if old_topology_natoms != old_system_natoms:
            msg = 'ExpandedEnsembleSampler: topology has %d atoms, while system has %d atoms' % (old_topology_natoms, old_system_natoms)
            raise Exception(msg)

        if self.scheme == 'ncmc-geometry-ncmc':
            if self.verbose: print("Updating chemical state with ncmc-geometry-ncmc scheme...")

            # Propose new chemical state.
            if self.verbose: print("Proposing new topology...")
            [system, topology, positions] = [self.sampler.thermodynamic_state.system, self.topology, self.sampler.sampler_state.positions]
            topology_proposal = self.proposal_engine.propose(system, topology)
            if self.verbose: print("Proposed transformation: %s => %s" % (topology_proposal.old_chemical_state_key, topology_proposal.new_chemical_state_key))

            # Determine state keys
            old_state_key = self.state_key
            new_state_key = topology_proposal.new_chemical_state_key

            # Determine log weight
            old_log_weight = self.get_log_weight(old_state_key)
            new_log_weight = self.get_log_weight(new_state_key)

            if self.verbose: print("Performing NCMC annihilation")
            # Alchemically eliminate atoms being removed.
            [ncmc_old_positions, ncmc_elimination_logp, potential_delete] = self.ncmc_engine.integrate(topology_proposal, positions, direction='delete', iteration=self.iteration)
            # Check that positions are not NaN
            if np.any(np.isnan(ncmc_old_positions)):
                raise Exception("Positions are NaN after NCMC delete with %d steps" % self._switching_nsteps)

            if self.verbose: print("Geometry engine proposal...")
            # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
            initial_time = time.time()
            geometry_old_positions = ncmc_old_positions
            geometry_new_positions, geometry_logp_propose = self.geometry_engine.propose(topology_proposal, geometry_old_positions, self.sampler.thermodynamic_state.beta)
            if self.verbose: print('proposal took %.3f s' % (time.time() - initial_time))

            if self.geometry_pdbfile is not None:
                print("Writing proposed geometry...")
                #self.geometry_pdbfile.write('MODEL     %4d\n' % (self.iteration+1)) # PyMOL doesn't render connectivity correctly this way
                from simtk.openmm.app import PDBFile
                PDBFile.writeFile(topology_proposal.new_topology, geometry_new_positions, file=self.geometry_pdbfile)
                #self.geometry_pdbfile.write('ENDMDL\n')
                self.geometry_pdbfile.flush()

            if self.verbose: print("Geometry engine logP_reverse calculation...")
            initial_time = time.time()
            geometry_logp_reverse = self.geometry_engine.logp_reverse(topology_proposal, geometry_new_positions, geometry_old_positions, self.sampler.thermodynamic_state.beta)
            geometry_logp = geometry_logp_reverse - geometry_logp_propose
            if self.verbose: print('calculation took %.3f s' % (time.time() - initial_time))

            if self.verbose: print("Performing NCMC insertion")
            # Alchemically introduce new atoms.
            initial_time = time.time()
            [ncmc_new_positions, ncmc_introduction_logp, potential_insert] = self.ncmc_engine.integrate(topology_proposal, geometry_new_positions, direction='insert', iteration=self.iteration)
            if self.verbose: print('NCMC took %.3f s' % (time.time() - initial_time))
            # Check that positions are not NaN
            if np.any(np.isnan(ncmc_new_positions)):
                raise Exception("Positions are NaN after NCMC insert with %d steps" % self._switching_nsteps)

            def print_energy_components(topology, system, positions):
                from parmed.openmm import load_topology, energy_decomposition_system
                structure = load_topology(topology, system=system, xyz=positions)
                energies = energy_decomposition_system(structure, system, nrg=unit.kilocalories_per_mole)
                for (name, energy) in energies:
                    print('%40s %12.3f kcal/mol' % (name, energy))

            # Compute change in eliminated potential contribution.
            switch_logp = - (potential_insert - potential_delete)
            if self.verbose:
                print('potential before geometry  : %12.3f kT' % potential_delete)
                print_energy_components(topology_proposal.old_topology, topology_proposal.old_system, geometry_old_positions)
                print('potential after geometry   : %12.3f kT' % potential_insert)
                print_energy_components(topology_proposal.new_topology, topology_proposal.new_system, geometry_new_positions)
                print('  GEOMETRY ENERGY CHANGE   : %+12.3f kT' % (potential_insert - potential_delete))
                print('---------------------------------------------------------')
                print('switch_logp                : %12.3f' % switch_logp)
                print('geometry_logp_propose      : %12.3f' % geometry_logp_propose)
                print('geometry_logp_reverse      : %12.3f' % geometry_logp_reverse)

            # Compute total log acceptance probability, including all components.
            logp_accept = topology_proposal.logp_proposal + geometry_logp + switch_logp + ncmc_elimination_logp + ncmc_introduction_logp + new_log_weight - old_log_weight
            if self.verbose:
                print("logp_accept = %+10.4e [logp_proposal %+10.4e geometry_logp %+10.4e switch_logp %+10.4e ncmc_elimination_logp %+10.4e ncmc_introduction_logp %+10.4e old_log_weight %+10.4e new_log_weight %+10.4e]"
                    % (logp_accept, topology_proposal.logp_proposal, geometry_logp, switch_logp, ncmc_elimination_logp, ncmc_introduction_logp, old_log_weight, new_log_weight))

            # Accept or reject.
            if np.isnan(logp_accept):
                accept = False
                print('logp_accept = NaN')
            else:
                accept = ((logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept)))
                if self.accept_everything:
                    print('accept_everything option is turned on; accepting')
                    accept = True

            if accept:
                self.sampler.thermodynamic_state.system = topology_proposal.new_system
                self.sampler.sampler_state.system = topology_proposal.new_system
                self.topology = topology_proposal.new_topology
                self.sampler.sampler_state.positions = ncmc_new_positions
                self.sampler.topology = topology
                self.state_key = topology_proposal.new_chemical_state_key
                self.naccepted += 1
                if self.verbose: print("    accepted")
            else:
                self.nrejected += 1
                if self.verbose: print("    rejected")

            elapsed_time = time.time() - initial_time

            # Write to storage.
            if self.storage:
                self.storage.write_quantity('logp_ncmc_elimination', ncmc_elimination_logp, iteration=self.iteration)
                self.storage.write_quantity('logp_ncmc_introduction', ncmc_introduction_logp, iteration=self.iteration)
                self.storage.write_quantity('update_state_elapsed_time', elapsed_time, iteration=self.iteration)
                self.storage.write_quantity('logp_switch', switch_logp, iteration=self.iteration)

        elif self.scheme == 'geometry-ncmc':
            raise(NotImplementedError("ExpandedEnsembleSampler scheme 'geometry-ncmc' has not been statistically validated and should not be used."))
            from perses.tests.utils import compute_potential
            if self.verbose: print("Updating chemical state with geometry-ncmc scheme...")
            # Propose new chemical state.
            if self.verbose: print("Proposing new topology...")
            [system, topology, positions] = [self.sampler.thermodynamic_state.system, self.topology, self.sampler.sampler_state.positions]
            topology_proposal = self.proposal_engine.propose(system, topology)
            if self.verbose: print("Proposed transformation: %s => %s" % (topology_proposal.old_chemical_state_key, topology_proposal.new_chemical_state_key))

            # Determine state keys
            old_state_key = self.state_key
            new_state_key = topology_proposal.new_chemical_state_key

            # Determine log weight
            old_log_weight = self.get_log_weight(old_state_key)
            new_log_weight = self.get_log_weight(new_state_key)

            potential_delete = self.sampler.thermodynamic_state.beta * compute_potential(system, positions, platform=self.ncmc_engine.platform)

            if self.verbose: print("Geometry engine proposal...")
            # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
            initial_time = time.time()
            geometry_old_positions = positions
            geometry_new_positions, geometry_logp_propose = self.geometry_engine.propose(topology_proposal, geometry_old_positions, self.sampler.thermodynamic_state.beta)
            if self.verbose: print('proposal took %.3f s' % (time.time() - initial_time))

            if self.geometry_pdbfile is not None:
                print("Writing proposed geometry...")
                #self.geometry_pdbfile.write('MODEL     %4d\n' % (self.iteration+1)) # PyMOL doesn't render connectivity correctly this way
                from simtk.openmm.app import PDBFile
                PDBFile.writeFile(topology_proposal.new_topology, geometry_new_positions, file=self.geometry_pdbfile)
                #self.geometry_pdbfile.write('ENDMDL\n')
                self.geometry_pdbfile.flush()

            if self.verbose: print("Geometry engine logP_reverse calculation...")
            initial_time = time.time()
            geometry_logp_reverse = self.geometry_engine.logp_reverse(topology_proposal, geometry_new_positions, geometry_old_positions, self.sampler.thermodynamic_state.beta)
            geometry_logp = geometry_logp_reverse - geometry_logp_propose
            if self.verbose: print('calculation took %.3f s' % (time.time() - initial_time))

            if self.verbose: print("Performing NCMC insertion")
            # Alchemically introduce new atoms.
            initial_time = time.time()
            [ncmc_new_positions, ncmc_introduction_logp, potential_insert] = self.ncmc_engine.integrate(topology_proposal, geometry_new_positions, direction='insert', iteration=self.iteration)
            if self.verbose: print('NCMC took %.3f s' % (time.time() - initial_time))
            # Check that positions are not NaN
            if np.any(np.isnan(ncmc_new_positions)):
                raise Exception("Positions are NaN after NCMC insert with %d steps" % self._switching_nsteps)

            def print_energy_components(topology, system, positions):
                from parmed.openmm import load_topology, energy_decomposition_system
                structure = load_topology(topology, system=system, xyz=positions)
                energies = energy_decomposition_system(structure, system, nrg=unit.kilocalories_per_mole)
                for (name, energy) in energies:
                    print('%40s %12.3f kcal/mol' % (name, energy))

            # Compute change in eliminated potential contribution.
            switch_logp = - (potential_insert - potential_delete)
            if self.verbose:
                print('potential before geometry  : %12.3f kT' % potential_delete)
                print_energy_components(topology_proposal.old_topology, topology_proposal.old_system, geometry_old_positions)
                print('potential after geometry   : %12.3f kT' % potential_insert)
                print_energy_components(topology_proposal.new_topology, topology_proposal.new_system, geometry_new_positions)
                print('---------------------------------------------------------')
                print('switch_logp                : %12.3f' % switch_logp)
                print('geometry_logp_propose      : %12.3f' % geometry_logp_propose)
                print('geometry_logp_reverse      : %12.3f' % geometry_logp_reverse)

            # Compute total log acceptance probability, including all components.
            logp_accept = topology_proposal.logp_proposal + geometry_logp + switch_logp + ncmc_introduction_logp + new_log_weight - old_log_weight
            if self.verbose:
                print("logp_accept = %+10.4e [logp_proposal %+10.4e geometry_logp %+10.4e switch_logp %+10.4e ncmc_introduction_logp %+10.4e old_log_weight %+10.4e new_log_weight %+10.4e]"
                    % (logp_accept, topology_proposal.logp_proposal, geometry_logp, switch_logp, ncmc_introduction_logp, old_log_weight, new_log_weight))

            # Accept or reject.
            if np.isnan(logp_accept):
                accept = False
                print('logp_accept = NaN')
            else:
                accept = ((logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept)))
                if self.accept_everything:
                    print('accept_everything option is turned on; accepting')
                    accept = True

            if accept:
                self.sampler.thermodynamic_state.system = topology_proposal.new_system
                self.sampler.sampler_state.system = topology_proposal.new_system
                self.topology = topology_proposal.new_topology
                self.sampler.sampler_state.positions = ncmc_new_positions
                self.sampler.topology = topology
                self.state_key = topology_proposal.new_chemical_state_key
                self.naccepted += 1
                if self.verbose: print("    accepted")
            else:
                self.nrejected += 1
                if self.verbose: print("    rejected")

            elapsed_time = time.time() - initial_time

            # Write to storage.
            if self.storage:
                self.storage.write_quantity('logp_ncmc_introduction', ncmc_introduction_logp, iteration=self.iteration)
                self.storage.write_quantity('update_state_elapsed_time', elapsed_time, iteration=self.iteration)
                self.storage.write_quantity('logp_switch', switch_logp, iteration=self.iteration)

        elif self.scheme == 'geometry-ncmc-geometry':
            if self.verbose: print("Updating chemical state with geometry-ncmc-geometry scheme...")

            # DEBUG: Check current topology can be built.
            try:
                self.proposal_engine._system_generator.build_system(self.topology)
            except Exception as e:
                msg = str(e)
                msg += '\n'
                msg += 'ExpandedEnsembleSampler.update_sampler: self.topology before ProposalEngine call cannot be built into a system'
                raise Exception(msg)

            # Propose new chemical state.
            if self.verbose: print("Proposing new topology...")
            [system, topology, positions] = [self.sampler.thermodynamic_state.system, self.topology, self.sampler.sampler_state.positions]
            topology_proposal = self.proposal_engine.propose(system, topology)
            if self.verbose: print("Proposed transformation: %s => %s" % (topology_proposal.old_chemical_state_key, topology_proposal.new_chemical_state_key))

            # Check to make sure no out-of-bounds atoms are present in new_to_old_atom_map
            natoms_old = topology_proposal.old_system.getNumParticles()
            natoms_new = topology_proposal.new_system.getNumParticles()
            if not set(topology_proposal.new_to_old_atom_map.values()).issubset(range(natoms_old)):
                msg = "Some old atoms in TopologyProposal.new_to_old_atom_map are not in span of old atoms (1..%d):\n" % natoms_old
                msg += str(topology_proposal.new_to_old_atom_map)
                raise Exception(msg)
            if not set(topology_proposal.new_to_old_atom_map.keys()).issubset(range(natoms_new)):
                msg = "Some new atoms in TopologyProposal.new_to_old_atom_map are not in span of old atoms (1..%d):\n" % natoms_new
                msg += str(topology_proposal.new_to_old_atom_map)
                raise Exception(msg)

            # Determine state keys
            old_state_key = self.state_key
            new_state_key = topology_proposal.new_chemical_state_key

            # Determine log weight
            old_log_weight = self.get_log_weight(old_state_key)
            new_log_weight = self.get_log_weight(new_state_key)

            if self.verbose: print("Geometry engine proposal...")
            # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
            initial_time = time.time()
            geometry_old_positions = positions
            geometry_new_positions, geometry_logp_propose = self.geometry_engine.propose(topology_proposal, geometry_old_positions, self.sampler.thermodynamic_state.beta)
            if self.verbose: print('proposal took %.3f s' % (time.time() - initial_time))

            if self.geometry_pdbfile is not None:
                print("Writing proposed geometry...")
                from simtk.openmm.app import PDBFile
                PDBFile.writeFile(topology_proposal.new_topology, geometry_new_positions, file=self.geometry_pdbfile)
                self.geometry_pdbfile.flush()

            if self.verbose: print("Performing NCMC switching")
            initial_time = time.time()
            [ncmc_new_positions, ncmc_old_positions, ncmc_logp] = self.ncmc_engine.integrate(topology_proposal, positions, geometry_new_positions)
            if self.verbose: print('NCMC took %.3f s' % (time.time() - initial_time))
            # Check that positions are not NaN
            if np.any(np.isnan(ncmc_new_positions)):
                raise Exception("Positions are NaN after NCMC insert with %d steps" % self._switching_nsteps)

            if self.verbose: print("Geometry engine logP_reverse calculation...")
            initial_time = time.time()
            geometry_logp_reverse = self.geometry_engine.logp_reverse(topology_proposal, ncmc_new_positions, ncmc_old_positions, self.sampler.thermodynamic_state.beta)
            geometry_logp = geometry_logp_reverse - geometry_logp_propose
            # why wtf
            if self.verbose: print('calculation took %.3f s' % (time.time() - initial_time))

            # Compute change in eliminated potential contribution.
            if self.verbose:
                #print('potential before geometry  : %12.3f kT' % potential_delete)
                #print('potential after geometry   : %12.3f kT' % potential_insert)
                print('---------------------------------------------------------')
                print('ncmc_logp                  : %12.3f' % ncmc_logp)
                print('geometry_logp_propose      : %12.3f' % geometry_logp_propose)
                print('geometry_logp_reverse      : %12.3f' % geometry_logp_reverse)

            # Compute total log acceptance probability, including all components.
            logp_accept = topology_proposal.logp_proposal + geometry_logp + ncmc_logp + new_log_weight - old_log_weight
            if self.verbose:
                print("logp_accept = %+10.4e [logp_proposal = %+10.4e, geometry_logp = %+10.4e, ncmc_logp = %+10.4e, old_log_weight = %+10.4e, new_log_weight = %+10.4e]"
                    % (logp_accept, topology_proposal.logp_proposal, geometry_logp, ncmc_logp, old_log_weight, new_log_weight))

            # Accept or reject.
            if np.isnan(logp_accept):
                accept = False
                print('logp_accept = NaN')
            else:
                accept = ((logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept)))
                if self.accept_everything:
                    print('accept_everything option is turned on; accepting')
                    accept = True

            if accept:
                self.sampler.thermodynamic_state.system = topology_proposal.new_system
                self.sampler.sampler_state.system = topology_proposal.new_system
                self.topology = topology_proposal.new_topology
                self.sampler.sampler_state.positions = ncmc_new_positions
                self.state_key = topology_proposal.new_chemical_state_key
                self.naccepted += 1
                if self.verbose: print("    accepted")
            else:
                self.nrejected += 1
                if self.verbose: print("    rejected")
            # Write to storage.
            if self.storage:
                self.storage.write_quantity('logp_ncmc', ncmc_logp, iteration=self.iteration)

        else:
            raise Exception("Expanded ensemble state proposal scheme '%s' unsupported" % self.scheme)

        if self.storage:
            self.storage.write_configuration('positions', self.sampler.sampler_state.positions, self.sampler.topology, iteration=self.iteration)
            self.storage.write_object('state_key', self.state_key, iteration=self.iteration)
            self.storage.write_object('proposed_state_key', topology_proposal.new_chemical_state_key, iteration=self.iteration)
            self.storage.write_quantity('naccepted', self.naccepted, iteration=self.iteration)
            self.storage.write_quantity('nrejected', self.nrejected, iteration=self.iteration)
            self.storage.write_quantity('logp_accept', logp_accept, iteration=self.iteration)
            self.storage.write_quantity('logp_topology_proposal', topology_proposal.logp_proposal, iteration=self.iteration)
            self.storage.write_quantity('logp_geometry', geometry_logp, iteration=self.iteration)
            self.storage.write_quantity('new_log_weight', new_log_weight, iteration=self.iteration)
            self.storage.write_quantity('old_log_weight', old_log_weight, iteration=self.iteration)


        # Update statistics.
        self.update_statistics()

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if self.verbose:
            print("-" * 80)
            print("Expanded Ensemble sampler iteration %8d" % self.iteration)
        self.update_positions()
        self.update_state()
        self.iteration += 1
        if self.verbose:
            print("-" * 80)

        if self.pdbfile is not None:
            print("Writing frame...")
            from simtk.openmm.app import PDBFile
            PDBFile.writeModel(self.topology, self.sampler.sampler_state.positions, self.pdbfile, self.iteration)
            self.pdbfile.flush()

        if self.storage:
            self.storage.sync()

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()

    def update_statistics(self):
        """
        Update sampler statistics.
        """
        if self.state_key not in self.number_of_state_visits:
            self.number_of_state_visits[self.state_key] = 0
        self.number_of_state_visits[self.state_key] += 1

################################################################################
# SAMS SAMPLER
################################################################################

class SAMSSampler(object):
    """
    Self-adjusted mixture sampling engine.

    Properties
    ----------
    state_keys : set of objects
        The names of states sampled by the sampler.
    logZ : dict() of keys : float
        logZ[key] is the log partition function (up to an additive constant) estimate for chemical state `key`
    update_method : str
        Update method.  One of ['default']
    iteration : int
        Iterations completed.
    verbose : bool
        If True, verbose debug output is printed.

    References
    ----------
    [1] Tan, Z. (2015) Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics, to appear. (Supplement)
    http://www.stat.rutgers.edu/home/ztan/Publication/SAMS_redo4.pdf

    Examples
    --------
    >>> # Create a test system
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a SystemGenerator and rebuild the System.
    >>> from perses.rjmc.topology_proposal import SystemGenerator
    >>> system_generator = SystemGenerator(['amber99sbildn.xml'], forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None })
    >>> test.system = system_generator.build_system(test.topology)
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298.0*unit.kelvin)
    >>> # Create an MCMC sampler
    >>> mcmc_sampler = MCMCSampler(thermodynamic_state, sampler_state)
    >>> # Turn off verbosity
    >>> mcmc_sampler.verbose = False
    >>> from perses.rjmc.geometry import FFAllAngleGeometryEngine
    >>> geometry_engine = FFAllAngleGeometryEngine(metadata={})
    >>> # Create an Expanded Ensemble sampler
    >>> from perses.rjmc.topology_proposal import PointMutationEngine
    >>> allowed_mutations = [[('2','ALA')],[('2','VAL'),('2','LEU')]]
    >>> proposal_engine = PointMutationEngine(test.topology, system_generator, max_point_mutants=1, chain_id='1', proposal_metadata=None, allowed_mutations=allowed_mutations)
    >>> exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, test.topology, 'ACE-ALA-NME', proposal_engine, geometry_engine)
    >>> # Create a SAMS sampler
    >>> sams_sampler = SAMSSampler(exen_sampler)
    >>> # Run the sampler
    >>> sams_sampler.run() # doctest: +ELLIPSIS
    ...
    """
    def __init__(self, sampler, logZ=None, log_target_probabilities=None, update_method='two-stage', storage=None):
        """
        Create a SAMS Sampler.

        Parameters
        ----------
        sampler : ExpandedEnsembleSampler
            The expanded ensemble sampler used to sample both configurations and discrete thermodynamic states.
        logZ : dict of key : float, optional, default=None
            If specified, the log partition functions for each state will be initialized to the specified dictionary.
        log_target_probabilities : dict of key : float, optional, default=None
            If specified, unnormalized target probabilities; default is all 0.
        update_method : str, optional, default='default'
            SAMS update algorithm
        storage : NetCDFStorageView, optional, default=None

        """
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.sampler = sampler
        if logZ is not None:
            self.logZ = logZ
        else:
            self.logZ = dict()
        if log_target_probabilities is not None:
            self.log_target_probabilities = log_target_probabilities
        else:
            self.log_target_probabilities = dict()
        self.update_method = update_method

        self.storage = None
        if storage is not None:
            self.storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # Initialize.
        self.iteration = 0
        self.verbose = False

    @property
    def state_keys(self):
        return self.logZ.keys()

    def update_sampler(self):
        """
        Update the underlying expanded ensembles sampler.
        """
        self.sampler.update()

    def update_logZ_estimates(self):
        """
        Update the logZ estimates according to self.update_method.
        """
        state_key = self.sampler.state_key

        # Add state key to dictionaries if we haven't visited this state before.
        if state_key not in self.logZ:
            self.logZ[state_key] = 0.0
        if state_key not in self.log_target_probabilities:
            self.log_target_probabilities[state_key] = 0.0

        # Update estimates of logZ.
        if self.update_method == 'one-stage':
            # Based on Eq. 9 of Ref. [1]
            gamma = 1.0 / float(self.iteration+1)
        elif self.update_method == 'two-stage':
            # Keep gamma large until second stage is activated.
            if not hasattr(self, 'second_stage_start') or (self.iteration < self.second_stage_start):
                # First stage.
                gamma = 1.0
                # TODO: Determine when to switch to second stage
            else:
                # Second stage.
                gamma = 1.0 / float(self.iteration - self.second_stage_start + 1)
        else:
            raise Exception("SAMS update method '%s' unknown." % self.update_method)
        self.logZ[state_key] += gamma / np.exp(self.log_target_probabilities[state_key])

        # Update log weights for sampler.
        self.sampler.log_weights = { state_key : - self.logZ[state_key] for state_key in self.logZ.keys() }

        if self.storage:
            self.storage.write_object('logZ', self.logZ, iteration=self.iteration)
            self.storage.write_object('log_weights', self.sampler.log_weights, iteration=self.iteration)

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if self.verbose:
            print("=" * 80)
            print("SAMS sampler iteration %5d" % self.iteration)
        self.update_sampler()
        self.update_logZ_estimates()
        if self.storage: self.storage.sync()
        self.iteration += 1
        if self.verbose:
            print("=" * 80)

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()

################################################################################
# MULTITARGET OPTIMIZATION SAMPLER
################################################################################

class MultiTargetDesign(object):
    """
    Multi-objective design using self-adjusted mixture sampling with additional recursion steps
    that update target weights on the fly.

    Parameters
    ----------
    samplers : list of SAMSSampler
        The SAMS samplers whose relative partition functions go into the design objective computation.
    sampler_exponents : dict of SAMSSampler : float
        samplers.keys() are the samplers, and samplers[key]
    log_target_probabilities : dict of hashable object : float
        log_target_probabilities[key] is the computed log objective function (target probability) for chemical state `key`
    verbose : bool
        If True, verbose output is printed.

    """
    def __init__(self, target_samplers, storage=None, verbose=False):
        """
        Initialize a multi-objective design sampler with the specified target sampler powers.

        Parameters
        ----------
        target_samplers : dict
            target_samplers[sampler] is the exponent associated with SAMS sampler `sampler` in the multi-objective design.
        storage : NetCDFStorage, optional, default=None
            If specified, will use the storage layer to write trajectory data.
        verbose : bool, optional, default=False
            If true, will print verbose output

        The target sampler weights for N samplers with specified exponents \alpha_n are given by

        \pi_{nk} \propto \prod_{n=1}^N Z_{nk}^{alpha_n}

        where \pi_{nk} is the target weight for sampler n state k,
        and Z_{nk} is the relative partition function of sampler n among states k.

        Examples
        --------
        Set up a mutation sampler to maximize implicit solvent hydration free energy.
        >>> from perses.tests.testsystems import AlanineDipeptideTestSystem
        >>> testsystem = AlanineDipeptideTestSystem()
        >>> # Set up target samplers.
        >>> target_samplers = { testsystem.sams_samplers['implicit'] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
        >>> # Set up the design sampler.
        >>> designer = MultiTargetDesign(target_samplers)

        """
        # Store target samplers.
        self.sampler_exponents = target_samplers
        self.samplers = list(target_samplers.keys())

        self.storage = None
        if storage is not None:
            self.storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # Initialize storage for target probabilities.
        self.log_target_probabilities = dict()
        self.verbose = verbose
        self.iteration = 0

    @property
    def state_keys(self):
        return self.log_target_probabilities.keys()

    def update_samplers(self):
        """
        Update all samplers.
        """
        for sampler in self.samplers:
            sampler.update()

    def update_target_probabilities(self):
        """
        Update all target probabilities.
        """
        # Gather list of all keys.
        state_keys = set()
        for sampler in self.samplers:
            for key in sampler.state_keys:
                state_keys.add(key)

        # Compute unnormalized log target probabilities.
        log_target_probabilities = { key : 0.0 for key in state_keys }
        for (sampler, log_weight) in self.sampler_exponents.items():
            for key in sampler.state_keys:
                log_target_probabilities[key] += log_weight * sampler.logZ[key]

        # Normalize
        log_sum = log_sum_exp(log_target_probabilities)
        for key in log_target_probabilities:
            log_target_probabilities[key] -= log_sum

        # Store.
        self.log_target_probabilities = log_target_probabilities

        if self.verbose:
            print("log_target_probabilities = %s" % str(self.log_target_probabilities))

        if self.storage:
            self.storage.write_object('log_target_probabilities', self.log_target_probabilities, iteration=self.iteration)

    def update(self):
        """
        Run one iteration of the sampler.
        """
        if self.verbose:
            print("*" * 80)
            print("MultiTargetDesign sampler iteration %8d" % self.iteration)
        self.update_samplers()
        self.update_target_probabilities()
        self.iteration += 1
        if self.storage: self.storage.sync()
        if self.verbose:
            print("*" * 80)

    def run(self, niterations=1):
        """
        Run the multi-target design sampler for the specified number of iterations.

        Parameters
        ----------
        niterations : int
            The number of iterations to run the sampler for.

        """
        # Update all samplers.
        for iteration in range(niterations):
            self.update()

################################################################################
# CONSTANT PH SAMPLER
################################################################################

class ProtonationStateSampler(object):
    """
    Protonation state sampler with given fixed target probabilities for ligand in solvent.

    Parameters
    ----------
    samplers : list of SAMSSampler
        The SAMS samplers whose relative partition functions go into the design objective computation.
    sampler_exponents : dict of SAMSSampler : float
        samplers.keys() are the samplers, and samplers[key]
    log_target_probabilities : dict of hashable object : float
        log_target_probabilities[key] is the computed log objective function (target probability) for chemical state `key`
    verbose : bool
        If True, verbose output is printed.

    """
    def __init__(self, complex_sampler, solvent_sampler, log_state_penalties, storage=None, verbose=False):
        """
        Initialize a protonation state sampler with fixed target probabilities for ligand in solvent.

        Parameters
        ----------
        complex_sampler : ExpandedEnsembleSampler
            Ligand in complex sampler
        solvent_sampler : SAMSSampler
            Ligand in solution sampler
        log_state_penalties : dict
            log_state_penalties[smiles] is the log state free energy (in kT) for ligand state 'smiles'
        storage : NetCDFStorage, optional, default=None
            If specified, will use the storage layer to write trajectory data.
        verbose : bool, optional, default=False
            If true, will print verbose output

        """
        # Store target samplers.
        self.log_state_penalties = log_state_penalties
        self.samplers = [complex_sampler, solvent_sampler]
        self.complex_sampler = complex_sampler
        self.solvent_sampler = solvent_sampler

        self.storage = None
        if storage is not None:
            self.storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # Initialize storage for target probabilities.
        self.log_target_probabilities = { key : - log_state_penalties[key] for key in log_state_penalties }
        self.verbose = verbose
        self.iteration = 0

    @property
    def state_keys(self):
        return self.log_target_probabilities.keys()

    def update_samplers(self):
        """
        Update all samplers.
        """
        for sampler in self.samplers:
            sampler.update()

    def update_target_probabilities(self):
        """
        Update all target probabilities.
        """
        # Update the complex sampler log weights using the solvent sampler log weights
        for key in self.solvent_sampler.state_keys:
            self.complex_sampler.log_weights[key] = self.solvent_sampler.sampler.log_weights[key]

        if self.verbose:
            print("log_weights = %s" % str(self.solvent_sampler.sampler.log_weights))

    def update(self):
        """
        Run one iteration of the sampler.
        """
        if self.verbose:
            print("*" * 80)
            print("ProtonationStateSampler iteration %8d" % self.iteration)
        self.update_samplers()
        self.update_target_probabilities()
        if self.storage: self.storage.sync()
        self.iteration += 1
        if self.verbose:
            print("*" * 80)

    def run(self, niterations=1):
        """
        Run the protonation state sampler for the specified number of iterations.

        Parameters
        ----------
        niterations : int
            The number of iterations to run the sampler for.

        """
        # Update all samplers.
        for iteration in range(niterations):
            self.update()

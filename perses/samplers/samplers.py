"""
Samplers for perses automated molecular design.

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

from perses.samplers import thermodynamics

################################################################################
# CONSTANTS
################################################################################

from thermodynamics import kB

################################################################################
# THERMODYNAMIC STATE
################################################################################

from thermodynamics import ThermodynamicState

#=============================================================================================
# MCMC sampler state
#=============================================================================================

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
    >>> import testsystems
    >>> test = testsystems.LennardJonesFluid()
    >>> # Create a sampler state manually.
    >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions, box_vectors=box_vectors)

    Create a sampler state for a system without box vectors.

    >>> # Create a test system
    >>> import testsystems
    >>> test = testsystems.LennardJonesCluster()
    >>> # Create a sampler state manually.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)

    TODO:
    * Can we remove the need to create a Context in initializing the sampler state by using the Reference platform and skipping energy calculations?

    """
    def __init__(self, system, positions, velocities=None, box_vectors=None, platform=None):
        self.system = copy.deepcopy(system)
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
        >>> import testsystems
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

        return self

    def createContext(self, integrator=None, platform=None):
        """
        Create an OpenMM Context object from the current sampler state.

        Parameters
        ----------
        integrator : simtk.openmm.Integrator, optional, default=None
           The integrator to use for Context creation.
           If not specified, a VerletIntegrator with 1 fs timestep is created.
        platform : simtk.openmm.Platform, optional, default=None
           If specified, the Platform to use for context creation.

        Returns
        -------
        context : simtk.openmm.Context
           The created OpenMM Context object

        Notes
        -----
        If the selected or default platform fails, the CPU and Reference platforms will be tried, in that order.

        Examples
        --------

        Create a context for a system with periodic box vectors.

        >>> # Create a test system
        >>> import testsystems
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
        >>> import testsystems
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

        # Create a Context.
        if platform:
            context = openmm.Context(self.system, integrator, platform)
        else:
            context = openmm.Context(self.system, integrator)

        # Set box vectors, if specified.
        if (self.box_vectors is not None):
            try:
                # try tuple of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0], self.box_vectors[1], self.box_vectors[2])
            except:
                # try numpy 3x3 matrix of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0,:], self.box_vectors[1,:], self.box_vectors[2,:])

        # Set positions.
        context.setPositions(self.positions)

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
        >>> import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Minimize
        >>> sampler_state.minimize()

        """
        timer = Timer()

        if (tolerance is None):
            tolerance = 1.0 * unit.kilocalories_per_mole / unit.angstroms

        if (maxIterations is None):
            maxIterations = 100

        # Use LocalEnergyMinimizer
        from simtk.openmm import LocalEnergyMinimizer
        timer.start("Context creation")
        context = self.createContext(platform=platform)
        logger.debug("LocalEnergyMinimizer: platform is %s" % context.getPlatform().getName())
        logger.debug("Minimizing with tolerance %s and %d max. iterations." % (tolerance, maxIterations))
        timer.stop("Context creation")
        timer.start("LocalEnergyMinimizer minimize")
        LocalEnergyMinimizer.minimize(context, tolerance, maxIterations)
        timer.stop("LocalEnergyMinimizer minimize")

        # Retrieve data.
        sampler_state = SamplerState.createFromContext(context)
        self.positions = sampler_state.positions
        self.potential_energy = sampler_state.potential_energy
        self.total_energy = sampler_state.total_energy

        del context

        timer.report_timing()

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

    References
    ----------
    [1]
    """
    def __init__(self, thermodynamic_state, sampler_state):
        """
        Create an MCMC sampler.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to simulate
        sampler_state : SamplerState
            The initial sampler state to simulate from.

        """
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.thermodynamic_state = thermodynamic_state
        self.sampler_state = sampler_state
        # Initialize
        self.iteration = 0
        # For GHMC integrator
        self.collision_rate = 5.0 / unit.picoseconds
        self.timestep = 1.0 * unit.femtoseconds
        self.nsteps = 500 # number of steps per update

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        from openmmtools.integrators import GHMCIntegrator
        integrator = GHMCIntegrator(temperature=self.thermodynamic_state.temperature, collision_rate=self.collision_rate, timestep=self.timestep)
        context = sampler_state.createContext(integrator=integrator)
        context.setVelocitiesToTemperature(self.thermodynamic_state.temperature)
        integrator.step(self.nsteps)
        self.sampler_state = SamplerState.createFromContext(context)
        del context, integrator

        # Increment iteration count
        self.iteration += 1

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

    References
    ----------
    [1]
    """
    def __init__(self, sampler, topology, state, proposal_engine, log_weights=None):
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
        log_weights : dict of object : float
            Log weights to use for expanded ensemble biases.
        """
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.sampler = sampler
        self.topology = topology
        self.state = state
        self.system_generator = system_generator
        self.proposal_engine = proposal_engine
        # Initialize
        self.thermodynamic_state = mcmc_sampler.thermodynamic_state
        self.keys = set()
        self.log_weight = dict() # log weights for chemical states
        self.update_method = 'default'
        self.iteration = 0

    def update_sampler(self):
        """
        Update the chemical state index.
        """
        proposal = self.proposal_engine.propose(self.sampler.thermodynamic_state.system, self.topology, self.sampler.sampler_state.positions, beta, metadata)

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        self.update_sampler()
        self.update_logZ_estimates()
        self.iteration += 1

################################################################################
# SAMS SAMPLER
################################################################################

class SAMSSampler(object):
    """
    Self-adjusted mixture sampling engine.

    Properties
    ----------
    keys : set of objects
        The names of states sampled by the sampler.
    logZ : dict() of keys : float
        logZ[key] is the log partition function (up to an additive constant) estimate for state `key`
    update_method : str
        Update method.  One of ['default']
    iteration : int
        Iterations completed.

    References
    ----------
    [1] Tan, Z. (2015) Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics, to appear. (Supplement)
    http://www.stat.rutgers.edu/home/ztan/Publication/SAMS_redo4.pdf

    """
    def __init__(self, thermodynamic_state, proposal_engine, topology, positions):
        """
        Create a SAMS Sampler.

        Parameters
        ----------
        thermodynamic_state
        proposal_engine

        """
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.thermodynamic_state = thermodynamic_state
        self.sampler = ExpandedEnsembleSampler(sampler, topology, state, proposal_engine)
        self.system_generator = system_generator
        self.proposal_engine = proposal_engine
        self.topology = topology
        self.positions = positions
        # Initialize.
        self.keys = set()
        self.logZ = dict()
        self.update_method = 'default'
        self.iteration = 0

    def update_sampler(self):
        """
        Update the underlying expanded ensembles sampler.
        """
        self.sampler.update()

    def update_logZ_estimates(self):
        """
        Update the logZ estimates according to self.update_method.
        """
        if self.update_method == 'default':
            # Based on Eq. 9 of Ref. [1]
            gamma = 1.0 / float(self.iteration+1)
            self.logZ[self.sampler.current_state] += gamma / self.target_probability[state]
        else:
            raise Exception("SAMS update method '%s' unknown." % self.update_method)

        # Shift values so logZ_min = 0
        logZ_min = np.min(self.logZ.values())
        for key in self.logZ:
            self.logZ[key] -= logZ_min

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        self.update_sampler()
        self.update_logZ_estimates()
        self.iteration += 1

################################################################################
# MULTITARGET OPTIMIZATION SAMPLER
################################################################################

class MultiTargetDesign(object):
    """
    Multi-objective design using self-adjusted mixture sampling with additional recursion steps
    that update target weights on the fly.

    """
    def __init__(self, target_samplers):
        """
        Initialize a multi-objective design sampler with the specified target sampler powers.

        Parameters
        ----------
        target_samplers : dict
            target_samplers[sampler] is the exponent associated with SAMS sampler `sampler` in the multi-objective design.

        The target sampler weights for N samplers with specified exponents \alpha_n are given by

        \pi_{nk} \propto \prod_{n=1}^N Z_{nk}^{alpha_n}

        where \pi_{nk} is the target weight for sampler n state k,
        and Z_{nk} is the relative partition function of sampler n among states k.

        Examples
        --------
        Set up a mutation sampler to maximize implicit solvent hydration free energy.
        >>> from perses.tests.testsystems import AlanineDipeptideSAMS
        >>> testsystem = AlanineDipeptideSAMS()
        >>> # Set up target samplers.
        >>> target_samplers = { testsystem.sams_samplers['implicit'] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
        >>> # Set up the design sampler.
        >>> designer = MultiTargetDesign(target_samplers)

        """
        # Store target samplers.
        self.target_samplers = target_samplers

        # Initialize storage for target probabilities.
        self.log_target_probabilities = dict()

    def update_samplers(self):
        """
        Update all samplers.
        """
        for sampler in self.target_samplers:
            sampler.update()

    def update_target_probabilities(self):
        """
        Update all target probabilities.
        """
        # Gather list of all keys.
        all_keys = set()
        for sampler in target_samplers:
            for key in sampler.keys:
                all_keys.add(key)

        # Compute unnormalized log target probabilities.
        log_target_probabilities = { key : 0.0 for key in all_keys }
        for (sampler, log_weight) in target_samplers.items():
            for key in sampler.keys:
                log_target_probabilities[key] += log_weight * sampler.logZ[key]

        # Normalize
        log_sum = log_sum_exp(log_target_probabilities)
        for key in log_target_probabiltiies:
            log_target_probabilities[key] -= log_sum

        # Store.
        self.log_target_probabilities = log_target_probabiltiies

    def update(self):
        """
        Run one iteration.
        """
        self.update_samplers()
        self.update_target_probabilities()

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

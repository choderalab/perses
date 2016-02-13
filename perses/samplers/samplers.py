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
    >>> # Run the sampler
    >>> sampler.run()

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
        self.nsteps = 50 # number of steps per update

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        from openmmtools.integrators import GHMCIntegrator
        integrator = GHMCIntegrator(temperature=self.thermodynamic_state.temperature, collision_rate=self.collision_rate, timestep=self.timestep)
        context = self.sampler_state.createContext(integrator=integrator)
        context.setVelocitiesToTemperature(self.thermodynamic_state.temperature)
        integrator.step(self.nsteps)
        self.sampler_state = SamplerState.createFromContext(context)
        del context, integrator

        # TODO: We currently are forced to update the default box vectors in System because we don't propagate them elsewhere in the code
        # so if they change during simulation, we're in trouble.  We should instead have the code use SamplerState throughout, and likely
        # should generalize SamplerState to include additional dynamical variables (like chemical state key?)
        if self.sampler_state.box_vectors is not None:
            self.thermodynamic_state.system.setDefaultPeriodicBoxVectors(*self.sampler_state.box_vectors)

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
    >>> system_generator = SystemGenerator(['amber99sbildn.xml'], forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : app.HBonds })
    >>> test.system = system_generator.build_system(test.topology)
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298.0*unit.kelvin)
    >>> # Create an MCMC sampler
    >>> mcmc_sampler = MCMCSampler(thermodynamic_state, sampler_state)
    >>> # Create an Expanded Ensemble sampler
    >>> from perses.rjmc.topology_proposal import PointMutationEngine
    >>> allowed_mutations = [[('2','ALA')],[('2','VAL'),('2','LEU')]]
    >>> proposal_engine = PointMutationEngine(system_generator, max_point_mutants=1, chain_id='1', proposal_metadata=None, allowed_mutations=allowed_mutations)
    >>> exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, test.topology, 'ACE-ALA-NME', proposal_engine)
    >>> # Run the sampler
    >>> exen_sampler.run()

    """
    def __init__(self, sampler, topology, state_key, proposal_engine, log_weights=None, scheme='ncmc-geometry-ncmc', options=dict()):
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
        scheme : str, optional, default='ncmc-geometry-ncmc'
            Update scheme. One of ['ncmc-geometry-ncmc', 'geometry-ncmc-geometry', 'geometry-ncmc']
        options : dict, optional, default=dict()
            Options for initializing switching scheme, such as 'timestep', 'nsteps', 'functions' for NCMC
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

        # Initialize
        self.iteration = 0
        option_names = ['timestep', 'nsteps', 'functions']
        for option_name in option_names:
            if option_name not in options:
                options[option_name] = None
        from perses.annihilation.ncmc_switching import NCMCEngine
        self.ncmc_engine = NCMCEngine(temperature=self.sampler.thermodynamic_state.temperature, timestep=options['timestep'], nsteps=options['nsteps'], functions=options['functions'])
        from perses.rjmc.geometry import FFAllAngleGeometryEngine
        self.geometry_engine = FFAllAngleGeometryEngine({'data': 0})

    @property
    def state_keys(self):
        return log_weights.keys()

    def update_positions(self):
        """
        Sample new positions.
        """
        self.sampler.update()

    def update_sampler(self):
        """
        Sample the thermodynamic state.
        """
        if self.scheme == 'ncmc-geometry-ncmc':
            # DEBUG: Check current topology can be built.
            try:
                self.proposal_engine._system_generator.build_system(self.topology)
            except Exception as e:
                msg = str(e)
                msg += '\n'
                msg += 'ExpandedEnsembleSampler.update_sampler: self.topology before ProposalEngine call cannot be built into a system'
                raise Exception(msg)

            # Propose new chemical state.
            [system, topology, positions] = [self.sampler.thermodynamic_state.system, self.topology, self.sampler.sampler_state.positions]
            topology_proposal = self.proposal_engine.propose(system, topology)

            # DEBUG: Check current topology can be built.
            try:
                self.proposal_engine._system_generator.build_system(topology_proposal.new_topology)
            except Exception as e:
                msg = str(e)
                msg += '\n'
                msg += 'ExpandedEnsembleSampler.update_sampler: toology_proposal.new_topology before ProposalEngine call cannot be built into a system'
                raise Exception(msg)

            # Determine log weight
            state_key = topology_proposal.new_chemical_state_key
            if state_key not in self.log_weights:
                self.log_weights[state_key] = 0.0
            log_weight = self.log_weights[state_key]

            # Alchemically eliminate atoms being removed.
            [ncmc_old_positions, ncmc_elimination_logp, potential_delete] = self.ncmc_engine.integrate(topology_proposal, positions, direction='delete')
            # Check that positions are not NaN
            if np.any(np.isnan(ncmc_old_positions)):
                raise Exception("Positions are NaN after NCMC delete with %d steps" % switching_nsteps)

            # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
            geometry_new_positions, geometry_logp  = self.geometry_engine.propose(topology_proposal, ncmc_old_positions, self.sampler.thermodynamic_state.beta)

            # Alchemically introduce new atoms.
            [ncmc_new_positions, ncmc_introduction_logp, potential_insert] = self.ncmc_engine.integrate(topology_proposal, geometry_new_positions, direction='insert')
            # Check that positions are not NaN
            if np.any(np.isnan(ncmc_new_positions)):
                raise Exception("Positions are NaN after NCMC insert with %d steps" % switching_nsteps)

            # Compute change in eliminated potential contribution.
            switch_logp = - (potential_insert - potential_delete) / kT

            # Compute total log acceptance probability, including all components.
            logp_accept = top_proposal.logp_proposal + geometry_logp + switch_logp + ncmc_elimination_logp + ncmc_introduction_logp + log_weight - current_log_weight
            print("Proposal from '%12s' -> '%12s' : logp_accept = %+10.4e [logp_proposal %+10.4e geometry_logp %+10.4e switch_logp %+10.4e ncmc_elimination_logp %+10.4e ncmc_introduction_logp %+10.4e log_weight %+10.4e current_log_weight %+10.4e]"
                % (smiles, top_proposal.molecule_smiles, logp_accept, top_proposal.logp_proposal, geometry_logp, switch_logp, ncmc_elimination_logp, ncmc_introduction_logp, log_weight, current_log_weight))

            # Accept or reject.
            accept = ((logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept)))
            if accept:
                self.sampler.thermodynamic_state.system = topology_proposal.new_system
                self.topology = topology_proposal.new_topology
                self.sampler.positions = ncmc_new_positions
                self.state_key = topology_proposal.new_chemical_state_key
                self.naccepted += 1
            else:
                self.nrejected += 1

        else:
            raise Exception("Expanded ensemble state proposal scheme '%s' unsupported" % self.scheme)

        # Update statistics.
        self.stats[self.state_key] += 1

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        self.update_sampler()
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
    >>> system_generator = SystemGenerator(['amber99sbildn.xml'], forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : app.HBonds })
    >>> test.system = system_generator.build_system(test.topology)
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298.0*unit.kelvin)
    >>> # Create an MCMC sampler
    >>> mcmc_sampler = MCMCSampler(thermodynamic_state, sampler_state)
    >>> # Create an Expanded Ensemble sampler
    >>> from perses.rjmc.topology_proposal import PointMutationEngine
    >>> allowed_mutations = [[('2','ALA')],[('2','VAL'),('2','LEU')]]
    >>> proposal_engine = PointMutationEngine(system_generator, max_point_mutants=1, chain_id='1', proposal_metadata=None, allowed_mutations=allowed_mutations)
    >>> exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, test.topology, 'ACE-ALA-NME', proposal_engine)
    >>> # Create a SAMS sampler
    >>> sams_sampler = SAMSSampler(exen_sampler)
    >>> # Run the sampler
    >>> sams_sampler.run()

    """
    def __init__(self, sampler, logZ=None, update_method='default'):
        """
        Create a SAMS Sampler.

        Parameters
        ----------
        sampler : ExpandedEnsembleSampler
            The expanded ensemble sampler used to sample both configurations and discrete thermodynamic states.
        logZ : dict of key : float, optional, default=None
            If specified, the log partition functions for each state will be initialized to the specified dictionary.
        update_method : str, optional, default='default'
            SAMS update algorithm

        """
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.sampler = sampler
        if logZ is not None:
            self.logZ = self.logZ
        else:
            self.logZ = dict()
        self.update_method = update_method

        # Initialize.
        self.iteration = 0

    @property
    def state_keys(self):
        return logZ.keys()

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

        # Initialize storage for target probabilities.
        self.log_target_probabilities = dict()

    @property
    def state_keys(self):
        return log_target_probabilities.keys()

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
        for key in log_target_probabiltiies:
            log_target_probabilities[key] -= log_sum

        # Store.
        self.log_target_probabilities = log_target_probabiltiies

    def update(self):
        """
        Run one iteration of the sampler.
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

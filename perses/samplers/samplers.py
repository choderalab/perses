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
import mdtraj as md
import numpy as np
from openmmtools import testsystems
import copy
import time
from openmmtools.constants import kB
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.mcmc import MCMCSampler

from perses.annihilation.ncmc_switching import NCMCEngine
from perses.dispersed import feptasks
from perses.storage import NetCDFStorageView
from perses.samplers import thermodynamics
from perses.tests.utils import quantity_is_finite

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

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
# EXPANDED ENSEMBLE SAMPLER
################################################################################

class ExpandedEnsembleSampler(object):
    """
    Method of expanded ensembles sampling engine.

    The acceptance criteria is given in the reference document. Roughly, the proposal scheme is:

    * Draw a proposed chemical state k', and calculate reverse proposal probability
    * Conditioned on k' and the current positions x, generate new positions with the GeometryEngine
    * With new positions, jump to a hybrid system at lambda=0
    * Anneal from lambda=0 to lambda=1, accumulating work
    * Jump from the hybrid system at lambda=1 to the k' system, and compute reverse GeometryEngine proposal
    * Add weight of chemical states k and k' to acceptance probabilities

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
    def __init__(self, sampler, topology, state_key, proposal_engine, geometry_engine, log_weights=None, options=None, platform=None, envname=None, storage=None):
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
        self._pressure = sampler.thermodynamic_state.pressure
        self._temperature = sampler.thermodynamic_state.temperature
        self.topology = md.Topology.from_openmm(topology)
        self.state_key = state_key
        self.proposal_engine = proposal_engine
        self.log_weights = log_weights
        if self.log_weights is None: self.log_weights = dict()

        self.storage = None
        if storage is not None:
            self.storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # Initialize
        self.iteration = 0
        option_names = ['timestep', 'nsteps', 'functions', 'nsteps_mcmc', 'splitting']

        if options is None:
            options = dict()
        for option_name in option_names:
            if option_name not in options:
                options[option_name] = None
        
        if options['splitting']:
            self._ncmc_splitting = options['splitting']
        else:
            self._ncmc_splitting = "V R O H R V"

        if options['nsteps']:
            self._switching_nsteps = options['nsteps']
            self.ncmc_engine = NCMCEngine(temperature=self.sampler.thermodynamic_state.temperature, timestep=options['timestep'], nsteps=options['nsteps'], functions=options['functions'], integrator_splitting=self._ncmc_splitting, platform=platform, storage=self.storage)
        else:
            self._switching_nsteps = 0

        if options['nsteps_mcmc']:
            self._n_iterations_per_update = options['nsteps_mcmc']
        else:
            self._n_iterations_per_update = 100

        self.geometry_engine = geometry_engine
        self.naccepted = 0
        self.nrejected = 0
        self.number_of_state_visits = dict()
        self.verbose = False
        self.pdbfile = None # if not None, write PDB file
        self.geometry_pdbfile = None # if not None, write PDB file of geometry proposals
        self.accept_everything = False # if True, will accept anything that doesn't lead to NaNs
        self.logPs = list()
        self.sampler.minimize(max_iterations=40)

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

    def _system_to_thermodynamic_state(self, system):
        """
        Given an OpenMM system object, create a corresponding ThermodynamicState that has the same
        temperature and pressure as the current thermodynamic state.

        Arguments
        ---------
        system : openmm.System
            The OpenMM system for which to create the thermodynamic state
        
        Returns
        -------
        new_thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state object representing the given system
        """
        return ThermodynamicState(system, temperature=self._temperature, pressure=self._pressure)

    def _geometry_forward(self, topology_proposal, old_sampler_state):
        """
        Run geometry engine to propose new positions and compute logP

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        old_sampler_state : openmmtools.states.SamplerState
            Configurational properties of the old system atoms.

        Returns
        -------
        new_sampler_state : openmmtools.states.SamplerState
            Configurational properties of new atoms proposed by geometry engine calculation.
        geometry_logp_propose : float
            The log probability of the forward-only proposal
        """
        if self.verbose: print("Geometry engine proposal...")
        # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
        initial_time = time.time()
        new_positions, geometry_logp_propose = self.geometry_engine.propose(topology_proposal, old_sampler_state.positions, self.sampler.thermodynamic_state.beta)
        if self.verbose: print('proposal took %.3f s' % (time.time() - initial_time))

        if self.geometry_pdbfile is not None:
            print("Writing proposed geometry...")
            from simtk.openmm.app import PDBFile
            PDBFile.writeFile(topology_proposal.new_topology, new_positions, file=self.geometry_pdbfile)
            self.geometry_pdbfile.flush()

        new_sampler_state = SamplerState(new_positions, box_vectors=old_sampler_state.box_vectors)  

        return new_sampler_state, geometry_logp_propose

    def _geometry_reverse(self, topology_proposal, new_sampler_state, old_sampler_state):
        """
        Run geometry engine reverse calculation to determine logP
        of proposing the old positions based on the new positions

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        new_sampler_state : openmmtools.states.SamplerState
            Configurational properties of the new atoms.
        old_sampler_state : openmmtools.states.SamplerState
            Configurational properties of the old atoms.

        Returns
        -------
        geometry_logp_reverse : float
            The log probability of the proposal for the given transformation
        """
        if self.verbose: print("Geometry engine logP_reverse calculation...")
        initial_time = time.time()
        geometry_logp_reverse = self.geometry_engine.logp_reverse(topology_proposal, new_sampler_state.positions, old_sampler_state.positions, self.sampler.thermodynamic_state.beta)
        if self.verbose: print('calculation took %.3f s' % (time.time() - initial_time))
        return geometry_logp_reverse

    def _ncmc_hybrid(self, topology_proposal, old_sampler_state, new_sampler_state):
        """
        Run a hybrid NCMC protocol from lambda = 0 to lambda = 1

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        old_sampler_State : openmmtools.states.SamplerState
            SamplerState of old system at the beginning of NCMCSwitching
        new_sampler_state : openmmtools.states.SamplerState
            SamplerState of new system at the beginning of NCMCSwitching

        Returns
        -------
        old_final_sampler_state : openmmtools.states.SamplerState
            SamplerState of old system at the end of switching
        new_final_sampler_state : openmmtools.states.SamplerState
            SamplerState of new system at the end of switching
        logP_work : float
            The NCMC work contribution to the log acceptance probability (Eq. 44)
        logP_energy : float
            The contribution of switching to and from the hybrid system to the acceptance probability (Eq. 45)
        """
        if self.verbose: print("Performing NCMC switching")
        initial_time = time.time()
        [old_final_sampler_state, new_final_sampler_state, logP_work, logP_energy] = self.ncmc_engine.integrate(topology_proposal, old_sampler_state, new_sampler_state, iteration=self.iteration)
        if self.verbose: print('NCMC took %.3f s' % (time.time() - initial_time))
        # Check that positions are not NaN
        if new_sampler_state.has_nan():
            raise Exception("Positions are NaN after NCMC insert with %d steps" % self._switching_nsteps)
        return old_final_sampler_state, new_final_sampler_state, logP_work, logP_energy

    def _geometry_ncmc_geometry(self, topology_proposal, sampler_state, old_log_weight, new_log_weight):
        """
        Use a hybrid NCMC protocol to switch from the old system to new system
        Will calculate new positions for the new system first, then give both
        sets of positions to the hybrid NCMC integrator, and finally use the
        final positions of the old and new systems to calculate the reverse
        geometry probability

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        sampler_state : openmmtools.states.SamplerState
            Configurational properties of old atoms at the beginning of the NCMC switching.
        old_log_weight : float
            Chemical state weight from SAMSSampler
        new_log_weight : float
            Chemical state weight from SAMSSampler

        Returns
        -------
        logP_accept : float
            Log of acceptance probability of entire Expanded Ensemble switch (Eq. 25 or 46)
        ncmc_new_sampler_state : openmmtools.states.SamplerState
            Configurational properties of new atoms at the end of the NCMC switching.
        """
        if self.verbose: print("Updating chemical state with geometry-ncmc-geometry scheme...")

        from perses.tests.utils import compute_potential

        logP_chemical_proposal = topology_proposal.logp_proposal

        old_thermodynamic_state = self.sampler.thermodynamic_state
        new_thermodynamic_state = self._system_to_thermodynamic_state(topology_proposal.new_system)

        initial_reduced_potential = feptasks.compute_reduced_potential(old_thermodynamic_state, sampler_state)

        new_geometry_sampler_state, logP_geometry_forward = self._geometry_forward(topology_proposal, sampler_state)
        
        #if we aren't doing any switching, then skip running the NCMC engine at all.
        if self._switching_nsteps == 0:
            ncmc_old_sampler_state = sampler_state
            ncmc_new_sampler_state = new_geometry_sampler_state
            logP_work = 0.0
            logP_initial_hybrid = 0.0
            logP_final_hybrid = 0.0
        else:
            ncmc_old_sampler_state, ncmc_new_sampler_state, logP_work, logP_initial_hybrid, logP_final_hybrid = self._ncmc_hybrid(topology_proposal, sampler_state, new_geometry_sampler_state)

        if logP_work > -np.inf and logP_initial_hybrid > -np.inf and logP_final_hybrid > -np.inf:
            logP_geometry_reverse = self._geometry_reverse(topology_proposal, ncmc_new_sampler_state, ncmc_old_sampler_state)
            logP_to_hybrid = logP_initial_hybrid + initial_reduced_potential

            final_reduced_potential = feptasks.compute_reduced_potential(new_thermodynamic_state, ncmc_new_sampler_state)
            logP_from_hybrid = -final_reduced_potential - logP_final_hybrid
            logP_sams_weight = new_log_weight - old_log_weight

            # Compute total log acceptance probability according to Eq. 46
            logP_accept = logP_to_hybrid - logP_geometry_forward + logP_work + logP_from_hybrid + logP_geometry_reverse + logP_sams_weight
        else:
            logP_geometry_reverse = 0.0
            logP_final = 0.0
            logP_to_hybrid = 0.0
            logP_from_hybrid = 0.0
            logP_sams_weight = new_log_weight - old_log_weight
            logP_accept = logP_to_hybrid - logP_geometry_forward + logP_work + logP_from_hybrid + logP_geometry_reverse + logP_sams_weight
            #TODO: mark failed proposals as unproposable

        if self.verbose:
            print("logP_accept = %+10.4e [logP_to_hybrid = %+10.4e, logP_chemical_proposal = %10.4e, logP_reverse = %+10.4e, -logP_forward = %+10.4e, logP_work = %+10.4e, logP_from_hybrid = %+10.4e, logP_sams_weight = %+10.4e]"
                % (logP_accept, logP_to_hybrid, logP_chemical_proposal, logP_geometry_reverse, -logP_geometry_forward, logP_work, logP_from_hybrid, logP_sams_weight))
        # Write to storage.
        if self.storage:
            self.storage.write_quantity('logP_accept', logP_accept, iteration=self.iteration)
            # Write components to storage
            self.storage.write_quantity('logP_ncmc_work', logP_work, iteration=self.iteration)
            self.storage.write_quantity('logP_from_hybrid', logP_from_hybrid, iteration=self.iteration)
            self.storage.write_quantity('logP_to_hybrid', logP_to_hybrid, iteration=self.iteration)
            self.storage.write_quantity('logP_chemical_proposal', logP_chemical_proposal, iteration=self.iteration)
            self.storage.write_quantity('logP_reverse', logP_geometry_reverse, iteration=self.iteration)
            self.storage.write_quantity('logP_forward', logP_geometry_forward, iteration=self.iteration)
            # Write some aggregate statistics to storage to make contributions to acceptance probability easier to analyze
            self.storage.write_quantity('logP_groups_chemical', logP_chemical_proposal, iteration=self.iteration)
            self.storage.write_quantity('logP_groups_geometry', logP_geometry_reverse - logP_geometry_forward, iteration=self.iteration)

        return logP_accept, ncmc_new_sampler_state

    def update_positions(self, n_iterations=1):
        """
        Sample new positions.
        """
        self.sampler.run(n_iterations=n_iterations)

    def update_state(self):
        """
        Sample the thermodynamic state.
        """

        initial_time = time.time()

        # Propose new chemical state.
        if self.verbose: print("Proposing new topology...")
        [system, topology, positions] = [self.sampler.thermodynamic_state.get_system(remove_thermostat=True), self.topology, self.sampler.sampler_state.positions]
        omm_topology = topology.to_openmm() #convert to OpenMM topology for proposal engine
        omm_topology.setPeriodicBoxVectors(self.sampler.sampler_state.box_vectors) #set the box vectors because in OpenMM topology has these...
        topology_proposal = self.proposal_engine.propose(system, omm_topology)
        if self.verbose: print("Proposed transformation: %s => %s" % (topology_proposal.old_chemical_state_key, topology_proposal.new_chemical_state_key))

        # Determine state keys
        old_state_key = self.state_key
        new_state_key = topology_proposal.new_chemical_state_key

        # Determine log weight
        old_log_weight = self.get_log_weight(old_state_key)
        new_log_weight = self.get_log_weight(new_state_key)

        logp_accept, ncmc_new_sampler_state = self._geometry_ncmc_geometry(topology_proposal, self.sampler.sampler_state, old_log_weight, new_log_weight)

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
            self.sampler.thermodynamic_state.set_system(topology_proposal.new_system, fix_state=True)
            self.sampler.sampler_state.system = topology_proposal.new_system
            self.topology = md.Topology.from_openmm(topology_proposal.new_topology)
            self.sampler.sampler_state = ncmc_new_sampler_state
            self.sampler.topology = self.topology
            self.state_key = topology_proposal.new_chemical_state_key
            self.naccepted += 1
            if self.verbose: print("    accepted")
        else:
            self.nrejected += 1
            if self.verbose: print("    rejected")

        if self.storage:
            self.storage.write_configuration('positions', self.sampler.sampler_state.positions, self.topology, iteration=self.iteration)
            self.storage.write_object('state_key', self.state_key, iteration=self.iteration)
            self.storage.write_object('proposed_state_key', topology_proposal.new_chemical_state_key, iteration=self.iteration)
            self.storage.write_quantity('naccepted', self.naccepted, iteration=self.iteration)
            self.storage.write_quantity('nrejected', self.nrejected, iteration=self.iteration)
            self.storage.write_quantity('logp_accept', logp_accept, iteration=self.iteration)
            self.storage.write_quantity('logp_topology_proposal', topology_proposal.logp_proposal, iteration=self.iteration)


        # Update statistics.
        self.update_statistics()

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if self.verbose:
            print("-" * 80)
            print("Expanded Ensemble sampler iteration %8d" % self.iteration)
        self.update_positions(n_iterations=self._n_iterations_per_update)
        self.update_state()
        self.iteration += 1
        if self.verbose:
            print("-" * 80)

        if self.pdbfile is not None:
            print("Writing frame...")
            from simtk.openmm.app import PDBFile
            PDBFile.writeModel(self.topology.to_openmm(), self.sampler.sampler_state.positions, self.pdbfile, self.iteration)
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
    def __init__(self, sampler, logZ=None, log_target_probabilities=None, update_method='two-stage', storage=None, second_stage_start=1000):
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
        second_state_start : int, optional, default None
            At what iteration number to switch to the optimal gain decay

        """
        from scipy.misc import logsumexp
        # Keep copies of initializing arguments.
        # TODO: Make deep copies?
        self.sampler = sampler
        self.chemical_states = None
        self._reference_state = None
        try:
            self.chemical_states = self.sampler.proposal_engine.chemical_state_list
        except NotImplementedError:
            logger.warn("The proposal engine has not properly implemented the chemical state property; SAMS will add states on the fly.")

        if self.chemical_states:
            #Select a reference state that will always be subtracted (ensure that dict ordering does not change)
            self._reference_state = self.chemical_states[0]

            #initialize the logZ dictionary with zeroes for each chemical state
            self.logZ = {chemical_state : 0.0 for chemical_state in self.chemical_states}

            #Initialize log target probabilities with log(1/n_states)
            self.log_target_probabilities = {chemical_state : np.log(len(self.chemical_states)) for chemical_state in self.chemical_states}

            #If initial weights are specified, override any weight with what is provided
            #However, if the chemical state is not in the reachable chemical state list,throw an exception
            if logZ is not None:
                for (chemical_state, logZ_value) in logZ:
                    if chemical_state not in self.chemical_states:
                        raise ValueError("Provided a logZ initial value for an un-proposable chemical state")
                    self.logZ[chemical_state] = logZ_value

            if log_target_probabilities is not None:
                for (chemical_state, log_target_probability) in log_target_probabilities:
                    if chemical_state not in self.chemical_states:
                        raise ValueError("Provided a log target probability for an un-proposable chemical state.")
                    self.log_target_probabilities[chemical_state] = log_target_probability

                #normalize target probabilities
                #this is likely not necessary, but it is copying the algorithm in Ref 1
                log_sum_target_probabilities = logsumexp((list(self.log_target_probabilities.values())))
                self.log_target_probabilities = {chemical_state : log_target_probability - log_sum_target_probabilities for chemical_state, log_target_probability in self.log_target_probabilities}
        else:
            self.logZ = dict()
            self.log_target_probabilities = dict()

        self.update_method = update_method

        self.storage = None
        if storage is not None:
            self.storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

        # Initialize.
        self.iteration = 0
        self.verbose = False

        self.second_stage_start = 0
        if second_stage_start is not None:
            self.second_stage_start = second_stage_start

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
            logger.warn("A new state key is being added to the logZ; note that this makes the resultant algorithm different from SAMS")
            self.logZ[state_key] = 0.0
        if state_key not in self.log_target_probabilities:
            logger.warn("A new state key is being added to the target probabilities; note that this makes the resultant algorithm different from SAMS")
            self.log_target_probabilities[state_key] = 0.0

        # Update estimates of logZ.
        if self.update_method == 'one-stage':
            # Based on Eq. 9 of Ref. [1]
            gamma = 1.0 / float(self.iteration+1)
        elif self.update_method == 'two-stage':
            # Keep gamma large until second stage is activated.
            if self.iteration < self.second_stage_start:
                # First stage.
                gamma = 1.0
                # TODO: Determine when to switch to second stage
            else:
                # Second stage.
                gamma = 1.0 / float(self.iteration - self.second_stage_start + 1)
        else:
            raise Exception("SAMS update method '%s' unknown." % self.update_method)

        #get the (t-1/2) update from equation 9 in ref 1
        self.logZ[state_key] += gamma / np.exp(self.log_target_probabilities[state_key])

        if self._reference_state:
            #the second step of the (t-1/2 update), subtracting the reference state from everything else.
            #we can only do this for cases where all states have been enumerated
            self.logZ = {state_key : logZ_estimate - self.logZ[self._reference_state] for state_key, logZ_estimate in self.logZ.items()}

        # Update log weights for sampler.
        self.sampler.log_weights = { state_key : - self.logZ[state_key] for state_key in self.logZ.keys()}

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

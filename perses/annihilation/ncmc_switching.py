from __future__ import print_function
import numpy as np
import copy
import logging
import traceback
from simtk import openmm, unit
from perses.dispersed.feptasks import NonequilibriumSwitchingMove, compute_reduced_potential
from perses.storage import NetCDFStorageView
from perses.annihilation.new_relative import HybridTopologyFactory
from perses.tests.utils import quantity_is_finite
from openmmtools.constants import kB
from openmmtools.cache import LRUCache, ContextCache
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
from openmmtools.alchemy import AlchemicalState


# make something hyperbolic or something to go from on to off to on
default_hybrid_functions = {
    'lambda_sterics_core' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_sterics_insert' : 'select(step(0.5-lambda), 1.0, 2*lambda)',
    'lambda_sterics_delete' : 'select(step(0.5-lambda), 1.0, 1 - 2.0*(lambda - 0.5))',
    'lambda_electrostatics_insert' : 'select(step(0.5-lambda),2.0*(lambda-0.5),0.0)',
    'lambda_electrostatics_delete' : 'select(step(0.5-lambda), 0.0, 1 - 2.0*lambda)',
    'lambda_bonds' : 'lambda',
    'lambda_angles' : 'lambda',
    'lambda_torsions' : 'lambda'
    }

default_temperature = 300.0*unit.kelvin
default_nsteps = 1
default_timestep = 1.0 * unit.femtoseconds
default_steps_per_propagation = 1
_logger = logging.getLogger("NCMCEngine")

class NaNException(Exception):
    def __init__(self, *args, **kwargs):
        super(NaNException,self).__init__(*args,**kwargs)

class NCMCEngine(object):
    """
    NCMC switching engine

    Examples
    --------

    Create a transformation for an alanine dipeptide test system where the N-methyl group is eliminated.

    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from perses.rjmc.topology_proposal import TopologyProposal
    >>> new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    >>> topology_proposal = TopologyProposal(old_system=testsystem.system, old_topology=testsystem.topology, old_chemical_state_key='AA', new_chemical_state_key='AA', new_system=testsystem.system, new_topology=testsystem.topology, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())
    >>> ncmc_engine = NCMCEngine(temperature=300.0*unit.kelvin, functions=default_functions, nsteps=50, timestep=1.0*unit.femtoseconds)
    >>> positions = testsystem.positions
    >>> [positions, logP_delete, potential_delete] = ncmc_engine.integrate(topology_proposal, positions, direction='delete')
    >>> [positions, logP_insert, potential_insert] = ncmc_engine.integrate(topology_proposal, positions, direction='insert')

    """

    def __init__(self, temperature=default_temperature, functions=None, nsteps=default_nsteps,
                 steps_per_propagation=default_steps_per_propagation, timestep=default_timestep,
                 constraint_tolerance=None, platform=None, write_ncmc_interval=None, measure_shadow_work=False,
                 integrator_splitting='V R O H R V', storage=None, verbose=False, LRUCapacity=10, pressure=None):
        """
        This is the base class for NCMC switching between two different systems.

        Arguments
        ---------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature at which switching is to be run
        functions : dict of str:str, optional, default=default_functions
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int, optional, default=1
            The number of steps to use for switching.
        steps_per_propagation : int, optional, default=1
            The number of intermediate propagation steps taken at each switching step
        timestep : simtk.unit.Quantity with units compatible with femtoseconds, optional, default=1*femtosecond
            The timestep to use for integration of switching velocity Verlet steps.
        constraint_tolerance : float, optional, default=None
            If not None, this relative constraint tolerance is used for position and velocity constraints.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, the platform to use for OpenMM simulations.
        write_ncmc_interval : int, optional, default=None
            If a positive integer is specified, a snapshot frame will be written to storage with the specified interval on NCMC switching.
            'storage' must also be specified.
        measure_shadow_work : bool, optional, default False
            Whether to measure shadow work
        integrator_splitting : str, optional, default='V R O H R V'
            NCMC internal integrator splitting based on OpenMMTools Langevin splittings
        storage : NetCDFStorageView, optional, default=None
            If specified, write data using this class.
        verbose : bool, optional, default=False
            If True, print debug information.
        LRUCapacity : int, default 10
            Capacity of LRU cache for hybrid systems
        pressure : float, default None
            The pressure to use for the simulation. If None, no barostat
        """
        # Handle some defaults.
        if functions == None:
            functions = default_hybrid_functions
        if nsteps == None:
            nsteps = default_nsteps
        if timestep == None:
            timestep = default_timestep
        if temperature == None:
            temperature = default_temperature

        self._temperature = temperature
        self._functions = copy.deepcopy(functions)
        self._nsteps = nsteps
        self._timestep = timestep
        self._constraint_tolerance = constraint_tolerance
        self._platform = platform
        self._integrator_splitting = integrator_splitting
        self._steps_per_propagation = steps_per_propagation
        self._verbose = verbose
        self._pressure = pressure
        self._disable_barostat = False
        self._hybrid_cache = LRUCache(capacity=LRUCapacity)
        self._measure_shadow_work = measure_shadow_work

        self._nattempted = 0

        self._storage = None
        if storage is not None:
            self._storage = NetCDFStorageView(storage, modname=self.__class__.__name__)
            self._save_configuration = True
        else:
            self._save_configuration = False
        if write_ncmc_interval is not None:
            self._write_ncmc_interval = write_ncmc_interval
        else:
            self._write_ncmc_interval = self._nsteps
        self._work_save_interval = write_ncmc_interval

    @property
    def beta(self):
        kT = kB * self._temperature
        beta = 1.0 / kT
        return beta

    def _compute_energy_contribution(self, hybrid_thermodynamic_state, initial_sampler_state, final_sampler_state):
        """
        Compute NCMC energy contribution to log probability.

        See Eqs. 62 and 63 (two-stage) and Eq. 45 (hybrid) of reference document.
        In both cases, the contribution is u(final_positions, final_lambda) - u(initial_positions, initial_lambda).

        Parameters
        ----------
        hybrid_thermodynamic_state : openmmtools.states.CompoundThermodynamicState
            The thermodynamic state of the hybrid sampler.
        initial_sampler_state : openmmtools.states.SamplerState
            The sampler state of the nonalchemical system at the start of the NCMC protocol with box vectors
        final_sampler_state : openmmtools.states.SamplerState
            The sampler state of the nonalchemical system at the end of the NCMC protocol

        Returns
        -------
        logP_energy : float
            The NCMC energy contribution to log probability.
        """
        hybrid_thermodynamic_state.set_alchemical_parameters(0.0)
        initial_reduced_potential = compute_reduced_potential(hybrid_thermodynamic_state, initial_sampler_state)

        hybrid_thermodynamic_state.set_alchemical_parameters(1.0)
        final_reduced_potential = compute_reduced_potential(hybrid_thermodynamic_state, final_sampler_state)

        return final_reduced_potential - initial_reduced_potential

    def _topology_proposal_to_thermodynamic_states(self, topology_proposal):
        """
        Convert a topology proposal to thermodynamic states for the end systems. This will be used to compute the
        "logP_energy" quantity.

        Arguments
        ---------
        topology_proposal : perses.rjmc.TopologyProposal
            topology proposal for whose endpoint systems we want ThermodynamicStates
        
        Returns
        -------
        old_thermodynamic_state : openmmtools.states.ThermodynamicState
            The old system (nonalchemical) thermodynamic state
        new_thermodynamic_state : openmmtools.states.ThermodynamicState
            The new system (nonalchemical) thermodynamic state
        """
        systems = [topology_proposal.old_system, topology_proposal.new_system]
        thermostates = []
        for system in systems:
            thermodynamic_state = ThermodynamicState(system, temperature=self._temperature, pressure=self._pressure)
            thermostates.append(thermodynamic_state)
        
        return thermostates[0], thermostates[1]

    def make_alchemical_system(self, topology_proposal, current_positions, new_positions):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal. This method generates a hybrid system using the new 
        HybridTopologyFactory. It memoizes so that calling multiple times (within a recent time period)
        will immediately return a cached object.

        Arguments
        ---------
        topology_proposal : perses.rjmc.TopologyProposal
            Unmodified real system corresponding to appropriate leg of transformation.
        current_positions : np.ndarray of float
            Positions of "old" system
        new_positions : np.ndarray of float
            Positions of "new" system atoms

        Returns
        -------
        hybrid_factory : perses.annihilation.new_relative.HybridTopologyFactory
            a factory object containing the hybrid system
        """
        try:
            hybrid_factory = self._hybrid_cache[topology_proposal]

            #If we've retrieved the factory from the cache, update it to include the relevant positions
            hybrid_factory._old_positions = current_positions
            hybrid_factory._new_positions = new_positions
            hybrid_factory._compute_hybrid_positions()
        except KeyError:
            try:
                hybrid_factory = HybridTopologyFactory(topology_proposal, current_positions, new_positions)
                self._hybrid_cache[topology_proposal] = hybrid_factory
            except:
                hybrid_factory = None


        return hybrid_factory

    def integrate(self, topology_proposal, initial_sampler_state, proposed_sampler_state, iteration=None):
        """
        Performs NCMC switching to either delete or insert atoms according to the provided `topology_proposal`.

        For `delete`, the system is first modified from fully interacting to alchemically modified, and then NCMC switching is used to eliminate atoms.
        For `insert`, the system begins with eliminated atoms in an alchemically noninteracting form and NCMC switching is used to turn atoms on, followed by making system real.

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        initial_sampler_state : openmmtools.states.SamplerState representing the initial (old) system
            Configurational properties of the atoms at the beginning of the NCMC switching.
        proposed_sampler_state : openmmtools.states.SamplerState representing the proposed (post-geometry new) system
            Configurational properties new system atoms at beginning of NCMC switching
        iteration : int, optional, default=None
            Iteration number, for storage purposes.

        Returns
        -------
        final_old_sampler_state : openmmtools.State.SamplerState
            The final configurational properties of the old system after hybrid alchemical switching
        final_sampler_state : openmmtools.states.SamplerState
            The final configurational properties after `nsteps` steps of alchemical switching, and reversion to the nonalchemical system
        logP_work : float
            The NCMC work contribution to the log acceptance probability (Eqs. 62 and 63)
        logP_energy : float
            The contribution of transforming to and from the hybrid system to the log acceptance probability (Eqs. 62 and 63)
        """

        assert not initial_sampler_state.has_nan() and not proposed_sampler_state.has_nan()

        #generate or retrieve the hybrid topology factory:
        hybrid_factory = self.make_alchemical_system(topology_proposal, initial_sampler_state.positions, proposed_sampler_state.positions)

        if hybrid_factory is None:
            _logger.warning("Unable to construct hybrid system for {} -> {}".format(topology_proposal.old_chemical_state_key, topology_proposal.new_chemical_state_key))
            return initial_sampler_state, proposed_sampler_state, -np.inf, -np.inf


        topology = hybrid_factory.hybrid_topology

        #generate the corresponding thermodynamic and sampler states so that we can use the NonequilibriumSwitchingMove:
        
        #First generate the thermodynamic state:
        hybrid_system = hybrid_factory.hybrid_system
        hybrid_thermodynamic_state = ThermodynamicState(hybrid_system, temperature=self._temperature, pressure=self._pressure)

        #Now create an AlchemicalState from the hybrid system:
        alchemical_state = AlchemicalState.from_system(hybrid_system)
        alchemical_state.set_alchemical_parameters(0.0)

        #Now create a compound thermodynamic state that combines the hybrid thermodynamic state with the alchemical state:
        compound_thermodynamic_state = CompoundThermodynamicState(hybrid_thermodynamic_state, composable_states=[alchemical_state])

        #construct a sampler state from the hybrid positions and the box vectors of the initial sampler state:
        initial_hybrid_positions = hybrid_factory.hybrid_positions
        initial_hybrid_box_vectors = initial_sampler_state.box_vectors

        initial_hybrid_sampler_state = SamplerState(initial_hybrid_positions, box_vectors=initial_hybrid_box_vectors)
        final_hybrid_sampler_state = copy.deepcopy(initial_hybrid_sampler_state)

        #create the nonequilibrium move:
        ne_move = NonequilibriumSwitchingMove(self._functions, self._integrator_splitting, self._temperature, self._nsteps, self._timestep, 
                                              work_save_interval=self._write_ncmc_interval, top=topology,subset_atoms=None,
                                              save_configuration=self._save_configuration, measure_shadow_work=self._measure_shadow_work)
        
        #run the NCMC protocol
        ne_move.apply(compound_thermodynamic_state, final_hybrid_sampler_state)

        #get the total work:
        logP_work = - ne_move.current_total_work

        # Compute contribution of transforming to and from the hybrid system:
        compound_thermodynamic_state.set_alchemical_parameters(0.0)
        initial_reduced_potential = compute_reduced_potential(compound_thermodynamic_state, initial_hybrid_sampler_state)

        compound_thermodynamic_state.set_alchemical_parameters(1.0)
        final_reduced_potential = compute_reduced_potential(compound_thermodynamic_state, final_hybrid_sampler_state)

        #compute the output SamplerState, which has the atoms only for the new system post-NCMC:
        new_positions = hybrid_factory.new_positions(final_hybrid_sampler_state.positions)
        new_box_vectors = final_hybrid_sampler_state.box_vectors
        final_sampler_state = SamplerState(new_positions, box_vectors=new_box_vectors)

        #compute the output SamplerState for the atoms only in the old system (required for geometry_logP_reverse)
        old_positions = hybrid_factory.old_positions(final_hybrid_sampler_state.positions)
        old_box_vectors = copy.deepcopy(new_box_vectors) #these are the same as the new system
        final_old_sampler_state = SamplerState(old_positions, box_vectors=old_box_vectors)

        if self._save_configuration:
            trajectory = ne_move.trajectory.xyz
            topology = ne_move.trajectory.topology
            varname = "ncmcpositions"
            nframes = ne_move.trajectory.n_frames

            for frame in range(nframes):
                self._storage.write_configuration(varname, trajectory[frame, :, :], topology, iteration=iteration, frame=frame, nframes=nframes)

        # Return
        return [final_old_sampler_state, final_sampler_state, logP_work, initial_reduced_potential, final_reduced_potential]

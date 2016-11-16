from __future__ import print_function
import numpy as np
import copy
import logging
import traceback
from simtk import openmm, unit
from openmmtools.integrators import GHMCIntegrator
from perses.storage import NetCDFStorageView
from perses.tests.utils import quantity_is_finite

default_functions = {
    'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
    'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
    'lambda_bonds' : '0.9*lambda + 0.1', # don't fully soften bonds
    'lambda_angles' : '0.9*lambda + 0.1', # don't fully soften angles
    'lambda_torsions' : 'lambda'
    }


functions_disable_all = {
    'lambda_sterics' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_bonds' : 'lambda',
    'lambda_angles' : 'lambda',
    'lambda_torsions' : 'lambda'
    }

default_hybrid_functions = {
    'lambda_sterics' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_bonds' : 'lambda',
    'lambda_angles' : 'lambda',
    'lambda_torsions' : 'lambda'
    }

default_temperature = 300.0*unit.kelvin
default_nsteps = 1
default_timestep = 1.0 * unit.femtoseconds
default_steps_per_propagation = 1

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

    def __init__(self, temperature=default_temperature, functions=None, nsteps=default_nsteps, steps_per_propagation=default_steps_per_propagation, timestep=default_timestep, constraint_tolerance=None, platform=None, write_ncmc_interval=None, integrator_type='GHMC', storage=None, verbose=False):
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
        integrator_type : str, optional, default='GHMC'
            NCMC internal integrator type ['GHMC', 'VV']
        storage : NetCDFStorageView, optional, default=None
            If specified, write data using this class.
        verbose : bool, optional, default=False
            If True, print debug information.
        """
        # Handle some defaults.
        if functions == None:
            functions = default_functions
        if nsteps == None:
            nsteps = default_nsteps
        if timestep == None:
            timestep = default_timestep
        if temperature == None:
            temperature = default_temperature

        self.temperature = temperature
        self.functions = copy.deepcopy(functions)
        self.nsteps = nsteps
        self.timestep = timestep
        self.constraint_tolerance = constraint_tolerance
        self.platform = platform
        self.integrator_type = integrator_type
        self.steps_per_propagation = steps_per_propagation
        self.verbose = verbose

        if steps_per_propagation != 1:
            raise Exception('steps_per_propagation must be 1 until CustomIntegrator is debugged')

        self.nattempted = 0

        self._storage = None
        if storage is not None:
            self._storage = NetCDFStorageView(storage, modname=self.__class__.__name__)
        self.write_ncmc_interval = write_ncmc_interval

    @property
    def beta(self):
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * self.temperature
        beta = 1.0 / kT
        return beta

    def _getAvailableParameters(self, system, prefix='lambda'):
        """
        Return a list of available alchemical context parameters defined in the system

        Parameters
        ----------
        system : simtk.openmm.System
            The system for which available context parameters are to be determined
        prefix : str, optional, default='lambda'
            Prefix required for parameters to be returned.

        Returns
        -------
        parameters : list of str
            The list of available context parameters in the system

        """
        parameters = list()
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if hasattr(force, 'getNumGlobalParameters'):
                for parameter_index in range(force.getNumGlobalParameters()):
                    parameter_name = force.getGlobalParameterName(parameter_index)
                    if parameter_name[0:(len(prefix)+1)] == (prefix + '_'):
                        parameters.append(parameter_name)
        return parameters

    def _computeAlchemicalCorrection(self, integrator, context, system, initial_positions, final_positions, direction):
        """
        Compute log probability for correction from transforming real system to/from alchemical system.

        If direction is `insert`, the contribution at `final_positions` is computed as (real - alchemical).
        If direction is `delete`, the contribution at `initial_positions` is computed as (alchemical - real).

        Parameters
        ----------
        itegrator : NCMCAlchemicalIntegrator subclasses
            NCMC switching integrator to annihilate or introduce particles alchemically.
        context : openmm.Context 
            Alchemical context
        system : simtk.unit.System
            Real fully-interacting system.
        initial_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The positions of the alchemical system at the start of the NCMC protocol
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The positions of the alchemical system at the end of the NCMC protocol
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])

        Returns
        -------
        switch_logp : float
            beta * the potential of the alchemically annhiliated conformation
            The difference between switch_logp of the insert and delete operations
            represents the log acceptance probability of the switch between
            the old and new chemical states
        logP_alchemical_correction : float
            The log acceptance probability of the switch between physical
            and alchemical systems of the same chemical state
        """
        # Get initial and final real and alchemical potentials
        from perses.tests.utils import compute_potential
        initial_alchemical_potential = self.beta * integrator.getGlobalVariableByName("Einitial") * unit.kilojoules_per_mole
        final_alchemical_potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        if direction == 'insert':
            final_unmodified_potential = self.beta * compute_potential(system, final_positions, platform=self.platform)
            logP_alchemical_correction = final_unmodified_potential - final_alchemical_potential
            switch_logp = initial_alchemical_potential
        elif direction == 'delete':
            initial_unmodified_potential = self.beta * compute_potential(system, initial_positions, platform=self.platform)
            logP_alchemical_correction = initial_alchemical_potential - initial_unmodified_potential
            switch_logp = final_alchemical_potential

        # Check potentials are finite
        if np.isnan(initial_alchemical_potential) or np.isnan(final_alchemical_potential):
            msg = "A required potential of %s operation is NaN:\n" % direction
            msg += "initial_alchemical_potential: %.3f kT\n" % initial_alchemical_potential
            msg += "final_alchemical_potential: %.3f kT\n" % final_alchemical_potential
            raise NaNException(msg)

        return switch_logp, logP_alchemical_correction

    def _choose_system_from_direction(self, topology_proposal, direction):
        """
        Based on the direction, return a topology, indices of alchemical
        atoms, and system which relate to the chemical state being modified.

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])

        Returns
        -------
        topology : openmm.app.Topology
            Alchemical topology being modified
        indices : list(int)
            List of the indices of atoms that are turned on / off
        unmodified_system : simtk.openmm.System
            Unmodified real system corresponding to appropriate leg of transformation.
        """
        # Select reference topology, indices, and system based on whether we are deleting or inserting.
        if direction == 'delete':
            return topology_proposal.old_topology, topology_proposal.unique_old_atoms, topology_proposal.old_system
        elif direction == 'insert':
            return topology_proposal.new_topology, topology_proposal.unique_new_atoms, topology_proposal.new_system

    def make_alchemical_system(self, unmodified_system, alchemical_atoms, direction='insert'):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal

        Arguments
        ---------
        unmodified_system : simtk.openmm.System
            Unmodified real system corresponding to appropriate leg of transformation.
        alchemical_atoms : list(int)
            List of the indices of atoms that are turned on / off
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])

        Returns
        -------
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        """
        # Create an alchemical factory.
        from alchemy import AbsoluteAlchemicalFactory
        alchemical_factory = AbsoluteAlchemicalFactory(unmodified_system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=True, alchemical_torsions=True, alchemical_bonds=True, alchemical_angles=True, softcore_beta=0.0)

        # Return the alchemically-modified system in fully-interacting form.
        alchemical_system = alchemical_factory.createPerturbedSystem()
        return alchemical_system

    def _integrate_switching(self, integrator, context, topology, indices, iteration, direction):
        """
        Runs `self.nsteps` integrator steps

        For `delete`, lambda will go from 1 to 0
        For `insert`, lambda will go from 0 to 1

        Parameters
        ----------
        itegrator : NCMCAlchemicalIntegrator subclasses
            NCMC switching integrator to annihilate or introduce particles alchemically.
        context : openmm.Context 
            Alchemical context
        topology : openmm.app.Topology
            Alchemical topology being modified
        indices : list(int)
            List of the indices of atoms that are turned on / off
        iteration : int or None
            Iteration number, for storage purposes.
        direction : str
            Direction of alchemical switching:
                'insert' causes lambda to switch from 0 to 1 over nsteps steps of integration
                'delete' causes lambda to switch from 1 to 0 over nsteps steps of integration

        Returns
        -------
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The final positions after `nsteps` steps of alchemical switching
        logP_NCMC : float
            The log acceptance probability of the NCMC moves
        """
        # Integrate switching
        try:
            # Write atom indices that are changing.
            if self._storage:
                self._storage.write_object('atomindices', indices, iteration=iteration)

            # Allocate storage for work.
            work = np.zeros([self.nsteps+1], np.float64) # work[n] is the accumulated work up to step n

            # Write trajectory frame.
            if self._storage and self.write_ncmc_interval:
                positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                self._storage.write_configuration('positions', positions, topology, iteration=iteration, frame=0, nframes=(nsteps+1))

            # Perform NCMC integration.
            for step in range(self.nsteps):
                # Take a step. 
                integrator.step(1)

                # Store accumulated work
                work[step+1] = integrator.getWork(context)

                # Write trajectory frame.
                if self._storage and self.write_ncmc_interval and (self.write_ncmc_interval % (step+1) == 0):
                    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                    assert quantity_is_finite(positions) == True
                    self._storage.write_configuration('positions', positions, topology, iteration=iteration, frame=(step+1), nframes=(nsteps+1))

            # Store work values.
            if self._storage:
                self._storage.write_array('work_%s' % direction, work, iteration=iteration)

        except Exception as e:
            # Trap NaNs as a special exception (allowing us to reject later, if desired)
            if str(e) == "Particle coordinate is nan":
                msg = "Particle coordinate is nan during NCMC integration while using integrator_type '%s'" % self.integrator_type
                if self.integrator_type == 'GHMC':
                    msg += '\n'
                    msg += 'This should NEVER HAPPEN with GHMC!'
                raise NaNException(msg)
            else:
                traceback.print_exc()
                raise e

        # Store final positions and log acceptance probability.
        final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        assert quantity_is_finite(final_positions) == True
        logP_NCMC = integrator.getLogAcceptanceProbability(context)
        return final_positions, logP_NCMC

    def _zero_steps_return(self, initial_positions, system):
        """
        Handle the special case of instantaneous insertion / deletion

        Parameters
        ----------
        initial_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms at the beginning of the NCMC switching.
        system : simtk.openmm.System
            The system with appropriate atoms alchemically modified

        Returns
        -------
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The final positions are equivalent to the initial positions  after 0 steps of alchemical switching
        logP : float
            The log acceptance probability is 0.0
        potential : simtk.unit.Quantity with units compatible with kilocalories_per_mole
            The potential energy of the alchemically eliminated conformation.
        """
        # TODO: Check this is correct.
        # TODO: Can we simplify this so there are not two code branches here?
        logP = 0.0
        final_positions = copy.deepcopy(initial_positions)
        from perses.tests.utils import compute_potential
        potential = self.beta * compute_potential(system, initial_positions, platform=self.platform)
        return [final_positions, logP, potential]

    def _choose_integrator(self, alchemical_system, functions, direction):
        """
        Instantiate the appropriate type of NCMC integrator, setting
        constraint tolerance if specified.

        Parameters
        ----------
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        functions : dict
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        direction : str
            Direction of alchemical switching:
                'insert' causes lambda to switch from 0 to 1 over nsteps steps of integration
                'delete' causes lambda to switch from 1 to 0 over nsteps steps of integration

        Returns
        -------
        integrator : simtk.openmm.CustomIntegrator
            NCMC switching integrator to annihilate or introduce particles alchemically.
        """
        # Create an NCMC velocity Verlet integrator.
        if self.integrator_type == 'VV':
            integrator = NCMCVVAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, steps_per_propagation=self.steps_per_propagation, timestep=self.timestep, direction=direction)
        elif self.integrator_type == 'GHMC':
            integrator = NCMCGHMCAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, steps_per_propagation=self.steps_per_propagation, timestep=self.timestep, direction=direction)
        else:
            raise Exception("integrator_type '%s' unknown" % self.integrator_type)

        # Set the constraint tolerance if specified.
        if self.constraint_tolerance is not None:
            integrator.setConstraintTolerance(self.constraint_tolerance)

        return integrator

    def _create_context(self, system, integrator, positions):
        """
        Instantiate context for alchemical system.

        Parameters
        ----------
        system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        itegrator : NCMCAlchemicalIntegrator subclasses
            NCMC switching integrator to annihilate or introduce particles alchemically.
        positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms at the beginning of the NCMC switching.

        Returns
        -------
        context : openmm.Context 
            Alchemical context
        """
        # Create a context on the specified platform.
        if self.platform is not None:
            context = openmm.Context(system, integrator, self.platform)
        else:
            context = openmm.Context(system, integrator)
        context.setPositions(positions)
        context.applyConstraints(integrator.getConstraintTolerance())
        # Set velocities to temperature and apply velocity constraints.
        context.setVelocitiesToTemperature(self.temperature)
        context.applyVelocityConstraints(integrator.getConstraintTolerance())
        return context

    def _get_functions(self, system):
        """
        Select subset of switching functions based on which alchemical parameters are present in the system.

        Parameters
        ----------
        system : simtk.openmm.System
            The system with appropriate atoms alchemically modified

        Returns
        -------
        functions : dict
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        """
        available_parameters = self._getAvailableParameters(system)
        functions = { parameter_name : self.functions[parameter_name] for parameter_name in self.functions if (parameter_name in available_parameters) }
        return functions

    def _clean_up_integration(self, logP_NCMC, logP_alchemical_correction, alchemical_system, context, integrator):
        """
        The final steps of the NCMC protocol are to incorporate the alchemical
        correction into the log acceptance probability of the ncmc integrator,
        delete the alchemical system, context and integrator, and increase
        the counter of number of NCMC attempts.

        Parameters
        ----------
        logP_NCMC : float
            The log acceptance probability of the NCMC moves
        logP_alchemical_correction : float
            The log acceptance probability of the switch between physical
            and alchemical systems of the same chemical state
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        context : openmm.Context 
            Alchemical context
        itegrator : NCMCAlchemicalIntegrator subclasses
            NCMC switching integrator to annihilate or introduce particles alchemically.

        Returns
        -------
        logP_ncmc : float
            The log acceptance probability of the switch
        """
        # Compute total logP
        logP_ncmc = logP_NCMC + logP_alchemical_correction

        # Clean up alchemical system.
        del alchemical_system, context, integrator

        # Keep track of statistics.
        self.nattempted += 1
        return logP_ncmc

    def integrate(self, topology_proposal, initial_positions, direction='insert', platform=None, iteration=None):
        """
        Performs NCMC switching to either delete or insert atoms according to the provided `topology_proposal`.

        For `delete`, the system is first modified from fully interacting to alchemically modified, and then NCMC switching is used to eliminate atoms.
        For `insert`, the system begins with eliminated atoms in an alchemically noninteracting form and NCMC switching is used to turn atoms on, followed by making system real.
        The contribution of transforming the real system to/from an alchemical system is included.

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        initial_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms at the beginning of the NCMC switching.
        direction : str, optional, default='insert'
            Direction of alchemical switching:
                'insert' causes lambda to switch from 0 to 1 over nsteps steps of integration
                'delete' causes lambda to switch from 1 to 0 over nsteps steps of integration
        platform : simtk.openmm.Platform, optional, default=None
            If not None, this platform is used for integration.
        iteration : int, optional, default=None
            Iteration number, for storage purposes.

        Returns
        -------
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The final positions after `nsteps` steps of alchemical switching
        logP : float
            The log acceptance probability of the switch
        potential : simtk.unit.Quantity with units compatible with kilocalories_per_mole
            For `delete`, `beta` * the potential energy of the final (alchemically eliminated) conformation.
            For `insert`, `beta` * the potential energy of the initial (alchemically eliminated) conformation.

        """
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        assert quantity_is_finite(initial_positions) == True

        topology, indices, system = self._choose_system_from_direction(topology_proposal, direction)

        # Handle special case of instantaneous insertion/deletion.
        if (self.nsteps == 0):
            return self._zero_steps_return(initial_positions, system)

        # Create alchemical system.
        alchemical_system = self.make_alchemical_system(system, indices, direction=direction)

        functions = self._get_functions(alchemical_system)
        integrator = self._choose_integrator(alchemical_system, functions, direction)
        context = self._create_context(alchemical_system, integrator, initial_positions)

        # Integrate switching
        final_positions, logP_NCMC = self._integrate_switching(integrator, context, topology, indices, iteration, direction)

        switch_logp, logP_alchemical_correction = self._computeAlchemicalCorrection(integrator, context, system, initial_positions, final_positions, direction)

        logP_ncmc = self._clean_up_integration(logP_NCMC, logP_alchemical_correction, alchemical_system, context, integrator)

        # Return
        return [final_positions, logP_ncmc, switch_logp]

class NCMCHybridEngine(NCMCEngine):
    """
    NCMC switching engine which switches directly from old to new systems
    via a hybrid alchemical topology

    Examples
    --------
    ## EXAMPLE UNCHANGED FROM BASE CLASS ##
    Create a transformation for an alanine dipeptide test system where the N-methyl group is eliminated.
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from perses.rjmc.topology_proposal import TopologyProposal
    >>> new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    >>> topology_proposal = TopologyProposal(old_system=testsystem.system, old_topology=testsystem.topology, old_chemical_state_key='AA', new_chemical_state_key='AA', new_system=testsystem.system, new_topology=testsystem.topology, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())
    >>> ncmc_engine = NCMCHybridEngine(temperature=300.0*unit.kelvin, functions=default_functions, nsteps=50, timestep=1.0*unit.femtoseconds)

    positions = testsystem.positions
    (need a geometry proposal in here now)
    [positions, new_old_positions, logP_insert, potential_insert] = ncmc_engine.integrate(topology_proposal, positions, proposed_positions)
    """

    def __init__(self, temperature=default_temperature, functions=None, 
                 nsteps=default_nsteps, timestep=default_timestep, 
                 constraint_tolerance=None, platform=None, 
                 write_ncmc_interval=None, integrator_type='GHMC'):
        """
        Subclass of NCMCEngine which switches directly between two different
        systems using an alchemical hybrid topology.

        Arguments
        ---------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature at which switching is to be run
        functions : dict of str:str, optional, default=default_functions
            functions[parameter] is the function (parameterized by 't' which
            switched from 0 to 1) that controls how alchemical context
            parameter 'parameter' is switched
        nsteps : int, optional, default=1
            The number of steps to use for switching.
        timestep : simtk.unit.Quantity with units compatible with femtoseconds,
            optional, default=1*femtosecond
            The timestep to use for integration of switching velocity
            Verlet steps.
        constraint_tolerance : float, optional, default=None
            If not None, this relative constraint tolerance is used for
            position and velocity constraints.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, the platform to use for OpenMM simulations.
        write_ncmc_interval : int, optional, default=None
            If a positive integer is specified, a PDB frame will be written
            with the specified interval on NCMC switching, with a different
            PDB file generated for each attempt.
        integrator_type : str, optional, default='GHMC'
            NCMC internal integrator type ['GHMC', 'VV']
        """
        if functions is None:
            functions = default_hybrid_functions

        super(NCMCHybridEngine, self).__init__(temperature=temperature, functions=functions, nsteps=nsteps,
                                               timestep=timestep, constraint_tolerance=constraint_tolerance,
                                               platform=platform, write_ncmc_interval=write_ncmc_interval,
                                               integrator_type=integrator_type)

    def _computeAlchemicalCorrection(self, integrator, context,
                                     unmodified_old_system, unmodified_new_system,
                                     initial_positions, final_positions,
                                     direction='insert'):
        """
        Compute log probability for correction from transforming real system
        to AND from alchemical system.

        Parameters
        ----------
        itegrator : NCMCAlchemicalIntegrator subclasses
            NCMC switching integrator to annihilate or introduce particles alchemically.
        context : openmm.Context 
            Alchemical context
        unmodified_old_system : simtk.unit.System
            Real fully-interacting system.
        unmodified_new_system : simtk.unit.System
            Real fully-interacting system.
        initial_positions : simtk.unit.Quantity of dimensions [nparticles,3]
            with units compatible with angstroms
            The initial positions before NCMC switching.
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3]
            with units compatible with angstroms
            The final positions after NCMC switching.
        direction : str, optional, default='insert'
            Not used in calculation
        Returns
        -------
        logP_alchemical_correction : float
            The log acceptance probability of the switch
        """
        from perses.tests.utils import compute_potential
        initial_logP_correction = (self.beta * integrator.getGlobalVariableByName("Einitial") * unit.kilojoules_per_mole) - self.beta * compute_potential(unmodified_old_system, initial_positions, platform=self.platform)
        final_logP_correction = self.beta * self.compute_potential(unmodified_new_system, final_positions, platform=self.platform) - (self.beta * context.getState(getEnergy=True).getPotentialEnergy())

        logP_alchemical_correction = initial_logP_correction + final_logP_correction
        return logP_alchemical_correction

    def make_alchemical_system(self, topology_proposal, old_positions,
                               new_positions):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal
        Arguments
        ---------
        topology_proposal : TopologyProposal namedtuple
            Contains old topology, proposed new topology, and atom mapping
        Returns
        -------
        unmodified_system : simtk.openmm.System
            Unmodified real system corresponding to appropriate leg of
            transformation.
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        """

        atom_map = topology_proposal.old_to_new_atom_map

        #take the unique atoms as those not in the {new_atom : old_atom} atom map
        unmodified_old_system = copy.deepcopy(topology_proposal.old_system)
        unmodified_new_system = copy.deepcopy(topology_proposal.new_system)
        old_topology = topology_proposal.old_topology
        new_topology = topology_proposal.new_topology

        # Create an alchemical factory.
        from perses.annihilation.relative import HybridTopologyFactory
        alchemical_factory = HybridTopologyFactory(unmodified_old_system,
                                                   unmodified_new_system,
                                                   old_topology, new_topology,
                                                   old_positions,
                                                   new_positions, atom_map)

        # Return the alchemically-modified system in fully-interacting form.
#        alchemical_system, _, alchemical_positions, final_atom_map, initial_atom_map = alchemical_factory.createPerturbedSystem()
        alchemical_system, alchemical_topology, alchemical_positions, final_atom_map, initial_atom_map = alchemical_factory.createPerturbedSystem()
        return [unmodified_old_system, unmodified_new_system,
                alchemical_system, alchemical_topology, alchemical_positions, final_atom_map,
                initial_atom_map]

    def _convert_hybrid_positions_to_final(self, positions, atom_map):
        final_positions = unit.Quantity(np.zeros([len(atom_map.keys()),3]), unit=unit.nanometers)
        for finalatom, hybridatom in atom_map.items():
            final_positions[finalatom] = positions[hybridatom]
        return final_positions

    def _zero_steps_return(self, initial_positions, proposed_positions, topology_proposal):
        # Special case of instantaneous insertion/deletion.
        final_positions = copy.deepcopy(proposed_positions)
        from perses.tests.utils import compute_potential
        potential_del = -self.beta * compute_potential(topology_proposal.old_system, initial_positions, platform=self.platform)
        potential_ins = -self.beta * compute_potential(topology_proposal.new_system, proposed_positions, platform=self.platform)
        potential = potential_ins - potential_del
        return [final_positions, initial_positions, potential]

    def integrate(self, topology_proposal, initial_positions, proposed_positions, platform=None):
        """
        Performs NCMC switching to either delete or insert atoms according to the provided `topology_proposal`.
        The contribution of transforming the real system to/from an alchemical system is included.
        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        initial_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms at the beginning of the NCMC switching.
        proposed_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the new system atoms proposed by geometry engine.
        platform : simtk.openmm.Platform, optional, default=None
            If not None, this platform is used for integration.
        Returns
        -------
        final_positions : simtk.unit.Quantity of dimensions [natoms, 3] with units of distance
            The final positions after `nsteps` steps of alchemical switching
        new_old_positions : simtk.unit.Quantity of dimensions [natoms, 3] with units of distance.
            The final positions of the atoms of the old system after `nsteps`
            steps of alchemical switching
        logP : float
            The log acceptance probability of the switch
        """
        direction = 'insert'
        if (self.nsteps == 0):
            return self._zero_steps_return(initial_positions, proposed_positions, topology_proposal)

        # Create alchemical system.
        [unmodified_old_system,
         unmodified_new_system,
         alchemical_system,
         alchemical_topology,
         alchemical_positions,
         final_to_hybrid_atom_map,
         initial_to_hybrid_atom_map] = self.make_alchemical_system(
                                            topology_proposal, initial_positions,
                                            proposed_positions)

        indices = [initial_to_hybrid_atom_map[idx] for idx in topology_proposal.unique_old_atoms] + [final_to_hybrid_atom_map[idx] for idx in topology_proposal.unique_new_atoms]
        functions = self._get_functions(alchemical_system)
        integrator = self._choose_integrator(alchemical_system, functions, direction)
        context = self._create_context(alchemical_system, integrator, alchemical_positions)

        final_hybrid_positions, logP_NCMC = self._integrate_switching(integrator, context, alchemical_topology, indices, None, direction)
        final_positions = self._convert_hybrid_positions_to_final(final_hybrid_positions, final_to_hybrid_atom_map)
        new_old_positions = self._convert_hybrid_positions_to_final(final_hybrid_positions, initial_to_hybrid_atom_map)

        # Compute contribution from transforming real system to/from alchemical system.
        logP_alchemical_correction = self._computeAlchemicalCorrection(
                                              integrator,
                                              context,
                                              unmodified_old_system,
                                              unmodified_new_system,
                                              initial_positions,
                                              final_positions,
                                          )

        logP_ncmc = self._clean_up_integration(logP_NCMC, logP_alchemical_correction, alchemical_system, context, integrator)

        # Return
        return [final_positions, new_old_positions, logP_ncmc]


class NCMCAlchemicalIntegrator(openmm.CustomIntegrator):
    """
    Helper base class for NCMC alchemical integrators.
    """
    def __init__(self, temperature, system, functions, nsteps, steps_per_propagation, timestep, direction):
        """
        Initialize base class for NCMC alchemical integrators.

        Parameters
        ----------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature to use for computing the NCMC acceptance probability.
        system : simtk.openmm.System
            The system to be simulated.
        functions : dict of str : str
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int
            The number of switching timesteps per call to integrator.step(1).
        steps_per_propagation : int
            The number of propagation steps taken at each value of lambda
        timestep : simtk.unit.Quantity with units compatible with femtoseconds
            The timestep to use for each NCMC step.
        direction : str, optional, default='insert'
            One of ['insert', 'delete'].
            For `insert`, the parameter 'lambda' is switched from 0 to 1.
            For `delete`, the parameter 'lambda' is switched from 1 to 0.

        """
        super(NCMCAlchemicalIntegrator, self).__init__(timestep)

        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)
        self.direction = direction

        # Compute kT in natural openmm units.
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        self.kT = kT

        self.has_statistics = False # no GHMC statistics by default

        self.nsteps = nsteps

        # Make a list of parameters in the system
        self.system_parameters = list()
        self.alchemical_functions = functions
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if hasattr(force, 'getNumGlobalParameters'):
                for parameter_index in range(force.getNumGlobalParameters()):
                    self.system_parameters.append(force.getGlobalParameterName(parameter_index))

    def addAlchemicalResetStep(self):
        """
        Reset alchemical state to initial state.
        """
        # Set the master 'lambda' alchemical parameter to the initial state
        if self.direction == 'insert':
            self.addComputeGlobal('lambda', '0.0')
        elif self.direction == 'delete':
            self.addComputeGlobal('lambda', '1.0')

        # Update all slaved alchemical parameters
        self.addUpdateAlchemicalParametersStep()

    def addAlchemicalPerturbationStep(self):
        """
        Add alchemical perturbation step.
        """
        # Set the master 'lambda' alchemical parameter to the current fractional state
        if self.nsteps == 0:
            # Toggle alchemical state
            if self.direction == 'insert':
                self.addComputeGlobal('lambda', '1.0')
            elif self.direction == 'delete':
                self.addComputeGlobal('lambda', '0.0')
        else:
            # Use fractional state
            if self.direction == 'insert':
                self.addComputeGlobal('lambda', '(step+1)/nsteps')
            elif self.direction == 'delete':
                self.addComputeGlobal('lambda', '(nsteps - step - 1)/nsteps')

        # Update all slaved alchemical parameters
        self.addUpdateAlchemicalParametersStep()

    def addUpdateAlchemicalParametersStep(self):
        """
        Update Context parameters according to provided functions.
        """
        for context_parameter in self.alchemical_functions:
            if context_parameter in self.system_parameters:
                self.addComputeGlobal(context_parameter, self.alchemical_functions[context_parameter])

    def addVelocityVerletStep(self):
        """
        Add velocity Verlet step.
        """
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

    def addGHMCStep(self):
        """
        Add a GHMC step.
        """
        self.hasStatistics = True

        # TODO: This could be precomputed to save time
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
        self.addComputeGlobal("Eold_GHMC", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + v*dt")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        self.addConstrainVelocities()
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew_GHMC", "ke + energy")
        # Compute acceptance probability
        # DEBUG: Check positions are finite
        self.addComputeGlobal("accept", "step(exp(-(Enew_GHMC-Eold_GHMC)/kT) - uniform)")
        self.beginIfBlock("accept != 1")
        # Reject sample, inverting velcoity
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

    def get_step(self):
        return self.getGlobalVariableByName("step")

    def reset(self):
        """
        Reset step counter and total work
        """
        self.setGlobalVariableByName("total_work", 0.0)
        self.setGlobalVariableByName("protocol_work", 0.0)
        self.setGlobalVariableByName("shadow_work", 0.0)
        if (self.nsteps > 0):
            self.setGlobalVariableByName("step", 0)
            if self.has_statistics:
                self.setGlobalVariableByName("naccept", 0)
                self.setGlobalVariableByName("ntrials", 0)

    def getStatistics(self, context):
        if (self.has_statistics):
            return (self.getGlobalVariableByName("naccept"), self.getGlobalVariableByName("ntrials"))
        else:
            return (0,0)

    def getWork(self, context):
        """Retrieve accumulated work (in units of kT)
        """
        return self.getGlobalVariableByName("total_work")

    def getShadowWork(self, context):
        """Retrieve accumulated shadow work (in units of kT)
        """
        return self.getGlobalVariableByName("shadow_work")

    def getProtocolWork(self, context):
        """Retrieve accumulated protocol work (in units of kT)
        """
        return self.getGlobalVariableByName("protocol_work")

    def getLogAcceptanceProbability(self, context):
        logp_accept = -1.0*self.getGlobalVariableByName("total_work")
        return logp_accept

class NCMCVVAlchemicalIntegrator(NCMCAlchemicalIntegrator):
    """
    Use NCMC switching to annihilate or introduce particles alchemically.

    TODO:
    ----
    * We may need to avoid unrolling integration steps.

    Examples
    --------

    Annihilate a Lennard-Jones particle

    >>> # Create an alchemically-perturbed test system
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.LennardJonesCluster()
    >>> from alchemy import AbsoluteAlchemicalFactory
    >>> alchemical_atoms = [0]
    >>> factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms)
    >>> alchemical_system = factory.createPerturbedSystem()
    >>> # Create an NCMC switching integrator.
    >>> temperature = 300.0 * unit.kelvin
    >>> nsteps = 5
    >>> functions = { 'lambda_sterics' : 'lambda' }
    >>> ncmc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nsteps, direction='delete')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Run the integrator
    >>> ncmc_integrator.step(nsteps)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.getLogAcceptanceProbability(context)

    Turn on an atom and its associated angles and torsions in alanine dipeptide

    >>> # Create an alchemically-perturbed test system
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from alchemy import AbsoluteAlchemicalFactory
    >>> alchemical_atoms = [0,1,2,3] # terminal methyl group
    >>> factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)
    >>> alchemical_system = factory.createPerturbedSystem()
    >>> # Create an NCMC switching integrator.
    >>> temperature = 300.0 * unit.kelvin
    >>> nsteps = 10
    >>> functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : 'lambda^2' }
    >>> ncmc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nsteps, direction='delete')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Minimize
    >>> openmm.LocalEnergyMinimizer.minimize(context)
    >>> # Run the integrator
    >>> ncmc_integrator.step(nsteps)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.getLogAcceptanceProbability(context)

    """

    def __init__(self, temperature, system, functions, nsteps=0, steps_per_propagation=1, timestep=1.0*unit.femtoseconds, direction='insert'):
        """
        Initialize an NCMC switching integrator to annihilate or introduce particles alchemically.

        Parameters
        ----------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature to use for computing the NCMC acceptance probability.
        system : simtk.openmm.System
            The system to be simulated.
        functions : dict of str : str
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int, optional, default=10
            The number of switching timesteps per call to integrator.step(1).
        steps_per_propagation : int, optional, default=1
            The number of propagation steps taken at each value of lambda
        timestep : simtk.unit.Quantity with units compatible with femtoseconds
            The timestep to use for each NCMC step.
        direction : str, optional, default='insert'
            One of ['insert', 'delete'].
            For `insert`, the parameter 'lambda' is switched from 0 to 1.
            For `delete`, the parameter 'lambda' is switched from 1 to 0.

        Note that each call to integrator.step(1) executes the entire integration program; this should not be called with more than one step.

        A symmetric protocol is used, in which the protocol begins and ends with a velocity Verlet step.

        TODO:
        * Add a global variable that causes termination of future calls to step(1) after the first

        """
        super(NCMCVVAlchemicalIntegrator, self).__init__(temperature, system, functions, nsteps, steps_per_propagation, timestep, direction)

        #
        # Initialize global variables
        #

        # NCMC variables
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addGlobalVariable('total_work', 0.0) # cumulative total work in kT
        self.addGlobalVariable('shadow_work', 0.0) # cumulative shadow work in kT
        self.addGlobalVariable('protocol_work', 0.0) # cumulative protocol work in kT
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable('kinetic', 0.0) # kinetic energy
        self.addGlobalVariable("Einitial", 0) # initial energy after setting initial alchemical state
        self.addGlobalVariable("kT", self.kT.value_in_unit_system(unit.md_unit_system))  # thermal energy
        self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
        self.addGlobalVariable('step', 0) # current NCMC step number

        # VV variables
        if (nsteps > 0):
            # VV variables
            self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints
            self.addGlobalVariable('psteps', steps_per_propagation)
            self.addGlobalVariable('pstep', 0)

        if nsteps == 0:
            self.beginIfBlock('step = 0')            
            # Initialize alchemical state
            self.addAlchemicalResetStep()
            self.setGlobalVariableByName("total_work", 0.0)
            self.setGlobalVariableByName("protocol_work", 0.0)
            self.setGlobalVariableByName("shadow_work", 0.0)
            # Compute energy of initial alchemical state
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            # Constrain initial positions and velocities
            self.addConstrainPositions()
            self.addConstrainVelocities()
            # Allow context state to be updated
            #self.addUpdateContextState()
            # Compute instantaneous work
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)/kT")
            self.addComputeGlobal("total_work", "protocol_work")
            # Update step
            self.addComputeGlobal("step", "step+1")
            # End block
            self.endBlock()
        if nsteps > 0:
            # Initial step only
            self.beginIfBlock('step = 0')
            # Initialize alchemical state
            self.addAlchemicalResetStep()
            self.setGlobalVariableByName("total_work", 0.0)
            self.setGlobalVariableByName("protocol_work", 0.0)
            self.setGlobalVariableByName("shadow_work", 0.0)
            # Compute energy of initial alchemical state
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            # Constrain initial positions and velocities
            self.addConstrainPositions()
            self.addConstrainVelocities()
            # Allow context state to be updated
            #self.addUpdateContextState()
            # Accumulate shadow work while running propagation steps
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Eold", "energy + kinetic")
            # Execute propagation steps.
            self.addComputeGlobal('pstep', '0')
            self.beginWhileBlock('pstep < psteps')
            self.addVelocityVerletStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            # Compute shadow work contribution
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Enew", "energy + kinetic")
            self.addComputeGlobal("shadow_work", "shadow_work + (Enew-Eold)/kT")
            # End block
            self.endBlock()

            # All steps, including initial step
            self.beginIfBlock('step < nsteps')        
            # Accumulate protocol work
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)/kT")
            # Accumulate shadow work while running propagation steps
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Eold", "energy + kinetic")
            # Execute propagation steps.
            self.addComputeGlobal('pstep', '0')
            self.beginWhileBlock('pstep < psteps')
            self.addVelocityVerletStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            # Compute shadow work contribution
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Enew", "energy + kinetic")
            self.addComputeGlobal("shadow_work", "shadow_work + (Enew-Eold)/kT")
            # Increment step
            self.addComputeGlobal('step', 'step+1')
            # Compute total work
            self.addComputeGlobal("total_work", "shadow_work + protocol_work")
            # End block
            self.endBlock()

class NCMCGHMCAlchemicalIntegrator(NCMCAlchemicalIntegrator):
    """
    Use NCMC switching to annihilate or introduce particles alchemically.
    """

    def __init__(self, temperature, system, functions, nsteps=0, steps_per_propagation=1, collision_rate=9.1/unit.picoseconds, timestep=1.0*unit.femtoseconds, direction='insert'):
        """
        Initialize an NCMC switching integrator to annihilate or introduce particles alchemically.

        Parameters
        ----------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature to use for computing the NCMC acceptance probability.
        system : simtk.openmm.System
            The system to be simulated.
        functions : dict of str : str
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int, optional, default=0
            The number of switching timesteps per call to integrator.step(1).
        steps_per_propagation : int, optional, default=1
            The number of propagation steps taken at each value of lambda
        timestep : simtk.unit.Quantity with units compatible with femtoseconds
            The timestep to use for each NCMC step.
        direction : str, optional, default='insert'
            One of ['insert', 'delete'].
            For `insert`, the parameter 'lambda' is switched from 0 to 1.
            For `delete`, the parameter 'lambda' is switched from 1 to 0.

        Note that each call to integrator.step(1) executes the entire integration program; this should not be called with more than one step.

        A symmetric protocol is used, in which the protocol begins and ends with a velocity Verlet step.

        TODO:
        * Add a global variable that causes termination of future calls to step(1) after the first

        """
        super(NCMCGHMCAlchemicalIntegrator, self).__init__(temperature, system, functions, nsteps, steps_per_propagation, timestep, direction)

        gamma = collision_rate

        # NCMC variables
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics        self.addGlobalVariable('lambda', 0.0) # alchemical switching parameter
        self.addGlobalVariable('total_work', 0.0) # cumulative total work in kT
        self.addGlobalVariable('shadow_work', 0.0) # cumulative shadow work in kT
        self.addGlobalVariable('protocol_work', 0.0) # cumulative protocol work in kT
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("Einitial", 0) # initial energy after setting initial alchemical state
        self.addGlobalVariable("kT", self.kT.value_in_unit_system(unit.md_unit_system))  # thermal energy
        self.addGlobalVariable('step', 0) # current NCMC step number

        if (nsteps > 0):
            # GHMC variables
            self.addGlobalVariable("Eold_GHMC", 0)  # old GHMC energy
            self.addGlobalVariable("Enew_GHMC", 0)  # new GHMC energy
            self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
            self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints
            self.addGlobalVariable("b", np.exp(-gamma * timestep))  # velocity mixing parameter
            self.addPerDofVariable("sigma", 0)
            self.addGlobalVariable("ke", 0)  # kinetic energy
            self.addPerDofVariable("vold", 0)  # old velocities
            self.addPerDofVariable("xold", 0)  # old positions
            self.addGlobalVariable("accept", 0)  # accept or reject
            self.addGlobalVariable("naccept", 0)  # number accepted
            self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials
            self.addGlobalVariable("pstep", 0) # number of propagation steps taken
            self.addGlobalVariable("psteps", steps_per_propagation) # total number of propagation steps

        if nsteps == 0:
            self.beginIfBlock('step = 0')            
            # Initialize alchemical state
            self.addAlchemicalResetStep()
            self.setGlobalVariableByName("total_work", 0.0)
            self.setGlobalVariableByName("protocol_work", 0.0)
            self.setGlobalVariableByName("shadow_work", 0.0)
            # Compute energy of initial alchemical state
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            # Constrain initial positions and velocities
            self.addConstrainPositions()
            self.addConstrainVelocities()
            # Allow context state to be updated
            #self.addUpdateContextState()
            # Compute instantaneous work
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)/kT")
            self.addComputeGlobal("total_work", "protocol_work")
            # Update step
            self.addComputeGlobal("step", "step+1")
            # End block
            self.endBlock()
        if nsteps > 0:
            # Initial step only
            self.beginIfBlock('step = 0')
            # Initialize alchemical state
            self.addAlchemicalResetStep()
            self.setGlobalVariableByName("total_work", 0.0)
            self.setGlobalVariableByName("protocol_work", 0.0)
            self.setGlobalVariableByName("shadow_work", 0.0)
            # Compute energy of initial alchemical state
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            # Constrain initial positions and velocities
            self.addConstrainPositions()
            self.addConstrainVelocities()
            # Allow context state to be updated
            #self.addUpdateContextState()
            # Execute propagation steps.
            self.addComputeGlobal('pstep', '0')
            self.beginWhileBlock('pstep < psteps')
            self.addGHMCStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            # End block
            self.endBlock()

            # All steps, including initial step
            self.beginIfBlock('step < nsteps')        
            # Accumulate protocol work
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)/kT")
            # Execute propagation steps.
            self.addComputeGlobal('pstep', '0')
            self.beginWhileBlock('pstep < psteps')
            self.addGHMCStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            # Increment step
            self.addComputeGlobal('step', 'step+1')
            # Compute total work
            self.addComputeGlobal("total_work", "protocol_work")
            # End block
            self.endBlock()

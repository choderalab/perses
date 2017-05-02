from __future__ import print_function
import numpy as np
import copy
import logging
import traceback
from simtk import openmm, unit
from perses.storage import NetCDFStorageView
from perses.tests.utils import quantity_is_finite
from openmmtools.integrators import NonequilibriumLangevinIntegrator

default_insert_functions = {
    'lambda_sterics': '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
    'lambda_electrostatics': '2*(lambda - 0.5) * step(lambda - 0.5)',
    'lambda_bonds': '0.9*lambda + 0.1',  # don't fully soften bonds
    'lambda_angles': '0.9*lambda + 0.1',  # don't fully soften angles
    'lambda_torsions': 'lambda'
}

default_delete_functions = {
    'lambda_sterics': '2*(1-lambda) * step(0.5 - (1-lambda)) + (1.0 - step(0.5 - (1-lambda)))',
    'lambda_electrostatics': '2*((1-lambda) - 0.5) * step((1-lambda) - 0.5)',
    'lambda_bonds': '0.9*(1-lambda) + 0.1',  # don't fully soften bonds
    'lambda_angles': '0.9*(1-lambda) + 0.1',  # don't fully soften angles
    'lambda_torsions': '(1-lambda)'
}


functions_disable_all = {
    'lambda_sterics': 'lambda',
    'lambda_electrostatics': 'lambda',
    'lambda_bonds': 'lambda',
    'lambda_angles': 'lambda',
    'lambda_torsions': 'lambda'
}

# make something hyperbolic or something to go from on to off to on
default_hybrid_functions = {
    'lambda_sterics': 'lambda',
    'lambda_electrostatics': 'lambda',
    'lambda_bonds': 'lambda',
    'lambda_angles': 'lambda',
    'lambda_torsions': 'lambda'
}

default_temperature = 300.0 * unit.kelvin
default_nsteps = 1
default_timestep = 1.0 * unit.femtoseconds
default_steps_per_propagation = 1


class NaNException(Exception):
    def __init__(self, *args, **kwargs):
        super(NaNException, self).__init__(*args, **kwargs)


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

    def __init__(self, temperature=default_temperature, insert_functions=None, delete_functions=None, nsteps=default_nsteps,
                 steps_per_propagation=default_steps_per_propagation, timestep=default_timestep,
                 constraint_tolerance=None, platform=None, write_ncmc_interval=None, integrator_type='GHMC',
                 storage=None, verbose=False):
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
        if insert_functions == None:
            insert_functions = default_insert_functions
        if delete_functions == None:
            delete_functions = default_delete_functions
        if nsteps == None:
            nsteps = default_nsteps
        if timestep == None:
            timestep = default_timestep
        if temperature == None:
            temperature = default_temperature

        self.temperature = temperature
        self.insert_functions = copy.deepcopy(insert_functions)
        self.delete_functions = copy.deepcopy(delete_functions)
        self.nsteps = nsteps
        self.timestep = timestep
        self.constraint_tolerance = constraint_tolerance
        self.platform = platform
        self.integrator_type = integrator_type
        self.steps_per_propagation = steps_per_propagation
        self.verbose = verbose
        self.disable_barostat = False

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
                    if parameter_name[0:(len(prefix) + 1)] == (prefix + '_'):
                        parameters.append(parameter_name)
        return parameters

    def _computeEnergyContribution(self, integrator):
        """
        Compute NCMC energy contribution to log probability.

        See Eqs. 62 and 63 (two-stage) and Eq. 45 (hybrid) of reference document.
        In both cases, the contribution is u(final_positions, final_lambda) - u(initial_positions, initial_lambda).

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
        logP_energy : float
            The NCMC energy contribution to log probability.
        """
        logP = integrator.getGlobalVariableByName("final_reduced_potential") - integrator.getGlobalVariableByName(
            "initial_reduced_potential")

        if np.isnan(logP):
            msg = "A required potential of NCMC operation is NaN:\n"
            msg += "initial_reduced_potential: %.3f kT\n" % integrator.getGlobalVariableByName(
                "initial_reduced_potential")
            msg += "final_reduced_potential:   %.3f kT\n" % integrator.getGlobalVariableByName(
                "final_reduced_potential")
            raise NaNException(msg)

        return logP

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
        alchemical_factory = AbsoluteAlchemicalFactory(unmodified_system, ligand_atoms=alchemical_atoms,
                                                       annihilate_electrostatics=True, annihilate_sterics=True,
                                                       alchemical_torsions=True, alchemical_bonds=True,
                                                       alchemical_angles=True, softcore_beta=0.0)

        # Return the alchemically-modified system in fully-interacting form.
        alchemical_system = alchemical_factory.createPerturbedSystem()

        if self.disable_barostat:
            for force in alchemical_system.getForces():
                if hasattr(force, 'setFrequency'):
                    force.setFrequency(0)

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

            nsteps = max(1,
                         self.nsteps)  # we must take 1 step even if nsteps = 0 to run the integrator through one cycle

            # Allocate storage for work.
            total_work = np.zeros([nsteps + 1], np.float64)  # work[n] is the accumulated total work up to step n
            shadow_work = np.zeros([nsteps + 1], np.float64)  # work[n] is the accumulated shadow work up to step n
            protocol_work = np.zeros([nsteps + 1], np.float64)  # work[n] is the accumulated protocol work up to step n

            # Write trajectory frame.
            if self._storage and self.write_ncmc_interval:
                positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                self._storage.write_configuration('positions', positions, topology, iteration=iteration, frame=0,
                                                  nframes=(self.nsteps + 1))

            # Perform NCMC integration.
            for step in range(nsteps):
                # Take a step.
                try:
                    integrator.step(1)
                except Exception as e:
                    print(e)
                    for index in range(integrator.getNumGlobalVariables()):
                        name = integrator.getGlobalVariableName(index)
                        val = integrator.getGlobalVariable(index)
                        print(name, val)
                    for index in range(integrator.getNumPerDofVariables()):
                        name = integrator.getPerDofVariableName(index)
                        val = integrator.getPerDofVariable(index)
                        print(name, val)

                # Store accumulated work
                total_work[step + 1] = integrator.getTotalWork(context)
                shadow_work[step + 1] = integrator.getShadowWork(context)
                protocol_work[step + 1] = integrator.getProtocolWork(context)

                # Write trajectory frame.
                if self._storage and self.write_ncmc_interval and (self.write_ncmc_interval % (step + 1) == 0):
                    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                    assert quantity_is_finite(positions) == True
                    self._storage.write_configuration('positions', positions, topology, iteration=iteration,
                                                      frame=(step + 1), nframes=(self.nsteps + 1))

            # Store work values.
            if self._storage:
                self._storage.write_array('total_work_%s' % direction, total_work, iteration=iteration)
                self._storage.write_array('shadow_work_%s' % direction, shadow_work, iteration=iteration)
                self._storage.write_array('protocol_work_%s' % direction, protocol_work, iteration=iteration)

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

    def _choose_integrator(self, functions):
        """
        Instantiate the appropriate type of NCMC integrator, setting
        constraint tolerance if specified.

        Parameters
        ----------
        functions : dict
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched

        Returns
        -------
        integrator : simtk.openmm.CustomIntegrator
            NCMC switching integrator to annihilate or introduce particles alchemically.
        """
        # Create an NCMC velocity Verlet integrator.
        #if self.integrator_type == 'VV':
        #    integrator = NCMCAlchemicalIntegrator(functions, splitting="O V R H R V O", temperature=self.temperature, nsteps_neq=self.nsteps)
        if self.constraint_tolerance is not None:
            constraint_tolerance = self.constraint_tolerance
        else:
            constraint_tolerance = 1e-8
        if self.integrator_type == 'GHMC':
            integrator = NCMCAlchemicalIntegrator(functions, splitting="O { V R H R V } O", temperature=self.temperature, nsteps_neq=self.nsteps, constraint_tolerance=constraint_tolerance)
        elif self.integrator_type == "gBAOAB":
            integrator = NCMCAlchemicalIntegrator(functions, splitting="V R R R O R R R V", temperature=self.temperature, nsteps_neq=self.nsteps, constraint_tolerance=constraint_tolerance)
        else:
            raise Exception("integrator_type '%s' unknown" % self.integrator_type)

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
        # print('before setpositions:')
        # print('positions', context.getState(getPositions=True).getPositions(asNumpy=True))
        # print('velocities', context.getState(getVelocities=True).getVelocities(asNumpy=True))
        context.setPositions(positions)
        # print('after setpositions:')
        # print('positions', context.getState(getPositions=True).getPositions(asNumpy=True))
        # print('velocities', context.getState(getVelocities=True).getVelocities(asNumpy=True))
        context.applyConstraints(integrator.getConstraintTolerance())
        # print('after applyConstraints:')
        # print('positions', context.getState(getPositions=True).getPositions(asNumpy=True))
        # print('velocities', context.getState(getVelocities=True).getVelocities(asNumpy=True))
        # Set velocities to temperature and apply velocity constraints.
        # print('after setVelocitiesToTemperature:')
        context.setVelocitiesToTemperature(self.temperature)
        # print('positions', context.getState(getPositions=True).getPositions(asNumpy=True))
        # print('velocities', context.getState(getVelocities=True).getVelocities(asNumpy=True))
        context.applyVelocityConstraints(integrator.getConstraintTolerance())
        # print('after applyVelocityConstraints:')
        # print('positions', context.getState(getPositions=True).getPositions(asNumpy=True))
        # print('velocities', context.getState(getVelocities=True).getVelocities(asNumpy=True))

        # state = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True, getParameters=True)
        # def write_file(filename, contents):
        #    outfile = open(filename, 'w')
        #    outfile.write(contents)
        #    outfile.close()
        # write_file('system.xml', openmm.XmlSerializer.serialize(system))
        # write_file('integrator.xml', openmm.XmlSerializer.serialize(integrator))
        # write_file('state.xml', openmm.XmlSerializer.serialize(state))

        return context

    def _get_functions(self, system, functions):
        """
        Select subset of switching functions based on which alchemical parameters are present in the system.

        Parameters
        ----------
        system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        functions : dict
            The functions to subset

        Returns
        -------
        functions : dict
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        """
        available_parameters = self._getAvailableParameters(system)
        functions = {parameter_name: functions[parameter_name] for parameter_name in functions if
                     (parameter_name in available_parameters)}
        return functions

    def _clean_up_integration(self, alchemical_system, context, integrator):
        """
        Delete the alchemical system, context and integrator, and increase
        the counter of number of NCMC attempts.

        Parameters
        ----------
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        context : openmm.Context
            Alchemical context
        itegrator : NCMCAlchemicalIntegrator subclasses
            NCMC switching integrator to annihilate or introduce particles alchemically.

        Returns
        -------
        logP : float
            The log contribution to the acceptance probability for this NCMC stage
        """
        # Clean up alchemical system.
        del alchemical_system, context, integrator

        # Keep track of statistics.
        self.nattempted += 1

    def integrate(self, topology_proposal, initial_positions, direction='insert', platform=None, iteration=None):
        """
        Performs NCMC switching to either delete or insert atoms according to the provided `topology_proposal`.

        For `delete`, the system is first modified from fully interacting to alchemically modified, and then NCMC switching is used to eliminate atoms.
        For `insert`, the system begins with eliminated atoms in an alchemically noninteracting form and NCMC switching is used to turn atoms on, followed by making system real.

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
        logP_work : float
            The NCMC work contribution to the log acceptance probability (Eqs. 62 and 63)
        logP_energy : float
            The NCMC energy contribution to the log acceptance probability (Eqs. 62 and 63)

        """
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        assert quantity_is_finite(initial_positions) == True

        topology, indices, system = self._choose_system_from_direction(topology_proposal, direction)

        # Create alchemical system.
        alchemical_system = self.make_alchemical_system(system, indices, direction=direction)

        if direction == "delete":
            functions = self._get_functions(alchemical_system, self.delete_functions)
        else:
            functions = self._get_functions(alchemical_system, self.insert_functions)

        integrator = self._choose_integrator(functions)
        context = self._create_context(alchemical_system, integrator, initial_positions)

        # Integrate switching
        final_positions, logP_work = self._integrate_switching(integrator, context, topology, indices, iteration,
                                                               direction)

        # Compute contribution from switching between real and alchemical systems in correct order
        logP_energy = self._computeEnergyContribution(integrator)

        self._clean_up_integration(alchemical_system, context, integrator)

        # Return
        return [final_positions, logP_work, logP_energy]


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
                 write_ncmc_interval=None, integrator_type='GHMC',
                 storage=None):
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
                                               storage=storage, integrator_type=integrator_type)

    def make_alchemical_system(self, topology_proposal, old_positions,
                               new_positions):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal
        Arguments
        ---------
        topology_proposal : TopologyProposal namedtuple
            Contains old topology, proposed new topology, and atom mapping
        old_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms at the beginning of the NCMC switching.
        new_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms proposed by geometry engine.

        Returns
        -------
        unmodified_old_system : simtk.openmm.System
            Unmodified real system corresponding to old chemical state.
        unmodified_new_system : simtk.openmm.System
            Unmodified real system corresponding to new chemical state.
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        alchemical_topology : openmm.app.Topology
            Topology which includes unique atoms of old and new states.
        alchemical_positions : simtk.unit.Quantity of dimensions [nparticles,3]
            with units compatible with angstroms
            Positions for the alchemical hybrid topology
        final_atom_map : dict(int : int)
            Dictionary mapping the index of every atom in the new topology
            to its index in the hybrid topology
        initial_atom_map : dict(int : int)
            Dictionary mapping the index of every atom in the old topology
            to its index in the hybrid topology
        """

        atom_map = topology_proposal.old_to_new_atom_map

        # take the unique atoms as those not in the {new_atom : old_atom} atom map
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
        alchemical_system, alchemical_topology, alchemical_positions, final_atom_map, initial_atom_map = alchemical_factory.createPerturbedSystem()

        # Disable barostat so that it isn't used during NCMC
        if self.disable_barostat:
            for force in alchemical_system.getForces():
                if hasattr(force, 'setFrequency'):
                    force.setFrequency(0)

        return [unmodified_old_system, unmodified_new_system,
                alchemical_system, alchemical_topology, alchemical_positions, final_atom_map,
                initial_atom_map]

    def _convert_hybrid_positions_to_final(self, positions, atom_map):
        final_positions = unit.Quantity(np.zeros([len(atom_map.keys()), 3]), unit=unit.nanometers)
        for finalatom, hybridatom in atom_map.items():
            final_positions[finalatom] = positions[hybridatom]
        return final_positions

    def integrate(self, topology_proposal, initial_positions, proposed_positions, platform=None, iteration=None):
        """
        Performs NCMC switching to either delete or insert atoms according to the provided `topology_proposal`.

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
        logP_work : float
            The NCMC work contribution to the log acceptance probability (Eq. 44)
        logP_energy : float
            The NCMC energy contribution to the log acceptance probability (Eq. 45)
        """
        direction = 'insert'

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

        indices = [initial_to_hybrid_atom_map[idx] for idx in topology_proposal.unique_old_atoms] + [
            final_to_hybrid_atom_map[idx] for idx in topology_proposal.unique_new_atoms]
        functions = self._get_functions(alchemical_system)
        integrator = self._choose_integrator(alchemical_system, functions, direction)
        context = self._create_context(alchemical_system, integrator, alchemical_positions)

        final_hybrid_positions, logP_work = self._integrate_switching(integrator, context, alchemical_topology, indices,
                                                                      iteration, direction)
        final_positions = self._convert_hybrid_positions_to_final(final_hybrid_positions, final_to_hybrid_atom_map)
        new_old_positions = self._convert_hybrid_positions_to_final(final_hybrid_positions, initial_to_hybrid_atom_map)

        logP_energy = self._computeEnergyContribution(integrator)

        self._clean_up_integration(alchemical_system, context, integrator)

        # Return
        return [final_positions, new_old_positions, logP_work, logP_energy]


class NCMCAlchemicalIntegrator(NonequilibriumLangevinIntegrator):
    """Allows nonequilibrium switching based on force parameters specified in alchemical_functions.
    A variable named lambda is switched from 0 to 1 linearly throughout the nsteps of the protocol.
    The functions can use this to create more complex protocols for other global parameters.
    Propagator is based on Langevin splitting, as described below.
    One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt
        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass
        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal
    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive but slow-fluctuating forces. Since forces are only
    evaluated in the V step, we represent this by including in our "alphabet" V0, V1, ...)
    When the system contains holonomic constraints, these steps are confined to the constraint
    manifold.
    Examples
    --------
        - VVVR
            splitting="O V R V O"
        - BAOAB:
            splitting="V R O R V"
        - g-BAOAB, with K_r=3:
            splitting="V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            splitting="V0 V1 R R O R R V1 R R O R R V1 V0"
        - An NCMC algorithm with Metropolized integrator:
            splitting="O { V R H R V } O"
    Attributes
    ----------
    _kinetic_energy : str
        This is 0.5*m*v*v by default, and is the expression used for the kinetic energy
    References
    ----------
    [Nilmeier, et al. 2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7
    """

    def __init__(self,
                 alchemical_functions,
                 splitting="O { V R H R V } O",
                 temperature=298.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True,
                 nsteps_neq=100):
        """
        Parameters
        ----------
        alchemical_functions : dict of strings
            key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
            string that depends on the variable "lambda" which is switched from 0 to 1
        splitting : string, default: "O { V R H R V } O"
            Sequence of R, V, O (and optionally V{i}), and { }substeps to be executed each timestep. There is also an H option,
            which increments the global parameter `lambda` by 1/nsteps_neq for each step.
            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            ( will cause metropolization, and must be followed later by a ).
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Fictitious "bath" temperature
        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep
        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver
        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`
        measure_heat : boolean, default: True
            Accumulate the heat exchanged with the bath in each step, in the global `heat`
        nsteps_neq : int, default: 100
            Number of steps in nonequilibrium protocol. Default 100
        """

        self.addGlobalVariable("initial_reduced_potential", 0)
        self.addGlobalVariable("final_reduced_potential", 0)
        super(NCMCAlchemicalIntegrator, self).__init__(
            alchemical_functions,
            splitting=splitting,
            temperature=298.0 * unit.kelvin,
            collision_rate=1.0 / unit.picoseconds,
            timestep=1.0 * unit.femtoseconds,
            constraint_tolerance=1e-8,
            measure_shadow_work=False,
            measure_heat=True,
            nsteps_neq=100)

    def alchemical_reset_step(self):
        """
        Reset the lambda and other values to their start.
        """
        super(NCMCAlchemicalIntegrator, self).alchemical_reset_step()
        self.addComputeGlobal("initial_reduced_potential", "energy/kT")

    def alchemical_perturbation_step(self):
        """
        Add the alchemical perturbation step. Also calculate final_reduced_potential
        """
        super(NCMCAlchemicalIntegrator, self).alchemical_perturbation_step()
        self.addComputeGlobal("final_reduced_potential", "energy/kT")

    def get_step(self):
        return self.getGlobalVariableByName("step")

    def reset(self):
        """
        Reset everything.
        """
        self.setGlobalVariableByName("step", 0)
        self.setGlobalVariableByName("lambda", 0.0)
        self.setGlobalVariableByName("protocol_work", 0.0)
        if self._measure_shadow_work:
            self.setGlobalVariableByName("shadow_work", 0.0)
        self.setGlobalVariableByName("initial_reduced_potential", 0.0)
        self.setGlobalVariableByName("final_reduced_potential", 0.0)
        if self._metropolized_integrator:
            self.setGlobalVariableByName("naccept", 0)
            self.setGlobalVariableByName("ntrials", 0)

    def getStatistics(self):
        if (self._metropolized_integrator):
            return (self.getGlobalVariableByName("naccept"), self.getGlobalVariableByName("ntrials"))
        else:
            return (0, 0)

    def getTotalWork(self):
        """Retrieve accumulated total work (in units of kT)
        """
        if self._measure_shadow_work:
            return self.getGlobalVariableByName("shadow_work") + self.getGlobalVariableByName("protocol_work")
        else:
            return self.getGlobalVariableByName("protocol_work")

    def getShadowWork(self):
        """Retrieve accumulated shadow work (in units of kT)
        """
        if self._measure_shadow_work:
            return self.getGlobalVariableByName("shadow_work")
        else:
            return 0

    def getProtocolWork(self):
        """Retrieve accumulated protocol work (in units of kT)
        """
        return self.getGlobalVariableByName("protocol_work")

    def getLogAcceptanceProbability(self):
        logp_accept = -1.0 * self.getTotalWork()
        return logp_accept

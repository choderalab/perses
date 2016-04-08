from __future__ import print_function
import numpy as np
import copy
import logging
from simtk import openmm, unit

default_functions = {
    'lambda_sterics' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_bonds' : 'lambda',
    'lambda_angles' : 'lambda',
    'lambda_torsions' : 'lambda'
    }

default_temperature = 300.0*unit.kelvin
default_nsteps = 1
default_timestep = 1.0 * unit.femtoseconds

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

    def __init__(self, temperature=default_temperature, functions=default_functions, nsteps=default_nsteps, timestep=default_timestep, constraint_tolerance=None, platform=None):
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
        timestep : simtk.unit.Quantity with units compatible with femtoseconds, optional, default=1*femtosecond
            The timestep to use for integration of switching velocity Verlet steps.
        constraint_tolerance : float, optional, default=None
            If not None, this relative constraint tolerance is used for position and velocity constraints.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, the platform to use for OpenMM simulations.

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

    def _computeAlchemicalCorrection(self, unmodified_system, alchemical_system, initial_positions, final_positions, direction='insert'):
        """
        Compute log probability for correction from transforming real system to/from alchemical system.

        If direction is `insert`, the contribution at `final_positions` is computed as (real - alchemical).
        If direction is `delete`, the contribution at `initial_positions` is computed as (alchemical - real).

        Parameters
        ----------
        unmodified_system : simtk.unit.System
            Real fully-interacting system.
        alchemical_system : simtk.unit.System
            Alchemically modified system in fully-interacting form.
        initial_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The initial positions before NCMC switching.
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The final positions after NCMC switching.
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])

        Returns
        -------
        logP_alchemical_correction : float
            The log acceptance probability of the switch

        """

        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        def computePotentialEnergy(system, positions):
            """
            Compute potential energy of the specified system object at the specified positions.

            Constraints are applied before the energy is computed.

            Parameters
            ----------
            system : simtk.openmm.System
                The System object for which the potential energy is to be computed.
            positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
                Positions of the atoms for which energy is to be computed.

            Returns
            -------
            potential : simtk.unit.Quantity with units of energy
                The computed potential energy

            """
            # Create dummy integrator.
            integrator = openmm.VerletIntegrator(self.timestep)
            # Set the constraint tolerance if specified.
            if self.constraint_tolerance is not None:
                integrator.setConstraintTolerance(self.constraint_tolerance)
            # Create a context on the specified platform.
            if self.platform is not None:
                context = openmm.Context(alchemical_system, integrator, self.platform)
            else:
                context = openmm.Context(alchemical_system, integrator)
            context.setPositions(positions)
            context.applyConstraints(integrator.getConstraintTolerance())
            # Compute potential energy.
            potential = context.getState(getEnergy=True).getPotentialEnergy()
            # Clean up context and integrator.
            del context, integrator
            # Return potential energy.
            return potential

        # Compute correction from transforming real system to/from alchemical system
        if direction == 'delete':
            alchemical_potential_correction = computePotentialEnergy(alchemical_system, initial_positions) - computePotentialEnergy(unmodified_system, initial_positions)
        elif direction == 'insert':
            alchemical_potential_correction = computePotentialEnergy(unmodified_system, final_positions) - computePotentialEnergy(alchemical_system, final_positions)
        logP_alchemical_correction = -self.beta * alchemical_potential_correction

        return logP_alchemical_correction

    def make_alchemical_system(self, topology_proposal, direction='insert'):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal

        Arguments
        ---------
        topology_proposal : TopologyProposal namedtuple
            Contains old topology, proposed new topology, and atom mapping
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])

        Returns
        -------
        unmodified_system : simtk.openmm.System
            Unmodified real system corresponding to appropriate leg of transformation.
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified

        """
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        atom_map = topology_proposal.new_to_old_atom_map

        #take the unique atoms as those not in the {new_atom : old_atom} atom map
        if direction == 'delete':
            unmodified_system = topology_proposal.old_system
            alchemical_atoms = [atom for atom in range(unmodified_system.getNumParticles()) if atom not in atom_map.values()]
        elif direction == 'insert':
            unmodified_system = topology_proposal.new_system
            alchemical_atoms = [atom for atom in range(unmodified_system.getNumParticles()) if atom not in atom_map.keys()]
        else:
            raise Exception("direction must be one of ['delete', 'insert']; found '%s' instead" % direction)

        # DEBUG
        #print('alchemical atoms:')
        #print(alchemical_atoms)

        # Create an alchemical factory.
        from alchemy import AbsoluteAlchemicalFactory
        alchemical_factory = AbsoluteAlchemicalFactory(unmodified_system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=True, alchemical_bonds=None, alchemical_angles=None)

        # Return the alchemically-modified system in fully-interacting form.
        alchemical_system = alchemical_factory.createPerturbedSystem()
        return [unmodified_system, alchemical_system]

    def integrate(self, topology_proposal, initial_positions, direction='insert', platform=None):
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

        Returns
        -------
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The final positions after `nsteps` steps of alchemical switching
        logP : float
            The log acceptance probability of the switch
        potential : simtk.unit.Quantity with units compatible with kilocalories_per_mole
            For `delete`, the potential energy of the final (alchemically eliminated) conformation.
            For `insert`, the potential energy of the initial (alchemically eliminated) conformation.

        """
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        if (self.nsteps == 0):
            # Special case of instantaneous insertion/deletion.
            logP = 0.0
            final_positions = copy.deepcopy(initial_positions)
            from perses.tests.utils import compute_potential
            if direction == 'delete':
                potential = self.beta * compute_potential(topology_proposal.old_system, initial_positions, platform=self.platform)
            elif direction == 'insert':
                potential = self.beta * compute_potential(topology_proposal.new_system, initial_positions, platform=self.platform)
            return [final_positions, logP, potential]

        # Create alchemical system.
        [unmodified_system, alchemical_system] = self.make_alchemical_system(topology_proposal, direction=direction)

        # DEBUG: Compute initial potential of unmodified system and alchemical system to make sure finite.
        from perses.tests.utils import compute_potential
        #print(compute_potential(unmodified_system, initial_positions, platform=self.platform))
        #print(compute_potential(alchemical_system, initial_positions, platform=self.platform))

        # Select subset of switching functions based on which alchemical parameters are present in the system.
        available_parameters = self._getAvailableParameters(alchemical_system)
        functions = { parameter_name : self.functions[parameter_name] for parameter_name in self.functions if (parameter_name in available_parameters) }

        # Create an NCMC velocity Verlet integrator.
        integrator = NCMCAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, timestep=self.timestep, direction=direction)
        # Set the constraint tolerance if specified.
        if self.constraint_tolerance is not None:
            integrator.setConstraintTolerance(self.constraint_tolerance)
        # Create a context on the specified platform.
        if self.platform is not None:
            context = openmm.Context(alchemical_system, integrator, self.platform)
        else:
            context = openmm.Context(alchemical_system, integrator)
        context.setPositions(initial_positions)
        context.applyConstraints(integrator.getConstraintTolerance())
        # Set velocities to temperature and apply velocity constraints.
        context.setVelocitiesToTemperature(self.temperature)
        context.applyVelocityConstraints(integrator.getConstraintTolerance())

        # Set initial context parameters.
        if direction == 'insert':
            for parameter_name in available_parameters:
                context.setParameter(parameter_name, 0)
        elif direction == 'delete':
            for parameter_name in available_parameters:
                context.setParameter(parameter_name, 1)

        # Compute initial potential of alchemical state.
        initial_potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        if np.isnan(initial_potential):
            raise NaNException("Initial potential of 'insert' operation is NaN (unmodified potential was %.3f kT, alchemical potential was %.3f kT before changing lambda)" % (unmodified_potential, alchemical_potential))
        from perses.tests.utils import compute_potential_components
        #print("initial potential before '%s' : %f kT" % (direction, initial_potential))
        #print("initial potential components:   %s" % str(compute_potential_components(context))) # DEBUG

        # Take a single integrator step since all switching steps are unrolled in NCMCAlchemicalIntegrator.
        try:
            integrator.step(1)
        except Exception as e:
            # Trap NaNs as a special exception (allowing us to reject later, if desired)
            if str(e) == "Particle coordinate is nan":
                raise NaNException(str(e))
            else:
                raise e

        # Set final context parameters.
        if direction == 'insert':
            for parameter_name in available_parameters:
                context.setParameter(parameter_name, 1)
        elif direction == 'delete':
            for parameter_name in available_parameters:
                context.setParameter(parameter_name, 0)

        # Compute final potential of alchemical state.
        final_potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        if np.isnan(final_potential):
            raise NaNException("Final potential of 'delete' operation is NaN")
        #print("final potential before '%s' : %f kT" % (direction, final_potential))
        #print("final potential components: %s" % str(compute_potential_components(context))) # DEBUG
        #print('')

        # Store final positions and log acceptance probability.
        final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        logP_NCMC = integrator.getLogAcceptanceProbability()
        # DEBUG
        logging.debug("NCMC logP %+10.1f | initial_total_energy %+10.1f kT | final_total_energy %+10.1f kT." % (logP_NCMC, integrator.getGlobalVariableByName('initial_total_energy'), integrator.getGlobalVariableByName('final_total_energy')))
        # Clean up NCMC switching integrator.
        del context, integrator

        # Compute contribution from transforming real system to/from alchemical system.
        logP_alchemical_correction = self._computeAlchemicalCorrection(unmodified_system, alchemical_system, initial_positions, final_positions, direction=direction)

        # Compute total logP
        logP = logP_NCMC + logP_alchemical_correction

        # Clean up alchemical system.
        del alchemical_system

        # Select whether to return initial or final potential.
        if direction == 'insert':
            potential = initial_potential
        elif direction == 'delete':
            potential = final_potential

        # Return
        return [final_positions, logP, potential]

class NCMCAlchemicalIntegrator(openmm.CustomIntegrator):
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
    >>> functions = { 'lambda_sterics' : 'lambda' }
    >>> ncmc_integrator = NCMCAlchemicalIntegrator(temperature, alchemical_system, functions, direction='delete')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Run the integrator
    >>> ncmc_integrator.step(1)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.getLogAcceptanceProbability()

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
    >>> functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : 'lambda^2' }
    >>> ncmc_integrator = NCMCAlchemicalIntegrator(temperature, alchemical_system, functions, direction='delete')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Minimize
    >>> openmm.LocalEnergyMinimizer.minimize(context)
    >>> # Run the integrator
    >>> ncmc_integrator.step(1)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.getLogAcceptanceProbability()

    """

    def __init__(self, temperature, system, functions, nsteps=10, timestep=1.0*unit.femtoseconds, direction='insert'):
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
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        super(NCMCAlchemicalIntegrator, self).__init__(timestep * (nsteps+1))

        # Make a list of parameters in the system
        # TODO: We should be able to remove this.
        system_parameters = list()
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if hasattr(force, 'getNumGlobalParameters'):
                for parameter_index in range(force.getNumGlobalParameters()):
                    system_parameters.append(force.getGlobalParameterName(parameter_index))

        self.addGlobalVariable('kinetic', 0.0) # kinetic energy
        self.addGlobalVariable('initial_total_energy', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalVariable('final_total_energy', 0.0) # final total energy (kinetic + potential)
        self.addGlobalVariable('log_ncmc_acceptance_probability', 0.0) # log of NCMC acceptance probability
        self.addGlobalVariable('dti', timestep.value_in_unit_system(unit.md_unit_system)) # inner timestep
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints

        # Compute kT in natural openmm units.
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        kT = kT.value_in_unit_system(unit.md_unit_system)

        # Constrain initial positions and velocities.
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.addUpdateContextState()

        # Set initial parameters.
        if direction == 'insert':
            self.addComputeGlobal('lambda', '0.0')
        elif direction == 'delete':
            self.addComputeGlobal('lambda', '1.0')

        # Update Context parameters according to provided functions.
        for context_parameter in functions:
            if context_parameter in system_parameters:
                self.addComputeGlobal(context_parameter, functions[context_parameter])

        # Store initial total energy.
        self.addComputeSum("kinetic", "0.5*m*v*v")
        self.addComputeGlobal('initial_total_energy', 'kinetic + energy')

        #
        # Initial Velocity Verlet propagation step
        #

        if (nsteps > 0):
            self.addComputePerDof("v", "v+0.5*dti*f/m")
            self.addComputePerDof("x", "x+dti*v")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dti*f/m+(x-x1)/dti")
            self.addConstrainVelocities()

        # Compute direction that lambda is changing in.
        if (nsteps == 0):
            #
            # Alchemical perturbation step does not appear in step loop.
            #

            if direction == 'insert':
                self.addComputeGlobal('lambda', '1.0')
            elif direction == 'delete':
                self.addComputeGlobal('lambda', '0.0')

            # Update Context parameters according to provided functions.
            for context_parameter in functions:
                if context_parameter in system_parameters:
                    self.addComputeGlobal(context_parameter, functions[context_parameter])

        # Loop over NCMC steps (for nsteps > 1)
        if (nsteps > 0):
            self.addGlobalVariable('step', 0) # current NCMC step number
            self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
            self.beginWhileBlock('step < nsteps')

            #
            # Alchemical perturbation step
            #

            if direction == 'insert':
                self.addComputeGlobal('lambda', '(step+1)/nsteps')
            elif direction == 'delete':
                self.addComputeGlobal('lambda', '(nsteps - step - 1)/nsteps')

            # Update Context parameters according to provided functions.
            for context_parameter in functions:
                if context_parameter in system_parameters:
                    self.addComputeGlobal(context_parameter, functions[context_parameter])

            #
            # Velocity Verlet propagation step
            #

            self.addComputePerDof("v", "v+0.5*dti*f/m")
            self.addComputePerDof("x", "x+dti*v")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dti*f/m+(x-x1)/dti")
            self.addConstrainVelocities()

            #
            # End of loop
            #

            self.addComputeGlobal('step','step+1')
            self.endBlock()

        # Store final total energy.
        self.addComputeSum("kinetic", "0.5*m*v*v")
        self.addComputeGlobal('final_total_energy', 'kinetic + energy')

        # Compute log acceptance probability.
        self.addComputeGlobal('log_ncmc_acceptance_probability', '-1 * (final_total_energy - initial_total_energy) / %f' % kT)

        return

    def getLogAcceptanceProbability(self):
        return self.getGlobalVariableByName('log_ncmc_acceptance_probability')

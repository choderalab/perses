import numpy as np
import copy
from simtk import openmm, unit

default_functions = {
    'alchemical_sterics' : 'lambda',
    'alchemical_electrostatocs' : 'lambda',
    'alchemical_bonds' : 'lambda',
    'alchemical_angles' : 'lambda',
    'alchemical_torsions' : 'lambda'
    }

class NCMCEngine(object):
    def __init__(self, temperature=300.0*unit.kelvin, functions=default_functions, nsteps=1, timestep=1.0*unit.femtoseconds, constraint_tolerance=None):
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

        """
        self.temperature = temperature
        self.functions = copy.deepcopy(functions)
        self.nsteps = nsteps
        self.timestep = timestep
        self.constraint_tolerance = constraint_tolerance

    def _getAvailableParameters(self, system):
        """
        Return a list of available context parameters defined in the system

        Parameters
        ----------
        system : simtk.openmm.System
            The system for which available context parameters are to be determined

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
                    parameters.append(force.getGlobalParameterName(parameter_index))
        return parameters

    def integrate(self, alchemical_system, initial_positions, direction='insert', platform=None):
        """
        Performs NCMC switching according to the provided

        Parameters
        ----------
        alchemical_system : simtk.openmm.System object
            alchemically-modified system with atoms to be eliminated
        initial_positions : [n, 3] numpy.ndarray
            positions of the atoms in the old system
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

        """
        # Make a list of available context parameters.

        # Select subset of switching functions based on which alchemical parameters are present in the system.
        available_parameters = self._getAvailableParameters(alchemical_system)
        functions = { parameter_name : self.functions[parameter_name] for parameter_name in self.functions if (parameter_name in available_parameters) }

        # Create an NCMC velocity Verlet integrator.
        integrator = NCMCAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, timestep=self.timestep)
        # Set the constraint tolerance if specified.
        if self.constraint_tolerance is not None:
            integrator.setConstraintTolerance(self.constraint_tolerance)
        # Create a context on the specified platform.
        if platform is not None:
            context = openmm.Context(self.alchemical_system, integrator, platform)
        else:
            context = openmm.Context(self.alchemical_system, integrator)
        context.setPositions(self.initial_positions)
        # Set velocities to temperature and apply velocity constraints.
        context.setVelocitiesToTemperature(self.temperature)
        context.applyVelocityConstraints(integrator.getConstraintTolerance())
        # Only take a single integrator step since all switching steps are unrolled in NCMCAlchemicalIntegrator.
        integrator.step(1)
        # Store final positions and log acceptance probability.
        final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        logP = integrator.getLogAcceptanceProbability()
        # Clean up.
        del context, integrator
        # Return
        return [final_positions, logP]

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
    >>> functions = { 'alchemical_sterics' : 't' }
    >>> ncmc_integrator = NCMCAlchemicalIntegrator(alchemical_system, functions, mode='delete')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Run the integrator
    >>> ncmc_integrator.step(1)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.log_ncmc

    Turn on an atom and its associated angles and torsions in alanine dipeptide

    >>> # Create an alchemically-perturbed test system
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from alchemy import AbsoluteAlchemicalFactory
    >>> alchemical_atoms = [0,1,2,3] # terminal methyl group
    >>> factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)
    >>> alchemical_state = AlchemicalState(lambda_sterics=0, lambda_torsions=0, lambda_angles=0)
    >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)
    >>> # Create an NCMC switching integrator.
    >>> functions = { 'lambda_sterics' : 't', 'lambda_electrostatics' : 't**0.5', 'lambda_torsions' : 't', 'lambda_angles' : 't**2' }
    >>> ncmc_integrator = NCMCAlchemicalIntegrator(alchemical_system, functions, mode='insert')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Run the integrator
    >>> ncmc_integrator.step(1)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.log_ncmc


    """
    def __init__(self, temperature, system, functions, nsteps=10, timestep=1.0*unit.femtoseconds, mode='insert'):
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
        mode : str, optional, default='insert'
            One of ['insert', 'delete'].
            For `insert`, the parameter 'lambda' is switched from 0 to 1.
            For `delete`, the parameter 'lambda' is switched from 1 to 0.

        Note that each call to integrator.step(1) executes the entire integration program; this should not be called with more than one step.

        A symmetric protocol is used, in which the protocol begins and ends with a velocity Verlet step.

        TODO:
        * Add a global variable that causes termination of future calls to step(1) after the first

        """
        if (nsteps < 1):
            raise Exception("'nsteps' must be >= 1")

        if mode not in ['intert', 'delete']:
            raise Exception("mode must be one of ['insert', 'delete']; was '%s' instead" % mode)

        super(NCMCAlchemicalIntegrator, self).__init__((nsteps+1) * timestep)

        self.addGlobalParameter('initial_total_energy', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalParameter('final_total_energy', 0.0) # final total energy (kinetic + potential)
        self.addGlobalParameter('log_ncmc_acceptance_probability', 0.0) # log of NCMC acceptance probability

        if mode == 'insert':
            self.addGlobalParameter('lambda', 0.0) # parameter switched from 0 to 1 during course of integrating internal 'nsteps' of dynamics
        elif mode == 'delete':
            self.addGlobalParameter('lambda', 1.0) # parameter switched from 1 to 0 during course of integrating internal 'nsteps' of dynamics

        self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints

        # Compute kT in natural openmm units.
        from openmmtools.constants import kB
        kT = kB * temperature
        kT = kT.value_in_unit_system(unit.md_unit_system)

        # Constrain initial positions and velocities.
        self.addConstrainPositions()
        self.addConstrainVelocities()

        # Store initial total energy.
        self.addComputeGlobal('initial_total_energy', 'ke + potential')

        #
        # Initial Velocity Verlet propagation step
        #

        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

        # Unroll loop over NCMC steps.
        for step in range(nsteps):
            #
            # Alchemical perturbation step
            #

            if nsteps==1:
                delta_lambda = 1.0
            else:
                delta_lambda = float(step)/float(nsteps-1)

            if mode == 'insert':
                self.addComputeGlobal('lambda', 'lambda + %f' % delta_t)
            elif mode == 'delete':
                self.addComputeGlobal('lambda', 'lambda - %f' % delta_t)

            # Update Context parameters according to provided functions.
            for context_parameter in functions:
                self.addComputeGlobal(context_parameter, functions[context_parameter])

            #
            # Velocity Verlet propagation step
            #

            self.addUpdateContextState()
            self.addComputePerDof("v", "v+0.5*dt*f/m")
            self.addComputePerDof("x", "x+dt*v")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
            self.addConstrainVelocities()

        # Store final total energy.
        self.addComputeGlobal('initial_total_energy', 'ke + potential')

        # Compute log acceptance probability.
        self.addComputeGlobal('log_ncmc_acceptance_probability', '(final_total_energy - initial_total_energy) / %f' % kT)

    def getLogAcceptanceProbability(self):
        return self.getGlobalVariableByName('log_ncmc_acceptance_probability')

import numpy as np
import copy
from simtk import openmm, unit


default_functions = {
    'lambda_sterics' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_bonds' : 'lambda',
    'lambda_angles' : 'lambda',
    'lambda_torsions' : 'lambda'
    }

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
    >>> topology_proposal = TopologyProposal(old_system=testsystem.system, old_topology=testsystem.topology, old_positions=testsystem.positions, new_system=testsystem.system, new_topology=testsystem.topology, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())
    >>> from perses.annihilation.alchemical_engine import AlchemicalEliminationEngine
    >>> alchemical_engine = AlchemicalEliminationEngine()
    >>> alchemical_system = alchemical_engine.make_alchemical_system(testsystem.system, topology_proposal, direction='delete')
    >>> ncmc_engine = NCMCEngine(temperature=300.0*unit.kelvin, functions=default_functions, nsteps=50, timestep=1.0*unit.femtoseconds)
    >>> [final_positions, logP] = ncmc_engine.integrate(alchemical_system, testsystem.positions, direction='delete')

    """

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
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        # Make a list of available context parameters.

        # Select subset of switching functions based on which alchemical parameters are present in the system.
        available_parameters = self._getAvailableParameters(alchemical_system)
        functions = { parameter_name : self.functions[parameter_name] for parameter_name in self.functions if (parameter_name in available_parameters) }

        # Create an NCMC velocity Verlet integrator.
        integrator = NCMCAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, timestep=self.timestep, direction=direction)
        # Set the constraint tolerance if specified.
        if self.constraint_tolerance is not None:
            integrator.setConstraintTolerance(self.constraint_tolerance)
        # Create a context on the specified platform.
        if platform is not None:
            context = openmm.Context(alchemical_system, integrator, platform)
        else:
            context = openmm.Context(alchemical_system, integrator)
        context.setPositions(initial_positions)
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
    >>> temperature = 300.0 * unit.kelvin
    >>> functions = { 'alchemical_sterics' : 'lambda' }
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
        self.addGlobalVariable('dti', timestep.in_unit_system(unit.md_unit_system))
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
        self.addComputeGlobal('dti', 'dt/%f' % nsteps)

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

        # Unroll loop over NCMC steps (for nsteps > 1)
        for step in range(nsteps):
            #
            # Alchemical perturbation step
            #

            delta_lambda = 1.0/nsteps
            if direction == 'insert':
                self.addComputeGlobal('lambda', '%f' % (delta_lambda * (step+1)))
            elif direction == 'delete':
                self.addComputeGlobal('lambda', '%f' % (delta_lambda * (nsteps - step - 1)))

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

        # Store final total energy.
        self.addComputeSum("kinetic", "0.5*m*v*v")
        self.addComputeGlobal('final_total_energy', 'kinetic + energy')

        # Compute log acceptance probability.
        self.addComputeGlobal('log_ncmc_acceptance_probability', '(final_total_energy - initial_total_energy) / %f' % kT)

    def getLogAcceptanceProbability(self):
        return self.getGlobalVariableByName('log_ncmc_acceptance_probability')

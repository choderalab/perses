import numpy as np
from simtk import openmm, unit

class NCMCEngine(object):
    def __init__(self, alchemical_system, alchemical_protocol, initial_positions):
        """
        This is the base class for NCMC switching between two different systems.

        Arguments
        ---------
        alchemical_system : simtk.openmm.System object
            alchemically-modified system with atoms to be eliminated
        alchemical_protocol : dict?
            The protocol to use for alchemical introduction or elimination
        initial_positions : [n, 3] numpy.ndarray
            positions of the atoms in the old system

        Properties
        ---------
        log_ncmc : float, read-only
            The contribution of the NCMC move to the acceptance probability
        final_positions : [n,3] numpy.ndarray, read-only
            positions of the system after NCMC switching

        TODO:
        * We also need the temperature to compute log acceptance probability contributions
        * The 'alchemical_protocol' is really an NCMC switching protocol for the alchemical parameters.
          Do we want it to contain the dict of alchemical switching functions, nsteps, and timestep?
          Or should this be encoded by which subclass of NCMCEngine we want?
        * Do we want to also handle creation of the alchemical system here, or keep that separate?
        * Does it make sense to split this into object creation followed immediately by integration,
          or does it make more sense to configure the NCMCEngine with the switching functions at the beginning
          and then have 'integrate' start from the alchemical_system and initial_positions and return the final_positions and log_ncmc?
          That would look more like
          ```python
          # Initialization of run
          ncmc_deletion  = NCMCEngine(alchemical_deletion_protocol)
          ncmc_insertion = NCMCEngine(alchemical_insertion_protocol)
          ...
          # Inside the main loop
          ...
          [final_deletion_positions, log_ncmc_deletion] = ncmc_deletion.integrate(alchemical_deletion_system, initial_deletion_positions)
          # do trans-dimensional RJMC here
          [final_insertion_positions, log_ncmc_insertion] = ncmc_insertion.integrate(alchemical_insertion_system, initial_insertion_positions)
          ...
          ```

        """
        self.temperature = 300 * unit.kelvin # TODO: Need a way to specify temperature
        self.alchemical_system = alchemical_system
        self.initial_positions = initial_positions
        self._log_ncmc = None
        self._final_psoitions = None

    def functions(self):
        """
        Define the dictionary of functions controlling how alchemical switching proceeds.

        """
        raise NotImplementedException()

    def integrate(self, nsteps=10, timestep=1.0*unit.femtoseconds):
        """
        Performs NCMC switching according to the provided
        alchemical_protocol
        """
        # Perform the NCMC switching step.
        integrator = NCMCAlchemicalIntegrator(self.temperature, self.alchemical_system, self.functions(), nsteps=10, timestep=1.0*unit.femtoseconds)
        context = openmm.Context(self.alchemical_system, integrator)
        context.setPositions(self.initial_positions)
        integrator.step(1)
        # Store final positions and log acceptance probability.
        self._log_ncmc = integrator.getLogAcceptanceProbability()
        self._final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        # Clean up.
        del context, integrator

    @property
    def log_ncmc(self):
        return self._log_ncmc

    @property
    def final_positions(self):
        return self._final_positions

class LinearNCMCEngine(NCMCEngine):
    """
    NCMC engine utilizing linear switching.

    """
    def functions(self):
        return { 'lambda_sterics' : 't', 'lambda_electrostatics' : 't', 'lambda_torsions' : 't', 'lambda_angles' : 't**2' }

class NCMCAlchemicalIntegrator(openmm.CustomIntegrator):
    """
    Use NCMC switching to annihilate or introduce particles alchemically.

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
    >>> ncmc_integrator = NCMCAlchemicalIntegrator(alchemical_system, functions, mode='deletion')
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
    >>> ncmc_integrator = NCMCAlchemicalIntegrator(alchemical_system, functions, mode='insertion')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Run the integrator
    >>> ncmc_integrator.step(1)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.log_ncmc


    """
    def __init__(self, temperature, system, functions, nsteps=10, timestep=1.0*unit.femtoseconds, mode='insertion'):
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
        mode : str, optional, default='insertion'
            One of ['insertion', 'deletion'].  For insertion, the parameter 'lambda' is switched from 0 to 1.  For deletion, the parameter 'lambda' is switched from 1 to 0.

        Note that each call to integrator.step(1) executes the entire integration program; this should not be called with more than one step.

        A symmetric protocol is used, in which the protocol begins and ends with a velocity Verlet step.

        TODO:
        * Add a global variable that causes termination of future calls to step(1) after the first

        """
        if (nsteps < 1):
            raise Exception("'nsteps' must be >= 1")

        if mode not in ['intertion', 'deletion']:
            raise Exception("mode must be one of ['insertion', 'deletion']; was '%s' instead" % mode)

        super(NCMCAlchemicalIntegrator, self).__init__((nsteps+1) * timestep)

        self.addGlobalParameter('initial_total_energy', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalParameter('final_total_energy', 0.0) # final total energy (kinetic + potential)
        self.addGlobalParameter('log_ncmc_acceptance_probability', 0.0) # log of NCMC acceptance probability

        if mode == 'insertion':
            self.addGlobalParameter('lambda', 0.0) # parameter switched from 0 to 1 during course of integrating internal 'nsteps' of dynamics
        elif mode == 'deletion':
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

            if mode == 'insertion':
                self.addComputeGlobal('lambda', 'lambda + %f' % delta_t)
            elif mode == 'deletion':
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

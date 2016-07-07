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
    'lambda_sterics' : '2*lambda * step(0.5 - lambda)',
    'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
    #'lambda_sterics' : 'lambda',
    #'lambda_electrostatics' : 'lambda',
    'lambda_bonds' : '0.9*lambda + 0.1', # don't fully soften bonds
    'lambda_angles' : '0.9*lambda + 0.1', # don't fully soften angles
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

    def __init__(self, temperature=default_temperature, functions=None, nsteps=default_nsteps, steps_per_propagation=default_steps_per_propagation, timestep=default_timestep, constraint_tolerance=None, platform=None, write_pdb_interval=None, integrator_type='GHMC', storage=None):
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
        write_pdb_interval : int, optional, default=None
            If a positive integer is specified, a PDB frame will be written with the specified interval on NCMC switching,
            with a different PDB file generated for each attempt.
        integrator_type : str, optional, default='GHMC'
            NCMC internal integrator type ['GHMC', 'VV']
        storage : NetCDFStorageView, optional, default=None
            If specified, write data using this class.
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

        self.write_pdb_interval = write_pdb_interval

        self.nattempted = 0

        self._storage = None
        if storage is not None:
            self._storage = NetCDFStorageView(storage, modname=self.__class__.__name__)

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

    def _updateAlchemicalState(self, context, functions, value):
        """
        Update alchemical state using the specified lambda value.

        Parameters
        ----------
        context : simtk.openmm.Context
            The Context
        functions : dict
            A dictionary of functions
        value : float
            The alchemical lambda value

        TODO: Improve function evaluation to better match Lepton and be more flexible in exact replacement of 'lambda' tokens

        """
        from perses.annihilation import NumericStringParser
        nsp = NumericStringParser()
        for parameter in functions:
            function = functions[parameter]
            evaluated = nsp.eval(function.replace('lambda', str(value)))
            context.setParameter(parameter, evaluated)

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
                context = openmm.Context(system, integrator, self.platform)
            else:
                context = openmm.Context(system, integrator)
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
        alchemical_factory = AbsoluteAlchemicalFactory(unmodified_system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=True, alchemical_bonds=None, alchemical_angles=None, softcore_beta=0.0)

        # Return the alchemically-modified system in fully-interacting form.
        alchemical_system = alchemical_factory.createPerturbedSystem()
        return [unmodified_system, alchemical_system]

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
            For `delete`, the potential energy of the final (alchemically eliminated) conformation.
            For `insert`, the potential energy of the initial (alchemically eliminated) conformation.

        """
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        assert quantity_is_finite(initial_positions) == True

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

        assert quantity_is_finite(initial_positions) == True

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
        if self.integrator_type == 'VV':
            integrator = NCMCVVAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, steps_per_propagation=self.steps_per_propagation, timestep=self.timestep, direction=direction)
        elif self.integrator_type == 'GHMC':
            integrator = NCMCGHMCAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, steps_per_propagation=self.steps_per_propagation, timestep=self.timestep, direction=direction)
        else:
            raise Exception("integrator_type '%s' unknown" % self.integrator_type)
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

        # DEBUG
        from perses.tests.utils import compute_potential
        if direction == 'delete':
            unmodified_potential = self.beta * compute_potential(topology_proposal.old_system, initial_positions, platform=self.platform)
        elif direction == 'insert':
            unmodified_potential = self.beta * compute_potential(topology_proposal.new_system, initial_positions, platform=self.platform)

        # Compute initial potential of alchemical state.
        alchemical_potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()

        # Set initial context parameters.
        if direction == 'insert':
            self._updateAlchemicalState(context, functions, 0)
        elif direction == 'delete':
            self._updateAlchemicalState(context, functions, 1)

        # Compute initial potential of alchemical state.
        initial_potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        # DEBUG
        print("Initial potential of '%s' operation: unmodified potential was %.3f kT, alchemical potential was %.3f kT before changing lambda, %.3f kT after changing lambda" % (direction, unmodified_potential, alchemical_potential, initial_potential))
        #print("Initial potential is %s" % str(initial_potential))
        if np.isnan(initial_potential):
            raise NaNException("Initial potential of 'insert' operation is NaN (unmodified potential was %.3f kT, alchemical potential was %.3f kT before changing lambda)" % (unmodified_potential, alchemical_potential))
        from perses.tests.utils import compute_potential_components
        #print("initial potential before '%s' : %f kT" % (direction, initial_potential))
        #print("initial potential components:   %s" % str(compute_potential_components(context))) # DEBUG
        self.write_pdb_interval = False
        # Integrate switching
        try:
            # Write PDB file if requested.
            if self.write_pdb_interval:
                if direction == 'insert':
                    topology = topology_proposal.new_topology
                    indices = topology_proposal.unique_new_atoms
                else:
                    topology = topology_proposal.old_topology
                    indices = topology_proposal.unique_old_atoms

                # Write atom indices that are changing
                import pickle
                filename = 'ncmc-%s-%d-atomindices.pkl' % (direction, self.nattempted)
                outfile = open(filename, 'wb')
                pickle.dump(indices, outfile)
                outfile.close()

                from simtk.openmm.app import PDBFile
                filename = 'ncmc-%s-%d.pdb' % (direction, self.nattempted)
                outfile = open(filename, 'w')
                PDBFile.writeHeader(topology, file=outfile)
                modelIndex = 0
                PDBFile.writeModel(topology, context.getState(getPositions=True).getPositions(asNumpy=True), file=outfile, modelIndex=modelIndex)
                try:
                    for step in range(self.nsteps):
                        integrator.step(1)
                        if (step+1)%self.write_pdb_interval == 0:
                            modelIndex += 1
                            # TODO: Replace with storage layer
                            PDBFile.writeModel(topology, context.getState(getPositions=True).getPositions(asNumpy=True), file=outfile, modelIndex=modelIndex)
                except ValueError as e:
                    # System is exploding and coordinates won't fit in PDB ATOM fields
                    print(e)

                PDBFile.writeFooter(topology, file=outfile)
                outfile.close()
            else:
                work = np.zeros([self.nsteps+1], np.float64) # work[n] is the accumulated work up to step n
                for step in range(self.nsteps):
                    integrator.step(1)
                    #potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
                    #print("Potential at step %d is %s" % (step, str(potential)))
                    #current_step = integrator.get_step()
                    #print("and the integrator's current step is %d" % current_step)

                    # Store accumulated work
                    work[step+1] = - integrator.getLogAcceptanceProbability(context)

                    # DEBUG
                    Eold = integrator.getGlobalVariableByName("Eold")
                    Enew = integrator.getGlobalVariableByName("Enew")
                    xsum_old = integrator.getGlobalVariableByName("xsum_old")
                    xsum_new = integrator.getGlobalVariableByName("xsum_new")
                    xsum = integrator.getGlobalVariableByName("xsum")
                    print('NCMC step %8d  / %8d %8s : Eold %16.8e Enew %16.8e work %16.8e xsum_old %16.8e xsum_new %16.8e xsum %16.8e' % (step, self.nsteps, direction, Eold, Enew, work[step+1], xsum_old, xsum_new, xsum))
                    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                    assert quantity_is_finite(positions) == True

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

        # Set final context parameters.
        # Set initial context parameters.
        if direction == 'insert':
            self._updateAlchemicalState(context, functions, 1)
        elif direction == 'delete':
            self._updateAlchemicalState(context, functions, 0)

        # DEBUG
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        assert quantity_is_finite(positions) == True

        # Compute final potential of alchemical state.
        final_potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        if np.isnan(final_potential):
            raise NaNException("Final potential of %s operation is NaN" % direction)
        #print("final potential before '%s' : %f kT" % (direction, final_potential))
        #print("final potential components: %s" % str(compute_potential_components(context))) # DEBUG
        #print('')

        # Store final positions and log acceptance probability.
        final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        logP_NCMC = integrator.getLogAcceptanceProbability(context)
        # DEBUG
        #logging.debug("NCMC logP %+10.1f | initial_total_energy %+10.1f kT | final_total_energy %+10.1f kT." % (logP_NCMC, integrator.getGlobalVariableByName('initial_total_energy'), integrator.getGlobalVariableByName('final_total_energy')))
        # Clean up NCMC switching integrator.
        del context, integrator

        # DEBUG
        assert quantity_is_finite(final_positions) == True

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

        # Keep track of statistics.
        self.nattempted += 1

        # Return
        return [final_positions, logP, potential]

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
        if self.direction == 'insert':
            self.addComputeGlobal('lambda', '0.0')
        elif self.direction == 'delete':
            self.addComputeGlobal('lambda', '1.0')

        self.addUpdateAlchemicalParametersStep()

    def addAlchemicalPerturbationStep(self):
        """
        Add alchemical perturbation step.
        """
        if self.nsteps == 0:
            if self.direction == 'insert':
                self.addComputeGlobal('lambda', '1.0')
            elif self.direction == 'delete':
                self.addComputeGlobal('lambda', '0.0')
        else:
            if self.direction == 'insert':
                self.addComputeGlobal('lambda', '(step+1)/nsteps')
            elif self.direction == 'delete':
                self.addComputeGlobal('lambda', '(nsteps - step - 1)/nsteps')

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
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + v*dt")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        self.addConstrainVelocities()
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeSum("xsum", "(x-xold)^2")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform) * step(xsum)")
        self.beginIfBlock("accept != 1")
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

    def getLogAcceptanceProbability(self, context):
        logp_accept = -1.0*self.getGlobalVariableByName("total_work") * unit.kilojoules_per_mole / self.kT
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
        self.addGlobalVariable('total_work', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable('kinetic', 0.0) # kinetic energy

        # VV variables
        if (nsteps > 0):
            # VV variables
            self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
            self.addGlobalVariable('step', 0) # current NCMC step number
            self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints
            self.addGlobalVariable('psteps', steps_per_propagation)
            self.addGlobalVariable('pstep', 0)

        # Constrain initial positions and velocities.
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.addUpdateContextState()

        if nsteps == 0:
            self.addAlchemicalResetStep()
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
        if nsteps > 0:
            self.addComputeGlobal('pstep', '0')
            # Initial step only
            self.beginIfBlock('step = 0')
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Eold", "energy + kinetic")
            self.beginWhileBlock('pstep < psteps')
            self.addVelocityVerletStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Enew", "energy + kinetic")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
            self.endBlock()

            # All steps
            self.beginIfBlock('step < nsteps')
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Eold", "energy + kinetic")
            self.addAlchemicalPerturbationStep()
            self.beginWhileBlock('pstep < psteps')
            self.addVelocityVerletStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Enew", "energy + kinetic")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
            self.addComputeGlobal('step', 'step+1')
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
        self.addGlobalVariable('lambda', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalVariable('total_work', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        # DEBUG
        self.addGlobalVariable("xsum_old", 0.0)
        self.addGlobalVariable("xsum_new", 0.0)

        if (nsteps > 0):
            # GHMC variables
            self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
            self.addGlobalVariable('step', 0) # current NCMC step number
            self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints
            self.addGlobalVariable("kT", self.kT.value_in_unit_system(unit.md_unit_system))  # thermal energy
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
            self.addGlobalVariable("xsum", 0.0) # sum of (x-xold)^2

        # Constrain initial positions and velocities.
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.addUpdateContextState()

        if nsteps == 0:
            self.addAlchemicalResetStep()
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
        if nsteps > 0:
            #self.addComputeGlobal('pstep', '0')
            # Initial step only
            #self.beginIfBlock('step = 0')
            #self.beginWhileBlock('pstep < psteps')
            #self.addComputeSum("xsum_old", "x") # DEBUG
            #self.addGHMCStep()
            #self.addComputeSum("xsum_new", "x") # DEBUG
            #self.addComputeGlobal('pstep', 'pstep+1')
            #self.endBlock()
            #self.endBlock()

            # All steps
            self.beginIfBlock('step < nsteps')
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
            #self.beginWhileBlock('pstep < psteps')
            self.addComputeSum("xsum_old", "x") # DEBUG
            self.addGHMCStep()
            self.addComputeSum("xsum_new", "x") # DEBUG
            #self.addComputeGlobal('pstep', 'pstep+1')
            #self.endBlock()
            self.addComputeGlobal('step', 'step+1')
            self.endBlock()

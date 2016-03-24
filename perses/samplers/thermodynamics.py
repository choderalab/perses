#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Thermodynamic ensemble utilities.

DESCRIPTION

This module provides a utilities for simulating physical thermodynamic ensembles.

Provided classes include:

* ThermodynamicState - Data specifying a thermodynamic state obeying Boltzmann statistics (System/temperature/pressure/pH)

DEPENDENCIES

TODO

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import copy
import numpy as np

import simtk.openmm as mm
import simtk.unit as units
from openmmtools import testsystems

import logging
logger = logging.getLogger(__name__)

#=============================================================================================
# REVISION CONTROL
#=============================================================================================

__version__ = "$Revision: $"

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# Thermodynamic state description
#=============================================================================================


class ThermodynamicState(object):
    """Object describing a thermodynamic state obeying Boltzmann statistics.

    Examples
    --------

    Specify an NVT state for a water box at 298 K.

    >>> system_container = testsystems.WaterBox()
    >>> (system, positions) = system_container.system, system_container.positions
    >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin)

    Get the inverse temperature
    
    >>> beta = state.beta

    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin, pressure=1.0*units.atmospheres)

    Note that the pressure is only relevant for periodic systems.

    A barostat will be added to the system if none is attached.

    Notes
    -----

    This state object cannot describe states obeying non-Boltzamnn statistics, such as Tsallis statistics.

    ToDo
    ----

    * Implement a more fundamental ProbabilityState as a base class?
    * Implement pH.

    """
    def __init__(self, system, temperature, pressure=None):
        """Construct a thermodynamic state with given system and temperature.

        Parameters
        ----------

        system : simtk.openmm.System
            System object describing the potential energy function
            for the system (default: None)
        temperature : simtk.unit.Quantity, compatible with 'kelvin'
            Temperature for a system with constant temperature
        pressure : simtk.unit.Quantity,  compatible with 'atmospheres', optional, default=None
            If not None, specifies the pressure for constant-pressure systems.


        """

        # Initialize.
        self.system = None          # the System object governing the potential energy computation
        self.temperature = None     # the temperature
        self.pressure = None        # the pressure, or None if not isobaric

        self._cache_context = True  # if True, try to cache Context object
        self._context = None        # cached Context
        self._integrator = None     # cached Integrator

        # Store provided values.
        if system is not None:
            if type(system) is not mm.System:
                raise(TypeError("system must be an OpenMM System; instead found %s" % type(system)))
            self.system = copy.deepcopy(system) # TODO: Do this when deep copy works.
            # self.system = system # we make a shallow copy for now, which can cause trouble later
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure

        # If temperature and pressure are specified, make sure MonteCarloBarostat is attached.
        if temperature and pressure:
            # Try to find barostat.
            barostat = False
            for force_index in range(self.system.getNumForces()):
                force = self.system.getForce(force_index)
                # Dispatch forces
                if isinstance(force, mm.MonteCarloBarostat):
                    barostat = force
                    break
            if barostat:
                # Set temperature.
                # TODO: Set pressure too, once that option is available.
                barostat.setTemperature(temperature)
            else:
                # Create barostat.
                barostat = mm.MonteCarloBarostat(pressure, temperature)
                self.system.addForce(barostat)

        return

    def _create_context(self, platform=None):
        """Create Integrator and Context objects if they do not already exist.
        """

        # Check if we already have a Context defined.
        if self._context:
            #if platform and (platform != self._context.getPlatform()): # TODO: Figure out why requested and cached platforms differed in tests.
            if platform and (platform.getName() != self._context.getPlatform().getName()): # DEBUG: Only compare Platform names for now; change this later to incorporate GPU IDs.
                # Platform differs from the one requested; destroy it.
                logger.info((platform.getName(), self._context.getPlatform().getName()))
                logger.debug("Platform differs from the one requested; destroying and recreating...")
                del self._context, self._integrator
            else:
                # Cached context is what we expect; do nothing.
                return

        # Create an integrator.
        timestep = 1.0 * units.femtosecond
        self._integrator = mm.VerletIntegrator(timestep)

        # Create a new OpenMM context.
        if platform:
            self._context = mm.Context(self.system, self._integrator, platform)
        else:
            self._context = mm.Context(self.system, self._integrator)

        logger.debug("_create_context created a new integrator and context")


    def _cleanup_context(self):
        del self._context, self._integrator
        self._context = None
        self._integrator = None

    def _compute_potential(self, coordinates, box_vectors):
        # Set periodic box vectors first, or else coordinates will wrap improperly.
        if box_vectors is not None: self._context.setPeriodicBoxVectors(*box_vectors)

        # Set coordinates.
        self._context.setPositions(coordinates)

        # Retrieve potential energy.
        openmm_state = self._context.getState(getEnergy=True)
        potential_energy = openmm_state.getPotentialEnergy()

        return potential_energy

    def reduced_potential(self, coordinates, box_vectors=None, platform=None):
        """Compute the reduced potential for the given coordinates in this thermodynamic state.

        Parameters
        ----------

        coordinates : simtk.unit.Quantity wrapped numpy ndarray, shape=(N, 3)
            coordinates[n,k] is kth coordinate of particle n
        box_vectors :
            periodic box vectors

        Returns
        -------
        u : float
            The unitless reduced potential (which can be considered to have units of kT)

        Examples
        --------

        Compute the reduced potential of a Lennard-Jones cluster at 100 K.

        >>> system_container = testsystems.LennardJonesCluster()
        >>> (system, positions) = system_container.system, system_container.positions
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)
        >>> potential = state.reduced_potential(positions)

        Compute the reduced potential of a Lennard-Jones fluid at 100 K and 1 atm.

        >>> system_container = testsystems.LennardJonesFluid()
        >>> (system, positions) = system_container.system, system_container.positions
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin, pressure=1.0*units.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> potential = state.reduced_potential(positions, box_vectors)

        Compute the reduced potential of a water box at 298 K and 1 atm.

        >>> system_container = testsystems.WaterBox()
        >>> (system, positions) = system_container.system, system_container.positions
        >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin, pressure=1.0*units.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> potential = state.reduced_potential(positions, box_vectors)

        Notes
        -----

        The reduced potential is defined as in Ref. [1]

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is he inverse temperature
        U(x) is the potential energy
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of titratible groups)

        References
        ----------

        [1] Shirts MR and Chodera JD. Statistically optimal analysis of equilibrium states. J Chem Phys 129:124105, 2008.

        TODO
        ----

        * Instead of requiring configuration and box_vectors be passed separately, develop a Configuration or Snapshot class.

        """

        # If pressure is specified, ensure box vectors have been provided.
        if (self.pressure is not None) and (box_vectors is None):
            raise ValueError("box_vectors must be specified if constant-pressure ensemble.")

        # Make sure we have Context and Integrator objects.
        self._create_context(platform)

        # Compute energy.
        try:
            potential_energy = self._compute_potential(coordinates, box_vectors)
        except Exception as e:
            logger.info(e)

            # Our cached context failed, so try deleting it and creating it anew.
            self._cleanup_context()
            self._create_context()

            # Compute energy
            potential_energy = self._compute_potential(coordinates, box_vectors)

        # Compute inverse temperature.
        beta = 1.0 / (kB * self.temperature)

        # Compute reduced potential.
        reduced_potential = beta * potential_energy
        if self.pressure is not None:
            reduced_potential += beta * self.pressure * volume(box_vectors) * units.AVOGADRO_CONSTANT_NA

        # Clean up context if requested, or if we're using Cuda (which can only have one active Context at a time).
        if (not self._cache_context) or (self._context.getPlatform().getName() == 'Cuda'):
            self._cleanup_context()

        return reduced_potential

    @property
    def beta(self):
        return (1.0 / (kB * self.temperature))

    def reduced_potential_multiple(self, coordinates_list, box_vectors_list=None, platform=None):
        """Compute the reduced potential for the given sets of coordinates in this thermodynamic state.

        This can pontentially be more efficient than repeated calls to reduced_potential.

        Parameters
        ----------

        coordinates_list : list of simtk.unit.Quantity wrapped Nx3 numpy.arrays)
            coordinates_list[i][n,k] is kth coordinate of particle n from list i

        box_vectors :
            periodic box vectors

        Returns
        -------

        u_k : np.ndarray, shape=(K), dtype=float
            The unitless reduced potentials (which can be considered to have units of kT)

        Examples
        --------

        Compute the reduced potential of a Lennard-Jones cluster at multiple configurations at 100 K.

        >>> system_container = testsystems.LennardJonesCluster()
        >>> (system, positions) = system_container.system, system_container.positions
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)
        >>> # create example list of coordinates
        >>> import copy
        >>> coordinates_list = [ copy.deepcopy(positions) for i in range(10) ]
        >>> # compute potential for all sets of coordinates
        >>> potentials = state.reduced_potential_multiple(coordinates_list)

        Notes
        -----

        The reduced potential is defined as in Ref. [1]

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is he inverse temperature
        U(x) is the potential energy
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of titratible groups)

        References
        ----------

        [1] Shirts MR and Chodera JD. Statistically optimal analysis of equilibrium states. J Chem Phys 129:124105, 2008.

        TODO
        ----

        * Instead of requiring configuration and box_vectors be passed separately, develop a Configuration or Snapshot class.

        """

        # If pressure is specified, ensure box vectors have been provided.
        if (self.pressure is not None) and (box_vectors_list is None):
            raise ValueError("box_vectors must be specified if constant-pressure ensemble.")

        # Make sure we have Context and Integrator objects.
        self._create_context(platform)

        # Allocate storage.
        K = len(coordinates_list)
        u_k = np.zeros([K], np.float64)

        # Compute energies.
        for k in range(K):
            # Compute energy
            if box_vectors_list:
                potential_energy = self._compute_potential(coordinates_list[k], box_vectors_list[k])
            else:
                potential_energy = self._compute_potential(coordinates_list[k], None)

            # Compute inverse temperature.
            beta = 1.0 / (kB * self.temperature)

            # Compute reduced potential.
            u_k[k] = beta * potential_energy
            if self.pressure is not None:
                u_k[k] += beta * self.pressure * volume(box_vectors_list[k]) * units.AVOGADRO_CONSTANT_NA

        # Clean up context if requested, or if we're using Cuda (which can only have one active Context at a time).
        if (not self._cache_context) or (self._context.getPlatform().getName() == 'Cuda'):
            self._cleanup_context()

        return u_k

    def is_compatible_with(self, state):
        """Determine whether another state is in the same thermodynamic ensemble (e.g. NVT, NPT).

        Parameters
        ----------

        state : ThermodynamicState
            thermodynamic state whose compatibility is to be determined

        Returns
        -------

        is_compatible : bool
            True if 'state' is of the same ensemble (e.g. both NVT, both NPT), False otherwise

        Examples
        --------

        Create NVT and NPT states.

        >>> system_container = testsystems.LennardJonesCluster()
        >>> (system, positions) = system_container.system, system_container.positions
        >>> nvt_state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)
        >>> npt_state = ThermodynamicState(system=system, temperature=100.0*units.kelvin, pressure=1.0*units.atmospheres)

        Test compatibility.

        >>> test1 = nvt_state.is_compatible_with(nvt_state)
        >>> test2 = nvt_state.is_compatible_with(npt_state)
        >>> test3 = npt_state.is_compatible_with(nvt_state)
        >>> test4 = npt_state.is_compatible_with(npt_state)

        """

        is_compatible = True

        # Make sure systems have the same number of atoms.
        if (self.system.getNumParticles() != state.system.getNumParticles()):
            is_compatible = False

        # Make sure other terms are defined for both states.
        # TODO: Use introspection to get list of parameters?
        for parameter in ['temperature', 'pressure']:
            if (parameter in dir(self)) is not (parameter in dir(state)):
                # parameter is not shared by both states
                is_compatible = False

        return is_compatible

    def __repr__(self):
        """Returns a string representation of a state.

        Examples
        --------

        Create an NVT state.

        >>> system_container = testsystems.LennardJonesCluster()
        >>> (system, positions) = system_container.system, system_container.positions
        >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin)

        Return a representation of the state.

        >>> state_string = repr(state)

        """

        r = "<ThermodynamicState object"
        if self.temperature is not None:
            r += ", temperature = %s" % str(self.temperature)
        if self.pressure is not None:
            r += ", pressure = %s" % str(self.pressure)
        r += ">"

        return r

    def __str__(self):
        # TODO: Write a human-readable representation.

        return repr(self)

def volume(box_vectors):
    """Return the volume of the current configuration.

    Parameters
    ----------
    box_vectors : simtk.unit.Quantity
        Box vectors of the configuration in question.

    Returns
    -------

    volume : simtk.unit.Quantity
        The volume of the system (in units of length^3), or None if no box coordinates are defined

    Examples
    --------

    Compute the volume of a Lennard-Jones fluid at 100 K and 1 atm.

    >>> system_container = testsystems.LennardJonesFluid()
    >>> (system, positions) = system_container.system, system_container.positions
    >>> state = ThermodynamicState(system=system, temperature=100.0*units.kelvin, pressure=1.0*units.atmosphere)
    >>> box_vectors = system.getDefaultPeriodicBoxVectors()
    >>> v = volume(box_vectors)

    """

    # Compute volume of parallelepiped.
    [a,b,c] = box_vectors
    A = np.array([a/a.unit, b/a.unit, c/a.unit])
    volume = np.linalg.det(A) * a.unit**3
    return volume

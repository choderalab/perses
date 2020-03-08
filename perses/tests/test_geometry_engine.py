__author__ = 'Patrick B. Grinaway'

import simtk.openmm as openmm
import openeye.oechem as oechem
import openmoltools
import openeye.oeiupac as oeiupac
import openeye.oeomega as oeomega
import simtk.openmm.app as app
import simtk.unit as unit
import logging
import numpy as np
import parmed
from collections import namedtuple, OrderedDict
import copy
from unittest import skipIf
from pkg_resources import resource_filename
try:
    from urllib.request import urlopen
    from io import StringIO
except:
    from urllib2 import urlopen
    from cStringIO import StringIO
import os
try:
    from subprocess import getoutput  # If python 3
except ImportError:
    from commands import getoutput  # If python 2
from nose.plugins.attrib import attr
from openmmtools.constants import kB
from perses.rjmc import coordinate_numba

from perses.rjmc.geometry import check_dimensionality
from perses.utils.data import get_data_filename


from nose.tools import nottest #protein mutations will be omitted (for the time being)

################################################################################
# TODO: Look into this for later
# https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
################################################################################
# Suppress matplotlib logging
################################################################################

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

################################################################################
# Global parameters
################################################################################

#correct p-value threshold for some multiple hypothesis testing
pval_base = 0.01
ntests = 3.0
ncommits = 10000.0

pval_threshold = pval_base / (ntests * ncommits)
temperature = 300.0 * unit.kelvin # unit-bearing temperature
kT = kB * temperature # unit-bearing thermal energy
beta = 1.0/kT # unit-bearing inverse thermal energy
CARBON_MASS = 12.01 # float (implicitly in units of AMU)
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("Reference")
proposal_test = namedtuple("proposal_test", ["topology_proposal", "current_positions"])
running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'
LOGP_FORWARD_THRESHOLD = 1e3

################################################################################
# Tests
################################################################################

class GeometryTestSystem(object):
    """
    A base class for a special set of test systems for the GeometryEngines.
    These systems should, unlike PersesTestSystem, expose certain features.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        the openmm Topology of the relevant system
    system : simtk.openmm.System
        the openmm system, containing relevant forces
    structure : parmed.Structure
        a parmed structure object with all parameters of the system
    growth_order : list of int
        list of indices for the growth order
    positions : [n,3] ndarray of float
        positions of atoms
    energy : Quantity, kJ/mol
        The current potential energy of the system calculated with OpenMM
    """

    @property
    def topology(self):
        return self._topology

    @property
    def system(self):
        return self._system

    @property
    def structure(self):
        return self._structure

    @property
    def growth_order(self):
        return self._growth_order

    @property
    def positions(self):
        return self._positions

    @property
    def energy(self):
        self._context.setPositions(self._positions)
        return self._context.getState(getEnergy=True).getPotentialEnergy()

class FourAtomValenceTestSystem(GeometryTestSystem):
    """
    Test system containing four particles, including a bond, angle, and torsion term in the potential.

    The particles are 0-1-2-3 atom-bond-angle-torsion.

    The positions for the atoms were taken from an earlier test for the geometry engine.

    Properties
    ----------
    internal_coordinates : array of floats
        The [r, theta, phi] internal coordinates of atom 0, implicitly in units of nanometers, radians, and radians
    bond_parameters : tuple of (Quantity, Quantity)
        The (equilibrium bond length, equilibrium force constant) in units compatible with nanometers and kJ/(mol*nm^2), atoms 0-1
    angle_parameters : tuple of (Quantity, Quantity)
        The (equilibrium angle, force constant) in units compatible with radians and kJ/(mol*rad^2), atoms 0-1-2
    torsion_parameters : tuple of (int, Quantity, Quantity)
        The (periodicity, phase, force constant) in units compatible with radians and kJ/mol respectively, atoms 0-1-2-3
    """
    # TODO: Change everything to always work in unit-bearing quantities
    def __init__(self, bond=True, angle=True, torsion=True):
        """
        Create a test system with four particles, including the potential for a bond, angle, torsion term.

        The particles are 0-1-2-3 atom-bond-angle-torsion.
        The positions for the atoms were taken from an earlier test for the geometry engine.

        Parameters
        ---------
        bond : Boolean, default True
            Whether to include the bond force term
        angle : Boolean, default True
            Whether to include the angle force term
        torsion : Boolean, default True
            Whether to include the torsion force term
        """
        #make a simple set of positions. These are taken from another test used for testing the torsions in GeometryEngine
        self._default_positions = unit.Quantity(np.zeros([4,3]), unit=unit.nanometer)
        self._default_positions[0] = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
        self._default_positions[1] = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
        self._default_positions[2] = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
        self._default_positions[3] = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)

        #use parameters taken from various parts of the AlanineDipeptideTestSystem
        self._default_r0 = unit.Quantity(value=0.1522, unit=unit.nanometer)
        self._default_bond_k = unit.Quantity(value=265265.60000000003, unit=unit.kilojoules_per_mole/unit.nanometer**2)
        self._default_angle_theta0 = unit.Quantity(value=1.91113635, unit=unit.radian)
        self._default_angle_k = unit.Quantity(value=418.40000000000003, unit=unit.kilojoules_per_mole/unit.radian**2)
        self._default_torsion_periodicity = 2
        self._default_torsion_phase = unit.Quantity(value=np.pi/2.0, unit=unit.radians)
        self._default_torsion_k = unit.Quantity(value=20.0, unit=unit.kilojoules_per_mole)

        #set up a topology with the appropriate atoms (make them all carbon)
        self._topology = app.Topology()
        new_chain = self._topology.addChain("0")
        new_res = self._topology.addResidue("MOL", new_chain)
        atom1 = self._topology.addAtom("C1", app.Element.getByAtomicNumber(6), new_res, 0)
        atom2 = self._topology.addAtom("C2", app.Element.getByAtomicNumber(6), new_res, 1)
        atom3 = self._topology.addAtom("C3", app.Element.getByAtomicNumber(6), new_res, 2)
        atom4 = self._topology.addAtom("C4", app.Element.getByAtomicNumber(6), new_res, 3)

        #add the bonds to make a linear molecule 1-2-3-4
        self._topology.addBond(atom1, atom2)
        self._topology.addBond(atom2, atom3)
        self._topology.addBond(atom3, atom4)

        #create a system using the same particle information
        self._system = openmm.System()
        indices = [self._system.addParticle(CARBON_MASS) for i in range(4)]

        #the growth order includes only the 0th atom, since there are only four atoms total
        self._growth_order = [0]

        #if the user has specified that a bond force should be used, add it with the appropriate constants
        if bond:
            bond_force = openmm.HarmonicBondForce()
            self._system.addForce(bond_force)
            bond_force.addBond(0, 1, self._default_r0, self._default_bond_k)

        #if the user has specified that an angle force should be used, add it with the appropriate constants
        if angle:
            angle_force = openmm.HarmonicAngleForce()
            self._system.addForce(angle_force)
            angle_force.addAngle(0, 1, 2, self._default_angle_theta0, self._default_angle_k)

        #if the user has specified that a torsion force should be used, add it with the appropriate constants
        if torsion:
            torsion_force = openmm.PeriodicTorsionForce()
            self._system.addForce(torsion_force)
            torsion_force.addTorsion(0, 1, 2, 3, self._default_torsion_periodicity, self._default_torsion_phase, self._default_torsion_k)

        #Now make a ParmEd structure from the topology and system, which will include relevant force parameters
        self._structure = parmed.openmm.load_topology(self._topology, self._system)

        #initialize class memers with the appropriate values
        self._positions = self._default_positions
        self._integrator = openmm.VerletIntegrator(0.0)
        self._platform = openmm.Platform.getPlatformByName("Reference") #use reference for stability

        #create a context and set positions so we can get potential energies
        self._context = openmm.Context(self._system, self._integrator, self._platform)
        self._context.setPositions(self._positions)

    @property
    def internal_coordinates(self):
        """Internal coordinates (r, theta, phi) as tuple of floats (implicitly in simtk.unit.md_unit_system)"""
        # TODO: Why does getting the internal coordinates return floats while setting the internal coordinates requires unit-bearing quantities?
        positions_without_units = self._positions.value_in_unit(unit.nanometer).astype(np.float64)
        internals = coordinate_numba.cartesian_to_internal(positions_without_units[0], positions_without_units[1], positions_without_units[2], positions_without_units[3])
        return internals

    @internal_coordinates.setter
    def internal_coordinates(self, internal_coordinates):
        """Set positions given dimensionless internal coordinates (r, theta, phi), implicitly in simtk.unit.md_system units"""
        # TODO: Why does getting the internal coordinates return floats while setting the internal coordinates requires unit-bearing quantities?
        internals_without_units = np.array(internal_coordinates, dtype=np.float64)
        positions_without_units = self._positions.value_in_unit(unit.nanometer).astype(np.float64)
        new_cartesian_coordinates = coordinate_numba.internal_to_cartesian(positions_without_units[1], positions_without_units[2], positions_without_units[3], internal_coordinates)
        self._positions[0] = unit.Quantity(new_cartesian_coordinates, unit=unit.nanometer)

    @property
    def bond_parameters(self):
        """Bond parameters (r0, k) as tuple of (Quantity, Quantity) unit-bearing quantities"""
        return (self._default_r0, self._default_bond_k)

    @property
    def angle_parameters(self):
        """Angle parmeters (theta0, k) as tuple of (Quantity, Quantity) unit-bearing quantities"""
        return (self._default_angle_theta0, self._default_angle_k)

    @property
    def torsion_parameters(self):
        """Torsion parameters (periodicity, phase, k) as tuple of (Quantity, int, Quantity) unit-bearing quantities"""
        return (self._default_torsion_periodicity, self._default_torsion_phase, self._default_torsion_k)

def test_propose_bond():
    """
    Test the proposal of bonds by GeometryEngine by comparing to proposals from a normal distribution
    with mean r0 (equilibrium bond length) and variance sigma = sqrt(1.0/(k*beta)), where k is the force
    constant and beta is the inverse temperature. A Kolmogorov-Smirnov test is used for that comparison.
    """
    NSAMPLES = 1000 # number of samples to draw
    NDIVISIONS = 1000 # number of bond divisions

    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import scipy.stats as stats
    geometry_engine = FFAllAngleGeometryEngine()

    #create a test system with just a bond from the 1-2 atoms
    testsystem = FourAtomValenceTestSystem(bond=True, angle=False, torsion=False)

    #Retrive this bond force's parameters
    bond = testsystem.structure.bonds[0] #this bond has parameters
    bond_with_units = geometry_engine._add_bond_units(bond)
    (r0, k) = testsystem.bond_parameters

    #compute the equivalent parameters for a normal distribution
    sigma = unit.sqrt(1.0/(beta*k))
    r0_without_units = r0.value_in_unit(unit.nanometers)
    sigma_without_units = sigma.value_in_unit(unit.nanometers)

    #Allocate an array for sampled bond lengths and propose from the geometry engine
    bond_array = np.zeros(NSAMPLES, np.float64)
    for i in range(NSAMPLES):
        bond_array[i] = geometry_engine._propose_bond(bond_with_units, beta, NDIVISIONS)

    # Compute the CDF
    r_i, log_p_i, bin_width = geometry_engine._bond_log_pmf(bond_with_units, beta, NDIVISIONS)
    assert np.isclose(np.exp(log_p_i).sum(), 1.0), 'bond probability mass function is not normalized'
    def cdf(rs):
        cdfs = np.zeros(rs.shape)
        pdf = np.exp(log_p_i)
        for i, r in enumerate(rs):
            index = int((r-r_i[0])/bin_width)
            cdfs[i] = np.sum(pdf[:index]) + (r - r_i[index]) / bin_width * pdf[index]
        return cdfs

    #compare the sampled angles to a normal cdf with the appropriate parameters using
    #the Kolomogorov-Smirnov test. The null hypothesis is that they are drawn from the same
    #distribution (the test passes).
    (dval, pval) = stats.kstest(bond_array, cdf)
    if pval < pval_threshold:
        raise Exception("The bond may be drawn from the wrong distribution. p = %f" % pval)

def test_propose_angle():
    """
    Test the proposal of angles by GeometryEngine by comparing to proposals from a normal distribution
    with mean theta0 (equilibrium angle) and variance sigma = sqrt(1.0/(k*beta)), where k is the force
    constant and beta is the inverse temperature. A Kolmogorov-Smirnov test is used for that comparison.
    """
    NSAMPLES = 1000 # number of samples to draw
    NDIVISIONS = 180 # number of divisions for angle

    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import scipy.stats as stats
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a test system with only an angle force
    testsystem = FourAtomValenceTestSystem(bond=False, angle=True, torsion=False)

    #There is only one angle in this system--extract it
    angle = testsystem.structure.angles[0]
    angle_with_units = geometry_engine._add_angle_units(angle)

    #extract the parameters and convert them to the equivalents for a normal distribution
    #without units
    (theta0, k) = testsystem.angle_parameters
    sigma = unit.sqrt(1.0/(beta*k)) # standard deviation, in unit-bearing quantities
    theta0_without_units = theta0.value_in_unit(unit.radian) # reference angle in float (implicitly in md_unit_system)
    sigma_without_units = sigma.value_in_unit(unit.radian) # standard deviation in float (implicitly in md_unit_system)

    #allocate an array for proposing angles from the appropriate distribution
    angle_array = np.zeros(NSAMPLES)
    for i in range(NSAMPLES):
        angle_array[i] = geometry_engine._propose_angle(angle_with_units, beta, NDIVISIONS)

    # Compute the CDF
    theta_i, log_p_i, bin_width = geometry_engine._angle_log_pmf(angle_with_units, beta, NDIVISIONS)
    assert np.isclose(np.exp(log_p_i).sum(), 1.0), 'angle probability mass function is not normalized'
    def cdf(thetas):
        cdfs = np.zeros(thetas.shape)
        pdf = np.exp(log_p_i)
        for i, theta in enumerate(thetas):
            index = int((theta - theta_i[0]) / bin_width)
            cdfs[i] = np.sum(pdf[:index]) + (theta - theta_i[index]) / bin_width * pdf[index]
        return cdfs

    #compare the sampled angles to a normal cdf with the appropriate parameters using
    #the Kolomogorov-Smirnov test. The null hypothesis is that they are drawn from the same
    #distribution (the test passes).
    (dval, pval) = stats.kstest(angle_array, cdf)
    if pval < pval_threshold:
        raise Exception("The angle may be drawn from the wrong distribution. p = %f" % pval)

def test_bond_logp():
    """
    Compare the bond log probability calculated by the geometry engine to the log-unnormalized
    probability calculated by openmm (-beta*potential_energy, where beta is inverse temperature)
    including the r^2 Jacobian term.
    """
    NDIVISIONS = 1000 # number of divisions to test
    TOLERANCE = 0.05 # logP deviation tolerance

    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with only a bond force
    testsystem = FourAtomValenceTestSystem(bond=True, angle=False, torsion=False)

    #Extrct the bond and its parameters
    #r0 - equilibrium bond length
    #k - force constant
    bond = testsystem.structure.bonds[0] #this bond has parameters
    bond_with_units = geometry_engine._add_bond_units(bond)
    (r0, k) = testsystem.bond_parameters

    #Convert the bond parameters to the normal distribution `sigma` (variance) and use r0 as mean
    sigma = unit.sqrt(1.0/(beta*k))
    sigma_without_units = sigma.value_in_unit(unit.nanometer)
    r0_without_units = r0.value_in_unit(unit.nanometer)

    #Create a set of bond lengths to test from r0 - 6*sigma to r0 + 2*sigma
    EPSILON = 1.0e-3 # we can't have zero bond lengths
    r_min = max(EPSILON, r0_without_units - 2.0*sigma_without_units)
    r_max = r0_without_units + 2.0*sigma_without_units
    bond_range = np.linspace(r_min, r_max, NDIVISIONS)

    #Extract the internal coordinates and add units.
    #r - bond length
    #theta - bond angle
    #phi - torsion angle
    internal_coordinates = testsystem.internal_coordinates
    r, theta, phi = internal_coordinates # dimensionless, implicitly in md_unit_system

    #Loop through the bond lengths and make sure the unnormalized log probability calculated by the geometry engine (logq)
    #matches that calculated by openmm (-beta*potential_energy) after normalization within 1.0e-6 tolerance
    logp_openmm = np.zeros(bond_range.shape)
    logp_ge = np.zeros(bond_range.shape)
    for i, bond_length in enumerate(bond_range):
        logp_ge[i] = geometry_engine._bond_logp(bond_length, bond_with_units, beta, NDIVISIONS)
        internal_coordinates[0] = bond_length
        testsystem.internal_coordinates = internal_coordinates
        logp_openmm[i] = 2.0*np.log(bond_length) - beta*testsystem.energy # Jacobian is included

    # Because we don't know the normalizing constant for the OpenMM logp, find optimal additive constant
    logp_openmm += logp_ge.max() - logp_openmm.max()

    # Check that all values in computed range are close
    max_deviation = (abs(logp_openmm - logp_ge)).max()
    if max_deviation > TOLERANCE:
        msg  = "Max deviation exceeded: {} (tolerance {})".format(max_deviation, TOLERANCE)
        raise Exception(msg)

def test_angle_logp():
    """
    Compare the angle log probability calculated by the geometry engine to the log-unnormalized
    probability calculated by openmm (-beta*potential_energy, where beta is inverse temperature)
    including the sin(theta) Jacobian term.
    """
    NDIVISIONS = 3600
    TOLERANCE = 0.05 # logP deviation tolerance

    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with only an angle
    testsystem = FourAtomValenceTestSystem(bond=False, angle=True, torsion=False)

    #Retrieve that angle and add units to it
    angle = testsystem.structure.angles[0]
    angle_with_units = geometry_engine._add_angle_units(angle)
    (theta0, k) = testsystem.angle_parameters

    #Convert the bond parameters to the normal distribution `sigma` (variance) and use r0 as mean
    sigma = unit.sqrt(1.0/(beta*k))
    sigma_without_units = sigma.value_in_unit(unit.radians)
    theta0_without_units = theta0.value_in_unit(unit.radians)

    #Retrieve internal coordinates and add units
    #r - bond length
    #theta - bond angle
    #phi - torsion angle
    internal_coordinates = testsystem.internal_coordinates
    r, theta, phi = internal_coordinates # float, implicitly in md_unit_system

    #Get a range of test points for the angle from 0 to pi, and add units
    EPSILON = 0.1 # we can't exactly have 0 or pi
    theta_min = max(EPSILON, theta0_without_units - 2.0*sigma_without_units)
    theta_max = min(np.pi-EPSILON, theta0_without_units + 2.0*sigma_without_units)
    angle_test_range = np.linspace(theta_min, theta_max, NDIVISIONS)

    #Loop through the test points for the angle and calculate the log-unnormalized probability of each using
    #geometry engine and openmm (-beta*potential_energy) where beta is inverse temperature. Tolerance 1.0e-4,
    #because this tries a range of angles well outside what one would typically see.
    logp_openmm = np.zeros(angle_test_range.shape)
    logp_ge = np.zeros(angle_test_range.shape)
    for i, test_angle in enumerate(angle_test_range):
        logp_ge[i] = geometry_engine._angle_logp(test_angle, angle_with_units, beta, NDIVISIONS)
        internal_coordinates[1] = test_angle
        testsystem.internal_coordinates = internal_coordinates
        logp_openmm[i] = np.log(np.sin(test_angle)) - beta*testsystem.energy # Jacobian is included

    # Because we don't know the normalizing constant for the OpenMM logp, find optimal additive constant
    logp_openmm += logp_ge.max() - logp_openmm.max()

    max_deviation = (abs(logp_openmm - logp_ge)).max()
    if max_deviation > TOLERANCE:
        msg  = "Max deviation exceeded: {} (tolerance {})".format(max_deviation, TOLERANCE)
        # Plot discrepancies
        filename = 'angle_logp_discrepancies.png'
        msg += '\nWriting plot to {}'.format(filename)
        from matplotlib import pyplot as plt
        plt.plot(angle_test_range, logp_openmm, '.')
        plt.plot(angle_test_range, logp_ge, '.')
        plt.legend(['OpenMM', 'geometry engine'])
        plt.savefig(filename, dpi=300)
        # Raise exception
        raise Exception(msg)

def test_add_bond_units():
    """
    Test that the geometry engine adds the correct units and value to bonds when replacing the default non-unit-bearing parmed
    parameters by comparing the result to the known original parameters.
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with just a bond
    testsystem = FourAtomValenceTestSystem(bond=True, angle=False, torsion=False)

    #Extract this bond
    bond = testsystem.structure.bonds[0] #this bond has parameters
    bond_with_units = geometry_engine._add_bond_units(bond)

    #Get the pre-defined parameters in the testsystem
    (r0, k) = testsystem.bond_parameters
    k_units = k.unit

    #take the difference between the known parameters and the ones with units added
    #If units are added incorrectly, this will result in either incompatible units or a nonzero difference
    bond_difference = bond_with_units.type.req - r0
    force_constant_difference = bond_with_units.type.k - k

    #Test the difference between the given parameters and the one with units added, with a tolerance of 1.0e-6
    if np.abs(bond_difference.value_in_unit(unit.nanometers)) > 1.0e-6 or np.abs(force_constant_difference.value_in_unit(k_units)) > 1.0e-6:
        raise Exception("Did not add units correctly to bond.")

def test_add_angle_units():
    """
    Test that the geometry engine adds the correct units and value to angles when replacing the default non-unit-bearing parmed
    parameters by comparing the result to the known original parameters.
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a test system with just an angle
    testsystem = FourAtomValenceTestSystem(bond=False, angle=True, torsion=False)

    #Extract this angle
    angle = testsystem.structure.angles[0]

    #Have the GeometryEngine add units to the angle
    angle_with_units = geometry_engine._add_angle_units(angle)

    #Get the parameters from the testsystem
    (theta0, k) = testsystem.angle_parameters
    k_units = k.unit

    #Get the difference between the angle with units added and the original
    #This serves two purposes: First, checking that the value has not been disturbed
    #But second, that it has also been created with compatible units correctly.
    angle_difference = angle_with_units.type.theteq - theta0
    force_constant_difference = angle_with_units.type.k - k

    #Check that the absolute value of the differences between the unit-added angle and the reference are less than a 1.0e-6 threshold
    if np.abs(angle_difference.value_in_unit(unit.radians)) > 1.0e-6 or np.abs(force_constant_difference.value_in_unit(k_units)) > 1.0e-6:
        raise Exception("Did not add units correctly to angle.")

def test_add_torsion_units():
    """
    Test that the geometry engine adds the correct units and value to torsions when replacing the default non-unit-bearing parmed
    parameters by comparing the result to the known original parameters.
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with only a torsion
    testsystem = FourAtomValenceTestSystem(bond=False, angle=False, torsion=True)

    #Extract that torsion
    torsion = testsystem.structure.dihedrals[0]

    #Have the geometry engine add units to the torsion
    torsion_with_units = geometry_engine._add_torsion_units(torsion)

    #Extract the torsion parameters from the testsystem
    (periodicity, phase, k) = testsystem.torsion_parameters

    #Get the absolute values of differences between reference and unit-added torsion parameters
    #This checks not only that the value is correct, but also that the units are compatible
    periodicity_difference = np.abs(periodicity - torsion_with_units.type.per)
    phase_difference = np.abs(phase - torsion_with_units.type.phase)
    force_difference = np.abs(k - torsion_with_units.type.phi_k)

    #Make sure absolute values of differences are less than 1.0e-6 threshold
    if periodicity_difference > 1.0e-6 or phase_difference.value_in_unit(unit.radians) > 1.0e-6 or force_difference.value_in_unit(k.unit) > 1.0e-6:
        raise Exception("Did not add units correctly to torsion.")

def test_torsion_scan():
    """
    Test that the torsion scan is generating angles that correspond to what openmm calculates. It does this by first generating
    a set of torsion angles at evenly spaced intervals, then checking that the cartesian coordinates generated from those
    have the correct torsion angle according to OpenMM. The OpenMM check is achieved via a function that creates a system with
    a torsion force that is just "phi" allowing us to read out the internal value of the torsion.
    """
    TOLERANCE = 1.0e-6

    from perses.rjmc.geometry import FFAllAngleGeometryEngine

    #do 360 test points
    n_divisions = 360
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with only a torsion
    testsystem = FourAtomValenceTestSystem(bond=False, angle=False, torsion=True)

    #get the internal coordinates of the testsystem
    internals = testsystem.internal_coordinates
    r = internals[0] # float, implicitly in units of nanometers
    theta = internals[1] # float, implicitly in units of radians

    #get the torsion that we're going to rotate
    torsion = testsystem.structure.dihedrals[0]

    #get torsion atom indices
    torsion_atom_indices = [torsion.atom1.idx, torsion.atom2.idx, torsion.atom3.idx, torsion.atom4.idx]

    #perform the torsion scan with the genometry engine, which returns cartesian coordinates
    # phis are returned as float array, implicitly in units of radians
    # xyzs are returned as unit-bearing quantities
    xyzs, phis, bin_width = geometry_engine._torsion_scan(torsion_atom_indices, testsystem.positions, r, theta, n_divisions)

    # Check that the values of phi that OpenMM calculates matches the ones created by the GeometryEngine within 1.0e-6
    for i, phi in enumerate(phis):
        xyz_ge = xyzs[i]
        r_new, theta_new, phi_new = _get_internal_from_omm(xyz_ge, testsystem.positions[1], testsystem.positions[2], testsystem.positions[3])
        if np.abs(phi_new - phi) > TOLERANCE:
            raise Exception("Torsion scan did not match OpenMM torsion")
        if np.abs(r_new - r) > TOLERANCE or np.abs(theta_new - theta) > TOLERANCE:
            raise Exception("Theta or r was disturbed in torsion scan.")

def test_torsion_log_discrete_pdf():
    """
    Compare the discrete log pdf for the torsion created by the GeometryEngine to one calculated manually in Python.
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine

    #use 740 points
    n_divisions = 740
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with a bond, angle, and torsion
    testsystem = FourAtomValenceTestSystem(bond=True, angle=True, torsion=True)

    #Extract the internal coordinates in appropriate units
    internals = testsystem.internal_coordinates
    r, theta, phi = internals # float, implicitly in md_unit_system

    #Get the relevant torsion and add units to it
    torsion = testsystem.structure.dihedrals[0]
    torsion_with_units = geometry_engine._add_torsion_units(torsion)

    #get torsion atom indices
    torsion_atom_indices = [torsion.atom1.idx, torsion.atom2.idx, torsion.atom3.idx, torsion.atom4.idx]

    #Calculate the torsion log pmf according to the geometry engine
    torsion_log_discrete_pdf, phis, bin_width = geometry_engine._torsion_log_pmf(testsystem._context, torsion_atom_indices, testsystem.positions, r, theta, beta, n_divisions)

    #Calculate the torsion potential manually using Python
    manual_torsion_log_discrete_pdf = calculate_torsion_discrete_log_pdf_manually(beta, torsion_with_units, phis)

    #Get the absolute difference in the geometry engine discrete log pdf and the manually computed one
    deviation = np.abs(torsion_log_discrete_pdf - manual_torsion_log_discrete_pdf)

    #check that the difference is less than 1.0e-4 at all points
    if np.max(deviation) > 1.0e-4:
        raise Exception("Torsion pmf didn't match expected.")

def calculate_torsion_discrete_log_pdf_manually(beta, torsion, phis):
    """
    Manually calculate the torsion potential for a series of phis and a given beta.

    Arguments
    ---------
    beta : float
        inverse temperature
    torsion : parmed.Dihedral object
        the torsion of interest
    phis : array of float, implicitly in units of radians
        the series of torsion angles at which to evaluate the torsion probability
    """
    #initialize array for the log unnormalized probabilities
    torsion_logq = np.zeros(len(phis))

    #get the parameters of the torsion
    torsion_k = torsion.type.phi_k
    torsion_per = torsion.type.per
    torsion_phase = torsion.type.phase

    check_dimensionality(torsion_k, unit.kilojoule_per_mole)
    check_dimensionality(torsion_phase, unit.radians)

    #loop through the phis and calculate the log unnormalized probability at each
    for i, phi in enumerate(phis):
        torsion_logq[i] = -1.0*beta*torsion_k*(1+unit.cos(torsion_per*phi*unit.radians - torsion_phase))

    #get the unnormalized probabilities and the normalizing constant
    q = np.exp(torsion_logq)
    Z = np.sum(q)

    #subtract off the log of the normalizing constant to get the log probabilities
    torsion_discrete_log_pdf = torsion_logq-np.log(Z)
    return torsion_discrete_log_pdf

def test_torsion_logp():
    """
    Test that the continuous torsion probability density function integrates to unity
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine

    #take typical numbers for the number of divisions and the number of test points
    n_divisions = 360
    n_divisions_test = 740
    TOLERANCE = 1.0e-3 # probability deviation tolerance

    #instantiate the geometry engine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a valence test system with a bond, angle, and torsion
    testsystem = FourAtomValenceTestSystem(bond=True, angle=True, torsion=True)

    #Extract the internal coordinates from this, and wrap them in units
    internals = testsystem.internal_coordinates
    r, theta, phi = internals # float, dimensionless, implicitly in md_unit_system

    #Extract the torsion from the parmed structure representation (there's only one, so index 0)
    torsion = testsystem.structure.dihedrals[0]

    #Create a set of n_divisions_test phis evenly spaced from -pi to pi
    bin_width = (2.0*np.pi)/n_divisions_test
    phis = np.arange(-np.pi, +np.pi, bin_width) # dimensionless, implicitly in units of radians

    # Initialize an array for the continuous torsion log pdf
    log_pdf = np.zeros(n_divisions_test)

    #get torsion atom indices
    torsion_atom_indices = [torsion.atom1.idx, torsion.atom2.idx, torsion.atom3.idx, torsion.atom4.idx]

    #calculate the continuous log pdf of the torsion at each test point using the geometry engine with n_divisions
    for i in range(n_divisions_test):
        log_pdf[i] = geometry_engine._torsion_logp(testsystem._context, torsion_atom_indices, testsystem.positions, r, theta, phis[i], beta, n_divisions)

    #exponentiate and integrate the continuous torsion log pdf
    pdf = np.exp(log_pdf)
    torsion_sum = np.trapz(pdf, phis)

    #Ensure that the absolute difference between the integral and one is less than 0.001 (accounting for numerical issues)
    if np.abs(1.0 - torsion_sum) > TOLERANCE:
        raise Exception("The torsion continuous distribution does not integrate to one.")

def test_propose_torsion():
    """
    Test, using the kolmogorov-smirnov test, that the torsion angles drawn via the geometry engine match the expected
    distribution.
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import scipy.stats as stats

    #choose a reasonable number of divisions and samples
    n_divisions = 360
    n_samples = 1000

    #instantiate the geometry engine and a test system with a bond, angle, and torsion
    geometry_engine = FFAllAngleGeometryEngine()
    testsystem = FourAtomValenceTestSystem(bond=True, angle=True, torsion=True)

    #Retrieve the internal coordinates and assign the appropriate units
    internals = testsystem.internal_coordinates
    r, theta, phi = internals # float, dimensionless, with implied md_unit_system units

    #retrieve the torsion of interest (0--there is only one) from the parmed structure
    torsion = testsystem.structure.dihedrals[0]

    #get torsion atom indices
    torsion_atom_indices = [torsion.atom1.idx, torsion.atom2.idx, torsion.atom3.idx, torsion.atom4.idx]

    #calculate the log probability mass function for an array of phis
    # phis are numpy floats, with implied units of radians
    log_p_i, phi_i, bin_width = geometry_engine._torsion_log_pmf(testsystem._context, torsion_atom_indices, testsystem.positions, r, theta, beta, n_divisions)
    assert np.isclose(np.exp(log_p_i).sum(), 1.0), 'torsion probability mass function is not normalized'

    # Compute the CDF
    def cdf(phis):
        pdf = np.exp(log_p_i)
        cdfs = np.zeros(phis.shape)
        for i, phi in enumerate(phis):
            index = int((phi - phi_i[0]) / bin_width)
            cdfs[i] = np.sum(pdf[:index]) + (phi - phi_i[index]) / bin_width * pdf[index]
        return cdfs

    #Draw a set of samples from the torsion distribution using the GeometryEngine
    torsion_samples = np.zeros(n_samples)
    for i in range(n_samples):
        phi, logp = geometry_engine._propose_torsion(testsystem._context, torsion_atom_indices, testsystem.positions, r, theta, beta, n_divisions)
        assert (phi >= -np.pi) and (phi < +np.pi), "sampled phi of {} is outside allowed bounds of [-pi,+pi)".format(phi)
        torsion_samples[i] = phi

    #now check if the samples match the logp using the Kolmogorov-Smirnov test
    (dval, pval) = stats.kstest(torsion_samples, cdf)
    if pval < pval_threshold:
        msg = "Torsion may not have been drawn from the correct distribution: pval = {} (threshold {})".format(pval, pval_threshold)
        raise Exception(msg)

def _get_internal_from_omm(atom_coords, bond_coords, angle_coords, torsion_coords):
    """
    Given four atom positions in cartesians, will output the internal positions in spherical coords

    Arguments
    ---------
    atom_coords : unit.Quantity(np.array([x,y,z]), unit = unit.nanometers)
        x, y, and z cartesians of an atom
    bond_coords : unit.Quantity(np.array([x,y,z]), unit = unit.nanometers)
        x, y, and z cartesians of a bond atom
    angle_coords : unit.Quantity(np.array([x,y,z]), unit = unit.nanometers)
        x, y, and z cartesians of an angle atom
    torsion_coords : unit.Quantity(np.array([x,y,z]), unit = unit.nanometers)
        x, y, and z cartesians of a torsion atom

    Returns
    -------
    r : float
        radial distance in unit.nanometers
    theta : float
        bond angle in radians
    phi : float
        torsion angle in radians
    """
    #master system, will be used for all three
    sys = openmm.System()
    platform = openmm.Platform.getPlatformByName("Reference")
    for i in range(4):
        sys.addParticle(1.0*unit.amu)

    #first, the bond length:
    bond_sys = openmm.System()
    bond_sys.addParticle(1.0*unit.amu)
    bond_sys.addParticle(1.0*unit.amu)
    bond_force = openmm.CustomBondForce("r")
    bond_force.addBond(0, 1, [])
    bond_sys.addForce(bond_force)
    bond_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    bond_context = openmm.Context(bond_sys, bond_integrator, platform)
    bond_context.setPositions([atom_coords, bond_coords])
    bond_state = bond_context.getState(getEnergy=True)
    r = bond_state.getPotentialEnergy()/unit.kilojoule_per_mole
    del bond_sys, bond_context, bond_integrator

    #now, the angle:
    angle_sys = copy.deepcopy(sys)
    angle_force = openmm.CustomAngleForce("theta")
    angle_force.addAngle(0,1,2,[])
    angle_sys.addForce(angle_force)
    angle_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    angle_context = openmm.Context(angle_sys, angle_integrator, platform)
    angle_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
    angle_state = angle_context.getState(getEnergy=True)
    theta = angle_state.getPotentialEnergy()/unit.kilojoule_per_mole
    del angle_sys, angle_context, angle_integrator

    #finally, the torsion:
    torsion_sys = copy.deepcopy(sys)
    torsion_force = openmm.CustomTorsionForce("theta")
    torsion_force.addTorsion(0,1,2,3,[])
    torsion_sys.addForce(torsion_force)
    torsion_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    torsion_context = openmm.Context(torsion_sys, torsion_integrator, platform)
    torsion_context.setPositions([atom_coords, bond_coords, angle_coords, torsion_coords])
    torsion_state = torsion_context.getState(getEnergy=True)
    phi = torsion_state.getPotentialEnergy()/unit.kilojoule_per_mole
    del torsion_sys, torsion_context, torsion_integrator

    return r, theta, phi

def generate_molecule_from_smiles(smiles, idx=0):
    """
    Generate oemol with geometry from smiles
    """
    print("Molecule %d is %s" % (idx, smiles))
    mol = oechem.OEMol()
    mol.SetTitle("MOL%d" % idx)
    oechem.OESmilesToMol(mol, smiles)
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
    oechem.OEAssignFormalCharges(mol)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetStrictStereo(False)
    mol.SetTitle("MOL_%d" % idx)
    omega(mol)
    return mol

def oemol_to_openmm_system(oemol, molecule_name=None, small_molecule_forcefield='gaff-2.11'):
    """
    Create an OpenM System object from an oemol

    Parameters
    ----------
    oemol : openeye.oechem.OEMol
        The molecule to create an unconstrained vacuum System from
    molecule_name : str, optional, default=None
        This is always ignored
    small_molecule_forcefield : str, optional, deafult='gaff-2.11'
        Small molecule force field to feed to SystemGenerator

    Returns
    -------
    system : simtk.openmm.System
        The System object for the unconstrained vacuum small molecule
    positions : simtk.unit.Quantity with shape [natoms,3] with units compatible with nanometers
        The positions
    topology : simtk.unit.app.Topology
        The Topology corresponding to the small molecule

    """
    from perses.rjmc import topology_proposal
    from perses.utils.openeye import extractPositionsFromOEMol
    from openmmforcefields.generators import SystemGenerator
    system_generator = SystemGenerator(small_molecule_forcefield=small_molecule_forcefield, forcefield_kwargs={'constraints' : None})
    topology = forcefield_generators.generateTopologyFromOEMol(oemol)
    system = system_generator.create_system(topology)
    positions = extractPositionsFromOEMol(oemol)
    return system, positions, topology

def oemol_to_openmm_system_amber(oemol, molecule_name):
    """
    Create an openmm system out of an oemol

    Returns
    -------
    system : openmm.System object
        the system from the molecule
    positions : [n,3] np.array of floats
    """

    _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(oemol, tripos_mol2_filename=molecule_name + '.tripos.mol2', conformer=0, residue_name='MOL')
    gaff_mol2, frcmod = openmoltools.amber.run_antechamber(molecule_name, tripos_mol2_filename)
    prmtop_file, inpcrd_file = openmoltools.amber.run_tleap(molecule_name, gaff_mol2, frcmod)
    from parmed.amber import AmberParm
    prmtop = AmberParm(prmtop_file)
    system = prmtop.createSystem(implicitSolvent=None, removeCMMotion=False)
    crd = app.AmberInpcrdFile(inpcrd_file)
    return system, crd.getPositions(asNumpy=True), prmtop.topology

def align_molecules(mol1, mol2):
    """
    MCSS two OEmols. Return the mapping of new : old atoms
    """
    mcs = oechem.OEMCSSearch(oechem.OEMCSType_Exhaustive)
    atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_AtomicNumber | oechem.OEExprOpts_HvyDegree
    bondexpr = oechem.OEExprOpts_Aromaticity
    #atomexpr = oechem.OEExprOpts_HvyDegree
    #bondexpr = 0
    mcs.Init(mol1, atomexpr, bondexpr)
    mcs.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())
    unique = True
    match = [m for m in mcs.Match(mol2, unique)][0]
    new_to_old_atom_mapping = {}
    for matchpair in match.GetAtoms():
        old_index = matchpair.pattern.GetIdx()
        new_index = matchpair.target.GetIdx()
        new_to_old_atom_mapping[new_index] = old_index
    return new_to_old_atom_mapping

@attr('advanced')
@nottest
@skipIf(running_on_github_actions, "Skip advanced test on GH Actions")
def test_mutate_from_all_to_all(): # TODO: fix protein mutations
    """
    Make sure mutations are successful between every possible pair of before-and-after residues
    Mutate Ecoli F-ATPase alpha subunit to all 20 amino acids (test going FROM all possibilities)
    Mutate each residue to all 19 alternatives
    """
    import perses.rjmc.topology_proposal as topology_proposal
    import perses.rjmc.geometry as geometry
    from perses.tests.utils import compute_potential_components
    from openmmtools import testsystems as ts
    geometry_engine = geometry.FFAllAngleGeometryEngine()

    aminos = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

    for aa in aminos:
        topology, positions = _get_capped_amino_acid(amino_acid=aa)
        modeller = app.Modeller(topology, positions)

        ff_filename = "amber99sbildn.xml"
        max_point_mutants = 1

        ff = app.ForceField(ff_filename)
        system = ff.createSystem(modeller.topology)
        chain_id = '1'

        system_generator = topology_proposal.SystemGenerator([ff_filename])

        pm_top_engine = topology_proposal.PointMutationEngine(modeller.topology, system_generator, chain_id, max_point_mutants=max_point_mutants)

        current_system = system
        current_topology = modeller.topology
        current_positions = modeller.positions
        minimize_integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        platform = openmm.Platform.getPlatformByName("Reference")
        minimize_context = openmm.Context(current_system, minimize_integrator, platform)
        minimize_context.setPositions(current_positions)
        initial_state = minimize_context.getState(getEnergy=True)
        initial_potential = initial_state.getPotentialEnergy()
        openmm.LocalEnergyMinimizer.minimize(minimize_context)
        final_state = minimize_context.getState(getEnergy=True, getPositions=True)
        final_potential = final_state.getPotentialEnergy()
        current_positions = final_state.getPositions()
        #print("Minimized initial structure from %s to %s" % (str(initial_potential), str(final_potential)))

        for k, proposed_amino in enumerate(aminos):
            pm_top_engine._allowed_mutations = [[('2',proposed_amino)]]
            pm_top_proposal = pm_top_engine.propose(current_system, current_topology)
            new_positions, logp = geometry_engine.propose(pm_top_proposal, current_positions, beta)
            new_system = pm_top_proposal.new_system
            if np.isnan(logp):
                raise Exception("NaN in the logp")
            integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
            platform = openmm.Platform.getPlatformByName("Reference")
            context = openmm.Context(new_system, integrator, platform)
            context.setPositions(new_positions)
            state = context.getState(getEnergy=True)
            #print(compute_potential_components(context))
            potential = state.getPotentialEnergy()
            potential_without_units = potential / potential.unit
            #print(str(potential))
            if np.isnan(potential_without_units):
                raise Exception("Energy after proposal is NaN")

@attr('advanced')
@nottest
@skipIf(running_on_github_actions, "Skip advanced test on GH Actions")
def test_propose_lysozyme_ligands(): # TODO: fix protein mutations
    """
    Try proposing geometries for all T4 ligands from all T4 ligands
    """
    from perses.tests.testsystems import T4LysozymeInhibitorsTestSystem
    testsystem = T4LysozymeInhibitorsTestSystem()
    smiles_list = testsystem.molecules[:7]
    proposals = make_geometry_proposal_array(smiles_list, forcefield=['data/T4-inhibitors.xml', 'data/gaff.xml'])
    run_proposals(proposals)

@attr('advanced')
@nottest
@skipIf(running_on_github_actions, "Skip advanced test on GH Actions")
def test_propose_kinase_inhibitors(): # TODO: fix protein mutations
    from perses.tests.testsystems import KinaseInhibitorsTestSystem
    testsystem = KinaseInhibitorsTestSystem()
    smiles_list = testsystem.molecules[:7]
    proposals = make_geometry_proposal_array(smiles_list, forcefield=['data/kinase-inhibitors.xml', 'data/gaff.xml'])
    run_proposals(proposals)

def run_proposals(proposal_list):
    """
    Run a list of geometry proposal namedtuples, checking if they render
    NaN energies

    Parameters
    ----------
    proposal_list : list of namedtuple

    """
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import time
    start_time = time.time()
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine()
    for proposal in proposal_list:
        current_time = time.time()
        #print("proposing")
        top_proposal = proposal.topology_proposal
        current_positions = proposal.current_positions
        new_positions, logp = geometry_engine.propose(top_proposal, current_positions, beta)
        #print("Proposal time is %s" % str(time.time()-current_time))
        integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(top_proposal.new_system, integrator, platform)
        context.setPositions(new_positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()
        potential_without_units = potential / potential.unit
        #print(str(potential))
        #print(" ")
        #print(' ')
        #print(" ")
        if np.isnan(potential_without_units):
            print("NanN potential!")
        if np.isnan(logp):
            print("logp is nan")
        del context, integrator
        # TODO: Can we quantify how good the proposals are?

def make_geometry_proposal_array(smiles_list, small_molecule_forcefield='gaff-2.11'):
    """
    Make an array of topology_proposals for each molecule to each other
    in the smiles_list. Includes self-proposals so as to test that.

    Parameters
    ----------
    smiles_list : list of str
        list of smiles
    small_molecule_forcefield : str, optional, default='gaff-2.11'
        Small molecule force field for SystemGenerator

    Returns
    -------
    list of proposal_test namedtuple
    """
    topology_proposals = []
    #make oemol array:
    oemols = OrderedDict()
    syspostop = OrderedDict()

    for smiles in smiles_list:
        oemols[smiles] = generate_molecule_from_smiles(smiles)
    for smiles in oemols.keys():
        print("Generating %s" % smiles)
        syspostop[smiles] = oemol_to_openmm_system(oemols[smiles], small_molecule_forcefield=small_molecule_forcefield)

    #get a list of all the smiles in the appropriate order
    smiles_pairs = list()
    for smiles1 in smiles_list:
        for smiles2 in smiles_list:
            if smiles1==smiles2:
                continue
            smiles_pairs.append([smiles1, smiles2])

    for i, pair in enumerate(smiles_pairs):
        #print("preparing pair %d" % i)
        smiles_1 = pair[0]
        smiles_2 = pair[1]
        new_to_old_atom_mapping = align_molecules(oemols[smiles_1], oemols[smiles_2])
        sys1, pos1, top1 = syspostop[smiles_1]
        sys2, pos2, top2 = syspostop[smiles_2]
        import perses.rjmc.topology_proposal as topology_proposal
        sm_top_proposal = topology_proposal.TopologyProposal(new_topology=top2, new_system=sys2, old_topology=top1, old_system=sys1,
                                                                      old_chemical_state_key='',new_chemical_state_key='', logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_mapping, metadata={'test':0.0})
        sm_top_proposal._beta = beta
        proposal_tuple = proposal_test(sm_top_proposal, pos1)
        topology_proposals.append(proposal_tuple)
    return topology_proposals

def load_pdbid_to_openmm(pdbid):
    """
    create openmm topology without pdb file
    lifted from pandegroup/pdbfixer
    """
    url = 'http://www.rcsb.org/pdb/files/%s.pdb' % pdbid
    file = urlopen(url)
    contents = file.read().decode('utf-8')
    file.close()
    file = StringIO(contents)

    if _guessFileFormat(file, url) == 'pdbx':
        pdbx = app.PDBxFile(contents)
        topology = pdbx.topology
        positions = pdbx.positions
    else:
        pdb = app.PDBFile(file)
        topology = pdb.topology
        positions = pdb.positions

    return topology, positions

def _guessFileFormat(file, filename):
    """
    Guess whether a file is PDB or PDBx/mmCIF based on its filename and contents.
    authored by pandegroup
    """
    filename = filename.lower()
    if '.pdbx' in filename or '.cif' in filename:
        return 'pdbx'
    if '.pdb' in filename:
        return 'pdb'
    for line in file:
        if line.startswith('data_') or line.startswith('loop_'):
            file.seek(0)
            return 'pdbx'
        if line.startswith('HEADER') or line.startswith('REMARK') or line.startswith('TITLE '):
            file.seek(0)
            return 'pdb'
    file.seek(0)
    return 'pdb'

def run_geometry_engine(index=0):
    """
    Run the geometry engine a few times to make sure that it actually runs
    without exceptions. Convert n-pentane to 2-methylpentane
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import copy
    from perses.utils.openeye import iupac_to_oemol
    molecule_name_1 = 'benzene'
    molecule_name_2 = 'biphenyl'
    #molecule_name_1 = 'imatinib'
    #molecule_name_2 = 'erlotinib'

    molecule1 = iupac_to_oemol(molecule_name_1)
    molecule2 = iupac_to_oemol(molecule_name_2)
    new_to_old_atom_mapping = align_molecules(molecule1, molecule2)

    sys1, pos1, top1 = oemol_to_openmm_system(molecule1, molecule_name_1)
    sys2, pos2, top2 = oemol_to_openmm_system(molecule2, molecule_name_2)

    import perses.rjmc.geometry as geometry
    import perses.rjmc.topology_proposal as topology_proposal
    from perses.tests.utils import compute_potential_components

    sm_top_proposal = topology_proposal.TopologyProposal(new_topology=top2, new_system=sys2, old_topology=top1, old_system=sys1,
                                                                      old_chemical_state_key='',new_chemical_state_key='', logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_mapping, metadata={'test':0.0})
    sm_top_proposal._beta = beta
    geometry_engine = geometry.FFAllAngleGeometryEngine(metadata={})
    # Turn on PDB file writing.
    geometry_engine.write_proposal_pdb = True
    geometry_engine.pdb_filename_prefix = 't13geometry-proposal'
    test_pdb_file = open("%s_to_%s_%d.pdb" % (molecule_name_1, molecule_name_2, index), 'w')

    def remove_nonbonded_force(system):
        """Remove NonbondedForce from specified system."""
        force_indices_to_remove = list()
        for [force_index, force] in enumerate(system.getForces()):
            if force.__class__.__name__ == 'NonbondedForce':
                force_indices_to_remove.append(force_index)
        for force_index in force_indices_to_remove[::-1]:
            system.removeForce(force_index)

    valence_system = copy.deepcopy(sys2)
    remove_nonbonded_force(valence_system)
    integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    integrator_1 = openmm.VerletIntegrator(1*unit.femtoseconds)
    ctx_1 = openmm.Context(sys1, integrator_1)
    ctx_1.setPositions(pos1)
    ctx_1.setVelocitiesToTemperature(300*unit.kelvin)
    integrator_1.step(1000)
    pos1_new = ctx_1.getState(getPositions=True).getPositions(asNumpy=True)
    context = openmm.Context(sys2, integrator)
    context.setPositions(pos2)
    state = context.getState(getEnergy=True)
    print("Energy before proposal is: %s" % str(state.getPotentialEnergy()))
    openmm.LocalEnergyMinimizer.minimize(context)

    new_positions, logp_proposal = geometry_engine.propose(sm_top_proposal, pos1_new, beta)
    logp_reverse = geometry_engine.logp_reverse(sm_top_proposal, new_positions, pos1, beta)
    print(logp_reverse)

    app.PDBFile.writeFile(top2, new_positions, file=test_pdb_file)
    test_pdb_file.close()
    context.setPositions(new_positions)
    state2 = context.getState(getEnergy=True)
    print("Energy after proposal is: %s" %str(state2.getPotentialEnergy()))
    print(compute_potential_components(context))

    valence_integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    valence_ctx = openmm.Context(valence_system, valence_integrator, platform)
    valence_ctx.setPositions(new_positions)
    vstate = valence_ctx.getState(getEnergy=True)
    print("Valence energy after proposal is %s " % str(vstate.getPotentialEnergy()))
    final_potential = state2.getPotentialEnergy()
    return final_potential / final_potential.unit

def test_existing_coordinates():
    """
    for each torsion, calculate position of atom1
    """
    from perses.utils.openeye import iupac_to_oemol
    ATOM_POSITION_TOLERANCE = 1e-6
    molecule_name_2 = 'butane'
    molecule2 = iupac_to_oemol(molecule_name_2)
    sys, pos, top = oemol_to_openmm_system(molecule2, molecule_name_2)
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    structure = parmed.openmm.load_topology(top, sys)
    torsions = [torsion for torsion in structure.dihedrals if not torsion.improper]
    for torsion in torsions:
        atom1_position = pos[torsion.atom1.idx]
        atom2_position = pos[torsion.atom2.idx]
        atom3_position = pos[torsion.atom3.idx]
        atom4_position = pos[torsion.atom4.idx]
        internal_coordinates, _ = geometry_engine._cartesian_to_internal(atom1_position, atom2_position, atom3_position, atom4_position)
        recalculated_atom1_position, _ = geometry_engine._internal_to_cartesian(atom2_position, atom3_position, atom4_position, internal_coordinates[0], internal_coordinates[1], internal_coordinates[2])
        n = np.linalg.norm(atom1_position-recalculated_atom1_position)
        assert n < ATOM_POSITION_TOLERANCE, f"the recalculated cartesian atom 1 position is displaced from the original position by more than {ATOM_POSITION_TOLERANCE}"

def run_logp_reverse():
    """
    Make sure logp_reverse and logp_forward are consistent
    """
    molecule_name_1 = 'erlotinib'
    molecule_name_2 = 'imatinib'
    from perses.utils.openeye import iupac_to_oemol

    molecule1 = iupac_to_oemol(molecule_name_1)
    molecule2 = iupac_to_oemol(molecule_name_2)
    new_to_old_atom_mapping = align_molecules(molecule1, molecule2)

    sys1, pos1, top1 = oemol_to_openmm_system(molecule1, molecule_name_1)
    sys2, pos2, top2 = oemol_to_openmm_system(molecule2, molecule_name_2)
    test_pdb_file = open("reverse_test1.pdb", 'w')
    app.PDBFile.writeFile(top1, pos1, file=test_pdb_file)
    test_pdb_file.close()

    import perses.rjmc.geometry as geometry
    import perses.rjmc.topology_proposal as topology_proposal

    sm_top_proposal = topology_proposal.TopologyProposal(new_topology=top2, new_system=sys2, old_topology=top1, old_system=sys1,
                                                                    logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_mapping, new_chemical_state_key="CCC", old_chemical_state_key="CC", metadata={'test':0.0})
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    new_positions, logp_proposal = geometry_engine.propose(sm_top_proposal, pos1, beta)

    logp_reverse = geometry_engine.logp_reverse(sm_top_proposal, pos2, pos1, beta)
    print(logp_proposal)
    print(logp_reverse)
    print(logp_reverse-logp_proposal)

def _get_capped_amino_acid(amino_acid='ALA'):
    import tempfile
    import shutil
    tleapstr = """
    source oldff/leaprc.ff99SBildn
    system = sequence {{ ACE {amino_acid} NME }}
    saveamberparm system {amino_acid}.prmtop {amino_acid}.inpcrd
    quit
    """.format(amino_acid=amino_acid)
    from perses.tests.utils import enter_temp_directory
    with enter_temp_directory() as tmpdirname:
        tleap_file = open('tleap_commands', 'w')
        tleap_file.writelines(tleapstr)
        tleap_file.close()
        tleap_cmd_str = "tleap -f %s " % tleap_file.name

        #call tleap, log output to logger
        output = getoutput(tleap_cmd_str)
        logging.debug(output)

        prmtop = app.AmberPrmtopFile("{amino_acid}.prmtop".format(amino_acid=amino_acid))
        inpcrd = app.AmberInpcrdFile("{amino_acid}.inpcrd".format(amino_acid=amino_acid))
        topology = prmtop.topology
        positions = inpcrd.positions

        debug = False
        if debug:
            system = prmtop.createSystem()
            integrator = openmm.VerletIntegrator(1)
            context = openmm.Context(system, integrator)
            context.setPositions(positions)
            openmm.LocalEnergyMinimizer.minimize(context)
            state = context.getState(getEnergy=True)
            print("%s energy: %s" % (amino_acid, str(state.getPotentialEnergy())))

    return topology, positions

def _tleap_all():
    aminos = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
    for aa in aminos:
        _get_capped_amino_acid(aa)

def _oemol_from_residue(res):
    """
    Get an OEMol from a residue, even if that residue
    is polymeric. In the latter case, external bonds
    are replaced by hydrogens.

    Parameters
    ----------
    res : app.Residue
        The residue in question

    Returns
    -------
    oemol : openeye.oechem.OEMol
        an oemol representation of the residue with topology indices
    """
    import openeye.oechem as oechem
    from openmoltools.forcefield_generators import generateOEMolFromTopologyResidue
    external_bonds = list(res.external_bonds())
    if external_bonds:
        for bond in external_bonds:
            res.chain.topology._bonds.remove(bond)
    mol = generateOEMolFromTopologyResidue(res, geometry=False)
    oechem.OEAddExplicitHydrogens(mol)
    return mol

def _print_failed_SMILES(failed_mol_list):
    import openeye.oechem as oechem
    for mol in failed_mol_list:
        smiles = oechem.OEMolToSmiles(mol)
        print(smiles)

def _generate_ffxmls():
    import os
    print(os.getcwd())
    print("Parameterizing T4 inhibitors")
    from perses.tests.testsystems import T4LysozymeInhibitorsTestSystem
    from openmoltools import forcefield_generators
    testsystem_t4 = T4LysozymeInhibitorsTestSystem()
    smiles_list_t4 = testsystem_t4.molecules
    oemols_t4 = [generate_molecule_from_smiles(smiles, idx=i) for i, smiles in enumerate(smiles_list_t4)]
    ffxml_str_t4, failed_list = forcefield_generators.generateForceFieldFromMolecules(oemols_t4, ignoreFailures=True)
    ffxml_out_t4 = open('/Users/grinawap/T4-inhibitors.xml','w')
    ffxml_out_t4.write(ffxml_str_t4)
    ffxml_out_t4.close()
    if failed_list:
        print("Failed some T4 inhibitors")
        _print_failed_SMILES(failed_list)
    print("Parameterizing kinase inhibitors")
    from perses.tests.testsystems import KinaseInhibitorsTestSystem
    testsystem_kinase = KinaseInhibitorsTestSystem()
    smiles_list_kinase = testsystem_kinase.molecules
    oemols_kinase = [generate_molecule_from_smiles(smiles, idx=i) for i, smiles in enumerate(smiles_list_kinase)]
    ffxml_str_kinase, failed_kinase_list = forcefield_generators.generateForceFieldFromMolecules(oemols_kinase, ignoreFailures=True)
    ffxml_out_kinase = open("/Users/grinawap/kinase-inhibitors.xml",'w')
    ffxml_out_kinase.write(ffxml_str_kinase)
    ffxml_out_t4.close()


# Now test analytical bead systems more thoroughly
class LinearValenceTestSystem(GeometryTestSystem):
    """
    This testsystem has 3 to 5 particles, and the potential for a bond, angle, torsion term.
    The particles are 0-1-2-3 atom-bond-angle-torsion. The positions for the atoms were taken
    from an earlier test for the geometry engine.
    """

    def __init__(self, bond=True, angle=True, torsion=True, n_atoms=4, add_extra_angle=True):
        """
        Arguments
        ---------
        bond : Boolean, default True
            Whether to include the bond force term
        angle : Boolean, default True
            Whether to include the angle force term
        torsion : Boolean, default True
            Whether to include the torsion force term
        four_atom: Boolean, default True
            Whether to include a fourth atom

        Properties
        ----------
        internal_coordinates : array of floats
            The r, theta, phi internal coordinates of atom 0
        bond_parameters : tuple of (Quantity, Quantity)
            The equilibrium bond length and equilibrium constant, in nanometers and kJ/(mol*nm^2), atoms 0-1
        angle_parameters : tuple of (Quantity, Quantity)
            The equilibrium angle and constant, in radians and kJ/(mol*rad^2), atoms 0-1-2
        torsion_parameters : tuple of (int, Quantity, Quantity)
            The periodicity, along with the phase and force constant in radians and kJ/mol respectively, atoms 0-1-2-3
        """
        import simtk.openmm.app as app
        import parmed

        if n_atoms < 3 or n_atoms > 5:
            raise ValueError("Number of atoms must be 3, 4, or 5")

        #make a simple set of positions. These are taken from another test used for testing the torsions in GeometryEngine
        self._default_positions = unit.Quantity(np.zeros([n_atoms,3]), unit=unit.nanometer)
        self._default_positions[0] = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
        self._default_positions[1] = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
        self._default_positions[2] = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
        if n_atoms > 3:
            self._default_positions[3] = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)
        if n_atoms == 5:
            self._default_positions[3] = unit.Quantity(np.array([-0.057, 0.0951, -0.1863]), unit=unit.nanometers)

        #use parameters taken from various parts of the AlanineDipeptideTestSystem
        self._default_r0 = unit.Quantity(value=0.1522, unit=unit.nanometer)
        self._default_bond_k = unit.Quantity(value=265265.60000000003, unit=unit.kilojoule/(unit.nanometer**2*unit.mole))
        self._default_angle_theta0 = unit.Quantity(value=1.91113635, unit=unit.radian)
        self._default_angle_k = unit.Quantity(value=418.40000000000003, unit=unit.kilojoule/(unit.mole*unit.radian**2))
        self._default_torsion_periodicity = 2
        self._default_torsion_phase = unit.Quantity(value=np.pi/2.0, unit=unit.radians)
        self._default_torsion_k = unit.Quantity(value=1.0, unit=unit.kilojoule/unit.mole)
        #self._default_torsion_k = unit.Quantity(value=0.0, unit=unit.kilojoule/unit.mole)

        #set up a topology with the appropriate atoms (make them all carbon)
        self._topology = app.Topology()
        new_chain = self._topology.addChain("0")
        new_res = self._topology.addResidue("MOL", new_chain)
        atom1 = self._topology.addAtom("C1", app.Element.getByAtomicNumber(6), new_res, 0)
        atom2 = self._topology.addAtom("C2", app.Element.getByAtomicNumber(6), new_res, 1)
        atom3 = self._topology.addAtom("C3", app.Element.getByAtomicNumber(6), new_res, 2)
        if n_atoms > 3:
            atom4 = self._topology.addAtom("C4", app.Element.getByAtomicNumber(6), new_res, 3)
        if n_atoms == 5:
            atom5 = self._topology.addAtom("C5", app.Element.getByAtomicNumber(6), new_res, 4)

        #add the bonds to make a linear molecule 1-2-3-4
        self._topology.addBond(atom1, atom2)
        self._topology.addBond(atom2, atom3)
        if n_atoms > 3:
            self._topology.addBond(atom3, atom4)
        if n_atoms == 5:
            self._topology.addBond(atom3, atom5)

        #create a system using the same particle information
        self._system = openmm.System()
        indices = [self._system.addParticle(CARBON_MASS) for i in range(n_atoms)]

        #the growth order includes only the 0th atom, since there are only four atoms total
        self._growth_order = [0]

        #if the user has specified that a bond force should be used, add it with the appropriate constants
        if bond:
            bond_force = openmm.HarmonicBondForce()
            self._system.addForce(bond_force)
            bond_force.addBond(0, 1, self._default_r0, self._default_bond_k)
            bond_force.addBond(1, 2, self._default_r0, self._default_bond_k)
            if n_atoms > 3:
                bond_force.addBond(2, 3, self._default_r0, self._default_bond_k)
            if n_atoms == 5:
                bond_force.addBond(4, 2, self._default_r0, self._default_bond_k)

        #if the user has specified that an angle force should be used, add it with the appropriate constants
        if angle:
            angle_force = openmm.HarmonicAngleForce()
            self._system.addForce(angle_force)
            angle_force.addAngle(0, 1, 2, self._default_angle_theta0, self._default_angle_k)
            if n_atoms > 3:
                angle_force.addAngle(1, 2, 3, self._default_angle_theta0, self._default_angle_k)
            if n_atoms == 5:
                angle_force.addAngle(1, 2, 4, self._default_angle_theta0, self._default_angle_k)
                if add_extra_angle:
                    angle_force.addAngle(4, 2, 3, self._default_angle_theta0, self._default_angle_k)

        #if the user has specified that a torsion force should be used, add it with the appropriate constants
        if torsion and n_atoms > 3:
            torsion_force = openmm.PeriodicTorsionForce()
            self._system.addForce(torsion_force)
            torsion_force.addTorsion(0, 1, 2, 3, self._default_torsion_periodicity, self._default_torsion_phase, self._default_torsion_k)
            #torsion_force.addTorsion(0, 1, 2, 3, self._default_torsion_periodicity, self._default_torsion_phase, self._default_torsion_k)

            if n_atoms == 5:
                torsion_force.addTorsion(0, 1, 2, 4, self._default_torsion_periodicity, self._default_torsion_phase,
                                         self._default_torsion_k)

        #Now make a ParmEd structure from the topology and system, which will include relevant force parameters
        self._structure = parmed.openmm.load_topology(self._topology, self._system)

        #initialize class memers with the appropriate values
        self._positions = self._default_positions
        self._integrator = openmm.VerletIntegrator(1)
        self._platform = REFERENCE_PLATFORM #use reference for stability

        #create a context and set positions so we can get potential energies
        self._context = openmm.Context(self._system, self._integrator, self._platform)
        self._context.setPositions(self._positions)


    @property
    def internal_coordinates(self):
        from perses.rjmc import coordinate_numba
        positions_without_units = self._positions.value_in_unit(unit.nanometer)
        internals = coordinate_numba.cartesian_to_internal(positions_without_units[0], positions_without_units[1], positions_without_units[2], positions_without_units[3])
        return internals

    @internal_coordinates.setter
    def internal_coordinates(self, internal_coordinates):
        from perses.rjmc import coordinate_numba
        internals_without_units = np.zeros(3, dtype=np.float64)
        internals_without_units[0] = internal_coordinates[0].value_in_unit(unit.nanometer)
        internals_without_units[1] = internal_coordinates[1].value_in_unit(unit.radians)
        internals_without_units[2] = internal_coordinates[2].value_in_unit(unit.radians)
        positions_without_units = self._positions.value_in_unit(unit.nanometer)
        new_cartesian_coordinates = coordinate_numba.internal_to_cartesian(positions_without_units[1], positions_without_units[2], positions_without_units[3], internals_without_units)
        self._positions[0] = unit.Quantity(new_cartesian_coordinates, unit=unit.nanometer)

    @property
    def bond_parameters(self):
        return (self._default_r0, self._default_bond_k)

    @property
    def angle_parameters(self):
        return (self._default_angle_theta0, self._default_angle_k)

    @property
    def torsion_parameters(self):
        return (self._default_torsion_periodicity, self._default_torsion_phase, self._default_torsion_k)




class AnalyticalBeadSystems(object):

    """
    Test class for generating work distributions for n_bead systems, for which all rjmc work values should be constant.  It is designed to prepare
    work distributions (forward and reverse proposals) for 3-to-4, 4-to-5, and 3-to-5 bead systems.
    """
    def __init__(self, transformation, num_iterations):
        """
        Arguments
        ---------
        transformation: list
            [int, int+1] where int = 3 or 4
        num_iterations: int
            number of iid conformations of molecule A to transform into molecule B

        Properties
        ----------
        num_iterations: ''
        transformation: ''
        sys_pos_top: dict
            dict of [molecule A: openmm.system, openmm.positions, openmm.topology, molecule B: openmm.system, openmm.positions, openmm.topology]
        """

        self.num_iterations = num_iterations
        self.transformation = transformation

        self.sys_pos_top = dict()
        for _atom_number, _letter in zip(self.transformation, ['A', 'B']):
            _testsystem = LinearValenceTestSystem(n_atoms=_atom_number)
            _sys, _top, _pos = _testsystem.system, _testsystem.topology, self.minimize(_testsystem.system, _testsystem.positions)
            self.sys_pos_top[_letter] = (_sys, _pos, _top)

    def convert_to_md(self, openmm_positions):
        """
        Convert openmm position objects into numpy ndarrays

        Arguments
        ---------
        openmm_positions: openmm unit.Quantity object
            Positions generated from openmm simulation

        Returns
        -------
        md_positions_stacked: np.ndarray
            Positions in md_unit_system (nanometers)
        """
        _openmm_positions_no_units = [_posits.value_in_unit_system(unit.md_unit_system) for _posits in openmm_positions]
        md_positions_stacked = np.stack(_openmm_positions_no_units)

        return md_positions_stacked

    def minimize(self, system, positions):
        """
        Utility function to minimize a system

        Arguments
        ---------
        system: openmm system object
        positions: openmm unit.Quantity object
            openmm position (single frame)

        Returns
        -------
        minimized_positions: openmm unit.Quantity object
            Minimized positions in md_unit_system (nanometers)

        """
        _ctx = openmm.Context(system, openmm.VerletIntegrator(1.0))
        _ctx.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(_ctx)
        minimized_positions = _ctx.getState(getPositions=True).getPositions(asNumpy=True)
        del _ctx

        return minimized_positions

    def compute_rp(self, system, positions):
        """
        Utility function to compute the reduced potential

        Arguments
        ---------
        system: openmm system object
        positions: openmm unit.Quantity object
            openmm position (single frame)

        Returns
        -------
        rp: float
            reduced potential
        """
        from simtk.unit.quantity import is_dimensionless
        _i = openmm.VerletIntegrator(1.0)
        _ctx = openmm.Context(system, _i, REFERENCE_PLATFORM)
        _ctx.setPositions(positions)
        rp = beta*_ctx.getState(getEnergy=True).getPotentialEnergy()
        assert is_dimensionless(rp), "reduced potential is not dimensionless"
        del _ctx
        return rp

    def create_simple_topology_proposal(self, sys_pos_top, n_atoms_initial, n_atoms_final, direction='forward'):
        """
        Utility function to generate a topology proposal from a linear bead system

        Arguments
        ---------
        sys_pos_top: dict
            dictionary of openmm.system, openmm.positions, openmm.topology for molecules A and B
        n_atoms_initial: int
            number of atoms in molecule A
        n_atoms_final: int
            number of atoms in molecule B
        direction: str
            either the topology proposal is from A --> B or B --> A

        Returns
        -------
        rp: dict
            perses topology proposal dictionary
        """
        from perses.rjmc import topology_proposal as tp

        if direction=='forward' or direction=='forwards':
            _initial_system, _initial_position, _initial_topology = sys_pos_top['A']
            _final_system, _final_positions, _final_topology = sys_pos_top['B']
        elif direction=='backward'or direction=='backwards':
            _initial_system, _initial_position, _initial_topology = sys_pos_top['B']
            _final_system, _final_positions, _final_topology = sys_pos_top['A']
        else:
            raise ValueError("direction may only be 'forward(s)' or 'backward(s)'!")

        if n_atoms_initial == 3 and (n_atoms_final == 4 or n_atoms_final == 5):
            _new_to_old_atom_map = {0: 0, 1: 1, 2: 2}
        elif n_atoms_initial == 4 and n_atoms_final == 5:
            _new_to_old_atom_map = {0: 0, 1: 1, 2: 2, 3: 3}
        elif n_atoms_initial==4 and n_atoms_final==3:
            _new_to_old_atom_map = {0: 0, 1: 1, 2: 2}
        elif n_atoms_initial == 5 and n_atoms_final == 4:
            _new_to_old_atom_map = {0: 0, 1: 1, 2: 2, 3: 3}
        elif n_atoms_initial == 5 and n_atoms_final == 3:
            _new_to_old_atom_map = {0: 0, 1: 1, 2: 2}
        else:
            raise ValueError("This method only supports going from 3->4, 4->3, 4->5, 5->4 3->5, or 5->3")

        topology_proposal = tp.TopologyProposal(new_topology=_final_topology, new_system=_final_system, old_topology=_initial_topology,
                                                old_system=_initial_system, logp_proposal=0.0,
                                                new_to_old_atom_map=_new_to_old_atom_map, old_chemical_state_key=str(n_atoms_initial),
                                                new_chemical_state_key=str(n_atoms_final))
        return topology_proposal

    def run_rj_simple_system(self, configurations_initial, topology_proposal, n_replicates):
        """
        Function to execute reversibje jump MC

        Arguments
        ---------
        configurations_initial: openmm.Quantity
            n_replicate frames of equilibrium simulation of initial system
        topology_proposal: dict
            perses.topology_proposal object
        n_replicates: int
            number of replicates to simulate

        Returns
        -------
        logPs: numpy ndarray
            shape = (n_replicates, 4) where logPs[i] = (reduced potential of initial molecule, log proposal probability, reversed log proposal probability, reduced potential of proposed molecule)
        final_positions: list
            list of openmm position objects for final molecule proposal
        """
        import tqdm
        from perses.rjmc.geometry import FFAllAngleGeometryEngine
        final_positions = []
        logPs = np.zeros([n_replicates, 4])
        _geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=10000, n_angle_divisions=1800, n_torsion_divisions=3600, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = True)
        for _replicate_idx in tqdm.trange(n_replicates, disable=running_on_github_actions):
            _old_positions = configurations_initial[_replicate_idx, :, :]
            _new_positions, _lp = _geometry_engine.propose(topology_proposal, _old_positions, beta)
            _lp_reverse = _geometry_engine.logp_reverse(topology_proposal, _new_positions, _old_positions, beta)
            _initial_rp = self.compute_rp(topology_proposal.old_system, _old_positions)
            if not topology_proposal.unique_old_atoms: #the geometry engine doesn't run the backward proposal
                logPs[_replicate_idx, 0] = _geometry_engine.forward_atoms_with_positions_reduced_potential
                logPs[_replicate_idx, 3] = _geometry_engine.forward_final_context_reduced_potential
            elif not topology_proposal.unique_new_atoms: #the geometry engine doesn't run forward
                logPs[_replicate_idx, 0] = _geometry_engine.reverse_final_context_reduced_potential
                logPs[_replicate_idx, 3] = _geometry_engine.reverse_atoms_with_positions_reduced_potential
            else:
                logPs[_replicate_idx, 0] = _geometry_engine.reverse_final_context_reduced_potential
                logPs[_replicate_idx, 3] = _geometry_engine.forward_final_context_reduced_potential
            logPs[_replicate_idx, 1] = _lp
            logPs[_replicate_idx, 2] = _lp_reverse
            final_rp = self.compute_rp(topology_proposal.new_system, _new_positions)
            final_positions.append(_new_positions)
        return logPs, final_positions

    def create_iid_bead_systems(self, printer=False):
        """
        Function to simulate i.i.d conformations of the initial molecule

        Arguments
        ---------
        printer: boolean
            whether to print the stacked positions of the simulated initial molecule

        Returns
        -------
        iid_positions_A: openmm.Quantity
            num_iterations of independent initial molecule conformations
        """
        from openmmtools import integrators
        import tqdm

        _sysA, _posA, _topA = self.sys_pos_top['A']
        _integrator = integrators.LangevinIntegrator()
        _ctx = openmm.Context(_sysA, _integrator, REFERENCE_PLATFORM)
        _ctx.setPositions(_posA)

        _iid_positions_A = unit.Quantity(np.zeros([self.num_iterations, self.transformation[0],3]), unit=unit.nanometers)

        for _iteration in tqdm.trange(self.num_iterations, disable=running_on_github_actions):
            _integrator.step(1000)
            _state=_ctx.getState(getPositions=True)
            _iid_positions_A[_iteration,:,:]=_state.getPositions(asNumpy=True)

        _iid_positions_A_stacked=self.convert_to_md(_iid_positions_A)
        iid_positions_A=unit.Quantity(_iid_positions_A_stacked, unit=unit.nanometer)

        if printer:
            print('simulated_positions: ')
            print(_iid_positions_A_stacked)

        del _ctx

        return iid_positions_A

    def forward_transformation(self, iid_positions_A, printer=False):
        """
        Function to conduct run_rj_simple_system RJMC on each conformation of initial molecule (i.e. A --> B)

        Arguments
        ---------
        iid_positions_A: openmm.Quantity
            ndarray of iid conformations of molecule A
        printer: boolean
            whether to print the stacked positions of the proposed molecule

        Returns
        -------
        proposed positions: openmm.Quantity
            num_iterations of final proposal molecule
        self.work_forward: ndarray
            numpy array of forward works
        """
        _topology_proposal = self.create_simple_topology_proposal(self.sys_pos_top, n_atoms_initial = self.transformation[0], n_atoms_final = self.transformation[1], direction='forward')
        _data_forward, _proposed_positions = self.run_rj_simple_system(iid_positions_A, _topology_proposal, self.num_iterations)
        self.work_forward = _data_forward[:, 3] - _data_forward[:, 0] - _data_forward[:, 2] + _data_forward[:, 1]

        _proposed_positions_stacked=self.convert_to_md(_proposed_positions)
        proposed_positions=unit.Quantity(_proposed_positions_stacked, unit=unit.nanometers)

        if printer:
            print('proposed_positions: ')
            print(_proposed_positions_stacked)

        return proposed_positions


    def backward_transformation(self, proposed_positions, printer=False):
        """
        Function to conduct run_rj_simple_system RJMC on each conformation of proposed molecule (i.e. B --> A)
        backward_positions should be the same unit.Quantity as iid_positions_A (the test function has an assertion to maintain this)

        Arguments
        ---------
        proposed_positions: openmm.Quantity
            ndarray of proposed conformations of molecule B
        printer: boolean
            whether to print the stacked positions of the final proposal molecule positions

        Returns
        -------
        backward_positions: openmm.Quantity
            num_iterations of final proposal molecule
        self.work_reverse: ndarray
            numpy array of backward works
        """

        _topology_proposal = self.create_simple_topology_proposal(self.sys_pos_top, n_atoms_initial = self.transformation[1], n_atoms_final = self.transformation[0], direction='backward')
        _data_backward, _backward_positions = self.run_rj_simple_system(proposed_positions, _topology_proposal, self.num_iterations)
        self.work_reverse = _data_backward[:, 3] - _data_backward[:, 0] - _data_backward[:, 2] + _data_backward[:, 1]

        _backward_positions_stacked=self.convert_to_md(_backward_positions)
        backward_positions=unit.Quantity(_backward_positions_stacked, unit=unit.nanometers)

        if printer:
            print('backward positions (i.e. backward transformation positions: ')
            print(_backward_positions_stacked)

        return backward_positions



    def work_comparison(self, printer=False):
        """
        Function to compute variance of forward and backward works, and to add the work arrays pairwise

        Arguments
        ---------
        printer: boolean
            whether to print the forward, reverse, variance and comparison works

        Returns
        -------
        work_sum : np array
            array of floats of the pairwise addition of forward and backward works
        work_forward_stddev : float
            Standard deviation of forward work, implicitly in units of kT
        work_reverse_stddev : float
            Standard deviation of backward work, implicitly in units of kT
        """
        work_sum = self.work_forward + self.work_reverse
        work_forward_stddev = self.work_forward.std()
        work_reverse_stddev = self.work_reverse.std()

        if printer:
            print('work_forward: ', self.work_forward)
            print('work_reverse: ', self.work_reverse)
            print('work_sum: ', work_sum)
            print('work_forward_stddev: ', work_forward_stddev)
            print('work_reverse_stddev: ', work_reverse_stddev)
        return work_sum, work_forward_stddev, work_reverse_stddev



#@nottest
@skipIf(running_on_github_actions, "Skip deprecated test on GH Actions")
def test_AnalyticalBeadSystems(transformation=[[3,4], [4,5], [3,5]], num_iterations=100):
    """
    Function to assert that the forward and reverse works are equal and opposite, and that the variances of each work distribution is much less
    than the average work.  This is conducted on all three possible forward and reverse transformations.

    Also asserts that each iid configuration of molecule A is equal (within a threshold) to the final proposal position of B --> A (i.e. the backward_transformation proposal molecule)

    Arguments
    ---------
    transformation: list
        list of pairwise transformation proposals
    num_iterations: int
        number of iid conformations from which to conduct rjmc
    """
    import mdtraj as md

    for pair in transformation:
        test = AnalyticalBeadSystems(pair, num_iterations)
        _iid_positions_A = test.create_iid_bead_systems(printer=False)
        _iid_positions_A_stacked = np.stack([_posits.value_in_unit_system(unit.md_unit_system) for _posits in _iid_positions_A])
        _proposed_positions = test.forward_transformation(_iid_positions_A, printer=False)
        _backward_positions = test.backward_transformation(_proposed_positions, printer=False)
        _backward_positions_stacked = np.stack([_posits.value_in_unit_system(unit.md_unit_system) for _posits in _backward_positions])


        POSITION_THRESHOLD = 1.0e-6
        _position_differences = np.array([simulated_frame - final_frame for simulated_frame, final_frame in zip(_iid_positions_A_stacked,_backward_positions_stacked)])
        assert all(frame.sum() < POSITION_THRESHOLD for frame in _position_differences)

        WORK_STDDEV_THRESHOLD = 0.1
        WORK_SUM_THRESHOLD = 0.1
        work_sum, work_forward_stddev, work_reverse_stddev = test.work_comparison(printer = False)
        print("work forward stddev: {}".format(work_forward_stddev))
        print("work reverse stddev: {}".format(work_reverse_stddev))
        assert (work_forward_stddev <= WORK_STDDEV_THRESHOLD), "forward work stddev {} exceeds threshold {}".format(work_forward_stddev, WORK_STDDEV_THRESHOLD)
        assert (work_reverse_stddev <= WORK_STDDEV_THRESHOLD), "reverse work stddev {} exceeds threshold {}".format(work_reverse_stddev, WORK_STDDEV_THRESHOLD)
        assert np.all(abs(work_sum) <= WORK_SUM_THRESHOLD), "sum of works {} exceeds threshold {}".format(work_sum, WORK_SUM_THRESHOLD)

def test_logp_forward_check_for_vacuum_topology_proposal(current_mol_name = 'propane', proposed_mol_name = 'butane', num_iterations = 10, neglect_angles = True):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.  Assert that the logp_forward < 1e3.
    This assertion will fail if the proposal order tool proposed the placement of the a carbon before a previously defined carbon in the alkane.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators
    from perses.rjmc.topology_proposal import SystemGenerator, TopologyProposal, SmallMoleculeSetProposalEngine
    from perses.utils.openeye import createSystemFromIUPAC, iupac_to_oemol
    from openmoltools.openeye import generate_conformers
    from perses.utils.data import get_data_filename
    from perses.rjmc import geometry
    from perses.utils.smallmolecules import render_atom_mapping
    import tqdm

    current_mol, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(current_mol_name,title=current_mol_name[0:4])
    proposed_mol = iupac_to_oemol(proposed_mol_name)
    proposed_mol = generate_conformers(proposed_mol,max_confs=1)

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff})
    geometry_engine = geometry.FFAllAngleGeometryEngine(n_bond_divisions=100, n_angle_divisions=180, n_torsion_divisions=360, neglect_angles = neglect_angles)
    proposal_engine = SmallMoleculeSetProposalEngine(
        [current_mol, proposed_mol], system_generator, residue_name=current_mol_name[0:4])

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old, current_mol_id=0, proposed_mol_id=1)

    # show atom mapping
    filename = str(current_mol_name)+str(proposed_mol_name)+'.pdf'
    render_atom_mapping(filename, current_mol, proposed_mol, topology_proposal.new_to_old_atom_map)

    total_works = []
    for _ in tqdm.trange(num_iterations, disable=running_on_github_actions):
        #generate new positions with geometry engine
        new_positions, logp_forward = geometry_engine.propose(topology_proposal, pos_old, beta)
        logp_reverse = geometry_engine.logp_reverse(topology_proposal, new_positions, pos_old, beta)

        #now just render forward and backward work
        work_fwd = logp_forward + geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential
        work_bkwd = logp_reverse + geometry_engine.reverse_atoms_with_positions_reduced_potential - geometry_engine.reverse_final_context_reduced_potential

        total_work = logp_forward - logp_reverse + geometry_engine.forward_final_context_reduced_potential - geometry_engine.reverse_final_context_reduced_potential
        total_works.append(total_work)

        print("forward, backward works : {}, {}".format(work_fwd, work_bkwd))
        print("total_work: {}".format(total_work))
        assert abs(work_fwd - work_bkwd - total_work) < 1, "The difference of fwd and backward works is not equal to the total work (within 1kT)"
        assert logp_forward < 1e3, "A heavy atom was proposed in an improper order"

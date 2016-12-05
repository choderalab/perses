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

from perses.rjmc import coordinate_numba

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01

proposal_test = namedtuple("proposal_test", ["topology_proposal", "current_positions"])

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
    This testsystem has 4 particles, and the potential for a bond, angle, torsion term.
    The particles are 0-1-2-3 atom-bond-angle-torsion
    """

    def __init__(self, bond=True, angle=True, torsion=True):
        #make a simple topology
        self._default_positions = unit.Quantity(np.zeros([4,3]), unit=unit.nanometer)
        self._default_positions[0] = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
        self._default_positions[1] = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
        self._default_positions[2] = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
        self._default_positions[3] = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)

        self._default_r0 = unit.Quantity(value=0.1522, unit=unit.nanometer)
        self._default_bond_k = unit.Quantity(value=265265.60000000003, unit=unit.kilojoule/(unit.nanometer**2*unit.mole))
        self._default_angle_theta0 = unit.Quantity(value=1.91113635, unit=unit.radian)
        self._default_angle_k = unit.Quantity(value=418.40000000000003, unit=unit.kilojoule/(unit.mole*unit.radian**2))
        self._default_torsion_periodicity = 2
        self._default_torsion_phase = unit.Quantity(value=np.pi/2.0, unit=unit.radians)
        self._default_torsion_k = unit.Quantity(value=20.0, unit=unit.kilojoule/unit.mole)

        self._topology = app.Topology()
        new_chain = self._topology.addChain("0")
        new_res = self._topology.addResidue("MOL", new_chain)
        atom1 = self._topology.addAtom("C1", app.Element.getByAtomicNumber(6), new_res, 0)
        atom2 = self._topology.addAtom("C2", app.Element.getByAtomicNumber(6), new_res, 1)
        atom3 = self._topology.addAtom("C3", app.Element.getByAtomicNumber(6), new_res, 2)
        atom4 = self._topology.addAtom("C4", app.Element.getByAtomicNumber(6), new_res, 3)


        self._topology.addBond(atom1, atom2)
        self._topology.addBond(atom2, atom3)
        self._topology.addBond(atom3, atom4)

        self._system = openmm.System()
        indices = [self._system.addParticle(CARBON_MASS) for i in range(4)]

        self._growth_order = [0]

        if bond:
            bond_force = openmm.HarmonicBondForce()
            self._system.addForce(bond_force)
            bond_force.addBond(0, 1, self._default_r0, self._default_bond_k)

        if angle:
            angle_force = openmm.HarmonicAngleForce()
            self._system.addForce(angle_force)
            angle_force.addAngle(0, 1, 2, self._default_angle_theta0, self._default_angle_k)

        if torsion:
            torsion_force = openmm.PeriodicTorsionForce()
            self._system.addForce(torsion_force)
            torsion_force.addTorsion(0, 1, 2, 3, self._default_torsion_periodicity, self._default_torsion_phase, self._default_torsion_k)

        self._structure = parmed.openmm.load_topology(self._topology, self._system)
        self._positions = self._default_positions
        self._integrator = openmm.VerletIntegrator(1)
        self._platform = openmm.Platform.getPlatformByName("Reference")
        self._context = openmm.Context(self._system, self._integrator, self._platform)
        self._context.setPositions(self._positions)

    @property
    def internal_coordinates(self):
        positions_without_units = self._positions.value_in_unit(unit.nanometer)
        internals = coordinate_numba.cartesian_to_internal(positions_without_units[0], positions_without_units[1], positions_without_units[2], positions_without_units[3])
        return internals

    @internal_coordinates.setter
    def internal_coordinates(self, internal_coordinates):
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

def test_propose_angle():
    """
    Test the proposal of angles by GeometryEngine by comparing to proposals from a normal distribution
    with mean theta0 (equilibrium angle) and variance sigma = sqrt(1.0/(k*beta)), where k is the force
    constant and beta is the inverse temperature. A Kolmogorov-Smirnov test is used for that comparison.
    """
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
    sigma = unit.sqrt(1.0/(beta*k))
    sigma_without_units = sigma.value_in_unit(unit.radian)
    theta0_without_units = theta0.value_in_unit(unit.radian)

    #allocate an array for proposing angles from the appropriate distribution
    angle_array = np.zeros(1000)
    for i in range(1000):
        proposed_angle_with_units = geometry_engine._propose_angle(angle_with_units, beta)
        angle_array[i] = proposed_angle_with_units.value_in_unit(unit.radians)

    #compare the sampled angles to a normal cdf with the appropriate parameters using
    #the Kolomogorov-Smirnov test. The null hypothesis is that they are drawn from the same
    #distribution (the test passes).
    (dval, pval) = stats.kstest(angle_array,'norm', args=(theta0_without_units, sigma_without_units))
    if pval < 0.05:
        raise Exception("The angle may be drawn from the wrong distribution. p= %f" % pval)

def test_propose_bond():
    """
    Test the proposal of bonds by GeometryEngine by comparing to proposals from a normal distribution
    with mean r0 (equilibrium bond length) and variance sigma = sqrt(1.0/(k*beta)), where k is the force
    constant and beta is the inverse temperature. A Kolmogorov-Smirnov test is used for that comparison.
    """
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
    sigma_without_units = sigma.value_in_unit(unit.nanometers)
    r0_without_units = r0.value_in_unit(unit.nanometers)

    #Allocate an array for sampled bond lengths and propose from the geometry engine
    bond_array = np.zeros(1000)
    for i in range(1000):
        proposed_bond_with_units = geometry_engine._propose_bond(bond_with_units, beta)
        bond_array[i] = proposed_bond_with_units.value_in_unit(unit.nanometer)

    #compare the sampled angles to a normal cdf with the appropriate parameters using
    #the Kolomogorov-Smirnov test. The null hypothesis is that they are drawn from the same
    #distribution (the test passes).
    (dval, pval) = stats.kstest(bond_array, 'norm', args=(r0_without_units, sigma_without_units))
    if pval < 0.05:
        raise Exception("The bond may be drawn from the wrong distribution. p= %f" % pval)

def test_bond_logq():
    """
    Compare the bond logq (log unnormalized probability) calculated by the geometry engine to the log-unnormalized
    probability calculated by openmm (-beta*potential_energy, where beta is inverse temperature).
    """
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

    #Create a set of bond lengths to test from r0 - sigma to r0 + sigma
    bond_range = np.linspace(r0_without_units - sigma_without_units, r0_without_units + sigma_without_units, 100)

    #Extract the internal coordinates and add units.
    #r - bond length
    #theta - bond angle
    #phi - torsion angle
    internal_coordinates = testsystem.internal_coordinates
    r = unit.Quantity(internal_coordinates[0], unit=unit.nanometer)
    theta = unit.Quantity(internal_coordinates[1], unit=unit.radians)
    phi = unit.Quantity(internal_coordinates[2], unit=unit.radians)
    internals_with_units = [r, theta, phi]

    #Add units to the set of bond lengths
    bond_range_with_units = unit.Quantity(bond_range, unit=unit.nanometer)

    #Loop through the bond lengths and make sure the unnormalized log probability calculated by the geometry engine (logq)
    #matches that calculated by openmm (-beta*potential_energy) within 1.0e-6 tolerance
    for bond_length in bond_range_with_units:
        bond_logq_ge = geometry_engine._bond_logq(bond_length, bond_with_units, beta)
        internals_with_units[0] = bond_length
        testsystem.internal_coordinates = internals_with_units
        bond_logq_omm = -beta*testsystem.energy
        if (np.abs(bond_logq_omm-bond_logq_ge)) > 1.0e-6:
            raise Exception("Bond logq did not match openmm")

def test_angle_logq():
    """
    Compare the angle logq (log unnormalized probability) calculated by the geometry engine to the log-unnormalized
    probability calculated by openmm (-beta*potential_energy, where beta is inverse temperature).
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with only an angle
    testsystem = FourAtomValenceTestSystem(bond=False, angle=True, torsion=False)

    #Retrieve that angle and add units to it
    angle = testsystem.structure.angles[0]
    angle_with_units = geometry_engine._add_angle_units(angle)

    #Retrieve internal coordinates and add units
    #r - bond length
    #theta - bond angle
    #phi - torsion angle
    internal_coordinates = testsystem.internal_coordinates
    r = unit.Quantity(internal_coordinates[0], unit=unit.nanometer)
    theta = unit.Quantity(internal_coordinates[1], unit=unit.radians)
    phi = unit.Quantity(internal_coordinates[2], unit=unit.radians)
    internals_with_units = [r, theta, phi]

    #Get a range of test points for the angle from 0 to pi, and add units
    angle_test_range = np.linspace(0.0, np.pi, num=100)
    angle_test_range_with_units = unit.Quantity(angle_test_range, unit=unit.radians)

    #Loop through the test points for the angle and calculate the log-unnormalized probability of each using
    #geometry engine and openmm (-beta*potential_energy) where beta is inverse temperature. Tolerance 1.0e-4,
    #because this tries a range of angles well outside what one would typically see.
    for test_angle in angle_test_range_with_units:
        angle_logq_ge = geometry_engine._angle_logq(test_angle, angle_with_units, beta)
        internals_with_units[1] = test_angle
        testsystem.internal_coordinates = internals_with_units
        angle_logq_omm = -beta*testsystem.energy
        if (np.abs(angle_logq_ge - angle_logq_omm)) > 1.0e-4:
            raise Exception("Angle logq did not match openmm")

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
    from perses.rjmc.geometry import FFAllAngleGeometryEngine

    #do 360 test points
    n_divisions = 360
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a testsystem with only a torsion
    testsystem = FourAtomValenceTestSystem(bond=False, angle=False, torsion=True)

    #get the internal coordinates of the testsystem
    internals = testsystem.internal_coordinates
    r = unit.Quantity(internals[0], unit=unit.nanometer)
    theta = unit.Quantity(internals[1], unit=unit.radian)

    #get the torsion that we're going to rotate
    torsion = testsystem.structure.dihedrals[0]

    #perform the torsion scan with the genometry engine, which returns cartesian coordinates
    xyzs, phis = geometry_engine._torsion_scan(torsion, testsystem.positions, r, theta, n_divisions=n_divisions)
    phis_without_units = phis.value_in_unit(unit.radians)

    #check that the values of phi that OpenMM calculates matches the ones created by the GeometryEngine within 1.0e-6
    for i in range(n_divisions):
        xyz_ge = xyzs[i]
        r_new, theta_new, phi = _get_internal_from_omm(xyz_ge, testsystem.positions[1], testsystem.positions[2], testsystem.positions[3])
        if np.abs(phis_without_units[i] - phi) >1.0e-6:
            raise Exception("Torsion scan did not match OpenMM torsion")
        if np.abs(r_new - internals[0]) >1.0e-6 or np.abs(theta_new - internals[1]) > 1.0e-6:
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
    r = unit.Quantity(internals[0], unit=unit.nanometer)
    theta = unit.Quantity(internals[1], unit=unit.radian)

    #Get the relevant torsion and add units to it
    torsion = testsystem.structure.dihedrals[0]
    torsion_with_units = geometry_engine._add_torsion_units(torsion)

    #Calculate the torsion log pmf according to the geometry engine
    torsion_log_discrete_pdf, phis = geometry_engine._torsion_log_probability_mass_function(testsystem._context, torsion_with_units, testsystem.positions, r, theta, beta, n_divisions=n_divisions)

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
    phis : array of float
        the series of torsion angles at which to evaluate the torsion probability
    """
    #initialize array for the log unnormalized probabilities
    torsion_logq = np.zeros(len(phis))

    #get the parameters of the torsion
    torsion_k = torsion.type.phi_k
    torsion_per = torsion.type.per
    torsion_phase = torsion.type.phase

    #loop through the phis and calculate the log unnormalized probability at each
    for i in range(len(phis)):
        torsion_logq[i] = -1.0*beta*torsion_k*(1+unit.cos(torsion_per*phis[i] - torsion_phase))

    #get the unnormalized probabilities and the normalizing constant
    q = np.exp(torsion_logq)
    Z = np.sum(q)

    #subtract off the log of the normalizing constant to get the log probabilities
    torsion_discrete_log_pdf = torsion_logq-np.log(Z)
    return torsion_discrete_log_pdf

def test_torsion_logp():
    """
    Test that the continuous torsion log pdf integrates to one
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine

    #take typical numbers for the number of divisions and the number of test points
    n_divisions = 360
    n_divisions_test = 740

    #instantiate the geometry engine
    geometry_engine = FFAllAngleGeometryEngine()

    #Create a valence test system with a bond, angle, and torsion
    testsystem = FourAtomValenceTestSystem(bond=True, angle=True, torsion=True)

    #Extract the internal coordinates from this, and wrap them in units
    internals = testsystem.internal_coordinates
    r = unit.Quantity(internals[0], unit=unit.nanometer)
    theta = unit.Quantity(internals[1], unit=unit.radian)

    #Extract the torsion from the parmed structure representation (there's only one, so index 0)
    torsion = testsystem.structure.dihedrals[0]

    #Create a set of n_divisions_test phis evenly spaced from -pi to pi
    phis = unit.Quantity(np.arange(-np.pi, +np.pi, (2.0*np.pi)/n_divisions_test), unit=unit.radians)

    #initialize an array for the continuous torsion log pdf
    log_pdf = np.zeros(n_divisions_test)

    #calculate the continuous log pdf of the torsion at each test point using the geometry engine with n_divisions
    for i in range(n_divisions_test):
        log_pdf[i] = geometry_engine._torsion_logp(testsystem._context, torsion, testsystem.positions, r, theta, phis[i], beta, n_divisions=n_divisions)

    #exponentiate and integrate the continuous torsion log pdf
    pdf = np.exp(log_pdf)
    torsion_sum = np.trapz(pdf, phis)

    #Ensure that the absolute difference between the integral and one is less than 0.001 (accounting for numerical issues)
    if np.abs(1.0 - torsion_sum) > 1.0e-3:
        raise Exception("The torsion continuous distribution does not integrate to one.")

def test_propose_torsion():
    """
    Test, using the kolmogorov-smirnov test, that the torsion angles drawn via the geometry engine match the expected
    distribution.
    """
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import scipy.stats as stats

    #choose a reasonable number of divisions and samples
    n_divisions = 96
    n_samples = 1000

    #instantiate the geometry engine and a test system with a bond, angle, and torsion
    geometry_engine = FFAllAngleGeometryEngine()
    testsystem = FourAtomValenceTestSystem(bond=True, angle=True, torsion=True)

    #Retrieve the internal coordinates and assign the appropriate units
    internals = testsystem.internal_coordinates
    r = unit.Quantity(internals[0], unit=unit.nanometer)
    theta = unit.Quantity(internals[1], unit=unit.radian)

    #retrieve the torsion of interest (0--there is only one) from the parmed structure
    torsion = testsystem.structure.dihedrals[0]

    #calculate the log probability mass function for an array of phis
    logp_phis, phis = geometry_engine._torsion_log_probability_mass_function(testsystem._context, torsion, testsystem.positions, r, theta, beta, n_divisions=n_divisions)

    #remove units from phis
    phis_without_units = phis.value_in_unit(unit.radians)

    #create a cumulative distribution function for use in the ks test
    cdf_func = create_cdf(logp_phis, phis_without_units, n_divisions)

    #Draw a set of samples from the torsion distribution using the GeometryEngine
    torsion_samples = unit.Quantity(np.zeros(n_samples), unit=unit.radian)
    for i in range(n_samples):
        torsion_samples[i], logp = geometry_engine._propose_torsion(testsystem._context, torsion, testsystem.positions, r, theta, beta, n_divisions=n_divisions)

    #now check if the samples match the logp using the Kolmogorov-Smirnov test
    (dval, pval) = stats.kstest(torsion_samples, cdf_func)
    if pval < 0.05:
        raise Exception("Torsion may not have been drawn from the correct distribution.")

def create_cdf(log_pmf, phis, n_divisions):
    """
    Create a cdf callable for scipy.stats.kstest
    """
    p_phis = np.exp(log_pmf)
    dphi = 2.0*np.pi/n_divisions
    normalizing_constant = 0.0
    for idx, phi in enumerate(phis):
        normalizing_constant += p_phis[idx]*dphi
    def torsion_cdf(phi_array):
        cdf_vals = np.zeros_like(phi_array)
        for idx, phi in enumerate(phi_array):
            cdfval = 0.0
            nearest_phi_idx = np.argmin(np.abs(phi-phis))
            nearest_phi = phis[nearest_phi_idx]
            for i in range(nearest_phi_idx):
                cdfval += p_phis[i]*dphi
            #find the width of the last partial bin in the CDF
            final_bin_dphi = phi - (nearest_phi - dphi / 2.0)
            cdfval += p_phis[nearest_phi_idx]*final_bin_dphi
            cdf_vals[idx] = cdfval / normalizing_constant
        return cdf_vals

    return torsion_cdf

def _get_internal_from_omm(atom_coords, bond_coords, angle_coords, torsion_coords):
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

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing
    In the source distribution, these files are in ``perses/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the openmoltools folder).
    """

    fn = resource_filename('perses', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

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

def generate_initial_molecule(iupac_name):
    """
    Generate an oemol with a geometry
    """
    mol = oechem.OEMol()
    oeiupac.OEParseIUPACName(mol, iupac_name)
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
    omega = oeomega.OEOmega()
    omega.SetStrictStereo(False)
    omega.SetMaxConfs(1)
    omega(mol)
    return mol

def oemol_to_openmm_system(oemol, molecule_name=None, forcefield=['data/gaff.xml']):
    from perses.rjmc import topology_proposal
    from openmoltools import forcefield_generators
    xml_filenames = [get_data_filename(fname) for fname in forcefield]
    system_generator = topology_proposal.SystemGenerator(xml_filenames, forcefield_kwargs={'constraints' : None})
    topology = forcefield_generators.generateTopologyFromOEMol(oemol)
    system = system_generator.build_system(topology)
    positions = extractPositionsFromOEMOL(oemol)
    return system, positions, topology

def extractPositionsFromOEMOL(molecule):
    positions = unit.Quantity(np.zeros([molecule.NumAtoms(), 3], np.float32), unit.angstroms)
    coords = molecule.GetCoords()
    for index in range(molecule.NumAtoms()):
        positions[index,:] = unit.Quantity(coords[index], unit.angstroms)
    return positions

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
def test_mutate_quick():
    """
    Abbreviated version of test_mutate_all for travis.
    """
    import perses.rjmc.topology_proposal as topology_proposal
    import perses.rjmc.geometry as geometry
    from perses.tests.utils import compute_potential_components
    from openmmtools import testsystems as ts
    geometry_engine = geometry.FFAllAngleGeometryEngine()

    aminos = ['ALA','VAL','GLY','PHE','PRO','TRP']

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
        print("Minimized initial structure from %s to %s" % (str(initial_potential), str(final_potential)))

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
            print(compute_potential_components(context))
            potential = state.getPotentialEnergy()
            potential_without_units = potential / potential.unit
            print(str(potential))
            if np.isnan(potential_without_units):
                raise Exception("Energy after proposal is NaN")

@attr('advanced')
def test_mutate_from_all_to_all():
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
        print("Minimized initial structure from %s to %s" % (str(initial_potential), str(final_potential)))

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
            print(compute_potential_components(context))
            potential = state.getPotentialEnergy()
            potential_without_units = potential / potential.unit
            print(str(potential))
            if np.isnan(potential_without_units):
                raise Exception("Energy after proposal is NaN")

@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
def test_propose_lysozyme_ligands():
    """
    Try proposing geometries for all T4 ligands from all T4 ligands
    """
    from perses.tests.testsystems import T4LysozymeInhibitorsTestSystem
    testsystem = T4LysozymeInhibitorsTestSystem()
    smiles_list = testsystem.molecules[:7]
    proposals = make_geometry_proposal_array(smiles_list, forcefield=['data/T4-inhibitors.xml', 'data/gaff.xml'])
    run_proposals(proposals)

@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
def test_propose_kinase_inhibitors():
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

def make_geometry_proposal_array(smiles_list, forcefield=['data/gaff.xml']):
    """
    Make an array of topology_proposals for each molecule to each other
    in the smiles_list. Includes self-proposals so as to test that.

    Parameters
    ----------
    smiles_list : list of str
        list of smiles

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
        syspostop[smiles] = oemol_to_openmm_system(oemols[smiles], forcefield=forcefield)

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
    molecule_name_1 = 'benzene'
    molecule_name_2 = 'biphenyl'
    #molecule_name_1 = 'imatinib'
    #molecule_name_2 = 'erlotinib'

    molecule1 = generate_initial_molecule(molecule_name_1)
    molecule2 = generate_initial_molecule(molecule_name_2)
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
    molecule_name_2 = 'butane'
    molecule2 = generate_initial_molecule(molecule_name_2)
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
        _internal_coordinates, _ = geometry_engine._cartesian_to_internal(atom1_position, atom2_position, atom3_position, atom4_position)
        internal_coordinates = internal_in_unit(_internal_coordinates)
        recalculated_atom1_position, _ = geometry_engine._internal_to_cartesian(atom2_position, atom3_position, atom4_position, internal_coordinates[0], internal_coordinates[1], internal_coordinates[2])
        n = np.linalg.norm(atom1_position-recalculated_atom1_position)
        print(n)

def internal_in_unit(internal_coords):
    r = internal_coords[0]*unit.nanometers if type(internal_coords[0]) != unit.Quantity else internal_coords[0]
    theta = internal_coords[1]*unit.radians if type(internal_coords[1]) != unit.Quantity else internal_coords[1]
    phi = internal_coords[2]*unit.radians if type(internal_coords[2]) != unit.Quantity else internal_coords[2]
    return [r, theta, phi]

def test_coordinate_conversion():
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    #try to transform random coordinates to and from cartesian
    for i in range(200):
        indices = np.random.randint(100, size=4)
        atom_position = unit.Quantity(np.array([ 0.80557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
        bond_position = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
        angle_position = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
        torsion_position = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)
        rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
        r = rtp[0]*unit.nanometers
        theta = rtp[1]*unit.radians
        phi = rtp[2]*unit.radians
        xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        assert np.linalg.norm(xyz-atom_position) < 1.0e-12

def test_openmm_dihedral():
    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    import simtk.openmm as openmm
    integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
    sys = openmm.System()
    force = openmm.CustomTorsionForce("theta")
    for i in range(4):
        sys.addParticle(1.0*unit.amu)
    force.addTorsion(0,1,2,3,[])
    sys.addForce(force)
    atom_position = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
    bond_position = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
    angle_position = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
    torsion_position = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)
    rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(sys, integrator, platform)
    positions = [atom_position, bond_position, angle_position, torsion_position]
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()

    #rotate about the torsion:
    n_divisions = 100
    phis = unit.Quantity(np.arange(0, 2.0*np.pi, (2.0*np.pi)/n_divisions), unit=unit.radians)
    omm_phis = np.zeros(n_divisions)
    for i, phi in enumerate(phis):
        xyz_atom1, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, rtp[0]*unit.nanometers, rtp[1]*unit.radians, phi)
        context.setPositions([xyz_atom1, bond_position, angle_position, torsion_position])
        state = context.getState(getEnergy=True)
        omm_phis[i] = state.getPotentialEnergy()/unit.kilojoule_per_mole

    return 0

def test_try_random_itoc():

    import perses.rjmc.geometry as geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})
    import simtk.openmm as openmm
    integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
    sys = openmm.System()
    force = openmm.CustomTorsionForce("theta")
    for i in range(4):
        sys.addParticle(1.0*unit.amu)
    force.addTorsion(0,1,2,3,[])
    sys.addForce(force)
    atom_position = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
    bond_position = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
    angle_position = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
    torsion_position = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)
    for i in range(1000):
        atom_position += unit.Quantity(np.random.normal(size=3), unit=unit.nanometers)
        r, theta, phi = _get_internal_from_omm(atom_position, bond_position, angle_position, torsion_position)
        r = r*unit.nanometers
        theta = theta*unit.radians
        phi = phi*unit.radians
        recomputed_xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        new_r, new_theta, new_phi = _get_internal_from_omm(recomputed_xyz,bond_position, angle_position, torsion_position)
        crtp = geometry_engine._cartesian_to_internal(recomputed_xyz,bond_position, angle_position, torsion_position)
        # DEBUG
        # print(atom_position-recomputed_xyz)
        # TODO: Add a test here that can fail if something is wrong.

def run_logp_reverse():
    """
    Make sure logp_reverse and logp_forward are consistent
    """
    molecule_name_1 = 'erlotinib'
    molecule_name_2 = 'imatinib'
    #molecule_name_1 = 'benzene'
    #molecule_name_2 = 'biphenyl'

    molecule1 = generate_initial_molecule(molecule_name_1)
    molecule2 = generate_initial_molecule(molecule_name_2)
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
    """.format(amino_acid=amino_acid)
    cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
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

    os.chdir(cwd)
    shutil.rmtree(temp_dir)
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

if __name__ == "__main__":
    test_try_random_itoc()

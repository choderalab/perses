__author__='dominic rufa'

# In[0]
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
from openmmtools import integrators
import tqdm
from perses.rjmc import topology_proposal as tp
from perses.rjmc.geometry import FFAllAngleGeometryEngine

#Define some global variables
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01

"""
The following python script will run dynamics on the 3 --> 4 bead system as such:
    1.  run dynamics for time equilibrate_time and then propose a geometry switch, accumulating a generalized work.
"""

# In[1]

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
    This testsystem has 4 particles, and the potential for a bond, angle, torsion term.
    The particles are 0-1-2-3 atom-bond-angle-torsion. The positions for the atoms were taken
    from an earlier test for the geometry engine.

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

    def __init__(self, bond=True, angle=True, torsion=True, n_atoms=4, add_extra_angle=False):

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

            if n_atoms == 5:
                torsion_force.addTorsion(0, 1, 2, 4, self._default_torsion_periodicity, self._default_torsion_phase,
                                         self._default_torsion_k)

        #Now make a ParmEd structure from the topology and system, which will include relevant force parameters
        self._structure = parmed.openmm.load_topology(self._topology, self._system)

        #initialize class memers with the appropriate values
        self._positions = self._default_positions
        self._integrator = openmm.VerletIntegrator(1)
        self._platform = openmm.Platform.getPlatformByName("Reference") #use reference for stability

        #create a context and set positions so we can get potential energies
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
def minimize(system, positions):
    """
    Utility function to minimize a system
    """
    ctx = openmm.Context(system, openmm.VerletIntegrator(1.0))
    ctx.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(ctx)

    return ctx.getState(getPositions=True).getPositions(asNumpy=True)
def compute_rp(system, positions):
    i = openmm.VerletIntegrator(1.0)
    ctx = openmm.Context(system, i)
    ctx.setPositions(positions)
    return beta*ctx.getState(getEnergy=True).getPotentialEnergy()


# In[3]
"""
Running the code for 3 --> 4 bead system
"""
def production_and_work_calculation(system_size, num_production_runs, timesteps_per_production_run):

    #3 bead system
    print('generating 3 bead system...')
    testsystem=FourAtomValenceTestSystem(n_atoms=system_size) #invoke class to generate
    sys=testsystem.system
    pos=minimize(sys, testsystem.positions); print('    minimization of 3bead system complete')
    init_pos=pos
    top=testsystem.topology
    print('done generating 3 bead system')

    #4 bead sytem
    print('generating 4 bead system...')
    testsystem4=FourAtomValenceTestSystem(n_atoms=4) #invoke class to generate
    sys4=testsystem4.system
    pos4=minimize(sys4, testsystem4.positions); print('     minimization 4bead system complete')
    top4=testsystem4.topology
    print('done generating 4 bead system')



    integrator = integrators.LangevinIntegrator()
    platform = openmm.Platform.getPlatformByName("CPU")
    ctx = openmm.Context(sys, integrator)
    geometry_engine=FFAllAngleGeometryEngine()

    ##These are all logger objects
    simulated_positions = unit.Quantity(np.zeros([num_production_runs, sys.getNumParticles(), 3]), unit=unit.nanometers)
    new_proposed_position_logger=np.zeros([num_production_runs, 4, 3])
    logPs_forward=np.zeros([num_production_runs])
    logPs_reverse=np.zeros([num_production_runs])
    final_reduced_potentials=np.zeros([num_production_runs])
    proposed_reduced_potentials=np.zeros([num_production_runs])

    print('beginning production runs...')
    for production_run in tqdm.trange(num_production_runs): #number of samples to compute
        initial_position=pos
        ctx.setPositions(pos)

        integrator.step(timesteps_per_production_run); print('  completed production run for iteration: %d' %production_run)
        state = ctx.getState(getPositions=True, getEnergy=True)


        final_simulated_positions=state.getPositions(asNumpy=True)
        final_simulated_positions_no_units=np.stack([pos.value_in_unit_system(unit.md_unit_system) for pos in final_simulated_positions])
        #print('     final simulated positions: ')
        #print(final_simulated_positions_no_units)
        simulated_positions[production_run, :, :] = final_simulated_positions


        topology_proposal=tp.TopologyProposal(new_topology=top4, new_system=sys4, old_topology=top,
                                              old_system=sys, logp_proposal=0.0,
                                              new_to_old_atom_map={0:0, 1:1, 2:2}, old_chemical_state_key='3',
                                              new_chemical_state_key='4')
        #print('     completed topology_proposal')

        #propose new geometries and log probabilities of forward and back
        new_proposed_positions, logp_forward=geometry_engine.propose(topology_proposal, final_simulated_positions, beta)
        logp_reversed=geometry_engine.logp_reverse(topology_proposal, new_proposed_positions, final_simulated_positions, beta)
        #print('     new_proposed_positions: ', new_proposed_positions)
        #print('     logp_forward, logp_reversed: ', logp_forward, logp_reversed)
        new_proposed_positions_no_units=np.stack([pos.value_in_unit_system(unit.md_unit_system) for pos in new_proposed_positions])

        new_proposed_position_logger[production_run]=new_proposed_positions_no_units
        logPs_forward[production_run]=logp_forward
        logPs_reverse[production_run]=logp_reversed

        #compute the final and _proposed_positions reduced potential
        final_reduced_potential=compute_rp(sys, final_simulated_positions)
        proposed_reduced_potential=compute_rp(sys4, new_proposed_positions)

        #print('     final_reduced_potential, proposed_reduced_potential: ', final_reduced_potential, proposed_reduced_potential)


        final_reduced_potentials[production_run]=final_reduced_potential
        proposed_reduced_potentials[production_run]=proposed_reduced_potential

        #insert work function here
        pos=state.getPositions() #for next n_iteration
    print('production runs completed!'); print('')

production_and_work_calculation(3,2,10000)

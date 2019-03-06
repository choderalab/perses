#!/usr/bin/env python
# In[0]
import sys
#sys.path.append('/home/dominic/github/perses/')
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
import tqdm

#correct p-value threshold for some multiple hypothesis testing
pval_base = 0.01
ntests = 3.0
ncommits = 10000.0

pval_threshold = pval_base / (ntests * ncommits)
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

def simulate_simple_systems():
    """
    Simulate the 3, 4, and 5 particle simple systems
    :return:
    """
    system_sizes = [3,4]

    configuration_rp = {}
    sys_pos_top = {}

    for system_size in system_sizes:
        testsystem = FourAtomValenceTestSystem(n_atoms=system_size)
        sys = testsystem.system
        top = testsystem.topology
        initial_pos = testsystem.positions

        pos = minimize(sys, initial_pos)

        sys_pos_top[system_size] = (sys, pos, top)

        configurations, reduced_potentials = simulate_equilibrium(sys, pos, 3)

        configuration_rp[system_size] = (configurations, reduced_potentials)

    return sys_pos_top, configuration_rp

def compute_rp(system, positions):
    i = openmm.VerletIntegrator(1.0)
    ctx = openmm.Context(system, i)
    ctx.setPositions(positions)
    return beta*ctx.getState(getEnergy=True).getPotentialEnergy()

def run_rj_simple_system(configurations_initial, topology_proposal, n_replicates):
    import tqdm
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    #n_replicates = 100
    final_positions = []
    logPs = np.zeros([n_replicates, 4])
    geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_torsion_divisions=3600, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0)
    potential_components_nb = np.zeros([n_replicates, 2, 4])
    for replicate_idx in tqdm.trange(n_replicates):
        oldpos_idx = np.random.choice(range(len(configurations_initial)))
        old_positions = configurations_initial[oldpos_idx, :, :]
        new_positions, lp = geometry_engine.propose(topology_proposal, old_positions, beta)
        lp_reverse = geometry_engine.logp_reverse(topology_proposal, new_positions, old_positions, beta)
        initial_rp = compute_rp(topology_proposal.old_system, old_positions)
        logPs[replicate_idx, 0] = initial_rp
        logPs[replicate_idx, 1] = lp
        logPs[replicate_idx, 2] = lp_reverse
        final_rp = compute_rp(topology_proposal.new_system, new_positions)
        logPs[replicate_idx, 3] = final_rp
        final_positions.append(new_positions)
    return logPs, final_positions

def run_rj_simple_system_revised(configurations_initial, topology_proposal, n_replicates):
    import tqdm
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    #n_replicates = 100
    final_positions = []
    logPs = np.zeros([n_replicates, 4])
    geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_torsion_divisions=3600, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0)
    for replicate_idx in tqdm.trange(n_replicates):
        #oldpos_idx = np.random.choice(range(len(configurations_initial)))
        old_positions = configurations_initial[replicate_idx, :, :]
        new_positions, lp = geometry_engine.propose(topology_proposal, old_positions, beta)
        #print('new_positions: ', new_positions)
        lp_reverse = geometry_engine.logp_reverse(topology_proposal, new_positions, old_positions, beta)
        initial_rp = compute_rp(topology_proposal.old_system, old_positions)
        logPs[replicate_idx, 0] = initial_rp
        logPs[replicate_idx, 1] = lp
        logPs[replicate_idx, 2] = lp_reverse
        final_rp = compute_rp(topology_proposal.new_system, new_positions)
        logPs[replicate_idx, 3] = final_rp
        final_positions.append(new_positions)
    return logPs, final_positions

def create_simple_topology_proposal(sys_pos_top, n_atoms_initial, n_atoms_final, direction='forward'):
    """

    :param n_atoms_initial:
    :param n_atoms_final:
    :param configurations_rp:
    :param sys_pos_top:
    :return:
    """
    from perses.rjmc import topology_proposal as tp

    if direction=='forward' or direction=='forwards':
        initial_system, initial_position, initial_topology = sys_pos_top['A']
        final_system, final_positions, final_topology = sys_pos_top['B']
    elif direction=='backward'or direction=='backwards':
        initial_system, initial_position, initial_topology = sys_pos_top['B']
        final_system, final_positions, final_topology = sys_pos_top['A']
    else:
        raise ValueError("direction may only be 'forward(s)' or 'backward(s)'!")

    if n_atoms_initial == 3 and (n_atoms_final == 4 or n_atoms_final == 5):
        new_to_old_atom_map = {0: 0, 1: 1, 2: 2}
    elif n_atoms_initial == 4 and n_atoms_final == 5:
        new_to_old_atom_map = {0: 0, 1: 1, 2: 2, 3: 3}
    elif n_atoms_initial==4 and n_atoms_final==3:
        new_to_old_atom_map = {0: 0, 1: 1, 2: 2}
    elif n_atoms_initial==5 and n_atoms_final==4:
        new_to_old_atom_map = {0:0, 1:1, 2:2, 3:3}
    elif n_atoms_initial==5 and n_atoms_final==3:
        new_to_old_atom_map = {0:0, 1:1, 2:2}
    else:
        raise ValueError("This method only supports going from 3->4, 4->3, 4->5, 5->4 3->5, or 5->3")

    topology_proposal = tp.TopologyProposal(new_topology=final_topology, new_system=final_system, old_topology=initial_topology,
                                            old_system=initial_system, logp_proposal=0.0,
                                            new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key=str(n_atoms_initial),
                                            new_chemical_state_key=str(n_atoms_final))
    return topology_proposal

def minimize(system, positions):
    """
    Utility function to minimize a system
    """
    ctx = openmm.Context(system, openmm.VerletIntegrator(1.0))
    ctx.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(ctx)

    return ctx.getState(getPositions=True).getPositions(asNumpy=True)

def simulate_equilibrium(system, starting_configuration, n_iterations):
    """
    Simulate equilibrium Langevin dynamics at 300.0 kelvin

    Parameters
    ----------
    system : openmm.System
        The system to simulate
    starting_configuration : [n, 3] np.array of quantity
        The initial configuration of the system
    n_iterations : int
        The number of times to run 1000 steps of dynamics

    Returns
    ----------
    simulated_positions : np.array of quantity
        The positions from the simulations
    reduced_potentials : np.array
        The reduced potentials corresponding to each frame
    """
    from openmmtools import integrators
    import tqdm
    integrator = integrators.LangevinIntegrator()
    platform = openmm.Platform.getPlatformByName("CPU")
    ctx = openmm.Context(system, integrator)
    ctx.setPositions(starting_configuration)
    simulated_positions = unit.Quantity(np.zeros([n_iterations, system.getNumParticles(), 3]), unit=unit.nanometers)
    reduced_potentials = np.zeros([n_iterations])
    for iteration in tqdm.trange(n_iterations):
        integrator.step(1000)
        state = ctx.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True)
        reduced_potentials[iteration] = beta * state.getPotentialEnergy()
        simulated_positions[iteration, :, :] = positions

    return simulated_positions, reduced_potentials

def run_simple_transformations():
    """
    Run all simple transformations (3->4, 4->5, 3->5)
    :return:
    """
    sys_pos_top, configuration_rp = simulate_simple_systems()
    logp_final_positions = {}

    #proposals = [[3,4], [4,5], [3,5]]
    proposals=[[3,4]]

    for proposal in proposals:
        topology_proposal = create_simple_topology_proposal(sys_pos_top, n_atoms_initial=proposal[0], n_atoms_final=proposal[1])
        configurations = configuration_rp[proposal[0]][0]
        logp, final_positions = run_rj_simple_system(configurations, topology_proposal)
        final_positions_no_units = [pos.value_in_unit_system(unit.md_unit_system) for pos in
                                    final_positions]
        final_positions_stacked = np.stack(final_positions_no_units)
        logp_final_positions['{}-{}'.format(*proposal)] = (logp, final_positions_stacked)

    return sys_pos_top, configuration_rp, logp_final_positions

def run_and_save_tx(outfile="/home/dominic/saved_run.npy"):
    """

    :param outfile:
    :return:
    """
    sys_pos_top, configuration_rp, logp_final_positions = run_simple_transformations()
    np.save(outfile, (sys_pos_top, configuration_rp, logp_final_positions))

def compute_generalized_work(saved_workfile, initial_num_beads, final_num_beads):
    """
    Using the saved workfile, return the generalized work for each attempt:

    (logp_final - logp_initial) + (logp_reverse - logp_forward)

    Parameters
    ----------
    saved_workfile : str
        name of file where quantities were saved
    initial_num_beads : int
        number of beads in initial system
    final_num_beads : int
        number of beads in final system

    Returns
    -------
    g_work : np.array of float
        Generalized work of each attempt
    """
    saved_data = np.load(saved_workfile)[2]
    logp, final_positions_stacked = saved_data['{}-{}'.format(initial_num_beads, final_num_beads)]
    g_work = logp[:, 3] - logp[:, 0] + logp[:, 2] - logp[:, 1]

    return g_work


#############################################
# In[1] 3 --> 4 bead system


class AnalyticalBeadSystems(object):

    """
    We conduct tests on the analytical systems of 3-to-4 bead geometry proposals and 4-to-3 bead geometry proposals to determine that
    the variance is bounded by the numerical distribution being sampled, as well as that the work in the 3-to-4 bead proposal is the
    negative of the reverse.

    It should be noted that the work in either proposal should amount to some constant (consistent with a ratio of partition functions since
    log weights of each distribution being sampled is set to zero)
    """
    def __init__(self, transformation):
        from openmmtools import integrators
        import tqdm
        self.num_iterations=10
        self.transformation=transformation

        self.testsystemA=FourAtomValenceTestSystem(n_atoms=self.transformation[0])
        self.testsystemB=FourAtomValenceTestSystem(n_atoms=self.transformation[1])


        sysA = self.testsystemA.system
        topA = self.testsystemA.topology
        initial_posA = self.testsystemA.positions

        sysB = self.testsystemB.system
        topB = self.testsystemB.topology
        initial_posB = self.testsystemB.positions

        posA = minimize(sysA, initial_posA)
        posB=  minimize(sysB, initial_posB)

        sys_pos_top=dict()
        sys_pos_top['A']=(sysA,posA,topA)
        sys_pos_top['B']=(sysB, posB, topB)
        self.sys_pos_top=sys_pos_top


        self.platform=openmm.Platform.getPlatformByName("CPU")
        self.integrator = integrators.LangevinIntegrator()
        self.ctx=openmm.Context(sysA, self.integrator)

        self.ctx.setPositions(posA)

    def create_iid_bead_systems(self):
        self.simulated_positions=unit.Quantity(np.zeros([self.num_iterations, self.transformation[0],3]), unit=unit.nanometers)

        for iteration in tqdm.trange(self.num_iterations):
            self.integrator.step(1000)
            state=self.ctx.getState(getPositions=True)
            self.simulated_positions[iteration,:,:]=state.getPositions(asNumpy=True)
        #print('simulated_positions: ', self.simulated_positions)

    def _forward_transformation(self):
        topology_proposal=create_simple_topology_proposal(self.sys_pos_top, n_atoms_initial=self.transformation[0], n_atoms_final=self.transformation[1], direction='forward')
        data,proposed_positions=run_rj_simple_system_revised(self.simulated_positions, topology_proposal, self.num_iterations)
        self.work_forward = data[:, 3] - data[:, 0] - data[:, 2] + data[:, 1]

        proposed_positions_no_units = [posits.value_in_unit_system(unit.md_unit_system) for posits in
                                        proposed_positions]
        proposed_positions_stacked = np.stack(proposed_positions_no_units)
        self.proposed_positions=unit.Quantity(proposed_positions_stacked, unit=unit.nanometers)

        #print('proposed_positions_4: ', self.proposed_positions4)

    def _backward_transformation(self):
        topology_proposal=create_simple_topology_proposal(self.sys_pos_top, n_atoms_initial=self.transformation[1], n_atoms_final=self.transformation[0], direction='backward')
        data,reverted_positions=run_rj_simple_system_revised(self.proposed_positions, topology_proposal, self.num_iterations)
        self.work_reverse = data[:, 3] - data[:, 0] - data[:, 2] + data[:, 1]

        reverted_positions_no_units = [posits.value_in_unit_system(unit.md_unit_system) for posits in
                                        reverted_positions]
        reverted_positions_stacked = np.stack(reverted_positions_no_units)
        self.reverted_positions=unit.Quantity(reverted_positions_stacked, unit=unit.nanometers)

        #print('reverted_positions: ', self.reverted_positions)



    def assertion(self):
        work_comparison=[i+j for i,j in zip(self.work_forward, self.work_reverse)]
        work_forward_var=np.var(self.work_forward)
        work_reverse_var=np.var(self.work_reverse)
        #print('work_forward: ', self.work_forward)
        #print('work_reverse: ', self.work_reverse)
        return work_comparison, work_forward_var, work_reverse_var


def test_AnalyticalBeadSystems():
    a,b,c=(AnalyticalBeadSystems([3,4]), AnalyticalBeadSystems([4,5]), AnalyticalBeadSystems([3,5]))
    a.create_iid_bead_systems(); a._forward_transformation(); a._backward_transformation()
    b.create_iid_bead_systems(); b._forward_transformation(); b._backward_transformation()
    c.create_iid_bead_systems(); c._forward_transformation(); c._backward_transformation()

    work_comparison, work_forward_var, work_reverse_var=a.assertion()
    assert all(item<1e-6 for item in work_comparison) and work_forward_var<1e-6 and work_reverse_var<1e-6

    work_comparison, work_forward_var, work_reverse_var=b.assertion()
    assert all(item<1e-6 for item in work_comparison) and work_forward_var<1e-6 and work_reverse_var<1e-6

    work_comparison, work_forward_var, work_reverse_var=c.assertion()
    assert all(item<1e-6 for item in work_comparison) and work_forward_var<1e-6 and work_reverse_var<1e-6

    #print("work_comparison: ", work_comparison )
    #print("work_forward_var: ", work_forward_var)
    #print("work_reverse_var: ", work_reverse_var)

test_AnalyticalBeadSystems()

#!/usr/bin/env python
import sys
import simtk.openmm as openmm
import simtk.unit as unit
import numpy as np
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

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
CARBON_MASS = 12.01

istravis = os.environ.get('TRAVIS', None) == 'true'

REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("Reference")

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

    def __del__(self):
        if hasattr(self, '_context'):
            del self._context

class LinearValenceTestSystem(GeometryTestSystem):
    """
    This testsystem has 3 to 5 particles, and the potential for a bond, angle, torsion term.
    The particles are 0-1-2-3 atom-bond-angle-torsion. The positions for the atoms were taken
    from an earlier test for the geometry engine.
    """

    def __init__(self, bond=True, angle=True, torsion=True, n_atoms=4, add_extra_angle=False):
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
        _geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0)
        for _replicate_idx in tqdm.trange(n_replicates):
            _old_positions = configurations_initial[_replicate_idx, :, :]
            _new_positions, _lp = _geometry_engine.propose(topology_proposal, _old_positions, beta)
            _lp_reverse = _geometry_engine.logp_reverse(topology_proposal, _new_positions, _old_positions, beta)
            _initial_rp = self.compute_rp(topology_proposal.old_system, _old_positions)
            logPs[_replicate_idx, 0] = _initial_rp
            logPs[_replicate_idx, 1] = _lp
            logPs[_replicate_idx, 2] = _lp_reverse
            final_rp = self.compute_rp(topology_proposal.new_system, _new_positions)
            logPs[_replicate_idx, 3] = final_rp
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

        for _iteration in tqdm.trange(self.num_iterations):
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
        work_sum, work_forward_stddev, work_reverse_stddev = test.work_comparison()
        assert (work_forward_stddev <= WORK_STDDEV_THRESHOLD), "forward work stddev {} exceeds threshold {}".format(work_forward_stddev, WORK_STDDEV_THRESHOLD)
        assert (work_reverse_stddev <= WORK_STDDEV_THRESHOLD), "reverse work stddev {} exceeds threshold {}".format(work_reverse_stddev, WORK_STDDEV_THRESHOLD)
        assert np.all(abs(work_sum) <= WORK_SUM_THRESHOLD), "sum of works {} exceeds threshold {}".format(work_sum, WORK_SUM_THRESHOLD)

#test_AnalyticalBeadSystems()

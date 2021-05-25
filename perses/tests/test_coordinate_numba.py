import simtk.openmm as openmm
import simtk.unit as unit
import numpy as np
import os
from openmmtools.constants import kB
from perses.rjmc.geometry import check_dimensionality
from perses.tests.test_geometry_engine import _get_internal_from_omm
################################################################################
# Global parameters
################################################################################
temperature = 300.0 * unit.kelvin # unit-bearing temperature
kT = kB * temperature # unit-bearing thermal energy
beta = 1.0/kT # unit-bearing inverse thermal energy
CARBON_MASS = 12.01 # float (implicitly in units of AMU)
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("Reference")
running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'
#########################################
# Tests
#########################################

def test_coordinate_conversion():
    """
    test that the `_internal_to_cartesian` and `_cartesian_to_internal` functions in `geometry.py` convert with correct
    dimensionality and within an error of 1e-12 for random inputs
    """
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
        # Check internal coordinates do not have units
        r, theta, phi = rtp
        assert isinstance(r, float)
        assert isinstance(theta, float)
        assert isinstance(phi, float)
        # Check that we can reproduce original unit-bearing positions
        xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        assert check_dimensionality(xyz, unit.nanometers)
        assert np.linalg.norm(xyz-atom_position) < 1.0e-12

def test_openmm_dihedral():
    """
    Test FFAllAngleGeometryEngine _internal_to_cartesian and _cartesian_to_internal are consistent with OpenMM torsion angles.
    """
    TORSION_TOLERANCE = 1.0e-4 # permitted disagreement in torsions

    # Create geometry engine
    from perses.rjmc import geometry
    geometry_engine = geometry.FFAllAngleGeometryEngine({'test': 'true'})

    # Create a four-bead test system with a single custom force that measures the OpenMM torsion
    import simtk.openmm as openmm
    integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
    sys = openmm.System()
    force = openmm.CustomTorsionForce("theta")
    for i in range(4):
        sys.addParticle(1.0*unit.amu)
    force.addTorsion(0,1,2,3,[])
    sys.addForce(force)
    positions = unit.Quantity(np.array([
            [0.10557722, -1.10424644, -1.08578826],
            [0.0765,  0.1,  -0.4005],
            [0.0829, 0.0952, -0.2479],
            [-0.057,  0.0951, -0.1863],
            ]), unit.nanometers)
    atom_position = positions[0,:]
    bond_position = positions[1,:]
    angle_position = positions[2,:]
    torsion_position = positions[3,:]

    #atom_position = unit.Quantity(np.array([ 0.10557722 ,-1.10424644 ,-1.08578826]), unit=unit.nanometers)
    #bond_position = unit.Quantity(np.array([ 0.0765,  0.1  ,  -0.4005]), unit=unit.nanometers)
    #angle_position = unit.Quantity(np.array([ 0.0829 , 0.0952 ,-0.2479]) ,unit=unit.nanometers)
    #torsion_position = unit.Quantity(np.array([-0.057 ,  0.0951 ,-0.1863] ) ,unit=unit.nanometers)

    # Compute the dimensionless internal coordinates consistent with this geometry
    rtp, detJ = geometry_engine._cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
    (r, theta, phi) = rtp # dimensionless internal coordinates

    # Create a reference context
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(sys, integrator, platform)
    context.setPositions([atom_position, bond_position, angle_position, torsion_position])
    openmm_phi = context.getState(getEnergy=True).getPotentialEnergy()/unit.kilojoule_per_mole # this system converts torsion radians -> kJ/mol        
    assert np.linalg.norm(openmm_phi - phi) < TORSION_TOLERANCE or np.abs(np.linalg.norm(openmm_phi - phi))-2*np.pi < TORSION_TOLERANCE, '_cartesian_to_internal and OpenMM disagree on torsions'

    # Test _internal_to_cartesian by rotating around the torsion
    n_divisions = 100
    phis = np.arange(-np.pi, np.pi, (2.0*np.pi)/n_divisions) # _internal_to_cartesian only accepts dimensionless quantities
    for i, phi in enumerate(phis):
        # Note that (r, theta, phi) are dimensionless here
        xyz_atom1, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        positions[0,:] = xyz_atom1
        context.setPositions(positions)
        openmm_phi = context.getState(getEnergy=True).getPotentialEnergy()/unit.kilojoule_per_mole # this system converts torsion radians -> kJ/mol
        msg  = '_internal_to_cartesian and OpenMM disagree on torsions: \n'
        msg += '_internal_to_cartesian generated positions for: {}\n'.format(phi)
        msg += 'OpenMM: {}\n'.format(openmm_phi)
        msg += 'positions: {}'.format(positions)
        # check that difference in torsions is ~ 0 or ~ 2pi
        assert np.linalg.norm(openmm_phi - phi) < TORSION_TOLERANCE or np.abs(np.linalg.norm(openmm_phi - phi))-2*np.pi < TORSION_TOLERANCE, msg

        # Check that _cartesian_to_internal agrees
        rtp, detJ = geometry_engine._cartesian_to_internal(xyz_atom1, bond_position, angle_position, torsion_position)
        assert np.linalg.norm(phi - rtp[2]) < TORSION_TOLERANCE, '_internal_to_cartesian disagrees with _cartesian_to_internal'

    # Clean up
    del context

def test_try_random_itoc():
    """
    test whether a perturbed four-atom system gives the same internal and cartesian coords when recomputed with `_internal_to_cartesian`
    and `_cartesian_to_internal` as compared to the values output by `_get_internal_from_omm`
    """
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
        recomputed_xyz, _ = geometry_engine._internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        new_r, new_theta, new_phi = _get_internal_from_omm(recomputed_xyz,bond_position, angle_position, torsion_position)
        TOLERANCE = 1e-10
        difference = np.linalg.norm(np.array(atom_position/unit.nanometers) - np.array(recomputed_xyz/unit.nanometers))
        assert difference < TOLERANCE, f"the norm of the difference in positions recomputed with original cartesians ({difference}) is greater than tolerance of {TOLERANCE}"
        difference = np.linalg.norm(np.array([r, theta, phi]) - np.array([new_r, new_theta, new_phi]))
        assert difference < TOLERANCE, f"the norm of the difference in internals recomputed with original sphericals ({difference}) is greater than tolerance of {TOLERANCE}"

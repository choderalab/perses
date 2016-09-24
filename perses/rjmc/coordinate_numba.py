from numba import jit, float64
import numpy as np

@jit(float64[:](float64[:], float64[:]), nopython=True, nogil=True, cache=True)
def _cross_vec3(a, b):
    c = np.zeros(3)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

@jit(float64(float64[:]), nopython=True, nogil=True, cache=True)
def _norm(a):
    n_2 = np.dot(a, a)
    return np.sqrt(n_2)

@jit(float64[:,:](float64[:], float64), nopython=True, nogil=True, cache=True)
def _rotation_matrix(axis, angle):
    """
    This method produces a rotation matrix given an axis and an angle.
    """
    axis_norm = _norm(axis)
    for k in range(3):
        axis[k] = axis[k] / axis_norm
    axis_squared = axis**2
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.zeros((3,3), dtype=float64)

    rotation_matrix[0, 0] = cos_angle + axis_squared[0]*(1.0-cos_angle)
    rotation_matrix[0, 1] = axis[0]*axis[1]*(1.0-cos_angle) - axis[2]*sin_angle
    rotation_matrix[0, 2] = axis[0]*axis[2]*(1.0-cos_angle) + axis[1]*sin_angle

    rotation_matrix[1, 0] = axis[1]*axis[0]*(1.0-cos_angle) + axis[2]*sin_angle
    rotation_matrix[1, 1] = cos_angle + axis_squared[1]*(1.0-cos_angle)
    rotation_matrix[1, 2] = axis[1]*axis[2]*(1.0-cos_angle) - axis[0]*sin_angle

    rotation_matrix[2, 0] = axis[2]*axis[0]*(1.0-cos_angle) - axis[1]*sin_angle
    rotation_matrix[2, 1] = axis[2]*axis[1]*(1.0-cos_angle) + axis[0]*sin_angle
    rotation_matrix[2, 2] = cos_angle + axis_squared[2]*(1.0-cos_angle)

    return rotation_matrix

@jit(float64[:](float64[:], float64[:], float64[:], float64[:]), nopython=True, nogil=True, cache=True)
def internal_to_cartesian(bond_position, angle_position, torsion_position, internal_coordinates):

    r = internal_coordinates[0]
    theta = internal_coordinates[1]
    phi = internal_coordinates[2]
    a = angle_position - bond_position
    b = angle_position - torsion_position

    a_u = a / _norm(a)
    b_u = b / _norm(b)

    d_r = r*a_u

    normal = _cross_vec3(a_u, b_u)

    #construct the angle rotation matrix
    angle_axis = normal / _norm(normal)
    angle_rotation_matrix = _rotation_matrix(angle_axis, theta)

    #apply it
    d_ang = np.dot(angle_rotation_matrix, d_r)

    #construct the torsion rotation matrix and apply it
    torsion_axis = a_u
    torsion_rotation_matrix = _rotation_matrix(torsion_axis, -(phi+np.pi))
    #apply it
    d_torsion = np.dot(torsion_rotation_matrix, d_ang)

    #add the positions of the bond atom
    xyz = bond_position + d_torsion
    return xyz

@jit(float64[:,:](float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, nogil=True, cache=True)
def torsion_scan(bond_position, angle_position, torsion_position, internal_coordinates, phi_set):
    n_phis = len(phi_set)
    xyzs = np.zeros((n_phis, 3))
    for i in range(n_phis):
        internal_coordinates[2] = phi_set[i]
        xyzs[i] = internal_to_cartesian(bond_position, angle_position, torsion_position, internal_coordinates)
    return xyzs

@jit(float64[:](float64[:], float64[:], float64[:], float64[:]), nopython=True, nogil=True, cache=True)
def cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position):

            a = atom_position - bond_position
            b = angle_position - bond_position
            #3-4 bond
            c = angle_position - torsion_position
            a_u = a / _norm(a)
            b_u = b / _norm(b)
            c_u = c / _norm(c)

            #bond length
            r = _norm(a)

            #bond angle
            cos_theta = np.dot(a_u, b_u)
            if cos_theta > 1.0:
                cos_theta = 1.0
            elif cos_theta < -1.0:
                cos_theta = -1.0
            theta = np.arccos(cos_theta)

            #torsion angle
            plane1 = _cross_vec3(a_u, b_u)
            plane2 = _cross_vec3(b_u, c_u)

            cos_phi = np.dot(plane1, plane2) / (_norm(plane1)*_norm(plane2))
            if cos_phi < -1.0:
                cos_phi = -1.0
            elif cos_phi > 1.0:
                cos_phi = 1.0

            phi = np.arccos(cos_phi)

            if np.dot(a, plane2) <= 0:
                phi = -phi

            return np.array([r, theta, phi])

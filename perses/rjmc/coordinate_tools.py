import numpy as np

def _rotation_matrix(axis, angle):
    """
    This method produces a rotation matrix given an axis and an angle.
    """
    axis = axis/np.linalg.norm(axis)
    axis_squared = axis**2
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rot_matrix_row_one = np.array([cos_angle+axis_squared[0]*(1-cos_angle),
                                   axis[0]*axis[1]*(1-cos_angle) - axis[2]*sin_angle,
                                   axis[0]*axis[2]*(1-cos_angle)+axis[1]*sin_angle])

    rot_matrix_row_two = np.array([axis[1]*axis[0]*(1-cos_angle)+axis[2]*sin_angle,
                                  cos_angle+axis_squared[1]*(1-cos_angle),
                                  axis[1]*axis[2]*(1-cos_angle) - axis[0]*sin_angle])

    rot_matrix_row_three = np.array([axis[2]*axis[0]*(1-cos_angle)-axis[1]*sin_angle,
                                    axis[2]*axis[1]*(1-cos_angle)+axis[0]*sin_angle,
                                    cos_angle+axis_squared[2]*(1-cos_angle)])

    rotation_matrix = np.array([rot_matrix_row_one, rot_matrix_row_two, rot_matrix_row_three])
    return rotation_matrix

def _cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position):
            """
            Cartesian to internal function
            """
            a = atom_position - bond_position
            b = angle_position - bond_position
            #3-4 bond
            c = angle_position - torsion_position
            a_u = a / np.linalg.norm(a)
            b_u = b / np.linalg.norm(b)
            c_u = c / np.linalg.norm(c)

            #bond length
            r = np.linalg.norm(a)

            #bond angle
            cos_theta = np.dot(a_u, b_u)
            if cos_theta > 1.0:
                cos_theta = 1.0
            elif cos_theta < -1.0:
                cos_theta = -1.0
            theta = np.arccos(cos_theta)

            #torsion angle
            plane1 = np.cross(a_u, b_u)
            plane2 = np.cross(b_u, c_u)

            cos_phi = np.dot(plane1, plane2) / (np.linalg.norm(plane1)*np.linalg.norm(plane2))
            if cos_phi < -1.0:
                cos_phi = -1.0
            elif cos_phi > 1.0:
                cos_phi = 1.0

            phi = np.arccos(cos_phi)

            if np.dot(a, plane2) <= 0:
                phi = -phi

            if np.isnan(phi):
                raise Exception("phi is nan, cos_phi is %s" % str(cos_phi))

            return np.array([r, theta, phi])

def _internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi):
            a = angle_position - bond_position
            b = angle_position - torsion_position


            a_u = a / np.linalg.norm(a)
            b_u = b / np.linalg.norm(b)

            d_r = r*a_u

            normal = np.cross(a_u, b_u)

            #construct the angle rotation matrix
            angle_axis = normal / np.linalg.norm(normal)
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

import numpy as np
import simtk.unit as units

def _rotation_matrix(axis, angle):
    """
    This method produces a rotation matrix given an axis and an angle.
    """
    axis = axis/units.norm(axis)
    axis_squared = axis**2
    cos_angle = units.cos(angle)
    sin_angle = units.sin(angle)
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
            Cartesian to internal function (hah)
            """
            a = bond_position - atom_position
            b = angle_position - bond_position
            #3-4 bond
            c = angle_position - torsion_position
            a_u = a / units.norm(a)
            b_u = b / units.norm(b)
            c_u = c / units.norm(c)

            #bond length
            r = units.norm(a)

            #bond angle
            theta = units.acos(units.dot(-a_u, b_u))

            #torsion angle
            plane1 = np.cross(a, b)
            plane2 = np.cross(b, c)

            cos_phi = units.dot(plane1, plane2) / (units.norm(plane1)*units.norm(plane2))
            if cos_phi < -1.0:
                cos_phi = -1.0
            elif cos_phi > 1.0:
                cos_phi = 1.0

            phi = units.acos(cos_phi)

            if units.dot(np.cross(plane1, plane2), b_u) > 0:
                phi = -phi

            if np.isnan(phi/phi.unit):
                raise Exception("phi is nan")

            return np.array([r, theta, phi])

def _internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi):
            a = angle_position - bond_position
            b = angle_position - torsion_position

            a_u = a / units.norm(a)


            d_r = r*a_u

            normal = np.cross(a, b)

            #construct the angle rotation matrix
            angle_axis = normal / units.norm(normal)
            angle_rotation_matrix = _rotation_matrix(angle_axis, theta)

            #apply it
            d_ang = units.dot(angle_rotation_matrix, d_r)

            #construct the torsion rotation matrix and apply it
            torsion_axis = a_u
            torsion_rotation_matrix = _rotation_matrix(torsion_axis, phi)
            #apply it
            d_torsion = units.dot(torsion_rotation_matrix, d_ang)

            #add the positions of the bond atom
            xyz = bond_position + d_torsion
            return xyz

if __name__=="__main__":
    example_coordinates = units.Quantity(np.random.normal(size=[100,3]), unit=units.nanometers)
    #try to transform random coordinates to and from
    for i in range(20):
        indices = np.random.randint(100, size=4)
        atom_position = example_coordinates[indices[0]]
        bond_position = example_coordinates[indices[1]]
        angle_position = example_coordinates[indices[2]]
        torsion_position = example_coordinates[indices[3]]
        rtp = _cartesian_to_internal(atom_position, bond_position, angle_position, torsion_position)
        r = rtp[0]
        theta = rtp[1]
        phi = rtp[2]
        xyz = _internal_to_cartesian(bond_position, angle_position, torsion_position, r, theta, phi)
        print(units.norm(atom_position))
        #print(units.norm(xyz-atom_position))
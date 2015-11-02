__author__ = 'patrickgrinaway'
"""
Old stuff goes here
"""
    def _cartesian_to_internal(self, atom, bond_atom, angle_atom, torsion_atom, positions):
        """
        Convert the cartesian coordinates of an atom to internal
        """
        a = positions[bond_atom] - positions[atom]
        b = positions[angle_atom] - positions[bond_atom]
        c = positions[torsion_atom] - positions[angle_atom]
        atom_position_u = positions[atom] / np.linalg.norm(positions[atom])

        a_u = a / np.linalg.norm(a)
        b_u = b / np.linalg.norm(b)
        c_u = c / np.linalg.norm(c)

        #bond length
        r = np.linalg.norm(a)

        #bond angle
        theta = np.arccos(np.dot(-a_u, b_u))

        #torsion angle
        plane1 = np.cross(a, b)
        plane2 = np.cross(b, c)

        phi = np.arccos(np.dot(plane1, plane2)/(np.linalg.norm(plane1)*np.linalg.norm(plane2)))

        if np.dot(np.cross(plane1, plane2), b_u) < 0:
            phi = -phi

        return np.array([r, theta, phi])

    def _internal_to_cartesian(self, bond_atom, angle_atom, torsion_atom, r, theta, phi):
        """
        Convert internal coordinates to cartesian
        """
        #2-3 bond
        a = angle_atom - bond_atom

        #3-4 bond
        b = angle_atom - torsion_atom


        d = r*a/(np.sqrt(np.dot(a,a))*units.nanometers)

        normal = np.cross(a, b)
        a = a/a.unit #get rid of the units of a
        #rotate the vector about the normal
        d = np.dot(self.rotation_matrix(normal, theta), d)

        #rotate the vector about the 2-3 bond)
        d = np.dot(self.rotation_matrix(a, phi), d)

        atomic_coordinates = bond_atom + d * units.nanometers

        return atomic_coordinates

    def rotation_matrix(self, axis, angle):
        """
        Euler-Rodrigues formula for rotation matrix
        """
        # Normalize the axis
        angle = angle/angle.unit
        axis = axis/np.sqrt(np.dot(axis, axis))
        a = np.cos(angle/2)
        b, c, d = -axis*np.sin(angle/2)
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

    def _jacobian_symbolic(self, rthetaphi, xyz):
        """
        Calculate a symbolic determinant of jacobian using sympy
        for the xyz and rthetaphi

        Arguments
        ---------
        rthetaphi : list of symbol
            list of symbolic x, y, z of interest
        xyz : list of symbol
            list of symbolic r, theta, phi of interest
        """
        j_rows = []
        for sym in rthetaphi:
            j_rows.append([sym.diff(xyz[0]), sym.diff(xyz[1]), sym.diff(xyz[2])])
        jacobian = sympy.Matrix(j_rows)
        jacobian_determinant = jacobian.det(method='berkowitz')
        return jacobian_determinant
    def _internal_to_cartesian(self, bond_atom, angle_atom, torsion_atom, r, theta, phi, positions):
        """
        Convert the internal coordinates to cartesian via sympy, also produce detJ
        """
        positions = positions/positions.unit
        r = r/r.unit
        theta = theta/theta.unit - np.pi/2
        phi = phi/phi.unit
        N = vector.ReferenceFrame('N')
        bond_position = positions[bond_atom]
        angle_position = positions[angle_atom]
        torsion_position = positions[torsion_atom]
        r_sym, theta_sym, phi_sym = sympy.symbols('r_sym, theta_sym, phi_sym')

        bond_vec = N.x*bond_position[0] + N.y*bond_position[1] + N.z*bond_position[2]
        angle_vec = N.x*angle_position[0] + N.y*angle_position[1] + N.z*angle_position[2]
        torsion_vec = N.x*torsion_position[0] + N.y*torsion_position[1] + N.z*torsion_position[2]

        #2-3 bond
        a = angle_vec - bond_vec
        a_u = a.normalize()

        #3-4 bond
        b = angle_vec - torsion_vec
        b_u = b.normalize()

        d = r_sym*a_u

        normal = vector.cross(a, b)

        #rotate the vector about the normal (bond angle)
        angle_rotation = N.orientnew('angle_rotation', 'Axis',[theta_sym, normal])
        angle_dcm = N.dcm(angle_rotation)
        d_theta = angle_dcm * d.to_matrix(N)

        #rotate the vector about the 2-3 bond (torsion)
        torsion_rotation = N.orientnew('torsion_rotation', 'Axis', [phi_sym, a])
        torsion_dcm = N.dcm(torsion_rotation)
        d_phi = torsion_dcm*d.to_matrix(N)

        #lambdify the expression and calculate the new coordinates
        cartesian_atomic_position = sympy.lambdify("r_sym, theta_sym, phi_sym", d_phi, "numpy")(r, theta, phi)

        #now get a jacobian matrix, take the determinant and absolute value, and we're done
        d_x = d_phi[0]
        d_y = d_phi[1]
        d_z = d_phi[2]

        jacobian_r = [sympy.diff(d_x, 'r_sym'), sympy.diff(d_y, 'r_sym'), sympy.diff(d_z, 'r_sym')]
        jacobian_theta = [sympy.diff(d_x, 'theta_sym'), sympy.diff(d_y, 'theta_sym'), sympy.diff(d_z, 'theta_sym')]
        jacobian_phi = [sympy.diff(d_x, 'phi_sym'), sympy.diff(d_y, 'phi_sym'), sympy.diff(d_z, 'phi_sym')]
        jacobian = sympy.Matrix([jacobian_r, jacobian_theta, jacobian_phi])
        jacobian_determinant = jacobian.det(method='berkowitz')

        detJ = sympy.lambdify('r_sym, theta_sym, phi_sym', jacobian_determinant, )(r, theta, phi)

        return np.array(cartesian_atomic_position), np.abs(detJ)



    def _cartesian_to_internal(self, atom, bond_atom, angle_atom, torsion_atom, positions):
        """
        Use sympy to derive the r, theta, phi for
        """
        N = vector.ReferenceFrame('N')
        Xa, Ya, Za = sympy.symbols("Xa Ya Za")

        atomic_position = positions[atom]
        bond_position = positions[bond_atom]
        angle_position = positions[angle_atom]
        torsion_position = positions[torsion_atom]
        atom_vec = N.x*Xa+N.y*Ya+N.z*Za
        bond_vec = N.x*bond_position[0] + N.y*bond_position[1] + N.z*bond_position[2]
        angle_vec = N.x*angle_position[0] + N.y*angle_position[1] + N.z*angle_position[2]
        torsion_vec = N.x*torsion_position[0] + N.y*torsion_position[1] + N.z*torsion_position[2]

        a = bond_vec - atom_vec
        b = angle_vec - bond_vec
        c = torsion_vec - angle_vec

        atom_pos_u = atom_vec.normalize()
        a_u = a.normalize()
        b_u = b.normalize()
        c_u = c.normalize()

        #bond length
        r_exp = a.magnitude()

        #bond angle
        theta_exp = sympy.acos(vector.dot(-a_u, b_u))

        #torsion
        plane1 = vector.cross(a, b)
        plane2 = vector.cross(b, c)

        plane1_norm = plane1.magnitude()
        plane2_norm = plane2.magnitude()

        plane_dot_u = vector.dot(plane1, plane2) / (plane1_norm * plane2_norm)

        phi_exp = sympy.acos(plane_dot_u)

        r = sympy.lambdify("Xa, Ya, Za", r_exp, "numpy")(*atomic_position)
        theta = sympy.lambdify("Xa, Ya, Za", theta_exp, "numpy")(*atomic_position)
        phi = sympy.lambdify("Xa, Ya, Za", phi_exp, "numpy")(*atomic_position)

        #calculate the jacobian and its determinant
        j_rows = []
        for sym in [r_exp, theta_exp, phi_exp]:
            j_rows.append([sym.diff(Xa), sym.diff(Ya), sym.diff(Za)])
        jacobian = sympy.Matrix(j_rows)
        jacobian_determinant = jacobian.det(method='berkowitz')

        detJ = sympy.lambdify("Xa, Ya, Za",jacobian_determinant, "numpy")(*atomic_position)


        return np.array([r, theta, phi]), np.abs(detJ)
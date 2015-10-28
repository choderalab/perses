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

from lxml import etree
import copy

class XMLmodifier(object):
    """
    TODO: Rename this -- intended specifically for use with dual topologies representing the 
    combination of 2 molecules, instead of the general case of N molecules

    Note -- I'm pretty sure the only difference between this and the general one is the use of 
    lambda and 1-lambda instead of lambda(i)
    """
    def __init__(self, ffxml_filename, each_molecule_N):
        """
        """
        self.ffxml_filename = ffxml_filename
        self.each_molecule_N = each_molecule_N
        self.tree = etree.parse(ffxml_filename)
        self.root = self.tree.getroot()

        root = self.root
        for element in list(root):
            if element.tag == 'AtomTypes':
                self.atomtypes = element
            elif element.tag == 'Residues':
                self.residues = element
            elif element.tag == 'HarmonicBondForce':
                self.bondforce = element
            elif element.tag == 'HarmonicAngleForce':
                self.angleforce = element
            elif element.tag == 'PeriodicTorsionForce':
                self.torsionforce = element
            elif element.tag == 'NonbondedForce':
                self.nonbonded = element

        print("Parsed ffxml file")
        self.fixatomtypes()
        self.addbondforces()
        self.addcustomforces()

        new_filename = ffxml_filename[:-4]+"_MODIFIED.xml"
        self.tree.write(new_filename)


    def fixatomtypes(self):
        print("Saving real atom class references")
        atomtypes = self.atomtypes
        each_molecule_N = self.each_molecule_N
        # make each_molecule_N a dictionary mapping from the index of an atom in the dual
        # topology to an id # representing the molecule it originates from
        # (I'm not sure if this is more useful in general, if so it can be changed when
        # created instead of here)
        # dict {index of atom : index of molecule it comes from}
        molecule_id_of_atom = {}
        for i, start_end_indices in enumerate(each_molecule_N):
            start_index = start_end_indices[0]
            end_index = start_end_indices[1]+1
            for j in range(start_index, end_index):
                molecule_id_of_atom[j] = i

        self.substructure_length = each_molecule_N[0][1]
        self.molecule_id_of_atom = molecule_id_of_atom
        

        save_index_to_class = {}
        save_real_classes = {}

        # modify the atom class of atoms unique to certain ligands 
        # COMPLETE
        print("Renaming atom classes")
        for i, atom in enumerate(list(atomtypes)):
            # atoms in the substructure do not change and therefore don't need
            # to be customized
            real_class_name = atom.attrib['class']
            if i > self.substructure_length:
                molecule_id = str(molecule_id_of_atom[i])
                modified_class_name = real_class_name+'_'+molecule_id
                atom.attrib['class'] = modified_class_name
            else:
                modified_class_name = real_class_name
            save_index_to_class[i] = modified_class_name
            save_real_classes[modified_class_name] = real_class_name

        self.save_index_to_class = save_index_to_class
        self.save_real_classes = save_real_classes


    def addbondforces(self):
        "Recreating bond forces for new atom classes"
        root = self.root
        residues = self.residues
        save_index_to_class = self.save_index_to_class
        save_real_classes = self.save_real_classes

        # add entries to harmonic bond force with modified atom classes
        # COMPLETE
        self.all_bonds = {}
        for residue in list(residues):
            for element in list(residue):
                if element.tag == 'Bond':
                    atom1 = int(element.values()[0])
                    atom2 = int(element.values()[1])
                    if not self.all_bonds.has_key(atom1):
                        self.all_bonds[atom1] = []
                    if not self.all_bonds.has_key(atom2):
                        self.all_bonds[atom2] = []
                    self.all_bonds[atom1].append(atom2)
                    self.all_bonds[atom2].append(atom1)
                    newclass1 = save_index_to_class[atom1]
                    newclass2 = save_index_to_class[atom2]
                    oldclass1 = save_real_classes[newclass1]
                    oldclass2 = save_real_classes[newclass2]
                    if newclass1 == oldclass1 and newclass2 == oldclass2:
                        continue
                    # THIS IS MAD COOL
                    bond = root.xpath(u'//Bond[@class1=\'%s\'][@class2=\'%s\']' %(oldclass1, oldclass2))
                    if bond == []:
                        bond = root.xpath(u'//Bond[@class1=\'%s\'][@class2=\'%s\']' %(oldclass2, oldclass1))
                    bond = bond[0]
                    newbond = copy.deepcopy(bond)
                    newbond.attrib['class1'] = newclass1
                    newbond.attrib['class2'] = newclass2
                    self.bondforce.append(newbond)


    def addcustomforces(self):
        each_molecule_N = self.each_molecule_N

        for i, start_end_indices in enumerate(each_molecule_N):
        # did this need to be enumerated?
            if i == 0:
                continue
            print("Creating custom forces for molecule "+str(i))
            start_index = start_end_indices[0]
            end_index = start_end_indices[1]+1

            molecule_atom_indices = range(self.substructure_length+1)
            molecule_atom_indices.extend(range(start_index, end_index))
            if i == 1:
                scale_factor = "(1-lambda)"
            elif 1 == 2:
                scale_factor = "lambda"
            self.customangleforce(scale_factor, molecule_atom_indices)
            self.customtorsionforce(scale_factor, molecule_atom_indices)

            # TO DO: ADD CUSTOM NONBONDED FORCE CAPABILITY
            #self.customnonbondedforce(scale_factor, molecule_atom_indices)

    def customangleforce(self, scale_factor, molecule_atom_indices):
        angleforce = self.angleforce
        root = self.root
        save_index_to_class = self.save_index_to_class
        save_real_classes = self.save_real_classes

        # create the custom angle force element
        custom_angle_force = root.makeelement('CustomAngleForce', attrib={})
        custom_angle_force.attrib['energy'] = scale_factor+"*k*(theta-angle)^2"

        # add subelements for the parameters
        angle = custom_angle_force.makeelement('PerAngleParameter',attrib={'name':"angle"})
        custom_angle_force.append(angle)
        k = custom_angle_force.makeelement('PerAngleParameter',attrib={'name':"k"})
        custom_angle_force.append(k)
        scale = custom_angle_force.makeelement('GlobalParameter',attrib={ 'name':scale_factor, 'defaultValue':"0.5"})
        custom_angle_force.append(scale)

        for atom1 in self.all_bonds.keys():
            if atom1 not in molecule_atom_indices:
                continue
            newclass1 = save_index_to_class[atom1]
            oldclass1 = save_real_classes[newclass1]
            for atom2 in self.all_bonds[atom1]:
                if atom2 not in molecule_atom_indices:
                    continue
                newclass2 = save_index_to_class[atom2]
                oldclass2 = save_real_classes[newclass2]
                for atom3 in self.all_bonds[atom2]:
                    if atom3 not in molecule_atom_indices:
                        continue
                    if atom3 == atom1:
                        continue
                    newclass3 = save_index_to_class[atom3]
                    oldclass3 = save_real_classes[newclass3]
                    if newclass1 == oldclass1 and newclass2 == oldclass2 and newclass3 == oldclass3:
                        continue
                    # find the parameters for the original three atom classes
                    angle = root.xpath(u'//Angle[@class1=\'%s\'][@class2=\'%s\'][@class3=\'%s\']' %(oldclass1, oldclass2, oldclass3))
                    if angle == []:
                        angle = root.xpath(u'//Angle[@class1=\'%s\'][@class2=\'%s\'][@class3=\'%s\']' %(oldclass3, oldclass2, oldclass1))
                    if angle == []:
                        continue
                    angle = angle[0]
                    newangle = copy.deepcopy(angle)
                    newangle.attrib['class1'] = newclass1
                    newangle.attrib['class2'] = newclass2
                    newangle.attrib['class3'] = newclass3
                    if newclass1 != oldclass1 and newclass2 != oldclass2 and newclass3 != oldclass3:
                        # add to regular angleforce
                        angleforce.append(newangle)
                    else:
                        # add to custom_angle_force
                        custom_angle_force.append(newangle)

        angleforce.addprevious(custom_angle_force)


    def customtorsionforce(self, scale_factor, molecule_atom_indices):
        torsionforce = self.torsionforce
        root = self.root
        save_index_to_class = self.save_index_to_class
        save_real_classes = self.save_real_classes

        # create the custom torsion
        custom_torsion_force = root.makeelement('CustomTorsionForce', attrib={})
        custom_torsion_force.attrib['energy'] = scale_factor+"*k*(1+cos(periodicity*theta-phase))"

        # add subelements for the parameters
        # CURRENTLY UNSURE ABOUT THE MULTIPLE SETS OF PARAMETERS ISSUE
        k = custom_torsion_force.makeelement('PerAngleParameter',attrib={'name':"k"})
        custom_torsion_force.append(k)
        periodicity = custom_torsion_force.makeelement('PerAngleParameter',attrib={'name':"periodicity"})
        custom_torsion_force.append(periodicity)
        phase = custom_torsion_force.makeelement('PerAngleParameter',attrib={'name':"phase"})
        custom_torsion_force.append(phase) 

        for atom1 in self.all_bonds.keys():
            if atom1 not in molecule_atom_indices:
                continue
            newclass1 = save_index_to_class[atom1]
            oldclass1 = save_real_classes[newclass1]
            for atom2 in self.all_bonds[atom1]:
                if atom2 not in molecule_atom_indices:
                    continue
                newclass2 = save_index_to_class[atom2]
                oldclass2 = save_real_classes[newclass2]
                for atom3 in self.all_bonds[atom2]:
                    if atom3 not in molecule_atom_indices:
                        continue
                    if atom3 == atom1:
                        continue
                    newclass3 = save_index_to_class[atom3]
                    oldclass3 = save_real_classes[newclass3]
                    for atom4 in self.all_bonds[atom3]:
                        if atom4 not in molecule_atom_indices:
                            continue
                        if atom4 == atom1 or atom4 == atom2:
                            continue
                        newclass4 = save_index_to_class[atom4]
                        oldclass4 = save_real_classes[newclass4]
                        if newclass1 == oldclass1 and newclass2 == oldclass2 and newclass3 == oldclass3 and newclass4 == oldclass4:
                            continue
                        # find the parameters for the original three atom classes
                        dihedral = root.xpath(u'//Proper[@class1=\'%s\'][@class2=\'%s\'][@class3=\'%s\'][@class4=\'%s\']' %(oldclass1, oldclass2, oldclass3, oldclass4))
                        if dihedral == []:
                            dihedral = root.xpath(u'//Proper[@class1=\'%s\'][@class2=\'%s\'][@class3=\'%s\'][@class4=\'%s\']' %(oldclass4, oldclass3, oldclass2, oldclass1))
                        if dihedral == []:
                            dihedral = root.xpath(u'//Improper[@class1=\'%s\'][@class2=\'%s\'][@class3=\'%s\'][@class4=\'%s\']' %(oldclass1, oldclass2, oldclass3, oldclass4))
                        if dihedral == []:
                            dihedral = root.xpath(u'//Improper[@class1=\'%s\'][@class2=\'%s\'][@class3=\'%s\'][@class4=\'%s\']' %(oldclass4, oldclass3, oldclass2, oldclass1))
                        if dihedral == []:
                            continue
                        dihedral = dihedral[0]
                        newdihedral = copy.deepcopy(dihedral)
                        newdihedral.attrib['class1'] = newclass1
                        newdihedral.attrib['class2'] = newclass2
                        newdihedral.attrib['class3'] = newclass3
                        newdihedral.attrib['class4'] = newclass4
                        if newclass1 != oldclass1 and newclass2 != oldclass2 and newclass3 != oldclass3 and newclass4 != oldclass4:
                            # add to regular torsionforce
                            torsionforce.append(newdihedral)
                        else:
                            # add to custom_torsion_force
                            custom_torsion_force.append(newdihedral)

        torsionforce.addprevious(custom_torsion_force)



    def customnonbondedforce(self, scale_factor, molecule_atom_indices):
        nonbonded = self.nonbonded
        root = self.root

        # create the custom nonbonded force element
        custom_nonbonded_force = copy.deepcopy(nonbonded)
        custom_nonbonded_force.tag = 'CustomNonbondedForce'
        # I don't know what the energy expression is

        # assuming the energy expression gets in there right this is g2g:
        for idx, nonbonded_particle in enumerate(list(custom_nonbonded_force)):
            if idx not in range(start_index, end_index):
                custom_nonbonded_force.remove(nonbonded_particle)
        nonbonded.addprevious(custom_nonbonded_force)













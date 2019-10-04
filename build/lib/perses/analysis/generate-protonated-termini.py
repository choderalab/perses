#!/bin/env python

"""
Parse an AMBER OpenMM ffxml file to generate variants of amino acid residue templates with protonated termini
"""

import xml.etree.ElementTree as etree
from xml.etree.ElementTree import Element, ElementTree, SubElement

input_filename = 'protein.ff14SB.xml' # input ffxml filename
output_filename = 'protein.ff14SB.protonated-termini.xml' # output ffxml filename

# Create new tree
new_root = Element('ForceField')
new_residues = SubElement(new_root, 'Residues')
new_tree = ElementTree(new_root)

# Parse old tree
tree = etree.parse(input_filename)
residues = tree.getroot().find('Residues').findall('Residue')
for residue in residues:
    name = residue.attrib['name']
    atom_names = set([atom.attrib['name'] for atom in residue.findall('Atom')])
    if set(['N', 'CA', 'C']).issubset(atom_names):
        residue.attrib['name'] += '-CTER-PROT'
        # Append HXT to carbonyl carbon
        atom = Element('Atom', name='HXT', type='protein-HA', charge="0.0000")
        residue.append(atom)
        bond = Element('Bond', atomName1='C', atomName2='HXT')
        residue.append(bond)
        # Remove C-terminal external bond
        for bond in residue.findall('ExternalBond'):
            if bond.attrib['atomName'] == 'C':
                residue.remove(bond)

        new_residues.append(residue)

# Write new file
new_tree.write(output_filename)

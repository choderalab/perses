# import sys
# sys.path.append('../../')
#
#
from openmmtools import cache
from simtk import openmm
from perses.tests.testsystems import AlanineDipeptideTestSystem
from perses.tests.testsystems import AlkanesTestSystem
from perses.tests.testsystems import KinaseInhibitorsTestSystem

# testsystem = AlanineDipeptideTestSystem()
# testsystem = AlkanesTestSystem(storage_filename='output.nc')
testsystem = KinaseInhibitorsTestSystem()
# Build a system
system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
# Retrieve a SAMSSampler
sams_sampler = testsystem.sams_samplers['explicit'] ## For alkanes test system and kinase inhibitor test system
# sams_sampler = testsystem.sams_samplers['implicit'] ## for alanine dipeptide test system
testsystem.exen_samplers['vacuum'].run(niterations=20)

#
# #### Test pdb fixer (compare results with results from my implementation)
#
# from pdbfixer import PDBFixer as PDBFixer
# from pkg_resources import resource_filename
# import string
#
# pdb_filename = resource_filename('openmmtools', 'data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb')
# fixer = PDBFixer(filename=pdb_filename)
#
# # Add chain ids
# alphabet = list(string.ascii_uppercase)
# for chain in fixer.topology.chains():
#     chain.id = alphabet[chain.index]
#
# fixer.applyMutations(['ALA-2-VAL'], "A") # Note: the indices for residues start at 1, not 0
# fixer.findMissingResidues()
# fixer.findMissingAtoms()
# fixer.addMissingAtoms()
# fixer.addMissingHydrogens()
#
# topology = fixer.topology
# print()
# print("pdb fixer structure...")
# for chain in topology.chains():
#     print("chain: ", chain)
#     for residue in chain.residues():
#         print("residue: ", residue)
#         for atom in residue.atoms():
#             print("atom: ", atom)
# for bond in topology.bonds():
#     print(bond)

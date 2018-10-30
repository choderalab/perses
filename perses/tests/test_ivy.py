# import sys
# sys.path.append('../../')
#
#
from perses.tests.testsystems import AlanineDipeptideTestSystem
from perses.tests.utils import compute_potential_components
# import sys
# sys.stdout = open("stdout", mode='w')
#skip 'buffering' if you don't want the output to be flushed right away after written
testsystem = AlanineDipeptideTestSystem()
# Build a system
system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
# Retrieve a SAMSSampler
sams_sampler = testsystem.sams_samplers['implicit']
testsystem.exen_samplers['vacuum'].run(niterations=20)

# from perses.tests.testsystems import AlkanesTestSystem
# from perses.tests.utils import compute_potential_components
# # import sys
# # sys.stdout = open("stdout", mode='w')
# #skip 'buffering' if you don't want the output to be flushed right away after written
# testsystem = AlkanesTestSystem()
# # Build a system
# system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
# # Retrieve a SAMSSampler
# sams_sampler = testsystem.sams_samplers['vacuum']
# testsystem.exen_samplers['vacuum'].run(niterations=20)



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

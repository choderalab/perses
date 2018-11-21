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
# sams_sampler = testsystem.sams_samplers['implicit'] ## For alanine dipeptide test system
testsystem.exen_samplers['vacuum'].run(niterations=20)

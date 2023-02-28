"""
Runs a simulation with OpenMM.
"""
import sys

# Configure logging
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
from rich.console import Console
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
log = logging.getLogger("rich")

# Set parameters for simulation
from openmm import unit
temperature = 300 * unit.kelvin
pressure = 1 * unit.atmospheres
collision_rate = 1.0 / unit.picoseconds
timestep = 4.0 * unit.femtoseconds
equilibration_steps = 5000 # 20 ps
reporting_interval = 1250 # 5 ps

# Use docopt for CLI handling
# TODO: Once we refactor this to encapsulate behavior in functions (or classes) migrate to click: https://click.palletsprojects.com/en/8.1.x/
__doc__ = """Generate explicit-solvent molecular dynamics simulation for a given receptor and ligand.

Usage:
  simulate.py --receptor=FILE --ligand=FILE --nsteps=INT [--selection=SELECT] [--initial=FILE] [--minimized=FILE] [--final=FILE] [--xtctraj=FILE] [--dcdtraj=FILE] [--pdbtraj=FILE] 
  simulate.py (-h | --help)

Options:
  -h --help           Show this screen.
  --receptor=FILE     Receptor PDB filename.
  --ligand=FILE       Ligand SDF filename.
  --nsteps=INT        Number of steps to run.
  --selection=SELECT  MDTraj selection to use (e.g. 'not water') [default: all].
  --initial=FILE      Write initial complex PDB file.  
  --minimized=FILE    Write minimized complex PDB file.
  --final=FILE        Write final complex PDB file.
  --xtctraj=FILE      Generate XTC trajectory file.
  --dcdtraj=FILE      Generate DCD trajectory file.
  --pdbfile=FILE      Generate PDB trajectory file.

"""
from docopt import docopt
arguments = docopt(__doc__, version='simulate 0.1')

num_steps = int(arguments['--nsteps']) # number of integrator steps
n_snapshots = int(num_steps / reporting_interval) # calculate number of snapshots that will be generated
num_steps = n_snapshots * reporting_interval # recalculate number of steps to run

log.info(f":gear:  Processing {arguments['--receptor']} and {arguments['--ligand']}")
log.info(f':clock1:  Will run {num_steps*timestep / unit.nanoseconds:3f} ns of production simulation to generate {n_snapshots} snapshots')

# check whether we have a GPU platform and if so set the precision to mixed
speed = 0
from openmm import Platform
for i in range(Platform.getNumPlatforms()):
    p = Platform.getPlatform(i)
    # print(p.getName(), p.getSpeed())
    if p.getSpeed() > speed:
        platform = p
        speed = p.getSpeed()

if platform.getName() == 'CUDA' or platform.getName() == 'OpenCL':
    platform.setPropertyDefaultValue('Precision', 'mixed')
    log.info(f':dart:  Setting precision for platform {platform.getName()} to mixed')

# Read the molfile into RDKit, add Hs and create an openforcefield Molecule object
log.info(':pill:  Reading ligand')
from rdkit import Chem
rdkitmol = Chem.SDMolSupplier(arguments['--ligand'])[0]
log.info(f':mage:  Adding hydrogens')
rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True);
# ensure the chiral centers are all defined
Chem.AssignAtomChiralTagsFromStructure(rdkitmolh);
from openff.toolkit.topology import Molecule
ligand_mol = Molecule(rdkitmolh);

# Initialize a SystemGenerator
log.info(':wrench:  Initializing SystemGenerator')
from openmm import app
from openmmforcefields.generators import SystemGenerator
forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*unit.amu }
periodic_forcefield_kwargs = {'nonbondedMethod': app.PME}
system_generator = SystemGenerator(
    forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
    small_molecule_forcefield='openff-1.3.1',
    molecules=[ligand_mol], cache='cache.json',
    forcefield_kwargs=forcefield_kwargs, periodic_forcefield_kwargs=periodic_forcefield_kwargs);

# Use Modeller to combine the protein and ligand into a complex
log.info(':cut_of_meat:  Reading protein')
from openmm.app import PDBFile
protein_pdb = PDBFile(arguments['--receptor'])
log.info(':sandwich:  Preparing complex')
from openmm.app import Modeller
modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
# This next bit is black magic.
# Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
# an openforcefield Molecule object that was created from a RDKit molecule.
# The topology part is described in the openforcefield API but the positions part grabs the first (and only)
# conformer and passes it to Modeller. It works. Don't ask why!
if hasattr(ligand_mol.conformers[0], 'to_openmm'):
    # openff-toolkit 0.11.0
    modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm())
else:
    # openff-toolkit 0.11.0
    modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0])

# We need to temporarily create a Context in order to identify molecules for adding virtual bonds
log.info(f':microscope:  Identifying molecules')
import openmm
integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
molecules_atom_indices = context.getMolecules()
del context, integrator, system

# Solvate
log.info(':droplet:  Adding solvent...')
# we use the 'padding' option to define the periodic box. The PDB file does not contain any
# unit cell information so we just create a box that has a 9A padding around the complex.
modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=9.0*unit.angstroms)
log.info(':package:  System has %d atoms' % modeller.topology.getNumAtoms())

# Determine which atom indices we want to use
import mdtraj
mdtop = mdtraj.Topology.from_openmm(modeller.topology)
atom_selection = arguments['--selection']
log.info(f':clipboard:  Using selection: {atom_selection}')
output_indices = mdtop.select(atom_selection)
output_topology = mdtop.subset(output_indices).to_openmm()

# Create the system using the SystemGenerator
log.info(':globe_showing_americas:  Creating system...')
system = system_generator.create_system(modeller.topology, molecules=ligand_mol);

# Add virtual bonds so solute is imaged together
log.info(f':chains:  Adding virtual bonds between molecules')
custom_bond_force = openmm.CustomBondForce('0')
for molecule_index in range(len(molecules_atom_indices)-1):
    custom_bond_force.addBond(molecules_atom_indices[molecule_index][0], molecules_atom_indices[molecule_index+1][0], [])
system.addForce(custom_bond_force)

# Add barostat
from openmm import MonteCarloBarostat
system.addForce(MonteCarloBarostat(pressure, temperature));
log.info(f':game_die: Default Periodic box:')
for dim in range(3):
    log.info(f'  :small_blue_diamond: {system.getDefaultPeriodicBoxVectors()[dim]}')

# Create integrator
log.info(':building_construction:  Creating integrator...')
from openmm import LangevinMiddleIntegrator
integrator = LangevinMiddleIntegrator(temperature, collision_rate, timestep)

# Create simulation
log.info(':mage:  Creating simulation...')
from openmm.app import Simulation
simulation = Simulation(modeller.topology, system, integrator, platform=platform)
context = simulation.context
context.setPositions(modeller.positions)

# Write initial PDB, if requested.
if arguments['--initial']:
    log.info(f":page_facing_up:  Writing initial PDB to {arguments['--initial']}")
    output_positions = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
    with open(arguments['--initial'], 'w') as outfile:    
        PDBFile.writeFile(output_topology, output_positions[output_indices,:], file=outfile, keepIds=False);

# Minimize energy
log.info(':skier:  Minimizing ...')
simulation.minimizeEnergy();

# Write minimized PDB, if requested
if arguments['--minimized']:
    log.info(f":page_facing_up:  Writing minimized PDB to {arguments['--minimized']}")
    output_positions = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
    with open(arguments['--minimized'], 'w') as outfile:    
        PDBFile.writeFile(output_topology, output_positions[output_indices,:], file=outfile, keepIds=False);

# Equilibrate
log.info(':fire:  Heating ...')
simulation.context.setVelocitiesToTemperature(temperature);
simulation.step(equilibration_steps);

# Add reporter to generate DCD trajectory, if requested
if arguments['--dcdtraj']:
    log.info(f":page_facing_up:  Will write DCD trajectory to {arguments['--dcdtraj']}")
    from mdtraj.reporters import DCDReporter
    simulation.reporters.append(DCDReporter(arguments['--dcdtraj'], reporting_interval, atomSubset=output_indices))

# Add reporter to generate XTC trajectory, if requested
if arguments['--xtctraj']:
    log.info(f":page_facing_up:  Will write XTC trajectory to {arguments['--xtctraj']}")
    from mdtraj.reporters import XTCReporter
    simulation.reporters.append(XTCReporter(arguments['--xtctraj'], reporting_interval, atomSubset=output_indices))

# Add reporter to generate PDB trajectory, if requested
# NOTE: The PDBReporter does not currently support atom subsets
if arguments['--pdbtraj']:
    log.info(f":page_facing_up:  Will write PDB trajectory to {arguments['--pdbtraj']}")
    from openmm.app import PDBReporter
    simulation.reporters.append(PDBReporter(arguments['--pdbtraj'], reporting_interval))

# Run simulation
log.info(':coffee:  Starting simulation...')
from rich.progress import track
for snapshot_index in track(range(n_snapshots), ':rocket: Running production simulation...'):
    simulation.step(reporting_interval)

# Write final PDB, if requested
if arguments['--final']:
    log.info(f":page_facing_up:  Writing final PDB to {arguments['--final']}")
    output_positions = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
    with open(arguments['--final'], 'w') as outfile:    
        PDBFile.writeFile(output_topology, output_positions[output_indices,:], file=outfile, keepIds=False);

# Flush trajectories to force files to be closed
for reporter in simulation.reporters:
    del reporter

# Clean up to release GPU resources
del simulation.context
del simulation
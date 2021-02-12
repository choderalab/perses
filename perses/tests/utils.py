"""
Unit tests for NCMC switching engine.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################
import copy
from simtk import openmm, unit
from simtk.openmm import app
import sys
import numpy as np
from perses.rjmc import geometry
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
from openeye import oechem
if sys.version_info >= (3, 0):
    pass
else:
    pass
from openmmtools.constants import kB
from openmmtools import states
import contextlib
from openmmtools import utils
################################################################################
# CONSTANTS
################################################################################

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
ENERGY_THRESHOLD = 1e-1
DEFAULT_PLATFORM = utils.get_fastest_platform()

################################################################################
# UTILITIES
################################################################################]

import logging
_logger = logging.getLogger("tests-utils")
_logger.setLevel(logging.INFO)

@contextlib.contextmanager
def enter_temp_directory():
    """Create and enter a temporary directory; used as context manager."""
    import tempfile
    temp_dir = tempfile.mkdtemp()
    import os
    cwd = os.getcwd()
    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(cwd)
    import shutil
    shutil.rmtree(temp_dir)

# TODO: Move some of these utility routines to openmoltools.

class NaNException(Exception):
    pass

def quantity_is_finite(quantity):
    """
    Check that elements in quantity are all finite.

    Parameters
    ----------
    quantity : simtk.unit.Quantity
        The quantity to check

    Returns
    -------
    is_finite : bool
        If quantity is finite, returns True; otherwise False.

    """
    if np.any( np.isnan( np.array(quantity / quantity.unit) ) ):
        return False
    return True

def compare_at_lambdas(context, functions):
    """
    Compare the energy components at all lambdas = 1 and 0.
    """

    # First, set all lambdas to 0
    for parm in functions.keys():
        context.setParameter(parm, 0.0)

    energy_components_0 = compute_potential_components(context)

    for parm in functions.keys():
        context.setParameter(parm, 1.0)

    energy_components_1 = compute_potential_components(context)

    print("-----------------------")
    print("Energy components at lambda=0")

    for i in range(len(energy_components_0)):
        name, value = energy_components_0[i]
        print("%s\t%s" % (name, str(value)))

    print("-----------------------")
    print("Energy components at lambda=1")

    for i in range(len(energy_components_1)):
        name, value = energy_components_1[i]
        print("%s\t%s" % (name, str(value)))

    print("------------------------")



def get_atoms_with_undefined_stereocenters(molecule, verbose=False):
    """
    Return list of atoms with undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to check.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Returns
    -------
    atoms : list of openeye.oechem.OEAtom
        List of atoms with undefined stereochemistry.

    """
    from openeye.oechem import OEAtomStereo_Undefined, OEAtomStereo_Tetrahedral
    undefined_stereocenters = list()
    for atom in molecule.GetAtoms():
        chiral = atom.IsChiral()
        stereo = OEAtomStereo_Undefined
        if atom.HasStereoSpecified(OEAtomStereo_Tetrahedral):
            v = list()
            for nbr in atom.GetAtoms():
                v.append(nbr)
            stereo = atom.GetStereo(v, OEAtomStereo_Tetrahedral)

        if chiral and (stereo == OEAtomStereo_Undefined):
            undefined_stereocenters.append(atom)
            if verbose:
                print("Atom %d (%s) of molecule '%s' has undefined stereochemistry (chiral=%s, stereo=%s)." % (atom.GetIdx(), atom.GetName(), molecule.GetTitle(), str(chiral), str(stereo)))

    return undefined_stereocenters

def has_undefined_stereocenters(molecule, verbose=False):
    """
    Check if given molecule has undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to check.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    TODO
    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Returns
    -------
    result : bool
        True if molecule has undefined stereocenters.

    Examples
    --------
    Enumerate undefined stereocenters
    >>> smiles = "[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]"
    >>> from openeye.oechem import OEGraphMol, OESmilesToMol
    >>> molecule = OEGraphMol()
    >>> OESmilesToMol(molecule, smiles)
    >>> print has_undefined_stereocenters(smiles)
    True

    """
    # TODO move to utils
    atoms = get_atoms_with_undefined_stereocenters(molecule, verbose=verbose)
    if len(atoms) > 0:
        return True

    return False

def enumerate_undefined_stereocenters(molecule, verbose=False):
    """
    Check if given molecule has undefined stereocenters.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule whose stereocenters are to be expanded.
    verbose : bool, optional, default=False
        If True, will print verbose output about undefined stereocenters.

    Returns
    -------
    molecules : list of OEMol
        The molecules with fully defined stereocenters.

    TODO
    ----
    Add handling of chiral bonds:
    https://docs.eyesopen.com/toolkits/python/oechemtk/glossary.html#term-canonical-isomeric-smiles

    Examples
    --------
    Enumerate undefined stereocenters
    >>> smiles = "[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]"
    >>> from openeye.oechem import OEGraphMol, OESmilesToMol
    >>> molecule = OEGraphMol()
    >>> OESmilesToMol(molecule, smiles)
    >>> molecules = enumerate_undefined_stereocenters(smiles)
    >>> len(molecules)
    2

    """
    #TODO move to utils
    from openeye.oechem import OEAtomStereo_RightHanded, OEAtomStereo_LeftHanded, OEAtomStereo_Tetrahedral
    from itertools import product

    molecules = list()
    atoms = get_atoms_with_undefined_stereocenters(molecule, verbose=verbose)
    for stereocenters in product([OEAtomStereo_RightHanded, OEAtomStereo_LeftHanded], repeat=len(atoms)):
        for (index,atom) in enumerate(atoms):
            neighbors = list()
            for neighbor in atom.GetAtoms():
                neighbors.append(neighbor)
            atom.SetStereo(neighbors, OEAtomStereo_Tetrahedral, stereocenters[index])
        molecules.append(molecule.CreateCopy())

    return molecules

def test_sanitizeSMILES():
    """
    Test SMILES sanitization.
    """
    from perses.utils.smallmolecules import sanitizeSMILES
    smiles_list = ['CC', 'CCC', '[H][C@]1(NC[C@@H](CC1CO[C@H]2CC[C@@H](CC2)O)N)[H]']

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='drop')
    if len(sanitized_smiles_list) != 2:
        raise Exception("Molecules with undefined stereochemistry are not being properly dropped (size=%d)." % len(sanitized_smiles_list))

    sanitized_smiles_list = sanitizeSMILES(smiles_list, mode='expand')
    if len(sanitized_smiles_list) != 4:
        raise Exception("Molecules with undefined stereochemistry are not being properly expanded (size=%d)." % len(sanitized_smiles_list))

    # Check that all molecules can be round-tripped.
    from openeye.oechem import OEGraphMol, OESmilesToMol, OECreateIsoSmiString
    for smiles in sanitized_smiles_list:
        molecule = OEGraphMol()
        OESmilesToMol(molecule, smiles)
        isosmiles = OECreateIsoSmiString(molecule)
        if (smiles != isosmiles):
            raise Exception("Molecule '%s' was not properly round-tripped (result was '%s')" % (smiles, isosmiles))

def compute_potential(system, positions, platform=None):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    system : simtk.openmm.System
        The system object to check.
    positions : simtk.unit.Quantity of size (natoms,3) with units compatible with nanometers
        The positions to check.
    platform : simtk.openmm.Platform, optional, default=none
        If specified, this platform will be used.

    """
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    if platform is not None:
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    context.setPositions(positions)
    context.applyConstraints(integrator.getConstraintTolerance())
    potential = context.getState(getEnergy=True).getPotentialEnergy()
    del context, integrator
    if np.isnan(potential / unit.kilocalories_per_mole):
        raise NaNException("Potential energy is NaN")
    return potential

def compute_potential_components(context, beta = beta, platform = DEFAULT_PLATFORM):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    context : simtk.openmm.Context
        The context from which to extract, System, parameters, and positions.

    """
    # Make a deep copy of the system.
    import copy

    from perses.dispersed.utils import configure_platform
    platform = configure_platform(platform.getName(), fallback_platform_name='Reference', precision='double')

    system = context.getSystem()
    system = copy.deepcopy(system)
    # Get positions.
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Get Parameters
    parameters = context.getParameters()
    # Segregate forces.
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
    # Create new Context.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for (parameter, value) in parameters.items():
        context.setParameter(parameter, value)
    energy_components = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        forcename = force.__class__.__name__
        groups = 1<<index
        potential = beta * context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        energy_components.append((forcename, potential))
    del context, integrator
    return energy_components

def check_system(system):
    """
    Check OpenMM System object for pathologies, like duplicate atoms in torsions.

    Parameters
    ----------
    system : simtk.openmm.System

    """
    forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
    force = forces['PeriodicTorsionForce']
    for index in range(force.getNumTorsions()):
        [i, j, k, l, periodicity, phase, barrier] = force.getTorsionParameters(index)
        if len(set([i,j,k,l])) < 4:
            msg  = 'Torsion index %d of self._topology_proposal.new_system has duplicate atoms: %d %d %d %d\n' % (index,i,j,k,l)
            msg += 'Serialzed system to system.xml for inspection.\n'
            raise Exception(msg)
    from simtk.openmm import XmlSerializer
    serialized_system = XmlSerializer.serialize(system)
    outfile = open('system.xml', 'w')
    outfile.write(serialized_system)
    outfile.close()

def generate_endpoint_thermodynamic_states(system, topology_proposal, repartitioned_endstate=None):
    """
    Generate endpoint thermodynamic states for the system

    Parameters
    ----------
    system : openmm.System
        System object corresponding to thermodynamic state
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        TopologyProposal representing transformation
    repartitioned_endstate : int, default None
        If the htf was generated using RepartitionedHybridTopologyFactory, use this argument to specify the endstate at
        which it was generated. Otherwise, leave as None.

    Returns
    -------
    nonalchemical_zero_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda zero endpoint
    nonalchemical_one_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda one endpoint
    lambda_zero_thermodynamic_state : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda zero
    lambda_one_thermodynamic_State : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda one
    """
    # Create the thermodynamic state
    from perses.annihilation.lambda_protocol import RelativeAlchemicalState

    check_system(system)

    # Create thermodynamic states for the nonalchemical endpoints
    nonalchemical_zero_thermodynamic_state = states.ThermodynamicState(topology_proposal.old_system, temperature=temperature)
    nonalchemical_one_thermodynamic_state = states.ThermodynamicState(topology_proposal.new_system, temperature=temperature)

    # Create the base thermodynamic state with the hybrid system
    thermodynamic_state = states.ThermodynamicState(system, temperature=temperature)

    if repartitioned_endstate == 0:
        lambda_zero_thermodynamic_state = thermodynamic_state
        lambda_one_thermodynamic_state = None
    elif repartitioned_endstate == 1:
        lambda_zero_thermodynamic_state = None
        lambda_one_thermodynamic_state = thermodynamic_state
    else:
        # Create relative alchemical states
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(system)
        lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

        # Ensure their states are set appropriately
        lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
        lambda_one_alchemical_state.set_alchemical_parameters(1.0)

        # Now create the compound states with different alchemical states
        lambda_zero_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_zero_alchemical_state])
        lambda_one_thermodynamic_state = states.CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_one_alchemical_state])

    return nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state

def  generate_solvated_hybrid_test_topology(current_mol_name="naphthalene", proposed_mol_name="benzene", current_mol_smiles = None, proposed_mol_smiles = None, vacuum = False, render_atom_mapping = False,atom_expression=['Hybridization'],bond_expression=['Hybridization']):
    """
    This function will generate a topology proposal, old positions, and new positions with a geometry proposal (either vacuum or solvated) given a set of input iupacs or smiles.
    The function will (by default) read the iupac names first.  If they are set to None, then it will attempt to read a set of current and new smiles.
    An atom mapping pdf will be generated if specified.
    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule
    current_mol_smiles : str (default None)
        current mol smiles
    proposed_mol_smiles : str (default None)
        proposed mol smiles
    vacuum: bool (default False)
        whether to render a vacuum or solvated topology_proposal
    render_atom_mapping : bool (default False)
        whether to render the atom map of the current_mol_name and proposed_mol_name
    atom_expression : list(str), optional
        list of atom mapping criteria
    bond_expression : list(str), optional
        list of bond mapping criteria

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from openeye import oechem
    from openmoltools.openeye import iupac_to_oemol, smiles_to_oemol
    from openmoltools import forcefield_generators
    import perses.utils.openeye as openeye
    from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
    import simtk.unit as unit
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    from perses.utils.openeye import generate_expression
    from openmmforcefields.generators import SystemGenerator
    from openff.toolkit.topology import Molecule

    atom_expr = generate_expression(atom_expression)
    bond_expr = generate_expression(bond_expression)

    if current_mol_name != None and proposed_mol_name != None:
        try:
            old_oemol, new_oemol = iupac_to_oemol(current_mol_name), iupac_to_oemol(proposed_mol_name)
            old_smiles = oechem.OECreateSmiString(old_oemol,oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
            new_smiles = oechem.OECreateSmiString(new_oemol,oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
        except:
            raise Exception(f"either {current_mol_name} or {proposed_mol_name} is not compatible with 'iupac_to_oemol' function!")
    elif current_mol_smiles != None and proposed_mol_smiles != None:
        try:
            old_oemol, new_oemol = smiles_to_oemol(current_mol_smiles), smiles_to_oemol(proposed_mol_smiles)
            old_smiles = oechem.OECreateSmiString(old_oemol,oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
            new_smiles = oechem.OECreateSmiString(new_oemol,oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
        except:
            raise Exception(f"the variables are not compatible")
    else:
        raise Exception(f"either current_mol_name and proposed_mol_name must be specified as iupacs OR current_mol_smiles and proposed_mol_smiles must be specified as smiles strings.")

    old_oemol, old_system, old_positions, old_topology = openeye.createSystemFromSMILES(old_smiles, title = "MOL")

    # Correct the old positions
    old_positions = openeye.extractPositionsFromOEMol(old_oemol)
    old_positions = old_positions.in_units_of(unit.nanometers)


    new_oemol, new_system, new_positions, new_topology = openeye.createSystemFromSMILES(new_smiles, title = "NEW")


    ffxml = forcefield_generators.generateForceFieldFromMolecules([old_oemol, new_oemol])

    old_oemol.SetTitle('MOL'); new_oemol.SetTitle('MOL')

    old_topology = forcefield_generators.generateTopologyFromOEMol(old_oemol)
    new_topology = forcefield_generators.generateTopologyFromOEMol(new_oemol)

    if not vacuum:
        nonbonded_method = app.PME
        barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)
    else:
        nonbonded_method = app.NoCutoff
        barostat = None

    forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus}
    periodic_forcefield_kwargs = {'nonbondedMethod': nonbonded_method}
    small_molecule_forcefield = 'gaff-2.11'

    system_generator = SystemGenerator(forcefields = forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs, periodic_forcefield_kwargs=periodic_forcefield_kwargs,
                                         small_molecule_forcefield = small_molecule_forcefield, molecules=[Molecule.from_openeye(mol) for mol in [old_oemol, new_oemol]], cache=None)

    proposal_engine = SmallMoleculeSetProposalEngine([old_oemol, new_oemol], system_generator, residue_name = 'MOL',atom_expr=atom_expr, bond_expr=bond_expr,allow_ring_breaking=True)
    geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)

    if not vacuum:
        # Now to solvate
        modeller = app.Modeller(old_topology, old_positions)
        hs = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H'] and atom.residue.name not in ['MOL','OLD','NEW']]
        modeller.delete(hs)
        modeller.addHydrogens(forcefield=system_generator.forcefield)
        modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=9.0*unit.angstroms)
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()
        solvated_positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
        solvated_system = system_generator.create_system(solvated_topology)

        # Now to create proposal
        top_proposal = proposal_engine.propose(current_system = solvated_system, current_topology = solvated_topology, current_mol_id=0, proposed_mol_id=1)
        new_positions, _ = geometry_engine.propose(top_proposal, solvated_positions, beta)

        if render_atom_mapping:
            from perses.utils.smallmolecules import render_atom_mapping
            print(f"new_to_old: {proposal_engine.non_offset_new_to_old_atom_map}")
            render_atom_mapping(f"{old_smiles}to{new_smiles}.png", old_oemol, new_oemol, proposal_engine.non_offset_new_to_old_atom_map)

        return top_proposal, solvated_positions, new_positions

    else:
        vacuum_system = system_generator.create_system(old_topology)
        top_proposal = proposal_engine.propose(current_system=vacuum_system, current_topology=old_topology, current_mol_id=0, proposed_mol_id=1)
        new_positions, _ = geometry_engine.propose(top_proposal, old_positions, beta)
        if render_atom_mapping:
            from perses.utils.smallmolecules import render_atom_mapping
            print(f"new_to_old: {top_proposal._new_to_old_atom_map}")
            render_atom_mapping(f"{old_smiles}to{new_smiles}.png", old_oemol, new_oemol, top_proposal._new_to_old_atom_map)
        return top_proposal, old_positions, new_positions

def generate_vacuum_hostguest_proposal(current_mol_name="B2", proposed_mol_name="MOL"):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

    Parameters
    ----------
    current_mol_name : str, optional
        name of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators
    from openmmtools import testsystems
    from openmmforcefields.generators import SystemGenerator

    from perses.utils.openeye import smiles_to_oemol
    from perses.utils.data import get_data_filename

    host_guest = testsystems.HostGuestVacuum()
    unsolv_old_system, old_positions, top_old = host_guest.system, host_guest.positions, host_guest.topology

    ligand_topology = [res for res in top_old.residues()]
    current_mol = forcefield_generators.generateOEMolFromTopologyResidue(ligand_topology[1]) # guest is second residue in topology
    proposed_mol = smiles_to_oemol('C1CC2(CCC1(CC2)C)C')

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False},nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [current_mol, proposed_mol], system_generator, residue_name=current_mol_name,atom_expr=atom_expr,bond_expr=bond_expr)

    # Generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old, current_mol_id=0, proposed_mol_id=1)

    # Generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, old_positions, beta)

    return topology_proposal, old_positions, new_positions

def validate_rjmc_work_variance(top_prop, positions, geometry_method = 0, num_iterations = 10, md_steps = 250, compute_timeseries = False, md_system = None, prespecified_conformers = None):
    """
    Parameters
    ----------
    top_prop : perses.rjmc.topology_proposal.TopologyProposal object
        topology_proposal
    md_system : openmm.System object, default None
        system from which md is conducted; the default is the top_prop._old_system
    geometry_method : int
        which geometry proposal method to use
            0: neglect_angles = True (this is supposed to be the zero-variance method)
            1: neglect_angles = False (this will accumulate variance)
            2: use_sterics = True (this is experimental)
    num_iterations: int
        number of times to run md_steps integrator
    md_steps: int
        number of md_steps to run in each num_iteration
    compute_timeseries = bool (default False)
        whether to use pymbar detectEquilibration and subsampleCorrelated data from the MD run (the potential energy is the data)
    prespecified_conformers = None or unit.Quantity(np.array([num_iterations, system.getNumParticles(), 3]), unit = unit.nanometers)
        whether to input a unit.Quantity of conformers and bypass the conformer_generation/pymbar stage; None will default conduct this phase

    Returns
    -------
    conformers : unit.Quantity(np.array([num_iterations, system.getNumParticles(), 3]), unit = unit.nanometers)
        decorrelated positions of the md run
    rj_works : list
        work from each conformer proposal
    """
    from openmmtools import integrators
    import simtk.unit as unit
    import simtk.openmm as openmm
    from openmmtools.constants import kB
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    import tqdm

    temperature = 300.0 * unit.kelvin # unit-bearing temperature
    kT = kB * temperature # unit-bearing thermal energy
    beta = 1.0/kT # unit-bearing inverse thermal energy

    # First, we must extract the top_prop relevant quantities
    topology = top_prop._old_topology
    if md_system == None:
        system = top_prop._old_system
    else:
        system = md_system

    if prespecified_conformers == None:

        # Now we can specify conformations from MD
        integrator = integrators.LangevinIntegrator(collision_rate = 1.0/unit.picosecond, timestep = 4.0*unit.femtosecond, temperature = temperature)
        context = openmm.Context(system, integrator)
        context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(context)
        minimized_positions = context.getState(getPositions = True).getPositions(asNumpy = True)
        print(f"completed initial minimization")
        context.setPositions(minimized_positions)

        zeros = np.zeros([num_iterations, int(system.getNumParticles()), 3])
        conformers = unit.Quantity(zeros, unit=unit.nanometers)
        rps = np.zeros((num_iterations))

        print(f"conducting md sampling")
        for iteration in tqdm.trange(num_iterations):
            integrator.step(md_steps)
            state = context.getState(getPositions = True, getEnergy = True)
            new_positions = state.getPositions(asNumpy = True)
            conformers[iteration,:,:] = new_positions

            rp = state.getPotentialEnergy()*beta
            rps[iteration] = rp

        del context, integrator

        if compute_timeseries:
            print(f"computing production and data correlation")
            from pymbar import timeseries
            t0, g, Neff = timeseries.detectEquilibration(rps)
            series = timeseries.subsampleCorrelatedData(np.arange(t0, num_iterations), g = g)
            print(f"production starts at index {t0} of {num_iterations}")
            print(f"the number of effective samples is {Neff}")
            indices = t0 + series
            print(f"the filtered indices are {indices}")

        else:
            indices = range(num_iterations)
    else:
        conformers = prespecified_conformers
        indices = range(len(conformers))

    # Now we can define a geometry_engine
    if geometry_method == 0:
        geometry_engine = FFAllAngleGeometryEngine( metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = True)
    elif geometry_method == 1:
        geometry_engine = FFAllAngleGeometryEngine( metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)
    elif geometry_method == 2:
        geometry_engine = FFAllAngleGeometryEngine( metadata=None, use_sterics=True, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0, neglect_angles = False)
    else:
        raise Exception(f"there is no geometry method for {geometry_method}")

    rj_works = []
    print(f"conducting geometry proposals...")
    for indx in tqdm.trange(len(indices)):
        index = indices[indx]
        print(f"index {indx}")
        new_positions, logp_forward = geometry_engine.propose(top_prop, conformers[index], beta)
        logp_backward = geometry_engine.logp_reverse(top_prop, new_positions, conformers[index], beta)
        print(f"\tlogp_forward, logp_backward: {logp_forward}, {logp_backward}")
        added_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential
        subtracted_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential
        print(f"\tadded_energy, subtracted_energy: {added_energy}, {subtracted_energy}")
        work = logp_forward - logp_backward + added_energy - subtracted_energy
        rj_works.append(work)
        print(f"\ttotal work: {work}")

    return conformers, rj_works

def validate_endstate_energies(topology_proposal,
                               htf,
                               added_energy,
                               subtracted_energy,
                               beta=1.0/kT,
                               ENERGY_THRESHOLD=1e-6,
                               platform=DEFAULT_PLATFORM,
                               trajectory_directory=None,
                               repartitioned_endstate=None):
    """
    Function to validate that the difference between the nonalchemical versus alchemical state at lambda = 0,1 is
    equal to the difference in valence energy (forward and reverse).

    Parameters
    ----------
    topology_proposal : perses.topology_proposal.TopologyProposal object
        top_proposal for relevant transformation
    htf : perses.new_relative.HybridTopologyFactory object
        hybrid top factory for setting alchemical hybrid states
    added_energy : float
        reduced added valence energy
    subtracted_energy : float
        reduced subtracted valence energy
    beta : float, default 1.0/kT
        unit-bearing inverse thermal energy
    ENERGY_THRESHOLD : float, default 1e-6
        threshold for ratio in energy difference at a particular endstate
    platform : str, default utils.get_fastest_platform()
        platform to conduct validation on (e.g. 'CUDA', 'Reference', 'OpenCL')
    trajectory_directory : str, default None
        path to save the save the serialized state to. If None, the state will not be saved
    repartitioned_endstate : int, default None
        if the htf was generated using RepartitionedHybridTopologyFactory, use this argument to specify the endstate at
        which it was generated. Otherwise, leave as None.

    Returns
    -------
    zero_state_energy_difference : float
        reduced potential difference of the nonalchemical and alchemical lambda = 0 state (corrected for valence energy).
    one_state_energy_difference : float
        reduced potential difference of the nonalchemical and alchemical lambda = 1 state (corrected for valence energy).
    """
    import copy
    from perses.dispersed.utils import configure_platform
    from perses.utils import data
    platform = configure_platform(platform.getName(), fallback_platform_name='Reference', precision='double')

    # Create copies of old/new systems and set the dispersion correction
    top_proposal = copy.deepcopy(topology_proposal)
    forces = { top_proposal._old_system.getForce(index).__class__.__name__ : top_proposal._old_system.getForce(index) for index in range(top_proposal._old_system.getNumForces()) }
    forces['NonbondedForce'].setUseDispersionCorrection(False)
    forces = { top_proposal._new_system.getForce(index).__class__.__name__ : top_proposal._new_system.getForce(index) for index in range(top_proposal._new_system.getNumForces()) }
    forces['NonbondedForce'].setUseDispersionCorrection(False)

    # Create copy of hybrid system, define old and new positions, and turn off dispersion correction
    hybrid_system = copy.deepcopy(htf.hybrid_system)
    hybrid_system_n_forces = hybrid_system.getNumForces()
    for force_index in range(hybrid_system_n_forces):
        forcename = hybrid_system.getForce(force_index).__class__.__name__
        if forcename == 'NonbondedForce':
            hybrid_system.getForce(force_index).setUseDispersionCorrection(False)

    old_positions, new_positions = htf._old_positions, htf._new_positions

    # Generate endpoint thermostates
    nonalch_zero, nonalch_one, alch_zero, alch_one = generate_endpoint_thermodynamic_states(hybrid_system, top_proposal, repartitioned_endstate)

    # Compute reduced energies for the nonalchemical systems...
    attrib_list = [('real-old', nonalch_zero, old_positions, top_proposal._old_system.getDefaultPeriodicBoxVectors()),
                    ('hybrid-old', alch_zero, htf._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    ('hybrid-new', alch_one, htf._hybrid_positions, hybrid_system.getDefaultPeriodicBoxVectors()),
                    ('real-new', nonalch_one, new_positions, top_proposal._new_system.getDefaultPeriodicBoxVectors())]

    rp_list = []
    for (state_name, state, pos, box_vectors) in attrib_list:
        if not state:
            rp_list.append(None)
        else:
            integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
            context = state.create_context(integrator, platform)
            samplerstate = states.SamplerState(positions = pos, box_vectors = box_vectors)
            samplerstate.apply_to_context(context)
            rp = state.reduced_potential(context)
            rp_list.append(rp)
            energy_comps = compute_potential_components(context)
            for name, force in energy_comps:
               print("\t\t\t{}: {}".format(name, force))
            _logger.debug(f'added forces:{sum([energy for name, energy in energy_comps])}')
            _logger.debug(f'rp: {rp}')
            if trajectory_directory is not None:
                _logger.info(f'Saving {state_name} state xml to {trajectory_directory}/{state_name}-state.gz')
                state = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True, getParameters=True)
                data.serialize(state,f'{trajectory_directory}-{state_name}-state.gz')
            del context, integrator

    nonalch_zero_rp, alch_zero_rp, alch_one_rp, nonalch_one_rp = rp_list[0], rp_list[1], rp_list[2], rp_list[3]

    if repartitioned_endstate == 0:
        zero_error = nonalch_zero_rp - alch_zero_rp + added_energy
        one_error = None
        ratio = abs((zero_error) / (nonalch_zero_rp + alch_zero_rp + added_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ZERO state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-zero: {nonalch_zero_rp} \n alc-zero: {alch_zero_rp} \nadded-valence: {added_energy}"
    elif repartitioned_endstate == 1:
        zero_error = None
        one_error = nonalch_one_rp - alch_one_rp + subtracted_energy
        ratio = abs((one_error) / (nonalch_one_rp + alch_one_rp + subtracted_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ONE state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-one: {nonalch_one_rp} \n alc-one: {alch_one_rp} \nsubtracted-valence: {subtracted_energy}"
    else:
        zero_error = nonalch_zero_rp - alch_zero_rp + added_energy
        one_error = nonalch_one_rp - alch_one_rp + subtracted_energy
        ratio = abs((zero_error) / (nonalch_zero_rp + alch_zero_rp + added_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ZERO state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-zero: {nonalch_zero_rp} \n alc-zero: {alch_zero_rp} \nadded-valence: {added_energy}"
        ratio = abs((one_error) / (nonalch_one_rp + alch_one_rp + subtracted_energy))
        assert ratio < ENERGY_THRESHOLD, f"The ratio in energy difference for the ONE state is {ratio}.\n This is greater than the threshold of {ENERGY_THRESHOLD}.\n real-one: {nonalch_one_rp} \n alc-one: {alch_one_rp} \nsubtracted-valence: {subtracted_energy}"

    return zero_error, one_error


def track_torsions(hybrid_factory):
    """
    This is a useful function for finding bugs in the `relative.py` self.handle_periodic_torsion_force() function, which creates hybrid torsion forces for the hybrid system.
    It is observed that accounting for all of the annealing and unique old/new torsion terms is cumbersome and often fails, resulting in validate_endstate_energies throwing an ENERGY_THRESHOLD error of >1e-6.
    Often, the mismatch in energies is in a valence term (specifically a torsion).  This function will throw an assert error if there is a unique old/new torsion that is not in the hybrid_system's PeriodicTorsionForce object
    or if there is a core torsion force that is not being annealed properly.  Since the energy mismatch is valence in nature, the validate_endstate_energies assert error should be reproducible in the vacuum phase.  Hence, it is advisable
    to use the vacuum hybrid factory as the argument to this function.  The first successful implementation of this code was in the MCL1 ligands 0 and 22.
        1. for all mapped atoms, pulls the torsions of the old and new systems and asserts that there are two interpolating
           terms from one to the other
        2. for all unique old/new atoms, asserts that the corresponding terms match
    """
    old_system, new_system = hybrid_factory._old_system, hybrid_factory._new_system
    hybrid_system = hybrid_factory._hybrid_system
    hybrid_to_old, hybrid_to_new = hybrid_factory._hybrid_to_old_map, hybrid_factory._hybrid_to_new_map
    print(f"hybrid_to_old: {hybrid_to_old}")
    print(f"hybrid_to_new: {hybrid_to_new}")
    old_to_hybrid, new_to_hybrid = hybrid_factory._old_to_hybrid_map, hybrid_factory._new_to_hybrid_map
    unique_old_atoms = hybrid_factory._atom_classes['unique_old_atoms']
    unique_new_atoms = hybrid_factory._atom_classes['unique_new_atoms']
    core_atoms = hybrid_factory._atom_classes['core_atoms']

    # First, grab all of the old/new torsions
    num_old_torsions, num_new_torsions = old_system.getForce(2).getNumTorsions(), new_system.getForce(2).getNumTorsions()
    print(f"num old torsions, new torsions: {num_old_torsions}, {num_new_torsions}")

    old_torsions = [old_system.getForce(2).getTorsionParameters(i) for i in range(num_old_torsions)]
    new_torsions = [new_system.getForce(2).getTorsionParameters(i) for i in range(num_new_torsions)]

    # Reformat the last two entries
    old_torsions = [[old_to_hybrid[q] for q in i[:4]] + [float(i[4])] + [i[5]/unit.radian] + [i[6]/(unit.kilojoule/unit.mole)] for i in old_torsions]
    new_torsions = [[new_to_hybrid[q] for q in i[:4]] + [float(i[4])] + [i[5]/unit.radian] + [i[6]/(unit.kilojoule/unit.mole)] for i in new_torsions]

    print(f"old torsions:")
    for i in old_torsions:
        print(i)

    print(f"new torsions:")
    for i in new_torsions:
        print(i)

    # Now grab the annealing hybrid torsions
    num_annealed_torsions = hybrid_system.getForce(4).getNumTorsions()
    annealed_torsions = [hybrid_system.getForce(4).getTorsionParameters(i) for i in range(num_annealed_torsions)]
    print(f"annealed torsions:")
    for i in annealed_torsions:
        print(i)

    # Now grab the old/new hybrid torsions
    num_old_new_atom_torsions = hybrid_system.getForce(5).getNumTorsions()
    hybrid_old_new_torsions = [hybrid_system.getForce(5).getTorsionParameters(i) for i in range(num_old_new_atom_torsions)]
    hybrid_old_new_torsions = [i[:5] + [i[5]/unit.radian] + [i[6]/(unit.kilojoule/unit.mole)] for i in hybrid_old_new_torsions] #reformatted

    hybrid_old_torsions = [i for i in hybrid_old_new_torsions if set(unique_old_atoms).intersection(set(i[:4])) != set()]
    print(f"hybrid old torsions:")
    for i in hybrid_old_torsions:
        print(i)
    hybrid_new_torsions = [i for i in hybrid_old_new_torsions if set(unique_new_atoms).intersection(set(i[:4])) != set()]
    print(f"hybrid_new_torsions:")
    for i in hybrid_new_torsions:
        print(i)


    #assert len(hybrid_old_torsions) + len(hybrid_new_torsions) == len(hybrid_old_new_torsions), f"there are some hybrid_old_new torsions missing: "
    for i in hybrid_old_new_torsions:
        assert (i in hybrid_old_torsions) or (i in hybrid_new_torsions), f"hybrid old/new torsion {i} is in neither hybrid_old/hybrid_new torsion list"
        if i in hybrid_old_torsions:
            assert i not in hybrid_new_torsions, f"{i} in both old ({unique_old_atoms}) and new ({unique_new_atoms}) hybrid torsion lists"
        elif i in hybrid_new_torsions:
            assert i not in hybrid_old_torsions, f"{i} in both old ({unique_old_atoms}) and new ({unique_new_atoms}) hybrid torsion lists"

    # Now we can check the hybrid old and new torsions
    old_counter, new_counter = 0, 0
    for hybrid_torsion in hybrid_old_torsions:
        if hybrid_torsion in old_torsions:
            old_counter += 1
        elif hybrid_torsion[:4][::-1] + hybrid_torsion[4:] in old_torsions:
            old_counter +=1
        else:
            print(f"found a hybrid old torsion not in old_torsions: {hybrid_torsion}")

    unique_old_torsions = len([i for i in hybrid_old_new_torsions if set(i[:4]).intersection(unique_old_atoms) != set()])
    unique_annealing_old_torsions = len([i for i in annealed_torsions if set(i[:4]).intersection(unique_old_atoms) != set()])
    assert unique_annealing_old_torsions == 0, f"there are unique old atoms in the annealing torsions: {unique_annealing_old_torsions}"
    assert old_counter == unique_old_torsions, f"the old counter ({old_counter}) != unique old torsions ({unique_old_torsions})"



    for hybrid_torsion in hybrid_new_torsions:
        if hybrid_torsion in new_torsions:
            new_counter += 1
        elif hybrid_torsion[:4][::-1] + hybrid_torsion[4:] in new_torsions:
            new_counter += 1
        else:
            print(f"found a hybrid new torsion not in new_torsions: {hybrid_torsion}")

    unique_new_torsions = len([i for i in hybrid_old_new_torsions if set(i[:4]).intersection(unique_new_atoms) != set()])
    unique_annealing_new_torsions = len([i for i in annealed_torsions if set(i[:4]).intersection(unique_new_atoms) != set()])
    assert unique_annealing_new_torsions == 0, f"there are unique new atoms in the annealing torsions: {unique_annealing_new_torsions}"
    assert new_counter == unique_new_torsions, f"the new counter ({new_counter}) != unique new torsions ({unique_new_torsions})"



    # Now we can test the annealed torsions to determine whether all of the core torsions forces are properly annealed

    # First to assert that the set of all annealed torsion atoms is a subset of all core atoms
    all_annealed_torsion_atoms = []
    for torsion_set in [i[:4] for i in annealed_torsions]:
        for torsion_index in torsion_set:
            all_annealed_torsion_atoms.append(torsion_index)

    assert set(all_annealed_torsion_atoms).issubset(core_atoms), f"the set of all annealed atom indices is not a subset of core atoms"

    # Now to take all of the annealed torsions and assert
    print(f"checking annealed torsions...")
    for annealed_torsion in annealed_torsions:
        print(f"checking annealed torsion: {annealed_torsion}")
        if annealed_torsion[4][2] == 0.0 and annealed_torsion[4][5] != 0.0: #turning on torsion should be unique new
            assert (annealed_torsion[:4] + list(annealed_torsion[-1][3:]) in new_torsions) or (annealed_torsion[:4][::-1] + list(annealed_torsion[-1][3:]) in new_torsions), f"{annealed_torsion}"

        elif annealed_torsion[4][2] != 0.0 and annealed_torsion[4][5] == 0.0: #turning off torsion should be unique old
            assert (annealed_torsion[:4] + list(annealed_torsion[-1][:3]) in old_torsions) or (annealed_torsion[:4][::-1] + list(annealed_torsion[-1][:3]) in old_torsions), f"{annealed_torsion}"

        else:
            print(f"this is a strange annealed torsion: {annealed_torsion}")

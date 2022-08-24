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
import numpy as np
import os
import shutil
import tempfile
from perses.rjmc import geometry
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
from perses.dispersed import utils
from openmmtools.constants import kB
from openmmtools import states, integrators
import contextlib

################################################################################
# CONSTANTS
################################################################################

temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
ENERGY_THRESHOLD = 1e-1

################################################################################
# UTILITIES
################################################################################]

import logging
_logger = logging.getLogger("tests-utils")
_logger.setLevel(logging.INFO)


@contextlib.contextmanager
def enter_temp_directory():
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(cwd)
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
    from perses.dispersed.utils import compute_potential_components

    # First, set all lambdas to 0
    for parm in functions.keys():
        context.setParameter(parm, 0.0)

    energy_components_0 = compute_potential_components(context)

    for parm in functions.keys():
        context.setParameter(parm, 1.0)

    energy_components_1 = compute_potential_components(context)

    print("-----------------------")
    print("Energy components at lambda=0")

    for name, value in energy_components_0.items():
        print(f"{name}\t{value}")

    print("-----------------------")
    print("Energy components at lambda=1")

    for name, value in energy_components_1.items():
        print(f"{name}\t{value}")

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


def generate_solvated_hybrid_test_topology(current_mol_name="naphthalene", proposed_mol_name="benzene", current_mol_smiles = None, proposed_mol_smiles = None, vacuum = False, render_atom_mapping = False,atom_expression=['Hybridization'],bond_expression=['Hybridization']):
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
    forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 3 * unit.amus}
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
        modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=16.0*unit.angstroms)
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
    # TODO: Modify this to use openff.toolkit.topology.Molecule instead of openeye

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

    from openeye import oechem
    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False},nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff})
    geometry_engine = geometry.FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine([current_mol, proposed_mol], system_generator, residue_name=current_mol_name)

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


def validate_endstate_energies_md(input_htf, T_max=300 * unit.kelvin, endstate=0, n_steps=125000, save_freq=250):
    """
    Check that the hybrid system's energy (without unique old/new valence energy) matches the original system's
    energy for snapshots extracted (every 1 ps) from a MD simulation.

    E.g. at endstate=0, the hybrid system's energy (with unique new valence terms zeroed) should match the old system's energy.

    Parameters
    ----------
    input_htf : RESTCapableHybridTopologyFactory
        the RESTCapableHybridTopologyFactory to test
    T_max : unit.kelvin default=300 * unit.kelvin
        T_max at which to test the factory. This should not actually affect the energy differences, since T_max should equal T_min at the endstates
    endstate : int, default=0
        the endstate to test (0 or 1)
    n_steps : int, default=125000
        number of MD steps to run. (125000 steps == 1 ns)
    save_freq : int, default 250
        frequency at which to save positions
        by default, saves one snapshot every 250 steps
    """
    import tqdm
    from perses.annihilation.lambda_protocol import RESTCapableRelativeAlchemicalState, RESTCapableLambdaProtocol

    # Check that endstate is 0 or 1
    assert endstate in [0, 1], "Endstate must be 0 or 1"

    # Make deep copy to ensure original object remains unaltered
    htf = copy.deepcopy(input_htf)

    # Set temperature
    T_min = temperature

    # Get hybrid system, positions, box_vectors
    hybrid_system = htf.hybrid_system
    hybrid_positions = htf.hybrid_positions
    box_vectors = hybrid_system.getDefaultPeriodicBoxVectors()

    # For this test, we need to turn the LRC on for the CustomNonbondedForce scaled steric interactions,
    # since there is no way to turn the LRC on for the non-scaled interactions only in the real systems
    force_dict = {force.getName(): index for index, force in enumerate(hybrid_system.getForces())}
    custom_force = hybrid_system.getForce(force_dict['CustomNonbondedForce_sterics'])
    custom_force.setUseLongRangeCorrection(True)

    # Create compound thermodynamic state
    lambda_protocol = RESTCapableLambdaProtocol()
    alchemical_state = RESTCapableRelativeAlchemicalState.from_system(hybrid_system)
    thermostate = states.ThermodynamicState(hybrid_system, temperature=T_min)
    compound_thermodynamic_state = states.CompoundThermodynamicState(thermostate, composable_states=[alchemical_state])

    # Set alchemical parameters
    beta_0 = 1 / (kB * T_min)
    beta_m = 1 / (kB * T_max)
    global_lambda = endstate
    compound_thermodynamic_state.set_alchemical_parameters(global_lambda, beta_0, beta_m, lambda_protocol=lambda_protocol)

    # Create context
    integrator = integrators.LangevinIntegrator(temperature=T_min, collision_rate=1 / unit.picoseconds, timestep=4 * unit.femtoseconds)
    context = compound_thermodynamic_state.create_context(integrator)
    context.setPositions(hybrid_positions)
    context.setPeriodicBoxVectors(*box_vectors)
    context.setVelocitiesToTemperature(T_min)

    # Minimize
    openmm.LocalEnergyMinimizer.minimize(context)

    # Run MD
    hybrid = list()
    for _ in tqdm.tqdm(range(int(n_steps / 250))):
        integrator.step(250)
        pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
        hybrid.append(pos)

    # Make context for og system
    og_system = htf._topology_proposal.old_system if endstate == 0 else htf._topology_proposal.new_system
    thermodynamic_state = states.ThermodynamicState(og_system, temperature=T_min)
    integrator_og = integrators.LangevinIntegrator(temperature=T_min,
                                                   collision_rate=1 / unit.picoseconds,
                                                   timestep=4 * unit.femtoseconds)
    context_og = thermodynamic_state.create_context(integrator_og)

    # Zero the unique old/new valence in the hybrid system
    force_dict = {force.getName(): force for force in hybrid_system.getForces()}
    bond_force = force_dict['CustomBondForce']
    angle_force = force_dict['CustomAngleForce']
    torsion_force = force_dict['CustomTorsionForce']
    hybrid_to_bond_indices = htf._hybrid_to_new_bond_indices if endstate == 0 else htf._hybrid_to_old_bond_indices
    hybrid_to_angle_indices = htf._hybrid_to_new_angle_indices if endstate == 0 else htf._hybrid_to_old_angle_indices
    hybrid_to_torsion_indices = htf._hybrid_to_new_torsion_indices if endstate == 0 else htf._hybrid_to_old_torsion_indices
    for hybrid_idx, idx in hybrid_to_bond_indices.items():
        p1, p2, hybrid_params = bond_force.getBondParameters(hybrid_idx)
        hybrid_params = list(hybrid_params)
        hybrid_params[-2] *= 0  # zero K_old
        hybrid_params[-1] *= 0  # zero K_new
        bond_force.setBondParameters(hybrid_idx, p1, p2, hybrid_params)
    for hybrid_idx, idx in hybrid_to_angle_indices.items():
        p1, p2, p3, hybrid_params = angle_force.getAngleParameters(hybrid_idx)
        hybrid_params = list(hybrid_params)
        hybrid_params[-1] *= 0
        hybrid_params[-2] *= 0
        angle_force.setAngleParameters(hybrid_idx, p1, p2, p3, hybrid_params)
    for hybrid_idx, idx in hybrid_to_torsion_indices.items():
        p1, p2, p3, p4, hybrid_params = torsion_force.getTorsionParameters(hybrid_idx)
        hybrid_params = list(hybrid_params)
        hybrid_params[-1] *= 0
        hybrid_params[-2] *= 0
        torsion_force.setTorsionParameters(hybrid_idx, p1, p2, p3, p4, hybrid_params)

    # Make context for hybrid system
    lambda_protocol = RESTCapableLambdaProtocol()
    alchemical_state = RESTCapableRelativeAlchemicalState.from_system(hybrid_system)
    thermostate = states.ThermodynamicState(hybrid_system, temperature=T_min)
    compound_thermodynamic_state = states.CompoundThermodynamicState(thermostate, composable_states=[alchemical_state])

    beta_0 = 1 / (kB * T_min)
    beta_m = 1 / (kB * T_max)
    global_lambda = endstate
    compound_thermodynamic_state.set_alchemical_parameters(global_lambda, beta_0, beta_m,
                                                           lambda_protocol=lambda_protocol)

    integrator_hybrid = integrators.LangevinIntegrator(temperature=T_min,
                                                       collision_rate=1 / unit.picoseconds,
                                                       timestep=4 * unit.femtoseconds)
    context_hybrid = compound_thermodynamic_state.create_context(integrator_hybrid)

    # Get energies for each conformation
    # TODO: Instead of checking with np.isclose(), check whether the ratio of differences is less than a specified energy threshold (like in validate_endstate_energies())
    energies_og = list()
    energies_hybrid = list()
    for i, pos in enumerate(tqdm.tqdm(hybrid)):
        og_positions = htf.old_positions(pos) if endstate == 0 else htf.new_positions(pos)
        context_og.setPositions(og_positions)
        energy_og = context_og.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
        energies_og.append(energy_og)

        context_hybrid.setPositions(pos)
        energy_hybrid = context_hybrid.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
        energies_hybrid.append(energy_hybrid)

        assert np.isclose([energy_og], [energy_hybrid]), f"Energies are not equal at frame {i}"
        print(f"Success! Energies are equal at frame {i}!")

def validate_unsampled_endstates_point(htf, hybrid_system, endstate=0, minimize=False):
    """
    Test create_endstates() to make sure that a given unsampled endstate's hybrid system has the same energies
    for each force as the original hybrid system.

    Parameters
    ----------
    htf : HybridTopologyFactory or RESTCapableHybridTopologyFactory
        hybrid factory from which the unsampled endstate's hybrid system was created
    hybrid_system : openmm.System
        the hybrid system to test
    endstate : int, default 0
        the lambda value corresponding to the hybrid system's endstate, the allowed values are: 0, 1
    minimize : bool, default False
        whether to minimize the htf's hybrid positions before checking the energies

    """
    import copy
    from openmmtools.states import ThermodynamicState, SamplerState
    from perses.dispersed import feptasks
    from perses.dispersed.utils import compute_potential_components

    assert endstate in [0, 1], f"endstate must be 0 or 1, you supplied: {endstate}"

    # Make a copy of the htf
    htf = copy.deepcopy(htf)
    hybrid_positions = htf.hybrid_positions

    # Get original hybrid system
    hybrid_system_og = htf.hybrid_system

    # Get energy components of unsampled endstate hybrid system
    thermostate_hybrid = ThermodynamicState(system=hybrid_system, temperature=temperature)
    integrator_hybrid = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context_hybrid = thermostate_hybrid.create_context(integrator_hybrid)
    if minimize:
        sampler_state = SamplerState(hybrid_positions)
        feptasks.minimize(thermostate_hybrid, sampler_state)
        hybrid_positions = sampler_state.positions
    context_hybrid.setPositions(hybrid_positions)
    components_hybrid = compute_potential_components(context_hybrid, beta=beta)

    # Get energy components of original hybrid system
    thermostate_other = ThermodynamicState(system=hybrid_system_og, temperature=temperature)
    integrator_other = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context_other = thermostate_other.create_context(integrator_other)
    htf_class = htf.__class__.__name__
    if htf_class == 'HybridTopologyFactory':
        for k, v in context_other.getParameters().items():
            if 'lambda' in k:
                context_other.setParameter(k, endstate)
    elif htf_class == 'RESTCapableHybridTopologyFactory':
        for k, v in context_other.getParameters().items():
            if 'alchemical' in k:
                if 'old' in k:
                    context_other.setParameter(k, 1 - endstate)
                elif 'new' in k or 'reciprocal' in k:
                    context_other.setParameter(k, endstate)
    else:
        raise Exception(f"{htf_class} is not supported. Supported factories: HybridTopologyFactory, RESTCapableHybridTopologyFactory")
    context_other.setPositions(hybrid_positions)
    components_other = compute_potential_components(context_other, beta=beta)

    # Check that each of the valence force energies are concordant
    # TODO: Instead of checking with np.isclose(), check whether the ratio of differences is less than a specified energy threshold (like in validate_endstate_energies())
    bonded_keys = [hybrid_system_og.getForce(i).getName() for i in range(hybrid_system_og.getNumForces())
                   if 'Nonbonded' not in hybrid_system_og.getForce(i).getName() and 'exceptions' not in hybrid_system_og.getForce(i).getName()]
    for key in bonded_keys:
        other_value = components_other[key]
        hybrid_value = components_hybrid[key]
        print(f"{key} -- og: {other_value}, hybrid: {hybrid_value}")
        assert np.isclose(other_value, hybrid_value)

    # Check that the nonbonded (rest of the components) force energies are concordant
    nonbonded_other_values = [components_other[key] for key in components_other.keys() if key not in bonded_keys and 'Force' in key]  # Do not include thermostats and barostats
    print([key for key in components_other.keys() if key not in bonded_keys and 'Force' in key])
    print(f"Nonbondeds -- og: {np.sum(nonbonded_other_values)}, hybrid: {components_hybrid['NonbondedForce']}")
    assert np.isclose([components_hybrid['NonbondedForce']], np.sum(nonbonded_other_values))

    print(f"Success! Energies are equal at lambda {endstate}!")


def validate_unsampled_endstates_md(htf, hybrid_system, endstate=0, n_steps=125000, save_freq=250):
    """
    Test create_endstates() to make sure that a given unsampled endstate's hybrid system energy matches the
    original system's energy for snapshots extracted (every 1 ps) from a MD simulation.

    E.g. at endstate=0, the unsampled endstate hybrid system's energy  should match that of the original hybrid system.

    Parameters
    ----------
    htf : HybridTopologyFactory or RESTCapableHybridTopologyFactory
        hybrid factory from which the unsampled endstate's hybrid system was created
    hybrid_system : openmm.System
        the hybrid system to test
    endstate : int, default=0
        the endstate to test (0 or 1)
    n_steps : int, default=125000
        number of MD steps to run. (125000 steps == 1 ns)
    save_freq : int, default 250
        frequency at which to save positions
        by default, saves one snapshot every 250 steps

    """
    import tqdm
    from openmmtools import integrators
    from openmmtools.states import ThermodynamicState

    # Check that endstate is 0 or 1
    assert endstate in [0, 1], "Endstate must be 0 or 1"

    # Make a copy of the htf
    htf = copy.deepcopy(htf)

    # Get original hybrid system
    hybrid_system_og = htf.hybrid_system

    # Get positions, box_vectors
    hybrid_positions = htf.hybrid_positions
    box_vectors = hybrid_system_og.getDefaultPeriodicBoxVectors()

    # Create thermodynamic state
    thermostate = ThermodynamicState(hybrid_system, temperature=temperature)

    # Create context
    integrator = integrators.LangevinIntegrator(temperature=temperature, collision_rate=1 / unit.picoseconds, timestep=4 * unit.femtoseconds)
    context = thermostate.create_context(integrator)
    context.setPositions(hybrid_positions)
    context.setPeriodicBoxVectors(*box_vectors)
    context.setVelocitiesToTemperature(temperature)

    # Minimize
    openmm.LocalEnergyMinimizer.minimize(context)

    # Run MD
    hybrid = list()
    for _ in tqdm.tqdm(range(int(n_steps / save_freq))):
        integrator.step(save_freq)
        pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
        hybrid.append(pos)

    # Make context for original hybrid system
    thermostate = ThermodynamicState(hybrid_system_og, temperature=temperature)
    integrator_og = integrators.LangevinIntegrator(temperature=temperature,
                                                   collision_rate=1 / unit.picoseconds,
                                                   timestep=4 * unit.femtoseconds)
    context_og = thermostate.create_context(integrator_og)

    # Set global parameters
    htf_class = htf.__class__.__name__
    if htf_class == 'HybridTopologyFactory':
        for k, v in context_og.getParameters().items():
            if 'lambda' in k:
                context_og.setParameter(k, endstate)
    elif htf_class == 'RESTCapableHybridTopologyFactory':
        for k, v in context_og.getParameters().items():
            if 'alchemical' in k:
                if 'old' in k:
                    context_og.setParameter(k, 1 - endstate)
                elif 'new' in k or 'reciprocal' in k:
                    context_og.setParameter(k, endstate)
    else:
        raise Exception(
            f"{htf_class} is not supported. Supported factories: HybridTopologyFactory, RESTCapableHybridTopologyFactory")

    # Make context for hybrid system
    thermostate = ThermodynamicState(hybrid_system, temperature=temperature)
    integrator_hybrid = integrators.LangevinIntegrator(temperature=temperature,
                                                       collision_rate=1 / unit.picoseconds,
                                                       timestep=4 * unit.femtoseconds)
    context_hybrid = thermostate.create_context(integrator_hybrid)

    # Get energies for each conformation
    differences = list()
    for i, pos in enumerate(tqdm.tqdm(hybrid)):
        context_og.setPositions(pos)
        energy_og = context_og.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)

        context_hybrid.setPositions(pos)
        energy_hybrid = context_hybrid.getState(getEnergy=True).getPotentialEnergy().value_in_unit_system(unit.md_unit_system)

        differences.append(abs(energy_og - energy_hybrid))

    # Check that the standard deviation of the differences is < 1 kT
    stddev = np.std(differences)
    assert stddev < 1, f"The standard deviation of the differences is > 1 kT: {stddev} kT"
    print(f"Success! The standard deviation of the differences is < 1 kT: {stddev} kT")

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

    # Build force name to index dict and pull out forces
    old_system_force_names = {force.__class__.__name__ : index for index, force in enumerate(old_system.getForces())}
    new_system_force_names = {force.__class__.__name__ : index for index, force in enumerate(new_system.getForces())}

    old_system_torsions = old_system.getForce(old_system_force_names["PeriodicTorsionForce"])
    new_system_torsions = new_system.getForce(new_system_force_names["PeriodicTorsionForce"])

    # First, grab all of the old/new torsions
    num_old_torsions, num_new_torsions = old_system_torsions.getNumTorsions(), new_system_torsions.getNumTorsions()
    print(f"num old torsions, new torsions: {num_old_torsions}, {num_new_torsions}")

    old_torsions = [old_system_torsions.getTorsionParameters(i) for i in range(num_old_torsions)]
    new_torsions = [new_system_torsions.getTorsionParameters(i) for i in range(num_new_torsions)]

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


# Added for API backward compatibility -- 23 Aug 2022
check_system = utils.check_system
compute_potential_components = utils.compute_potential_components
generate_endpoint_thermodynamic_states = utils.generate_endpoint_thermodynamic_states
validate_endstate_energies = utils.validate_endstate_energies
validate_endstate_energies_point = utils.validate_endstate_energies_point
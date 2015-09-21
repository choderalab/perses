#!/usr/bin/env python

from perses import multitopology

import openeye.oechem as oe
import simtk.openmm as mm
from simtk import unit
import simtk.openmm.app as app
import numpy as np
import tempfile
import commands
import copy

from openmoltools import openeye

################################################################################
# SUBROUTINES
################################################################################

def assign_am1bcc_charges(molecule):
    """
    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule for which AM1-BCC charges are to be assigned

    Returns
    -------
    molecule : openeye.oechem.OEMol
        The charged molecule

    Notes
    -----
    From recipe for generating canonical AM1-BCC charges at:
    https://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html

    """

    # Create a copy.
    molecule = molecule.CreateCopy()

    # Expand conformations.
    from openeye import oeomega
    omega = oeomega.OEOmega()
    omega.SetIncludeInput(False)
    omega.SetCanonOrder(False)
    omega.SetSampleHydrogens(True)
    eWindow = 15.0
    omega.SetEnergyWindow(eWindow)
    omega.SetMaxConfs(800)
    omega.SetRMSThreshold(1.0)
    omega(molecule)

    # Assign partial charges.
    from openeye import oequacpac
    from openeye.oequacpac import OEAssignPartialCharges, OECharges_AM1BCCSym
    OEAssignPartialCharges(molecule, OECharges_AM1BCCSym)

    return molecule

def parameterize_molecule(molecule, implicitSolvent=app.OBC1, constraints=None, cleanup=True, verbose=False):
    """
    Parameterize the specified molecule for AMBER.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to be parameterized.
    implicitSolvent : default=app.OBC1
        The implicit solvent model to use; one of [None, HCT, OBC1, OBC2, GBn, GBn2]
    constraints : default=None
        Constraints to use; one of [None, HBonds, AllBonds, HAngles]
    cleanup : bool, optional, default=False
        If True, work done in a temporary working directory will be deleted.

    Returns
    -------
    system : simtk.openmm.System
        The OpenMM System of the molecule.
    topology : simtk.openmm.app.Topology
        The OpenMM topology of the molecule.
    positions :
        The positions of the molecule.
    gaff_molecule : oechem.OEMol
        The OEMol molecule with GAFF atom and bond types.

    """
    # Create molecule and geometry.
    molecule = gaff2xml.openeye.iupac_to_oemol(iupac_name)
    # Create a a temporary directory.
    working_directory = tempfile.mkdtemp()
    old_directory = os.getcwd()
    os.chdir(working_directory)
    # Parameterize molecule for AMBER (currently using old machinery for convenience)
    # TODO: Replace this with gaff2xml stuff
    amber_prmtop_filename = 'molecule.prmtop'
    amber_inpcrd_filename = 'molecule.inpcrd'
    amber_off_filename = 'molecule.off'
    oldmmtools.parameterizeForAmber(molecule, amber_prmtop_filename, amber_inpcrd_filename, charge_model=None, offfile=amber_off_filename)
    # Read in the molecule with GAFF atom and bond types
    print "Overwriting OEMol with GAFF atom and bond types..."
    gaff_molecule = oldmmtools.loadGAFFMolecule(molecule, amber_off_filename)

    # Load positions.
    inpcrd = app.AmberInpcrdFile(amber_inpcrd_filename)
    positions = inpcrd.getPositions()

    # Load system (with GB parameters).
    prmtop = app.AmberPrmtopFile(amber_prmtop_filename)
    system = prmtop.createSystem(implicitSolvent=implicitSolvent, constraints=constraints)

    # Clean up temporary files.
    os.chdir(old_directory)
    if cleanup:
        commands.getoutput('rm -r %s' % working_directory)
    else:
        print "Work done in %s..." % working_directory

    return [system, topology, positions, gaff_molecule]

def create_molecule(iupac_name):
    """
    Create an OEMol molecule from an IUPAC name.

    Parameters
    ----------
    iupac_name : str
        The IUPAC name of the molecule to be created.

    Returns
    -------
    molecule : openeye.oechem.OEMol
        A molecule with AM1-BCC charges.

    """

    molecule = gaff2xml.openeye.iupac_to_oemol(iupac_name)

    # Assign AM1-BCC charges using canonical scheme.
    # TODO: Replace wit updated gaff2xml scheme.
    molecule = assign_am1bcc_charges(molecule)

    # Assign conformations.
    from openeye import oeomega
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega(molecule)

    return molecule

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':
    # Create list of molecules that share a common core.
    molecule_names = ['benzene', 'toluene', 'methoxytoluene']
    molecules = [ create_molecule(name) for name in molecule_names ]

    # DEBUG: Write molecules to mol2 files for ease of debugging/visualization.
    for (index, molecule) in enumerate(molecules):
        openeye.molecule_to_mol2(molecule, tripos_mol2_filename='molecule-%05d.mol2' % index)

    # Create an OpenMM system to represent the environment.
    molecule = create_molecule('benzene') # TODO: Change to supramolecular host, like 18-crown-6?
    [system, topology, positions] = generate_openmm_system(molecule)
    # Translate the molecule out of the way so it doesn't overlap with anything.
    positions[:,2] += 15.0 * unit.angstroms # translate 15A along z-axis

    # Create merged topology for ligands and add them to the environment.
    [system, topology, positions] = create_merged_topology(system, topology, positions, molecules)

    # DEBUG: Write new atom identities.
    natoms = system.getNumParticles()
    for (index, atom) in enumerate(topology.atoms()):
        print '%8d %8s %8s %8s %8s %8s %8.3f %8.3f %8.3f' % (index, atom.name, atom.residue.chain.index, atom.element.name, atom.residue.index, atom.residue.name, positions[index,0]/unit.angstroms, positions[index,1]/unit.angstroms, positions[index,2]/unit.angstroms)
    app.PDBFile.writeFile(topology, positions, file=open('initial.pdb','w'))

    # Create an OpenMM test simulation of the merged-topology system to test stability.
    # A PDB file containing the trajectory is generated.
    temperature = 300.0 * unit.kelvin
    collision_rate = 20.0 / unit.picoseconds
    timestep = 1.0 * unit.femtoseconds
    integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = mm.Context(system, integrator)
    context.setParameter('alchemical_variant', 1) # Select variant index.
    context.setPositions(positions)
    niterations = 100
    nsteps_per_iteration = 100
    filename = 'trajectory.pdb'
    print "Writing out trajectory to %s ..." % filename
    outfile = open(filename, 'w')
    app.PDBFile.writeHeader(topology, file=outfile)
    for iteration in range(niterations):
        # Modify the lambda value.
        lambda_value = 1.0 - float(iteration) / float(niterations - 1)
        context.setParameter('alchemical_lambda', lambda_value)
        # Run some dynamics
        integrator.step(nsteps_per_iteration)
        # Get potential energy.
        state = context.getState(getPositions=True, getEnergy=True)
        # Report current information.
        print "Iteration %5d / %5d : lambda %8.5f : potential %8.3f kcal/mol" % (iteration, niterations, lambda_value, state.getPotentialEnergy() / unit.kilocalories_per_mole)
        # Write frame to trajectory.
        positions = state.getPositions()
        app.PDBFile.writeModel(topology, positions, file=outfile, modelIndex=(iteration+1))
    # Clean up.
    app.PDBFile.writeFooter(topology, file=outfile)
    outfile.close()
    del context, integrator



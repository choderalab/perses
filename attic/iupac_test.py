#!/usr/bin/env python
"""
Tool for creating a merged topology representation with discrete variant selection and continuous lambda parameter for relative alchemical transformations.

Example
-------

Create a few molecules sharing a common core.

>>> molecule_names = ['benzene', 'toluene', 'methoxytoluene']
>>> molecules = [ create_molecule(name) for name in molecule_names ]

Create an OpenMM system, topology, and positions to represent the environment that the molecules will be inserted into.
This example uses a nearby phenol molecule to represent the environment.

>>> molecule = create_molecule('phenol')
>>> [environment_system, environment_topology, environment_positions, environment_molecule_gaff] = parameterize_molecule(molecule)
>>> environment_positions[:,2] += 15.0 * unit.angstroms

Now create the merged topology, aligning the core to the specified reference molecule.

>>> factory = AlchemicalMergedTopologyFactory(environment_system, environment_topology, environment_positions)
>>> for molecule in molecules:
...    [system, topology, positions, gaff_molecule] = parameterize_molecule(molecule)
...    variant_index = factory.addMoleculeVariant(gaff_molecule, system, topology, positions)
>>> [system, topology, positions] = factory.generateMergedTopology(reference_molecule=molecules[0])

Notes
-----

In this scheme, the substructure that shares common atom types is assigned to the shared "core" that is not perturbed.

Context parameters
------------------
* `alchemical_variant` - index (0, 1, 2...) of variant currently selected to be active
* `alchemical_lambda` - alchemical parameter that interpolates between 0 (core only) and 1 (variant `alchemical_varid` is fully chemically present)

TODO
----
* Write standalone function to convert Tripos mol2 -> GAFF mol2 + AMBER prmtop/inpcrd


"""

################################################################################
# IMPORTS
################################################################################

import gaff2xml.openeye
import openeye
import simtk.openmm as mm
from simtk import unit
import simtk.openmm.app as app
import numpy as np
import tempfile
import commands
import copy

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

if __name__ == '__main__':
    # Create list of molecules that share a common core.
    molecule_names = ['benzene', 'toluene', 'methoxytoluene', '18-crown-6']
    for name in molecule_names:
        print name
        molecule = create_molecule(name)
        from oldmmtools import writeMolecule
        tripos_mol2_filename = name + '.mol2'
        writeMolecule(molecule, tripos_mol2_filename)

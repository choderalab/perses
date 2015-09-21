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

from openmoltools import openeye

import openeye.oechem as oe
import simtk.openmm as mm
from simtk import unit
import simtk.openmm.app as app
import numpy as np
import tempfile
import commands
import copy

################################################################################
# UTILITY TESTING SUBROUTINES
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
    molecule = openeye.iupac_to_oemol(iupac_name)
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

    molecule = openeye.iupac_to_oemol(iupac_name)

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
# MODULE CONSTANTS
################################################################################

ONE_4PI_EPS0 = 138.935456 # OpenMM constant for Coulomb interactions (openmm/platforms/reference/include/SimTKOpenMMRealType.h) in OpenMM units

################################################################################
# ALCHEMICAL FACTORIES
################################################################################

class AlchemicalMergedTopologyFactory(object):
    """\
    Factory for creating an OpenMM System object that utilizes an alchemically
    modified merged topology that can interpolate between many small molecules
    sharing a common core.

    Examples
    --------

    This example creates a merged topology file containing several small benzene derivatives where the environment is an 18-crown-6 host.

    Create molecules to be added to the system.

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

    Createa a system and set context parameters.

    >>> integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    >>> context = openmm.Context(system, integrator)
    >>> context.setParameter('alchemical_variant', 1) # select first molecule
    >>> context.setParameter('alchemical_lambda', 1.0) # turn molecule on fully

    Notes
    -----

    The procedure to generate merged topology files roughly follows this:

    System preparation:

    * Add new Force objects to the system to handle the interpolation between the different bonded terms in the core.

    Core:

    * Identify a common core or scaffold via a maximum common substructure (MCSS) search
    * Selecta set of charges for the core that minimize changes required to perturb to other molecules and preserve total charge
    * Add the core atoms and forcefield terms to the system

    Molecules:

    * For each variant molecule
      * Identify the atoms corresponding to the core and map them onto the core
      * Add a complete copy of the molecule to the system
      * Nonbonded terms for the molecule are handled by a CustomNonbondedForce
      * Valence terms for the molecule are handled either by Custom*Force objects (if part of the core) or standard forces (otherwise)
      * Harmonic restraints are added to keep core-corresponding restrained close to their corresponding core atoms

    Context parameters
    ------------------

    The new `system` object will now have two context parameters that can be modified:
    * `alchemical_variant` - index (0, 1, 2...) of variant currently selected to be active
    * `alchemical_lambda` - alchemical parameter that interpolates between 0 (core only) and 1 (variant `alchemical_variant` is fully chemically present)

    """
    def __init__(self, environment_system, environment_topology, environment_positions,
                       softcore_alpha=0.5, softcore_beta=12*unit.angstrom**2):
        """\
        Create a factory for generating merged topologies.

        Parameters
        ----------
        environment_system : simtk.openmm.System
            The OpenMM System representing the environment.
        environment_topology : simtk.openmm.app.Topology
            The OpenMM Topology representing the environment.
        environment_positions : simtk.unit.Quantity of (natoms,3) with units compatible with distance
            The positions for the environment atoms.
        softcore_alpha : float, optional, default=0.5
            The softcore parameter for Lennard-Jones softening.
        softcore_beta : simtk.unit.Quantity with units compatible with length**2, optional, default = 12*angstroms**2
            The softcore parameter for electrostatics softening.

        """

        # Make copies of environment to not destroy original objects.
        self.environment_system = copy.deepcopy(environment_system)
        self.environment_topology = copy.deepcopy(environment_topology)
        self.environment_positions = copy.deepcopy(environment_positions)

        # Softcore defaults.
        self.softcore_alpha = softcore_alpha
        self.softcore_beta = softcore_beta

        # Storage for molecule variants.
        self._molecules = list()
        self._systems = list()
        self._topologies = list()
        self._positions = list()

        return

    def _showMolecule(self, molecule):
        """\
        Show the molecule (for debugging).

        Parameters
        ----------
        molecule : openeye.oechem.OEMol
            The molecule to be printed.

        """
        atom_index = 1
        for atom in molecule.GetAtoms():
            print "%5d %6s %12s" % (atom_index, atom.GetName(), atom.GetType())
            atom_index += 1
        print ""
        return

    def _determineCommonSubstructure(self, ligands, min_atoms=4, verbose=False):
        """Find a common substructure shared by all molecules.

        The atom type name strings and integer bond types are used to obtain an exact match.

        Parameters
        ----------
        ligands : list of openeye.oechem.OEMol
            The set of ligands for which the common substructure is to be determined.
        min_atoms : int, optional, default=4
            Minimum number of atoms for substructure match
        verbose : bool, optional, default=False
            If True, verbose information is printed

        Returns
        -------
        common_substructure : openeye.oechem.OEMol
            A molecule fragment representing the common substructure


        Provenance
        ----------
        This function comes from the mmtools repo.
        Was modified by David L. Mobley on 15 Nov 2010.

        """
        # Determine number of ligands
        nligands = len(ligands)

        # First, initialize with first ligand.
        common_substructure = ligands[0].CreateCopy()

        # Show initial molecule.
        if verbose: self._showMolecule(common_substructure)

        # Now delete bits that don't match every other ligand.
        for ligand in ligands[1:]:
            # get ligand name
            ligand_name = ligand.GetTitle()

            # Create an OEMCSSearch from this molecule.
            from openeye.oechem import OEMCSSearch, OEExprOpts_StringType, OEExprOpts_IntType
            mcss = OEMCSSearch(ligand, OEExprOpts_StringType, OEExprOpts_IntType)

            # ignore substructures smaller than 4 atoms
            mcss.SetMinAtoms(min_atoms)

            # This modifies scoring function to prefer keeping cycles complete.
            from openeye.oechem import OEMCSMaxAtomsCompleteCycles
            mcss.SetMCSFunc( OEMCSMaxAtomsCompleteCycles() )

            # perform match
            for match in mcss.Match(common_substructure):
                nmatched = match.NumAtoms()

                if verbose: print "%(ligand_name)s : match size %(nmatched)d atoms" % vars()

                # build list of matched atoms in common substructure
                matched_atoms = list()
                for matchpair in match.GetAtoms():
                    atom = matchpair.target
                    matched_atoms.append(atom)

                # delete all unmatched atoms from common substructure
                for atom in common_substructure.GetAtoms():
                    if atom not in matched_atoms:
                        common_substructure.DeleteAtom(atom)

                # Show molecule after pruning.
                if verbose: self._showMolecule(common_substructure)

                # we only need to consider one match
                break

        # Rename common substructure.
        common_substructure.SetTitle('core')

        # return the common substructure
        return common_substructure

    def addMoleculeVariant(self, molecule, system, topology, positions):
        """\
        Add a variant molecule.

        Parameters
        ----------
        molecule : openeye.oechem.OEMol
            The OEMol molecule with GAFF atom types and bond types (or other desired types to use for matching).
            These molecules can be in any orientation or position; their cores can later be aligned to a reference molecule when the merged topology is constructed.
        system : simtk.openmm.System
            The System object corresponding to molecule.
        topology : simtk.openmm.app.Topology
            The topology corresponding to the molecule.
        positions :
            The positions of the molecule.
            TODO: Is this redundant with `molecule`?

        Returns
        -------
        variant_index : int
            The variant index for this molecule.

        """
        variant_index = len(self._molecules)
        self._molecules.append(molecule.CreateCopy())
        self._systems.append(copy.deepcopy(system))
        self._topologies.append(copy.deepcopy(topology))
        self._positions.append(copy.deepcopy(positions))
        return variant_index

    def generateMergedTopology(self, reference_molecule=None, verbose=False):
        """\
        Generate an alchemical merged topology for the added molecule variants.

        Parameters
        ----------
        reference_molecule : openeye.oechem.OEMol, optional, default=None
            If specified, a molecule whose positions the core is to be aligned.
        verbose : bool, optional, default=False
            If True, will print out lots of debug info.

        Returns
        -------
        system : simtk.openmm.System
            Modified version of system in which old system is recovered for global context paramaeter `lambda` = 0 and new molecule is substituted for `lambda` = 1.
        topology : system.openmm.Topology
            Topology corresponding to system.
        positions: simtk.unit.Quantity of (natoms,3) with units compatible with length
            Positions corresponding to constructed system.

        """
        # Copy molecules so as not to accidentally overwrite them.
        molecules = [ molecule.CreateCopy() for molecule in self._molecules ]

        # Determine common substructure using exact match of GAFF atom and bond types.
        if verbose: print "Determining common core substructure..."
        core = oldmmtools.determineCommonSubstructure(molecules, verbose=True)

        # Find RMS-fit charges for common intermediate.
        if verbose: print "Determining RMS-fit charges for common intermediate..."
        core = oldmmtools.determineMinimumRMSCharges(core, molecules)

        # DEBUG: Write out info for common core / scaffold.
        if verbose:
            print "Common core atoms, types, and partial charges:"
            print "\n%s" % core.GetTitle()
            for atom in core.GetAtoms():
                print "%6s : %3s %8.3f" % (atom.GetName(), atom.GetType(), atom.GetPartialCharge())

            # Write out common substructure in GAFF format.
            filename = 'core.gaff.mol2'
            print "Writing common intermediate with GAFF atomtypes to %s" % filename
            oldmmtools.writeMolecule(core, filename, preserve_atomtypes=True)

        # Set up MCSS to detect overlap with common substructure.
        atomexpr = oe.OEExprOpts_StringType # match GAFF atom type (str) exactly
        bondexpr = oe.OEExprOpts_IntType # match GAFF bond type (int) exactly
        min_atoms = 4
        mcss = oe.OEMCSSearch(core, atomexpr, bondexpr)
        mcss.SetMinAtoms(min_atoms) # ensure a minimum number of atoms match
        mcss.SetMCSFunc( oe.OEMCSMaxAtomsCompleteCycles() ) # prefer keeping cycles complete.
        mcss.SetMaxMatches(1)



        return [system, topology, positions]

def create_merged_topology(system, topology, positions,
                           molecules,
                           softcore_alpha=0.5, softcore_beta=12*unit.angstrom**2):
    """
    Create an OpenMM system that utilizes a merged topology that can interpolate between many small molecules sharing a common core.

    Notes
    -----
    * Currently, only one set of molecules is supported.
    * Residue mutations are not yet supported.

    Parameters
    ----------
    system : simtk.openmm.System
       The system representing the environment, not yet containing any molecules.
    topology : simtk.openmm.app.Topology
       The topology object corresponding to system.
    positions : simtk.unit.Quantity of numpy array natoms x 3 compatible with units angstroms
       The positions array corresponding to system and topology.
    molecules : list of openeye.oechem.OEMol
       Molecules to be added to the system.  These molecules must share a common core with identical GAFF atom types.
    softcore_alpha : float, optional, default=0.5
       Softcore parameter for Lennard-Jones softening.
    softcore_beta : simtk.unit.Quantity with units compatible with angstrom**2
       Softcore parameter for Coulomb interaction softening.

    Returns
    -------
    system : simtk.openmm.System
       Modified version of system in which old system is recovered for global context paramaeter `lambda` = 0 and new molecule is substituted for `lambda` = 1.
    topology : system.openmm.Topology
       Topology corresponding to system.

    """

    #
    # First, process first molecule so we can add common substructure atoms to System along with valence terms among core atoms.
    #

    # Use the first molecule as a reference molecule.
    # TODO: Replace this with a different reference molecule.
    molecule = molecules[0].CreateCopy()
    gaff_molecule = gaff_molecules[0].CreateCopy()

    # Determine common atoms in second molecule.
    matches = [ match for match in mcss.Match(gaff_molecule, True) ]
    match = matches[0] # we only need the first match

    # Make a list of the atoms in core.
    reverse_mapping = dict()
    print "molecule => core mapping"
    for matchpair in match.GetAtoms():
        core_index = matchpair.pattern.GetIdx()
        molecule_index = matchpair.target.GetIdx()
        print "%8d => %8d" % (molecule_index, core_index)
        reverse_mapping[core_index] = molecule_index
    # Make sure the list of atoms in molecule corresponds to the ordering of atoms within the core.
    core_atoms = [ reverse_mapping[core_index] for core_index in range(core.NumAtoms()) ]

    # Create OpenMM Topology and System objects for given molecules using GAFF/AM1-BCC.
    print "Generating OpenMM system for molecule..."
    [molecule_system, molecule_topology, molecule_positions] = generate_openmm_system(molecule)

    # TODO: Replace charges in molecule with RMS charges.

    # Add core fragment to system.
    core_mapping = add_molecule_to_system(system, molecule_system, core_atoms, variant=0)

    # Add molecule to topology as a new residue.
    chain = topology.addChain()
    residue = topology.addResidue('COR', chain)
    atoms = [ atom for atom in molecule.GetAtoms() ] # build a list of all atoms in reference molecule
    atoms = [ atoms[index] for index in core_atoms ] # select out only core atoms in proper order
    for atom in atoms:
        name = atom.GetName()
        atomic_number = atom.GetAtomicNum()
        element = app.Element.getByAtomicNumber(atomic_number)
        topology.addAtom(name, element, residue)

    # Append positions of new particles.
    positions = unit.Quantity(np.append(positions/positions.unit, molecule_positions[core_atoms,:]/positions.unit, axis=0), positions.unit)

    # Create a running list of atoms all variants should have interactions excluded with.
    atoms_to_exclude = core_mapping.values()

    #
    # Create restraint force to keep corresponding core atoms from molecules near their corresponding core atoms.
    # TODO: Can we replace these with length-zero constraints or virtual particles if OpenMM supports this in the future?
    #

    K = 10.0 * unit.kilocalories_per_mole / unit.angstrom**2 # spring constant
    energy_expression  = '(K/2) * r^2;'
    energy_expression += 'K = %f;' % K.value_in_unit_system(unit.md_unit_system)
    restraint_force = mm.CustomBondForce(energy_expression)

    #
    # Now, process all molecules.
    #

    variants = list()

    variant = 1
    for (molecule, gaff_molecule) in zip(molecules, gaff_molecules):
        print ""
        print "*******************************************************"
        print "Incorporating molecule %s" % molecule.GetTitle()
        print "*******************************************************"
        print ""

        # Determine common atoms in second molecule.
        matches = [ match for match in mcss.Match(gaff_molecule, True) ]
        print matches
        match = matches[0] # we only need the first match

        # Make a list of the atoms in core.
        core_atoms = list()
        core_atoms_bond_mapping = dict()
        print "molecule => core mapping"
        for matchpair in match.GetAtoms():
            core_index = matchpair.pattern.GetIdx()
            molecule_index = matchpair.target.GetIdx()
            core_atoms.append(molecule_index) # list of atoms in the molecule that correspond to core atoms
            print "%8d => %8d" % (molecule_index, core_index)
            core_atoms_bond_mapping[molecule_index] = core_mapping[core_index] # index of corresponding core atom in system, for restraining

        # Align molecule to overlay common core.
        overlay = True
        rmat  = oe.OEDoubleArray(9)
        trans = oe.OEDoubleArray(3)
        rms = oe.OERMSD(mcss.GetPattern(), gaff_molecule, match, overlay, rmat, trans)
        if rms < 0.0:
            raise Exception("RMS overlay failure")
        print "RMSD after overlay is %.3f A" % rms
        oe.OERotate(gaff_molecule, rmat)
        oe.OETranslate(gaff_molecule, trans)

        # Transfer positions to regular molecule with Tripos atom types.
        gaff_atoms = [ atom for atom in gaff_molecule.GetAtoms() ]
        atoms = [ atom for atom in molecule.GetAtoms() ]
        for (source_atom, dest_atom) in zip(gaff_atoms, atoms):
            molecule.SetCoords(atom, gaff_molecule.GetCoords(atom))

        # Create OpenMM Topology and System objects for given molecules using GAFF/AM1-BCC.
        print "Generating OpenMM system for molecule..."
        [molecule_system, molecule_topology, molecule_positions] = generate_openmm_system(molecule)

        # Append positions of new particles.
        positions = unit.Quantity(np.append(positions/positions.unit, molecule_positions/positions.unit, axis=0), positions.unit)

        # Add valence terms only.
        mapping = add_molecule_to_system(system, molecule_system, core_atoms, variant=variant, atoms_to_exclude=atoms_to_exclude)

        # DEBUG
        print "Atom mappings into System object:"
        print mapping

        # Add restraints to keep core atoms from this molecule near their corresponding core atoms.
        for index in core_atoms:
            restraint_force.addBond(mapping[index], core_atoms_bond_mapping[index], [])

        # Add molecule to topology as a new residue.
        # TODO: Can we simplify this by copying from molecule_topology instead?
        residue = topology.addResidue('LIG', chain)
        for atom in atoms:
            name = atom.GetName()
            atomic_number = atom.GetAtomicNum()
            element = app.Element.getByAtomicNumber(atomic_number)
            topology.addAtom(name, element, residue)

        # Increment variant index.
        variant += 1

        # Append to list of atoms to be excluded.
        atoms_to_exclude += mapping.values()

    print "Done!"

    # Add restraint force to core atoms.
    # NOTE: This cannot be added to system earlier because of dict lookup for 'CustomBondForce' in add_valence_terms().
    system.addForce(restraint_force)

    return [system, topology, positions]

def add_molecule_to_system(system, molecule_system, core_atoms, variant, atoms_to_exclude=[]):
    """
    Add the valence terms for the molecule from molecule_system.

    Parameters
    ----------
    system : simtk.openmm.System
       The system object to which the valence terms are to be added.
    molecule_system : simtk.openmm.System
       The system object from which core valence terms are to be taken.
    core_atoms : list of int
       The list of atom indices within molecule_system corresponding to core atoms.
    variant : int
       The variant index of this molecule if not a core fragment, or 0 if this is a core fragment and only core atoms are to be added.

    Returns
    -------
    mapping : dict of int
       mapping[index] is the atom index in `system` corresponding to atom `index` within `molecule_system`.

    """

    def _createCustomNonbondedForce(self, system, molecule_system, softcore_alpha=0.5, softcore_beta=12*unit.angstrom**2):
        """
        Create alchemically-modified version of NonbondedForce.

        Parameters
        ----------
        system : simtk.openmm.System
            Alchemically-modified system being built.  This object will be modified.
        molecule_system : simtk.openmm.System
            Source molecule system to copy from.
        softcore_alpha : float, optional, default = 0.5
            Alchemical softcore parameter for Lennard-Jones.
        softcore_beta : simtk.unit.Quantity with units compatible with angstroms**2, optional, default = 12*angstrom**2
            Alchemical softcore parameter for electrostatics.

        TODO
        ----
        Try using a single, common "reff" effective softcore distance for both Lennard-Jones and Coulomb.

        """

        alchemical_atom_indices = self.ligand_atoms

        # Create a copy of the NonbondedForce to handle non-alchemical interactions.
        nonbonded_force = copy.deepcopy(reference_force)
        system.addForce(nonbonded_force)

        # Create CustomNonbondedForce objects to handle softcore interactions between alchemically-modified system and rest of system.

        # Create atom groups.
        natoms = system.getNumParticles()
        atomset1 = set(alchemical_atom_indices) # only alchemically-modified atoms
        atomset2 = set(range(system.getNumParticles())) # all atoms, including alchemical region

        # CustomNonbondedForce energy expression.
        sterics_energy_expression = ""
        electrostatics_energy_expression = ""

        # Select functional form based on nonbonded method.
        method = reference_force.getNonbondedMethod()
        if method in [openmm.NonbondedForce.NoCutoff]:
            # soft-core Lennard-Jones
            sterics_energy_expression += "U_sterics = lambda_sterics*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            # soft-core Coulomb
            electrostatics_energy_expression += "U_electrostatics = ONE_4PI_EPS0*lambda_electrostatics*chargeprod/reff_electrostatics;"
        elif method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
            # soft-core Lennard-Jones
            sterics_energy_expression += "U_sterics = lambda_sterics*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            # reaction-field electrostatics
            epsilon_solvent = reference_force.getReactionFieldDielectric()
            r_cutoff = reference_force.getCutoffDistance()
            electrostatics_energy_expression += "U_electrostatics = lambda_electrostatics*ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);"
            k_rf = r_cutoff**(-3) * ((epsilon_solvent - 1) / (2*epsilon_solvent + 1))
            c_rf = r_cutoff**(-1) * ((3*epsilon_solvent) / (2*epsilon_solvent + 1))
            electrostatics_energy_expression += "k_rf = %f;" % (k_rf / k_rf.in_unit_system(unit.md_unit_system).unit)
            electrostatics_energy_expression += "c_rf = %f;" % (c_rf / c_rf.in_unit_system(unit.md_unit_system).unit)
        elif method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            # soft-core Lennard-Jones
            sterics_energy_expression += "U_sterics = lambda_sterics*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            # Ewald direct-space electrostatics
            [alpha_ewald, nx, ny, nz] = reference_force.getPMEParameters()
            if alpha_ewald == 0.0:
                # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
                delta = reference_force.getEwaldErrorTolerance()
                r_cutoff = reference_force.getCutoffDistance()
                alpha_ewald = np.sqrt(-np.log(2*delta)) / r_cutoff
            electrostatics_energy_expression += "U_electrostatics = lambda_electrostatics*ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
            electrostatics_energy_expression += "alpha_ewald = %f;" % (alpha_ewald / alpha_ewald.in_unit_system(unit.md_unit_system).unit)
            # TODO: Handle reciprocal-space electrostatics
        else:
            raise Exception("Nonbonded method %s not supported yet." % str(method))

        # Add additional definitions common to all methods.
        sterics_energy_expression += "reff_sterics = sigma*((softcore_alpha*(1.-lambda_sterics) + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
        sterics_energy_expression += "softcore_alpha = %f;" % softcore_alpha
        electrostatics_energy_expression += "reff_electrostatics = sqrt(softcore_beta*(1.-lambda_electrostatics) + r^2);" # effective softcore distance for electrostatics
        electrostatics_energy_expression += "softcore_beta = %f;" % (softcore_beta / softcore_beta.in_unit_system(unit.md_unit_system).unit)
        electrostatics_energy_expression += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0 # already in OpenMM units

        # Define mixing rules.
        sterics_mixing_rules = ""
        sterics_mixing_rules += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        sterics_mixing_rules += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
        electrostatics_mixing_rules = ""
        electrostatics_mixing_rules += "chargeprod = charge1*charge2;" # mixing rule for charges

        # Create CustomNonbondedForce to handle interactions between alchemically-modified atoms and rest of system.
        electrostatics_custom_nonbonded_force = openmm.CustomNonbondedForce("U_electrostatics;" + electrostatics_energy_expression + electrostatics_mixing_rules)
        electrostatics_custom_nonbonded_force.addGlobalParameter("lambda_electrostatics", 1.0);
        electrostatics_custom_nonbonded_force.addPerParticleParameter("charge") # partial charge
        sterics_custom_nonbonded_force = openmm.CustomNonbondedForce("U_sterics;" + sterics_energy_expression + sterics_mixing_rules)
        sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics", 1.0);
        sterics_custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon

        # Set parameters to match reference force.
        sterics_custom_nonbonded_force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
        electrostatics_custom_nonbonded_force.setUseSwitchingFunction(False)
        sterics_custom_nonbonded_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
        electrostatics_custom_nonbonded_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
        sterics_custom_nonbonded_force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
        sterics_custom_nonbonded_force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
        electrostatics_custom_nonbonded_force.setUseLongRangeCorrection(False)

        # Set periodicity and cutoff parameters corresponding to reference Force.
        if nonbonded_force.getNonbondedMethod() in [openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME, openmm.NonbondedForce.CutoffPeriodic]:
            sterics_custom_nonbonded_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
            electrostatics_custom_nonbonded_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
        else:
            sterics_custom_nonbonded_force.setNonbondedMethod( nonbonded_force.getNonbondedMethod() )
            electrostatics_custom_nonbonded_force.setNonbondedMethod( nonbonded_force.getNonbondedMethod() )

        # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
        # TODO: Exclude intra-alchemical region if we are separately handling that through a separate CustomNonbondedForce for decoupling.
        sterics_custom_nonbonded_force.addInteractionGroup(atomset1, atomset2)
        electrostatics_custom_nonbonded_force.addInteractionGroup(atomset1, atomset2)

        # Add custom forces.
        system.addForce(sterics_custom_nonbonded_force)
        system.addForce(electrostatics_custom_nonbonded_force)

        # Create CustomBondForce to handle exceptions for both kinds of interactions.
        custom_bond_force = openmm.CustomBondForce("U_sterics + U_electrostatics;" + sterics_energy_expression + electrostatics_energy_expression)
        custom_bond_force.addGlobalParameter("lambda_electrostatics", 1.0);
        custom_bond_force.addGlobalParameter("lambda_sterics", 1.0);
        custom_bond_force.addPerBondParameter("chargeprod") # charge product
        custom_bond_force.addPerBondParameter("sigma") # Lennard-Jones effective sigma
        custom_bond_force.addPerBondParameter("epsilon") # Lennard-Jones effective epsilon
        system.addForce(custom_bond_force)

        # Move NonbondedForce particle terms for alchemically-modified particles to CustomNonbondedForce.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Add parameters to custom force handling interactions between alchemically-modified atoms and rest of system.
            sterics_custom_nonbonded_force.addParticle([sigma, epsilon])
            electrostatics_custom_nonbonded_force.addParticle([charge])
            # Turn off Lennard-Jones contribution from alchemically-modified particles.
            if particle_index in alchemical_atom_indices:
                nonbonded_force.setParticleParameters(particle_index, 0*charge, sigma, 0*epsilon)

        # Move NonbondedForce exception terms for alchemically-modified particles to CustomNonbondedForce/CustomBondForce.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Exclude this atom pair in CustomNonbondedForce.
            sterics_custom_nonbonded_force.addExclusion(iatom, jatom)
            electrostatics_custom_nonbonded_force.addExclusion(iatom, jatom)
            # Move exceptions involving alchemically-modified atoms to CustomBondForce.
            if self.annihilate_sterics and (iatom in alchemical_atom_indices) and (jatom in alchemical_atom_indices):
                # Add special CustomBondForce term to handle alchemically-modified Lennard-Jones exception.
                custom_bond_force.addBond(iatom, jatom, [chargeprod, sigma, epsilon])
                # Zero terms in NonbondedForce.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, 0*chargeprod, sigma, 0*epsilon)

        # TODO: Add back NonbondedForce terms for alchemical system needed in case of decoupling electrostatics or sterics via second CustomBondForce.
        # TODO: Also need to change current CustomBondForce to not alchemically disappear system.

        return

    # Build dict of forces.
    def create_force_dict(system):
        return { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }

    molecule_forces = create_force_dict(molecule_system)
    forces          = create_force_dict(system)

    # Create Custom*Force classes if necessary.
    if 'CustomBondForce' not in forces:
        energy_expression  = 'lambda*(K/2)*(r-length)^2;'
        energy_expression += 'lambda = (1-alchemical_lambda)*delta(variant) + alchemical_lambda*delta(variant-alchemical_variant);'
        custom_force = mm.CustomBondForce(energy_expression)
        custom_force.addGlobalParameter('alchemical_lambda', 0.0)
        custom_force.addGlobalParameter('alchemical_variant', 0.0)
        custom_force.addPerBondParameter('variant')
        custom_force.addPerBondParameter('length')
        custom_force.addPerBondParameter('K')
        system.addForce(custom_force)
        forces['CustomBondForce'] = custom_force
    if 'CustomAngleForce' not in forces:
        energy_expression  = 'lambda*(K/2)*(theta-theta0)^2;'
        energy_expression += 'lambda = (1-alchemical_lambda)*delta(variant) + alchemical_lambda*delta(variant-alchemical_variant);'
        custom_force = mm.CustomAngleForce(energy_expression)
        custom_force.addGlobalParameter('alchemical_lambda', 0.0)
        custom_force.addGlobalParameter('alchemical_variant', 0.0)
        custom_force.addPerAngleParameter('variant')
        custom_force.addPerAngleParameter('theta0')
        custom_force.addPerAngleParameter('K')
        system.addForce(custom_force)
        forces['CustomAngleForce'] = custom_force
    if 'CustomTorsionForce' not in forces:
        energy_expression  = 'lambda*K*(1+cos(periodicity*theta-phase));'
        energy_expression += 'lambda = (1-alchemical_lambda)*delta(variant) + alchemical_lambda*delta(variant-alchemical_variant);'
        custom_force = mm.CustomTorsionForce(energy_expression)
        custom_force.addGlobalParameter('alchemical_lambda', 0.0)
        custom_force.addGlobalParameter('alchemical_variant', 0.0)
        custom_force.addPerTorsionParameter('variant')
        custom_force.addPerTorsionParameter('periodicity')
        custom_force.addPerTorsionParameter('phase')
        custom_force.addPerTorsionParameter('K')
        system.addForce(custom_force)
        forces['CustomTorsionForce'] = custom_force
    if 'CustomNonbondedForce' not in forces:
        # DEBUG
        # TODO: Create proper CustomNonbondedForce here.
        energy_expression = "0.0;"
        custom_force = mm.CustomNonbondedForce(energy_expression)
        custom_force.addGlobalParameter('alchemical_lambda', 0.0)
        custom_force.addGlobalParameter('alchemical_variant', 0.0)
        custom_force.addPerParticleParameter('variant')
        custom_force.addPerParticleParameter('charge')
        custom_force.addPerParticleParameter('sigma')
        custom_force.addPerParticleParameter('epsilon')
        system.addForce(custom_force)
        forces['CustomNonbondedForce'] = custom_force

        # Add parameters for existing particles.
        for index in range(system.getNumParticles()):
            [charge, sigma, epsilon] = forces['NonbondedForce'].getParticleParameters(index)
            custom_force.addParticle([0, charge, sigma, epsilon])

    # Add particles to system.
    mapping = dict() # mapping[index_in_molecule] = index_in_system
    for index_in_molecule in range(molecule_system.getNumParticles()):
        # Add all atoms, unless we're adding the core, in which case we just add core atoms.
        if (variant) or (index_in_molecule in core_atoms):
            # TODO: We may want to make masses lighter.
            index_in_system = system.addParticle(system.getParticleMass(index_in_molecule))
            mapping[index_in_molecule] = index_in_system

    # Constraints are not supported.
    # TODO: Later, consider supporting some constraints, such as those within core and those within variants.
    if (molecule_system.getNumConstraints() > 0):
        raise Exception("Constraints are not supported for alchemically modified molecule.")

    # Process forces.
    # Valence terms involving only core atoms are created as Custom*Force classes where lambda=0 activates the "core" image and lambda=1 activates the "variant" image.
    for (force_name, force) in molecule_forces.iteritems():
        print force_name
        if force_name == 'HarmonicBondForce':
            for index in range(force.getNumBonds()):
                [atom_i, atom_j, length, K] = force.getBondParameters(index)
                if set([atom_i, atom_j]).issubset(core_atoms):
                    forces['CustomBondForce'].addBond(mapping[atom_i], mapping[atom_j], [variant, length, K])
                elif (variant):
                    forces[force_name].addBond(mapping[atom_i], mapping[atom_j], length, K)

        elif force_name == 'HarmonicAngleForce':
            for index in range(force.getNumAngles()):
                [atom_i, atom_j, atom_k, theta0, K] = force.getAngleParameters(index)
                if set([atom_i, atom_j, atom_k]).issubset(core_atoms):
                    forces['CustomAngleForce'].addAngle(mapping[atom_i], mapping[atom_j], mapping[atom_k], [variant, theta0, K])
                elif (variant):
                    forces[force_name].addAngle(mapping[atom_i], mapping[atom_j], mapping[atom_k], theta0, K)

        elif force_name == 'PeriodicTorsionForce':
            for index in range(force.getNumTorsions()):
                [atom_i, atom_j, atom_k, atom_l, periodicity, phase, K] = force.getTorsionParameters(index)
                if set([atom_i, atom_j, atom_k, atom_l]).issubset(core_atoms):
                    forces['CustomTorsionForce'].addTorsion(mapping[atom_i], mapping[atom_j], mapping[atom_k], mapping[atom_l], [variant, periodicity, phase, K])
                elif (variant):
                    forces[force_name].addTorsion(mapping[atom_i], mapping[atom_j], mapping[atom_k], mapping[atom_l], periodicity, phase, K)

        elif force_name == 'NonbondedForce':
            for index in range(force.getNumParticles()):
                # TODO: Nonbonded terms will have to be handled as CustomNonbondedForce terms.
                [charge, sigma, epsilon] = force.getParticleParameters(index)
                if set([index]).issubset(core_atoms):
                    forces[force_name].addParticle(0.0*charge, sigma, 0.0*epsilon)
                    forces['CustomNonbondedForce'].addParticle([variant, charge, sigma, epsilon])
                elif (variant):
                    forces[force_name].addParticle(0.0*charge, sigma, 0.0*epsilon)
                    forces['CustomNonbondedForce'].addParticle([variant, charge, sigma, epsilon])

            for index in range(force.getNumExceptions()):
                [atom_i, atom_j, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
                if set([atom_i, atom_j]).issubset(core_atoms):
                    # TODO: Nonbonded exceptions will have to be handled as CustomBondForce terms.
                    forces[force_name].addException(mapping[atom_i], mapping[atom_j], 0.0 * unit.elementary_charge**2, 1.0 * unit.angstrom, 0.0 * unit.kilocalories_per_mole)
                elif (variant):
                    forces[force_name].addException(mapping[atom_i], mapping[atom_j], chargeProd, sigma, epsilon)

        # TODO: Add GB force processing.

    # Add exclusions to previous variants and core.
    for atom_i in mapping.values():
        for atom_j in atoms_to_exclude:
            forces['NonbondedForce'].addException(atom_i, atom_j, 0.0 * unit.elementary_charge**2, 1.0 * unit.angstrom, 0.0 * unit.kilocalories_per_mole)
            forces['CustomNonbondedForce'].addExclusion(atom_i, atom_j)

    print system.getNumParticles(), forces['NonbondedForce'].getNumParticles()

    return mapping

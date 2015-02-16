#!/usr/bin/env python
"""
Testbed for creating a merged topology representation with continuous lambda parameters for alchemical transformations.

Notes
-----

In this scheme, the substructure that shares common atom types is assigned to the shared "core" that is not perturbed.

Context parameters
------------------
* `alchemical_variant` - index (0, 1, 2...) of variant currently selected to be active
* `alchemical_lambda` - alchemical parameter that interpolates between 0 (core only) and 1 (variant `alchemical_varid` is fully chemically present)

"""

import gaff2xml.openeye
import openeye.oechem as oe
import simtk.openmm as mm
from simtk import unit
import simtk.openmm.app as app
import numpy as np
import copy

def create_molecule(iupac_name):
    molecule = gaff2xml.openeye.iupac_to_oemol(iupac_name)
    molecule = gaff2xml.openeye.get_charges(molecule, max_confs=1)
    #import openeye.oeomega as om
    #omega = om.OEOmega()
    #omega.SetMaxConfs(1)
    #omega(molecule)
    return molecule

def generate_openmm_system(molecule):
    """
    Generate OpenMM system, topology, and positions for a given molecule.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
       The molecule for which parameters are to be generated.

    Returns
    -------
    system : simtk.openmm.System
       The OpenMM system for the molecule.
    topology : simtk.openmm.app.Topology
       The topology object corresponding to the molecule.
    positions : simtk.unit.Quantity of dimension natoms x 3 with units compatible with length
       The atomic positions of the molecule.

    TODO
    ----
    * Add option to not generate charges.

    """

    trajs, ffxmls = gaff2xml.openeye.oemols_to_ffxml([molecule])
    ff = app.ForceField(ffxmls)
    # Get OpenMM Topology.
    topology = trajs[0].top.to_openmm()
    # Create OpenMM System object.
    system = ff.createSystem(topology)
    # Create positions.
    natoms = molecule.NumAtoms()
    positions = unit.Quantity(np.zeros([natoms,3], np.float32), unit.angstrom)
    for atom in molecule.GetAtoms():
        (x, y, z) = molecule.GetCoords(atom)
        index = atom.GetIdx()
        positions[index,0] = x * unit.angstrom
        positions[index,1] = y * unit.angstrom
        positions[index,2] = z * unit.angstrom
    # Return results.
    return [system, topology, positions]

ONE_4PI_EPS0 = 138.935456 # OpenMM constant for Coulomb interactions (openmm/platforms/reference/include/SimTKOpenMMRealType.h) in OpenMM units

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
        # TODO: Create this correctly.
        energy_expression = "0.0"
        custom_force = mm.CustomNonbondedForce(energy_expression)
        custom_force.addGlobalParameter('alchemical_lambda', 0.0)
        custom_force.addGlobalParameter('alchemical_variant', 0.0)
        custom_force.addPerParticleParameter('variant')
        custom_force.addPerParticleParameter('charge')
        custom_force.addPerParticleParameter('sigma')
        custom_force.addPerParticleParameter('epsilon')
        system.addForce(custom_force)
        forces['CustomNonbondedForce'] = custom_force

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

    print system.getNumParticles(), forces['NonbondedForce'].getNumParticles()

    return mapping

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

    # Copy molecules so as not to accidentally overwrite them.
    molecules = [ molecule.CreateCopy() for molecule in molecules ]

    # Normalize molecules.
    # TODO: May need to do more normalization/cleanup here.
    for molecule in molecules:
        oe.OEPerceiveChiral(molecule)

    # Make copies of environment to not destroy original objects.
    system = copy.deepcopy(system)
    topology = copy.deepcopy(topology)
    positions = copy.deepcopy(positions)

    # Parameterize small molecules with GAFF, replacing atom and bond types in OEMol object with GAFF types.
    import oldmmtools
    gaff_molecules = list()
    for (index, molecule) in enumerate(molecules):
        print "MOLECULE:"
        print molecule
        amber_prmtop_filename = 'molecule-%05d.prmtop' % index
        amber_inpcrd_filename = 'molecule-%05d.inpcrd' % index
        amber_off_filename = 'molecule-%05d.off' % index
        # Parameterize molecule for AMBER (currently using old machinery for convenience)
        # TODO: Replace this with gaff2xml stuff
        print "Parameterizing molecule %d / %d (%s) for GAFF with antechamber..." % (index, len(molecules), molecule.GetTitle())
        oldmmtools.parameterizeForAmber(molecule, amber_prmtop_filename, amber_inpcrd_filename, charge_model=None, offfile=amber_off_filename)
        # Read in the molecule with GAFF atom and bond types
        print "Overwriting OEMol with GAFF atom and bond types..."
        molecule = oldmmtools.loadGAFFMolecule(molecule, amber_off_filename)
        gaff_molecules.append(molecule)

    # Determine common substructure.
    print "Determining common substructure..."
    core = oldmmtools.determineCommonSubstructure(gaff_molecules, verbose=True)

    # Find RMS-fit charges for common intermediate.
    print "Determining RMS-fit charges for common intermediate..."
    core = oldmmtools.determineMinimumRMSCharges(core, molecules)

    # DEBUG
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

if __name__ == '__main__':
    # Create list of molecules.
    molecule_names = ['benzene', 'toluene', 'methoxytoluene']
    molecules = [ create_molecule(name) for name in molecule_names ]

    # Write molecules to mol2 files for ease of debugging.
    for (index, molecule) in enumerate(molecules):
        gaff2xml.openeye.molecule_to_mol2(molecule, tripos_mol2_filename='molecule-%05d.mol2' % index)

    # Create an OpenMM system to represent the environment.
    molecule = create_molecule('benzene')
    [system, topology, positions] = generate_openmm_system(molecule)
    # Be sure to translate positions.
    positions[:,2] += 15.0 * unit.angstroms

    # Add other molecules to the system.
    [system, topology, positions] = create_merged_topology(system, topology, positions, molecules)

    # Write.
    natoms = system.getNumParticles()
    for (index, atom) in enumerate(topology.atoms()):
        print '%8d %8s %8s %8s %8s %8s %8.3f %8.3f %8.3f' % (index, atom.name, atom.residue.chain.index, atom.element.name, atom.residue.index, atom.residue.name, positions[index,0]/unit.angstroms, positions[index,1]/unit.angstroms, positions[index,2]/unit.angstroms)
    app.PDBFile.writeFile(topology, positions, file=open('initial.pdb','w'))

    # Create an OpenMM test simulation.
    temperature = 300.0 * unit.kelvin
    collision_rate = 20.0 / unit.picoseconds
    timestep = 1.0 * unit.femtoseconds
    integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = mm.Context(system, integrator)
    context.setParameter('alchemical_variant', 1) # Select variant index.
    context.setPositions(positions)
    niterations = 100
    filename = 'trajectory.pdb'
    print "Writing out trajectory to %s ..." % filename
    outfile = open(filename, 'w')
    app.PDBFile.writeHeader(topology, file=outfile)
    for iteration in range(niterations):
        lambda_value = 1.0 - float(iteration) / float(niterations - 1)
        context.setParameter('alchemical_lambda', lambda_value)
        integrator.step(100)
        state = context.getState(getPositions=True, getEnergy=True)
        print "Iteration %5d / %5d : lambda %8.5f : potential %8.3f kcal/mol" % (iteration, niterations, lambda_value, state.getPotentialEnergy() / unit.kilocalories_per_mole)
        positions = state.getPositions()
        app.PDBFile.writeModel(topology, positions, file=outfile, modelIndex=(iteration+1))
    app.PDBFile.writeFooter(topology, file=outfile)
    outfile.close()
    del context, integrator



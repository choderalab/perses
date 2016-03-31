from __future__ import print_function
"""
Test systems for perses automated design.

Examples
--------

Alanine dipeptide in various environments (vacuum, implicit, explicit):

>>> from perses.tests.testsystems import AlaninDipeptideSAMS
>>> testsystem = AlanineDipeptideTestSystem()
>>> system_generator = testsystem.system_generator['explicit']
>>> sams_sampler = testsystem.sams_sampler['explicit']

TODO
----
* Have all PersesTestSystem subclasses automatically subjected to a battery of tests.
* Add short descriptions to each class through a class property.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################
from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
from functools import partial
from pkg_resources import resource_filename
from openeye import oechem, oeshape, oeomega
from openmmtools import testsystems
from perses.tests.utils import sanitizeSMILES
import copy

################################################################################
# CONSTANTS
################################################################################

from perses.samplers.thermodynamics import kB

################################################################################
# TEST SYSTEMS
################################################################################

class PersesTestSystem(object):
    """
    Create a consistent set of samplers useful for testing.

    Properties
    ----------
    environments : list of str
        Available environments
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments
    designer : MultiTargetDesign sampler
        Example MultiTargetDesign sampler

    """
    def __init__(self):
        self.environments = list()
        self.topologies = dict()
        self.positions = dict()
        self.system_generators = dict()
        self.proposal_engines = dict()
        self.thermodynamic_states = dict()
        self.mcmc_samplers = dict()
        self.exen_samplers = dict()
        self.sams_samplers = dict()
        self.designer = None

class AlanineDipeptideTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for testing PointMutationEngine on alanine dipeptide in various solvents.
    This is useful for testing a variety of components.

    Properties
    ----------
    environments : list of str
        Available environments: ['vacuum', 'explicit', 'implicit']
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments
    designer : MultiTargetDesign sampler
        Example MultiTargetDesign sampler for implicit solvent hydration free energies

    Examples
    --------

    >>> from perses.tests.testsystems import AlanineDipeptideTestSystem
    >>> testsystem = AlanineDipeptideTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['implicit']

    """
    def __init__(self):
        super(AlanineDipeptideTestSystem, self).__init__()
        environments = ['explicit', 'implicit', 'vacuum']

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator(['amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=False)
        system_generators['implicit'] = SystemGenerator(['amber99sbildn.xml', 'amber99_obc.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : app.OBC2, 'constraints' : None },
            use_antechamber=False)
        system_generators['vacuum'] = SystemGenerator(['amber99sbildn.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=False)

        # Create peptide in solvent.
        from openmmtools.testsystems import AlanineDipeptideExplicit, AlanineDipeptideImplicit, AlanineDipeptideVacuum
        from pkg_resources import resource_filename
        pdb_filename = resource_filename('openmmtools', 'data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb')
        from simtk.openmm.app import PDBFile
        topologies = dict()
        positions = dict()
        pdbfile = PDBFile(pdb_filename)
        topologies['vacuum'] = pdbfile.getTopology()
        positions['vacuum'] = pdbfile.getPositions(asNumpy=True)
        topologies['implicit'] = pdbfile.getTopology()
        positions['implicit'] = pdbfile.getPositions(asNumpy=True)

        # Create molecule in explicit solvent.
        modeller = app.Modeller(topologies['vacuum'], positions['vacuum'])
        modeller.addSolvent(system_generators['explicit'].getForceField(), model='tip3p', padding=9.0*unit.angstrom)
        topologies['explicit'] = modeller.getTopology()
        positions['explicit'] = modeller.getPositions()

        # Set up the proposal engines.
        from perses.rjmc.topology_proposal import PointMutationEngine
        proposal_metadata = { 'ffxmls' : ['amber99sbildn.xml'] }
        proposal_engines = dict()
        chain_id = '1'
        allowed_mutations = [[('2','ALA')],[('2','VAL')],[('2','LEU')],[('2','PHE')]]
        for environment in environments:
            proposal_engines[environment] = PointMutationEngine(system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        from perses.samplers.thermodynamics import ThermodynamicState
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        thermodynamic_states['explicit'] = ThermodynamicState(system=systems['explicit'], temperature=temperature, pressure=pressure)
        thermodynamic_states['implicit'] = ThermodynamicState(system=systems['implicit'], temperature=temperature)
        thermodynamic_states['vacuum']   = ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment == 'explicit':
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state)
            mcmc_samplers[environment].nsteps = 5 # reduce number of steps for testing
            mcmc_samplers[environment].verbose = True
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], options={'nsteps':5})
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment])
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['implicit'] : 1.0, sams_samplers['vacuum'] : -1.0 }
        designer = MultiTargetDesign(target_samplers)
        designer.verbose = True

        # Store things.
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.systems = systems
        self.system_generators = system_generators
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

def load_via_pdbfixer(filename=None, pdbid=None):
    """
    Load a PDB file via PDBFixer, keeping all heterogens and building in protons for any crystallographic waters.
    """
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=filename, pdbid=pdbid)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return [fixer.topology, fixer.positions]

class T4LysozymeTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for testing PointMutationEngine on T4 lysozyme in various solvents.
    Wild Type is T4 L99A

    Properties
    ----------
    environments : list of str
        Available environments: ['vacuum', 'explicit', 'implicit']
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments
    designer : MultiTargetDesign sampler
        Example MultiTargetDesign sampler for implicit solvent hydration free energies

    Examples
    --------

    >>> from perses.tests.testsystems import T4LysozymeTestSystem
    >>> testsystem = T4LysozymeTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['implicit']

    """
    def __init__(self):
        super(T4LysozymeTestSystem, self).__init__()
#        environments = ['explicit-complex', 'explicit-receptor', 'implicit-complex', 'implicit-receptor', 'vacuum-complex', 'vacuum-receptor']
        environments = ['explicit-complex', 'explicit-receptor', 'vacuum-complex', 'vacuum-receptor']


        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename,'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True)
        system_generators['explicit-complex'] = system_generators['explicit']
        system_generators['explicit-receptor'] = system_generators['explicit']
        system_generators['implicit'] = SystemGenerator([gaff_xml_filename,'amber99sbildn.xml', 'amber99_obc.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : app.OBC2, 'constraints' : None },
            use_antechamber=True)
        system_generators['implicit-complex'] = system_generators['implicit']
        system_generators['implicit-receptor'] = system_generators['implicit']
        system_generators['vacuum'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True)
        system_generators['vacuum-complex'] = system_generators['vacuum']
        system_generators['vacuum-receptor'] = system_generators['vacuum']

        # Create receptor in solvent.
        from pkg_resources import resource_filename
        pdb_filename = resource_filename('perses', 'data/181L.pdb')
        import pdbfixer
        from simtk.openmm.app import PDBFile, Modeller
        topologies = dict()
        positions = dict()
        [fixer_topology, fixer_positions] = load_via_pdbfixer(pdb_filename)
        modeller = Modeller(fixer_topology, fixer_positions)

        residues_to_delete = [ residue for residue in modeller.getTopology().residues() if residue.name in ['HED','CL','HOH'] ]
        modeller.delete(residues_to_delete)

        receptor_modeller = copy.deepcopy(modeller)
        ligand_modeller = copy.deepcopy(modeller)

        for chain in receptor_modeller.getTopology().chains():
            pass
        chains_to_delete = [chain]
        receptor_modeller.delete(chains_to_delete)
        topologies['receptor'] = receptor_modeller.getTopology()
        positions['receptor'] = receptor_modeller.getPositions()

        for chain in ligand_modeller.getTopology().chains():
            break
        chains_to_delete = [chain]
        ligand_modeller.delete(chains_to_delete)
        for residue in ligand_modeller.getTopology().residues():
            if residue.name == 'BNZ':
                break

        from openmoltools import forcefield_generators
        from utils import extractPositionsFromOEMOL, giveOpenmmPositionsToOEMOL
        # create OEMol version of benzene
        mol = oechem.OEMol()
        mol.SetTitle('BNZ')
        oechem.OESmilesToMol(mol,'C1=CC=CC=C1')
        # put positions from pdb into OEMol
        print(ligand_modeller.positions)
        giveOpenmmPositionsToOEMOL(ligand_modeller.positions, mol)
        oechem.OEAddExplicitHydrogens(mol)
        oechem.OETriposAtomNames(mol)
        oechem.OETriposBondTypeNames(mol)
        omega = oeomega.OEOmega()
        omega.SetStrictStereo(False)
        omega.SetMaxConfs(1)
        # require omega to preserve coordinates assigned to carbons (from pdb)
        omega.SetFromCT(False)
        omega.SetFixSmarts('[c]')
        omega(mol)
        new_positions = extractPositionsFromOEMOL(mol)
        print(new_positions)
        new_residue = forcefield_generators.generateTopologyFromOEMol(mol)

        modeller = copy.deepcopy(receptor_modeller)
        modeller.add(new_residue, new_positions)
        topologies['complex'] = modeller.getTopology()
        positions['complex'] = modeller.getPositions()
        with open('omegaed_181L.pdb','w') as fo:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, fo, keepIds=True)

        # Create all environments.
        for environment in ['implicit', 'vacuum']:
            for component in ['receptor', 'complex']:
                topologies[environment + '-' + component] = topologies[component]
                positions[environment + '-' + component] = positions[component]

        # Set up in explicit solvent.
        for component in ['receptor', 'complex']:
            modeller = app.Modeller(topologies[component], positions[component])
            modeller.addSolvent(system_generators['explicit'].getForceField(), model='tip3p', padding=9.0*unit.angstrom)
            atoms = list(modeller.topology.atoms())
            print('Solvated %s has %s atoms' % (component, len(atoms)))
            topologies['explicit' + '-' + component] = modeller.getTopology()
            positions['explicit' + '-' + component] = modeller.getPositions()

        # Set up the proposal engines.
        allowed_mutations = [
            [('99','GLY')],
            [('102','GLN')],
            [('102','HIS')],
            [('102','GLU')],
            [('102','LEU')],
            [('153','ALA')],
            [('108','VAL')],
            [('99','GLY'),('108','VAL')]
        ]
        from perses.rjmc.topology_proposal import PointMutationEngine
        proposal_metadata = { 'ffxmls' : ['amber99sbildn.xml'] }
        proposal_engines = dict()
        chain_id = 'A'
        for environment in environments:
            proposal_engines[environment] = PointMutationEngine(system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems
        systems = dict()
        for environment in environments:
            print(environment)
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        from perses.samplers.thermodynamics import ThermodynamicState
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        for component in ['receptor', 'complex']:
            thermodynamic_states['explicit' + '-' + component] = ThermodynamicState(system=systems['explicit' + '-' + component], temperature=temperature, pressure=pressure)
            #thermodynamic_states['implicit' + '-' + component] = ThermodynamicState(system=systems['implicit' + '-' + component], temperature=temperature)
            thermodynamic_states['vacuum' + '-' + component]   = ThermodynamicState(system=systems['vacuum' + '-' + component], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment[0:8] == 'explicit':
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state)
            mcmc_samplers[environment].nsteps = 5 # reduce number of steps for testing
            mcmc_samplers[environment].verbose = True
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], options={'nsteps':5})
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment])
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['explicit-complex'] : 1.0, sams_samplers['explicit-receptor'] : -1.0 }
        designer = MultiTargetDesign(target_samplers)
        designer.verbose = True

        # Store things.
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.systems = systems
        self.system_generators = system_generators
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

        minimize(self)

class MybTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for testing PointMutationEngine on Myb:peptide interaction in various solvents.

    Properties
    ----------
    environments : list of str
        Available environments: ['vacuum', 'explicit', 'implicit']
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments
    designer : MultiTargetDesign sampler
        Example MultiTargetDesign sampler for implicit solvent hydration free energies

    Examples
    --------

    >>> from perses.tests.testsystems import MybTestSystem
    >>> testsystem = MybTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum-peptide'].build_system(testsystem.topologies['vacuum-peptide'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['implicit-peptide']

    """
    def __init__(self):
        super(MybTestSystem, self).__init__()
        environments = ['explicit-complex', 'explicit-peptide', 'implicit-complex', 'implicit-peptide', 'vacuum-complex', 'vacuum-peptide']

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator(['amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=False)
        system_generators['explicit-complex'] = system_generators['explicit']
        system_generators['explicit-peptide'] = system_generators['explicit']
        system_generators['implicit'] = SystemGenerator(['amber99sbildn.xml', 'amber99_obc.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : app.OBC2, 'constraints' : None },
            use_antechamber=False)
        system_generators['implicit-complex'] = system_generators['implicit']
        system_generators['implicit-peptide'] = system_generators['implicit']
        system_generators['vacuum'] = SystemGenerator(['amber99sbildn.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=False)
        system_generators['vacuum-complex'] = system_generators['vacuum']
        system_generators['vacuum-peptide'] = system_generators['vacuum']

        # Create peptide in solvent.
        from pkg_resources import resource_filename
        pdb_filename = resource_filename('perses', 'data/1sb0.pdb')
        import pdbfixer
        from simtk.openmm.app import PDBFile, Modeller
        topologies = dict()
        positions = dict()
        #pdbfile = PDBFile(pdb_filename)
        [fixer_topology, fixer_positions] = load_via_pdbfixer(pdb_filename)
        topologies['complex'] = fixer_topology
        positions['complex'] = fixer_positions
        modeller = Modeller(topologies['complex'], positions['complex'])
        chains_to_delete = [ chain for chain in modeller.getTopology().chains() if chain.id == 'A' ] # remove chain A
        modeller.delete(chains_to_delete)
        topologies['peptide'] = modeller.getTopology()
        positions['peptide'] = modeller.getPositions()

        # Create all environments.
        for environment in ['implicit', 'vacuum']:
            for component in ['peptide', 'complex']:
                topologies[environment + '-' + component] = topologies[component]
                positions[environment + '-' + component] = positions[component]

        # Set up in explicit solvent.
        for component in ['peptide', 'complex']:
            modeller = app.Modeller(topologies[component], positions[component])
            modeller.addSolvent(system_generators['explicit'].getForceField(), model='tip3p', padding=9.0*unit.angstrom)
            topologies['explicit' + '-' + component] = modeller.getTopology()
            positions['explicit' + '-' + component] = modeller.getPositions()

        # Set up the proposal engines.
        allowed_mutations = list()
        for resid in ['91', '99', '103', '105']:
            for resname in ['ALA', 'LEU', 'VAL', 'PHE', 'CYS', 'THR', 'TRP', 'TYR', 'GLU', 'ASP', 'LYS', 'ARG', 'ASN']:
                allowed_mutations.append([(resid, resname)])
        from perses.rjmc.topology_proposal import PointMutationEngine
        proposal_metadata = { 'ffxmls' : ['amber99sbildn.xml'] }
        proposal_engines = dict()
        chain_id
        for environment in environments:
            proposal_engines[environment] = PointMutationEngine(system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        from perses.samplers.thermodynamics import ThermodynamicState
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        for component in ['peptide', 'complex']:
            thermodynamic_states['explicit' + '-' + component] = ThermodynamicState(system=systems['explicit' + '-' + component], temperature=temperature, pressure=pressure)
            thermodynamic_states['implicit' + '-' + component] = ThermodynamicState(system=systems['implicit' + '-' + component], temperature=temperature)
            thermodynamic_states['vacuum' + '-' + component]   = ThermodynamicState(system=systems['vacuum' + '-' + component], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment[0:8] == 'explicit':
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state)
            mcmc_samplers[environment].nsteps = 5 # reduce number of steps for testing
            mcmc_samplers[environment].verbose = True
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], options={'nsteps':5})
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment])
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['implicit-complex'] : 1.0, sams_samplers['implicit-peptide'] : -1.0 }
        designer = MultiTargetDesign(target_samplers)
        designer.verbose = True

        # Store things.
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.systems = systems
        self.system_generators = system_generators
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

class AblImatinibTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for testing PointMutationEngine on Abl:imatinib.

    Properties
    ----------
    environments : list of str
        Available environments: ['vacuum', 'explicit', 'implicit']
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments
    designer : MultiTargetDesign sampler
        Example MultiTargetDesign sampler for implicit solvent hydration free energies

    Examples
    --------

    >>> from perses.tests.testsystems import AblImatinibTestSystem
    >>> testsystem = AblImatinibTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum-inhibitor'].build_system(testsystem.topologies['vacuum-inhibitor'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['vacuum-inhibitor']

    """
    def __init__(self):
        super(AblImatinibTestSystem, self).__init__()
        solvents = ['vacuum', 'explicit'] # TODO: Add 'implicit' once GBSA parameterization for small molecules is working
#        solvents = ['vacuum'] # DEBUG
        components = ['receptor', 'complex'] # TODO: Add 'ATP:kinase' complex to enable resistance design
        padding = 9.0*unit.angstrom
        explicit_solvent_model = 'tip3p'
        setup_path = 'data/abl-imatinib'
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres

        # Construct list of all environments
        environments = list()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                environments.append(environment)

        # Create a system generator for desired forcefields
        from perses.rjmc.topology_proposal import SystemGenerator
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True)
        system_generators['implicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'amber99_obc.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : app.OBC2, 'constraints' : None },
            use_antechamber=True)
        system_generators['vacuum'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True)
        # Copy system generators for all environments
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                system_generators[environment] = system_generators[solvent]

        # Load topologies and positions for all components
        from simtk.openmm.app import PDBFile, Modeller
        topologies = dict()
        positions = dict()
        for component in components:
            pdb_filename = resource_filename('perses', os.path.join(setup_path, '%s.pdb' % component))
            pdbfile = PDBFile(pdb_filename)
            topologies[component] = pdbfile.topology
            positions[component] = pdbfile.positions

        # Construct positions and topologies for all solvent environments
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                if solvent == 'explicit':
                    # Create MODELLER object.
                    modeller = app.Modeller(topologies[component], positions[component])
                    modeller.addSolvent(system_generators[solvent].getForceField(), model='tip3p', padding=9.0*unit.angstrom)
                    topologies[environment] = modeller.getTopology()
                    positions[environment] = modeller.getPositions()
                else:
                    environment = solvent + '-' + component
                    topologies[environment] = topologies[component]
                    positions[environment] = positions[component]

        # Set up resistance mutation proposal engines
        allowed_mutations = list()
        # TODO: Expand this beyond the ATP binding site
        for resid in ['22', '37', '52', '55', '65', '81', '125', '128', '147', '148']:
            for resname in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']:
                allowed_mutations.append([(resid, resname)])
        from perses.rjmc.topology_proposal import PointMutationEngine
        proposal_metadata = { 'ffxmls' : ['amber99sbildn.xml'] }
        proposal_engines = dict()
        chain_id = 'A'
        for solvent in solvents:
            for component in ['complex', 'receptor']: # Mutations only apply to components that contain the kinase
                environment = solvent + '-' + component
                proposal_engines[environment] = PointMutationEngine(system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems ror all environments
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Create SAMS samplers
        from perses.samplers.thermodynamics import ThermodynamicState
        from perses.samplers.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        thermodynamic_states = dict()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])

                if solvent == 'explicit':
                    thermodynamic_state = ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                    sampler_state = SamplerState(system=systems[environment], positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
                else:
                    thermodynamic_state = ThermodynamicState(system=systems[environment], temperature=temperature)
                    sampler_state = SamplerState(system=systems[environment], positions=positions[environment])

                mcmc_samplers[environment] = MCMCSampler(thermodynamic_state, sampler_state)
                mcmc_samplers[environment].nsteps = 5 # reduce number of steps for testing
                mcmc_samplers[environment].verbose = True
                exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], options={'nsteps':5})
                exen_samplers[environment].verbose = True
                sams_samplers[environment] = SAMSSampler(exen_samplers[environment])
                sams_samplers[environment].verbose = True
                thermodynamic_states[environment] = thermodynamic_state

        # Create test MultiTargetDesign sampler.
        # TODO: Replace this with inhibitor:kinase and ATP:kinase ratio
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['vacuum-complex'] : 1.0, sams_samplers['vacuum-receptor'] : -1.0 }
        designer = MultiTargetDesign(target_samplers)
        designer.verbose = True

        # Store things.
        self.components = components
        self.solvents = solvents
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.systems = systems
        self.system_generators = system_generators
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

        # This system must currently be minimized.
        minimize(self)

def minimize(testsystem):
    """
    Minimize all structures in test system.

    Parameters
    ----------
    testystem : PersesTestSystem
        The testsystem to minimize.

    """
    for environment in testsystem.environments:
        print("Minimizing '%s'..." % environment)
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = openmm.Context(testsystem.systems[environment], integrator)
        context.setPositions(testsystem.positions[environment])
        print ("Initial energy is %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
        TOL = 1.0
        MAX_STEPS = 50
        openmm.LocalEnergyMinimizer.minimize(context, TOL, MAX_STEPS)
        print ("Final energy is   %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
        # Update positions.
        testsystem.positions[environment] = context.getState(getPositions=True).getPositions(asNumpy=True)
        # Update sampler states.
        testsystem.mcmc_samplers[environment].sampler_state.positions = testsystem.positions[environment]
        # Clean up.
        del context, integrator

class SmallMoleculeLibraryTestSystem(PersesTestSystem):
    """
    Create a consistent set of samplers useful for testing SmallMoleculeProposalEngine on alkanes in various solvents.
    This is useful for testing a variety of components.

    Properties
    ----------
    environments : list of str
        Available environments: ['vacuum', 'explicit']
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments
    designer : MultiTargetDesign sampler
        Example MultiTargetDesign sampler for explicit solvent hydration free energies
    molecules : list
        Molecules in library. Currently only SMILES format is supported.

    Examples
    --------

    >>> from perses.tests.testsystems import AlkanesTestSystem
    >>> testsystem = AlkanesTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['explicit']

    """
    def __init__(self):
        super(SmallMoleculeLibraryTestSystem, self).__init__()
        # Expand molecules without explicit stereochemistry and make canonical isomeric SMILES.
        molecules = sanitizeSMILES(self.molecules)
        environments = ['explicit', 'vacuum']

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None })
        system_generators['vacuum'] = SystemGenerator([gaff_xml_filename],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None })

        #
        # Create topologies and positions
        #
        topologies = dict()
        positions = dict()

        from openmoltools import forcefield_generators
        forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
        forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

        # Create molecule in vacuum.
        from perses.tests.utils import createOEMolFromSMILES, extractPositionsFromOEMOL
        smiles = molecules[0] # current sampler state
        molecule = createOEMolFromSMILES(smiles)
        topologies['vacuum'] = forcefield_generators.generateTopologyFromOEMol(molecule)
        positions['vacuum'] = extractPositionsFromOEMOL(molecule)

        # Create molecule in solvent.
        modeller = app.Modeller(topologies['vacuum'], positions['vacuum'])
        modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
        topologies['explicit'] = modeller.getTopology()
        positions['explicit'] = modeller.getPositions()

        # Set up the proposal engines.
        from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
        proposal_metadata = { }
        proposal_engines = dict()
        for environment in environments:
            proposal_engines[environment] = SmallMoleculeSetProposalEngine(molecules, system_generators[environment])

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        from perses.samplers.thermodynamics import ThermodynamicState
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        thermodynamic_states['explicit'] = ThermodynamicState(system=systems['explicit'], temperature=temperature, pressure=pressure)
        thermodynamic_states['vacuum']   = ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment == 'explicit':
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state)
            mcmc_samplers[environment].nsteps = 5 # reduce number of steps for testing
            mcmc_samplers[environment].verbose = True
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], options={'nsteps':5})
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment])
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['explicit'] : 1.0, sams_samplers['vacuum'] : -1.0 }
        designer = MultiTargetDesign(target_samplers)

        # Store things.
        self.molecules = molecules
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.system_generators = system_generators
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

class AlkanesTestSystem(SmallMoleculeLibraryTestSystem):
    """
    Library of small alkanes in various solvent environments.
    """
    def __init__(self):
        self.molecules = ['CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC']
        super(AlkanesTestSystem, self).__init__()

class KinaseInhibitorsTestSystem(SmallMoleculeLibraryTestSystem):
    """
    Library of clinical kinase inhibitors in various solvent environments.
    """
    def __init__(self):
        # Read SMILES from CSV file of clinical kinase inhibitors.
        from pkg_resources import resource_filename
        smiles_filename = resource_filename('perses', 'data/clinical-kinase-inhibitors.csv')
        import csv
        molecules = list()
        with open(smiles_filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                name = row[0]
                smiles = row[1]
                molecules.append(smiles)
        self.molecules = molecules
        # Intialize
        super(KinaseInhibitorsTestSystem, self).__init__()

class T4LysozymeInhibitorsTestSystem(SmallMoleculeLibraryTestSystem):
    """
    Library of T4 lysozyme L99A inhibitors in various solvent environments.
    """
    def read_smiles(self, filename):
        import csv
        molecules = list()
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for row in csvreader:
                name = row[0]
                smiles = row[1]
                reference = row[2]
                molecules.append(smiles)
        return molecules

    def __init__(self):
        # Read SMILES from CSV file of clinical kinase inhibitors.
        from pkg_resources import resource_filename
        molecules = list()
        molecules += self.read_smiles(resource_filename('perses', 'data/L99A-binders.txt'))
        molecules += self.read_smiles(resource_filename('perses', 'data/L99A-non-binders.txt'))
        self.molecules = molecules
        # Intialize
        super(T4LysozymeInhibitorsTestSystem, self).__init__()

class ValenceSmallMoleculeLibraryTestSystem(PersesTestSystem):
    """
    Create a consistent set of samplers useful for testing SmallMoleculeProposalEngine on alkanes with a valence-only forcefield.

    Properties
    ----------
    environments : list of str
        Available environments: ['vacuum']
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments
    designer : MultiTargetDesign sampler
        Example MultiTargetDesign sampler for explicit solvent hydration free energies
    molecules : list
        Molecules in library. Currently only SMILES format is supported.

    Examples
    --------

    >>> from perses.tests.testsystems import ValenceSmallMoleculeLibraryTestSystem
    >>> testsystem = ValenceSmallMoleculeLibraryTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['vacuum']

    """
    def __init__(self):
        super(ValenceSmallMoleculeLibraryTestSystem, self).__init__()
        molecules = ['CC', 'CCC','CCCC', 'CCCCC','CC(C)CC', 'CC(CC)CC', 'C(C)CCC', 'C(CC)CCC']
        environments = ['vacuum']

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff-valence-only.xml')
        system_generators['vacuum'] = SystemGenerator([gaff_xml_filename],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None })

        #
        # Create topologies and positions
        #
        topologies = dict()
        positions = dict()

        from openmoltools import forcefield_generators
        forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
        forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

        # Create molecule in vacuum.
        from perses.tests.utils import createOEMolFromSMILES, extractPositionsFromOEMOL
        smiles = molecules[0] # current sampler state
        molecule = createOEMolFromSMILES(smiles)
        topologies['vacuum'] = forcefield_generators.generateTopologyFromOEMol(molecule)
        positions['vacuum'] = extractPositionsFromOEMOL(molecule)

        # Set up the proposal engines.
        from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
        proposal_metadata = { }
        proposal_engines = dict()
        for environment in environments:
            proposal_engines[environment] = SmallMoleculeSetProposalEngine(molecules, system_generators[environment])

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        from perses.samplers.thermodynamics import ThermodynamicState
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        thermodynamic_states['vacuum']   = ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment == 'explicit':
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = SamplerState(system=systems[environment], positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state)
            mcmc_samplers[environment].nsteps = 500 # reduce number of steps for testing
            mcmc_samplers[environment].verbose = True
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], options={'nsteps':0})
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment])
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['vacuum'] : 1.0, sams_samplers['vacuum'] : -1.0 }
        designer = MultiTargetDesign(target_samplers)

        # Store things.
        self.molecules = molecules
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.system_generators = system_generators
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

def check_topologies(testsystem):
    """
    Check that all SystemGenerators can build systems for their corresponding Topology objects.
    """
    for environment in testsystem.environments:
        topology = testsystem.topologies[environment]
        try:
            testsystem.system_generators[environment].build_system(topology)
        except Exception as e:
            msg = str(e)
            msg += '\n'
            msg += "topology for environment '%s' cannot be built into a system" % environment
            from perses.tests.utils import show_topology
            show_topology(topology)
            raise Exception(msg)

def checktestsystem(testsystem_class):
    # Instantiate test system.
    testsystem = testsystem_class()
    # Check topologies
    check_topologies(testsystem)

def test_testsystems():
    """
    Test instantiation of all test systems.
    """
    testsystem_names = ['T4LysozymeInhibitorsTestSystem', 'KinaseInhibitorsTestSystem', 'AlkanesTestSystem', 'AlanineDipeptideTestSystem']
    niterations = 5 # number of iterations to run
    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        f = partial(checktestsystem, testsystem_class)
        f.description = "Testing %s" % (testsystem_name)
        yield f

def run_t4():
    """
    Run T4 lysozyme test system.
    """
    testsystem = T4LysozymeTestSystem()
    solvent = 'explicit'
    for component in ['complex', 'receptor']:
        testsystem.exen_samplers[solvent + '-' + component].pdbfile = open('t4-' + component + '.pdb', 'w')
        testsystem.exen_samplers[solvent + '-' + component].options={'nsteps':0} # instantaneous MC
        testsystem.mcmc_samplers[solvent + '-' + component].verbose = True # use fewer MD steps to speed things up
        testsystem.mcmc_samplers[solvent + '-' + component].nsteps = 50 # use fewer MD steps to speed things up
        testsystem.sams_samplers[solvent + '-' + component].run(niterations=5)
    testsystem.designer.verbose = True
    testsystem.designer.run(niterations=5)

def run_myb():
    """
    Run myb test system.
    """
    testsystem = MybTestSystem()
    solvent = 'implicit'
    testsystem.exen_samplers[solvent + '-peptide'].pdbfile = open('myb-vacuum.pdb', 'w')
    testsystem.exen_samplers[solvent + '-complex'].pdbfile = open('myb-complex.pdb', 'w')
    testsystem.exen_samplers[solvent + '-complex'].options={'nsteps':0}
    testsystem.exen_samplers[solvent + '-peptide'].options={'nsteps':0}
    testsystem.mcmc_samplers[solvent + '-complex'].nsteps = 50
    testsystem.mcmc_samplers[solvent + '-peptide'].nsteps = 50
    testsystem.sams_samplers[solvent + '-complex'].run(niterations=5)
    #testsystem.designer.verbose = True
    #testsystem.designer.run(niterations=500)
    #testsystem.exen_samplers[solvent + '-peptide'].verbose=True
    #testsystem.exen_samplers[solvent + '-peptide'].run(niterations=100)

def run_abl_imatinib():
    """
    Run abl test system.
    """
    testsystem = AblImatinibTestSystem()
    #for environment in testsystem.environments:
    for environment in ['vacuum-complex']:
        print(environment)
        testsystem.exen_samplers[environment].pdbfile = open('abl-imatinib-%s.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].geometry_pdbfile = open('abl-imatinib-%s-geometry-proposals.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].options={'nsteps':20000, 'timestep' : 1.0 * unit.femtoseconds}
        testsystem.exen_samplers[environment].accept_everything = False # accept everything that doesn't lead to NaN for testing
        testsystem.mcmc_samplers[environment].nsteps = 20000
        testsystem.mcmc_samplers[environment].timestep = 1.0 * unit.femtoseconds
        #testsystem.mcmc_samplers[environment].run(niterations=5)
        testsystem.exen_samplers[environment].run(niterations=100)
        #testsystem.sams_samplers[environment].run(niterations=5)

    #testsystem.designer.verbose = True
    #testsystem.designer.run(niterations=500)
    #testsystem.exen_samplers[solvent + '-peptide'].verbose=True
    #testsystem.exen_samplers[solvent + '-peptide'].run(niterations=100)

def run_kinase_inhibitors():
    """
    Run kinase inhibitors test system.
    """
    testsystem = KinaseInhibitorsTestSystem()
    environment = 'vacuum'
    testsystem.exen_samplers[environment].pdbfile = open('kinase-inhibitors-vacuum.pdb', 'w')
    testsystem.exen_samplers[environment].geometry_pdbfile = open('kinase-inhibitors-%s-geometry-proposals.pdb' % environment, 'w')
    testsystem.exen_samplers[environment].options={'nsteps':0}
    testsystem.mcmc_samplers[environment].nsteps = 50
    testsystem.exen_samplers[environment].geometry_engine.write_proposal_pdb = True # write proposal PDBs
    testsystem.sams_samplers[environment].run(niterations=100)

def run_valence_system():
    """
    Run valence molecules test system.
    """
    testsystem = ValenceSmallMoleculeLibraryTestSystem()
    environment = 'vacuum'
    testsystem.exen_samplers[environment].pdbfile = open('valence.pdb', 'w')
    testsystem.exen_samplers[environment].options={'nsteps':0}
    testsystem.mcmc_samplers[environment].nsteps = 50
    testsystem.sams_samplers[environment].run(niterations=5)

if __name__ == '__main__':
    run_t4()
    #run_valence_system()
    #run_kinase_inhibitors()
    #run_abl_imatinib()
    #run_myb()

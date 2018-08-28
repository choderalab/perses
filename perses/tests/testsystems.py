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
from openmmtools import states
from openmmtools.mcmc import MCMCSampler, LangevinSplittingDynamicsMove
from perses.tests.utils import sanitizeSMILES, canonicalize_SMILES
from perses.storage import NetCDFStorage, NetCDFStorageView
from perses.rjmc.topology_proposal import OESMILES_OPTIONS
from perses.rjmc.geometry import FFAllAngleGeometryEngine
import tempfile
import copy
from openmmtools.constants import kB

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
    def __init__(self, storage_filename=None, mode='w', ncmc_nsteps=5, mcmc_nsteps=100):
        """Create a testsystem.

        Parameters
        ----------
        storage_filename : str, optional, default=None
           If specified, bind to this storage file.
        mode : str, optional, default='w'
           File open mode, 'w' for (over)write, 'a' for append.

        """
        self.storage = None
        if storage_filename is not None:
            self.storage = NetCDFStorage(storage_filename, mode='w')
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
        self.geometry_engine = FFAllAngleGeometryEngine(metadata={})
        self._splitting = "V R O R V"
        self._timestep = 1.0*unit.femtosecond
        self._ncmc_nsteps = ncmc_nsteps
        self._mcmc_nsteps = mcmc_nsteps
        self._move = LangevinSplittingDynamicsMove(timestep=self._timestep, splitting=self._splitting)


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
    def __init__(self, constraints=app.HBonds, **kwargs):
        super(AlanineDipeptideTestSystem, self).__init__(**kwargs)
        environments = ['explicit', 'implicit', 'vacuum']
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres

        # Use sterics in proposals
        self.geometry_engine.use_sterics = True

        # Write atom-by-atom geometry output.
        self.geometry_engine.write_proposal_pdb = True
        self.geometry_engine.pdb_filename_prefix = 'geometry'

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator(['amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : constraints },
            use_antechamber=False, barostat=barostat)
        system_generators['implicit'] = SystemGenerator(['amber99sbildn.xml', 'amber99_obc.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : app.OBC2, 'constraints' : constraints },
            use_antechamber=False)
        system_generators['vacuum'] = SystemGenerator(['amber99sbildn.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : constraints },
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
        proposal_metadata = {
            'ffxmls' : ['amber99sbildn.xml'], # take sidechain definitions from this ffxml file
            'always_change' : True # don't propose self-transitions
            }
        proposal_engines = dict()
        chain_id = ' '
        allowed_mutations = [[('2','VAL')],[('2','LEU')],[('2','ILE')]]
        for environment in environments:
            proposal_engines[environment] = PointMutationEngine(topologies[environment],system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        
        thermodynamic_states = dict()
        thermodynamic_states['explicit'] = states.ThermodynamicState(system=systems['explicit'], temperature=temperature, pressure=pressure)
        thermodynamic_states['implicit'] = states.ThermodynamicState(system=systems['implicit'], temperature=temperature)
        thermodynamic_states['vacuum']   = states.ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            storage = None
            if self.storage:
                storage = NetCDFStorageView(self.storage, envname=environment)

            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment == 'explicit':
                sampler_state = states.SamplerState(positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = states.SamplerState(positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state, copy.deepcopy(self._move))
             # reduce number of steps for testing
            mcmc_samplers[environment].timestep = 1.0 * unit.femtoseconds
            
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps': 0}, storage=storage)
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['implicit'] : 1.0, sams_samplers['vacuum'] : -1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)
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

class AlanineDipeptideValenceTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for testing PointMutationEngine on alanine dipeptide in various solvents.
    Only valence terms are included---no sterics.

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
        Example MultiTargetDesign sampler for implicit solvent hydration free energies

    Examples
    --------

    >>> from perses.tests.testsystems import AlanineDipeptideValenceTestSystem
    >>> testsystem = AlanineDipeptideValenceTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum'].build_system(testsystem.topologies['vacuum'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['vacuum']

    """
    def __init__(self, **kwargs):
        super(AlanineDipeptideValenceTestSystem, self).__init__(**kwargs)
        environments = ['vacuum']

        # Write atom-by-atom geometry output.
        self.geometry_engine.write_proposal_pdb = False
        #self.geometry_engine.pdb_filename_prefix = 'geometry2'

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        from pkg_resources import resource_filename
        valence_xml_filename = resource_filename('perses', 'data/amber99sbildn-valence-only.xml')
        system_generators['vacuum'] = SystemGenerator([valence_xml_filename],
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

        # Set up the proposal engines.
        from perses.rjmc.topology_proposal import PointMutationEngine
        proposal_metadata = {
            'ffxmls' : ['amber99sbildn.xml'], # take sidechain definitions from this ffxml file
            'always_change' : True # don't propose self-transitions
            }
        proposal_engines = dict()
        chain_id = ' '
        allowed_mutations = [[('2','PHE')]]
        proposal_metadata = {"always_change":True}
        for environment in environments:
            proposal_engines[environment] = PointMutationEngine(topologies[environment],system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations, always_change=True)

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        thermodynamic_states['vacuum'] = states.ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            storage = None
            if self.storage:
                storage = NetCDFStorageView(self.storage, envname=environment)

            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            sampler_state = states.SamplerState(positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state, copy.deepcopy(self._move))
             # reduce number of steps for testing
            
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':50}, storage=storage)
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['vacuum'] : 1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)
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

class T4LysozymeMutationTestSystem(PersesTestSystem):
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
    def __init__(self, **kwargs):
        super(T4LysozymeMutationTestSystem, self).__init__(**kwargs)
#        environments = ['explicit-complex', 'explicit-receptor', 'implicit-complex', 'implicit-receptor', 'vacuum-complex', 'vacuum-receptor']
        environments = ['explicit-complex', 'explicit-receptor', 'vacuum-complex', 'vacuum-receptor']
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename,'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True, barostat=barostat)
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
        from perses.tests.utils import extractPositionsFromOEMOL, giveOpenmmPositionsToOEMOL
        import perses.rjmc.geometry as geometry
        from perses.rjmc.topology_proposal import TopologyProposal
        # create OEMol version of benzene
        mol = oechem.OEMol()
        #mol.SetTitle('BNZ') # should be set to residue.name in generateTopologyFromOEMol, not working
        oechem.OESmilesToMol(mol,'C1=CC=CC=C1')
        oechem.OEAddExplicitHydrogens(mol)
        oechem.OETriposAtomNames(mol)
        oechem.OETriposBondTypeNames(mol)

        new_residue = forcefield_generators.generateTopologyFromOEMol(mol)
        for res in new_residue.residues():
            res.name = 'BNZ'
        bnz_new_sys = system_generators['vacuum'].build_system(new_residue)
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        temperature = 300.0 * unit.kelvin
        kT = kB * temperature
        beta = 1.0/kT
        adding_hydrogen_proposal = TopologyProposal(new_topology=new_residue, new_system =bnz_new_sys, old_topology=ligand_modeller.topology, old_system =bnz_new_sys, logp_proposal = 0.0, new_to_old_atom_map = {0:0,1:1,2:2,3:3,4:4,5:5}, old_chemical_state_key='',new_chemical_state_key='')
        geometry_engine = geometry.FFAllAngleGeometryEngine()
        new_positions, logp = geometry_engine.propose(adding_hydrogen_proposal, ligand_modeller.positions, beta)

        modeller = copy.deepcopy(receptor_modeller)
        modeller.add(new_residue, new_positions)
        topologies['complex'] = modeller.getTopology()
        positions['complex'] = modeller.getPositions()

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
            proposal_engines[environment] = PointMutationEngine(topologies[environment], system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems
        systems = dict()
        for environment in environments:
            print(environment)
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        
        thermodynamic_states = dict()
        for component in ['receptor', 'complex']:
            thermodynamic_states['explicit' + '-' + component] = states.ThermodynamicState(system=systems['explicit' + '-' + component], temperature=temperature, pressure=pressure)
            #thermodynamic_states['implicit' + '-' + component] = ThermodynamicState(system=systems['implicit' + '-' + component], temperature=temperature)
            thermodynamic_states['vacuum' + '-' + component]   = states.ThermodynamicState(system=systems['vacuum' + '-' + component], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            storage = None
            if self.storage:
                storage = NetCDFStorageView(self.storage, envname=environment)

            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment[0:8] == 'explicit':
                sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = states.SamplerState(positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state, copy.deepcopy(self._move))
             # reduce number of steps for testing
            
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':self._ncmc_nsteps, 'mcmc_nsteps':self._mcmc_nsteps}, storage=storage)
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['explicit-complex'] : 1.0, sams_samplers['explicit-receptor'] : -1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)
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
    def __init__(self, **kwargs):
        super(MybTestSystem, self).__init__(**kwargs)
        environments = ['explicit-complex', 'explicit-peptide', 'implicit-complex', 'implicit-peptide', 'vacuum-complex', 'vacuum-peptide']
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres

        # Use sterics in proposals
        self.geometry_engine.use_sterics = True

        # Write atom-by-atom geometry output.
        self.geometry_engine.write_proposal_pdb = True
        self.geometry_engine.pdb_filename_prefix = 'geometry'

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
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
        proposal_metadata = {
            'ffxmls' : ['amber99sbildn.xml'], # take sidechain definitions from this ffxml file
            'always_change' : True # don't propose self-transitions
            }
        proposal_engines = dict()
        chain_id = 'B'
        for environment in environments:
            proposal_engines[environment] = PointMutationEngine(topologies[environment], system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        
        thermodynamic_states = dict()
        for component in ['peptide', 'complex']:
            thermodynamic_states['explicit' + '-' + component] = states.ThermodynamicState(system=systems['explicit' + '-' + component], temperature=temperature, pressure=pressure)
            thermodynamic_states['implicit' + '-' + component] = states.ThermodynamicState(system=systems['implicit' + '-' + component], temperature=temperature)
            thermodynamic_states['vacuum' + '-' + component]   = states.ThermodynamicState(system=systems['vacuum' + '-' + component], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            storage = None
            if self.storage:
                storage = NetCDFStorageView(self.storage, envname=environment)

            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment[0:8] == 'explicit':
                sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = states.SamplerState(positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state, copy.deepcopy(self._move))
            00 # reduce number of steps for testing
            
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':0}, storage=storage)
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['vacuum-complex'] : 1.0, sams_samplers['vacuum-peptide'] : -1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)
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

class AblImatinibResistanceTestSystem(PersesTestSystem):
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

    >>> from perses.tests.testsystems import AblImatinibResistanceTestSystem
    >>> testsystem = AblImatinibResistanceTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum-inhibitor'].build_system(testsystem.topologies['vacuum-inhibitor'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['vacuum-inhibitor']

    """
    def __init__(self, **kwargs):
        super(AblImatinibResistanceTestSystem, self).__init__(**kwargs)
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
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True, barostat=barostat)
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
                proposal_engines[environment] = PointMutationEngine(topologies[environment], system_generators[environment], chain_id, proposal_metadata=proposal_metadata, allowed_mutations=allowed_mutations)

        # Generate systems ror all environments
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Create SAMS samplers
        
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        thermodynamic_states = dict()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])

                storage = None
                if self.storage:
                    storage = NetCDFStorageView(self.storage, envname=environment)

                if solvent == 'explicit':
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                    sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
                else:
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature)
                    sampler_state = states.SamplerState(positions=positions[environment])

                mcmc_samplers[environment] = MCMCSampler(thermodynamic_state, sampler_state, copy.deepcopy(self._move))
                 # reduce number of steps for testing
                
                exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, self.geometry_engine, proposal_engines[environment], options={'nsteps':self._ncmc_nsteps, 'mcmc_nsteps':self._mcmc_nsteps}, storage=storage)
                exen_samplers[environment].verbose = True
                sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
                sams_samplers[environment].verbose = True
                thermodynamic_states[environment] = thermodynamic_state

        # Create test MultiTargetDesign sampler.
        # TODO: Replace this with inhibitor:kinase and ATP:kinase ratio
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['vacuum-complex'] : 1.0, sams_samplers['vacuum-receptor'] : -1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)
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

class AblAffinityTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for optimizing kinase inhibitor affinity to Abl.

    TODO: Generalize to standard inhibitor:protein test system and extend to T4 lysozyme small molecules.

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

    >>> from perses.tests.testsystems import AblAffinityTestSystem
    >>> testsystem = AblAffinityestSystem()
    # Build a system
    >>> system = testsystem.system_generators['vacuum-inhibitor'].build_system(testsystem.topologies['vacuum-inhibitor'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['vacuum-inhibitor']

    """
    def __init__(self, **kwargs):
        super(AblAffinityTestSystem, self).__init__(**kwargs)
        solvents = ['vacuum', 'explicit'] # TODO: Add 'implicit' once GBSA parameterization for small molecules is working
        solvents = ['vacuum'] # DEBUG
        components = ['inhibitor', 'complex'] # TODO: Add 'ATP:kinase' complex to enable resistance design
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
        # Add current molecule
        molecules.append('Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)C[NH+]5CCN(CC5)C')
        self.molecules = molecules

        # Expand molecules without explicit stereochemistry and make canonical isomeric SMILES.
        molecules = sanitizeSMILES(self.molecules)
        molecules = canonicalize_SMILES(molecules)

        # Create a system generator for desired forcefields
        from perses.rjmc.topology_proposal import SystemGenerator
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True, barostat=barostat)
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
            print(pdb_filename)
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

        # Set up the proposal engines.
        from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
        proposal_metadata = { }
        proposal_engines = dict()
        for environment in environments:
            storage = None
            if self.storage:
                storage = NetCDFStorageView(self.storage, envname=environment)
            proposal_engines[environment] = SmallMoleculeSetProposalEngine(molecules, system_generators[environment], residue_name='MOL', storage=storage)

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        
        thermodynamic_states = dict()
        for component in components:
            for solvent in solvents:
                environment = solvent + '-' + component
                if solvent == 'explicit':
                    thermodynamic_states[environment] = states.ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                else:
                    thermodynamic_states[environment]   = states.ThermodynamicState(system=systems[environment], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])

                storage = None
                if self.storage:
                    storage = NetCDFStorageView(self.storage, envname=environment)

                if solvent == 'explicit':
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                    sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
                else:
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature)
                    sampler_state = states.SamplerState(positions=positions[environment])

                mcmc_samplers[environment] = MCMCSampler(thermodynamic_state, sampler_state, copy.deepcopy(self._move))
                 # reduce number of steps for testing
                
                exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':self._ncmc_nsteps, 'mcmc_nsteps':self._mcmc_nsteps}, storage=storage)
                exen_samplers[environment].verbose = True
                sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
                sams_samplers[environment].verbose = True
                thermodynamic_states[environment] = thermodynamic_state

        # Create test MultiTargetDesign sampler.
        # TODO: Replace this with inhibitor:kinase and ATP:kinase ratio
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['vacuum-complex'] : 1.0, sams_samplers['vacuum-inhibitor'] : -1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)
        designer.verbose = True

        # Store things.
        self.molecules = molecules
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.system_generators = system_generators
        self.systems = systems
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

        # This system must currently be minimized.
        minimize(self)

class AblImatinibProtonationStateTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for sampling protonation states of the Abl:imatinib complex.

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

    >>> from perses.tests.testsystems import AblImatinibProtonationStateTestSystem
    >>> testsystem = AblImatinibProtonationStateTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['explicit-inhibitor'].build_system(testsystem.topologies['explicit-inhibitor'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['explicit-inhibitor']

    """
    def __init__(self, **kwargs):
        super(AblImatinibProtonationStateTestSystem, self).__init__(**kwargs)
        solvents = ['vacuum', 'explicit'] # TODO: Add 'implicit' once GBSA parameterization for small molecules is working
        components = ['inhibitor', 'complex'] # TODO: Add 'ATP:kinase' complex to enable resistance design
        #solvents = ['vacuum'] # DEBUG: Just try vacuum for now
        #components = ['inhibitor'] # DEBUG: Just try inhibitor for now
        padding = 9.0*unit.angstrom
        explicit_solvent_model = 'tip3p'
        setup_path = 'data/constant-pH/abl-imatinib'
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres

        # Construct list of all environments
        environments = list()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                environments.append(environment)

        # Read mol2 file containing protonation states and extract canonical isomeric SMILES from this.
        from pkg_resources import resource_filename
        molecules = list()
        mol2_filename = resource_filename('perses', os.path.join(setup_path, 'Imatinib-epik-charged.mol2'))
        ifs = oechem.oemolistream(mol2_filename)
        mol = oechem.OEMol()
        while oechem.OEReadMolecule(ifs, mol):
            smiles = oechem.OEMolToSmiles(mol)
            molecules.append(smiles)
        # Read log probabilities
        log_state_penalties = dict()
        state_penalties_filename = resource_filename('perses', os.path.join(setup_path, 'Imatinib-state-penalties.out'))
        for (smiles, log_state_penalty) in zip(molecules, np.fromfile(state_penalties_filename, sep='\n')):
            log_state_penalties[smiles] = log_state_penalty

        # Add current molecule
        smiles = 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)C[NH+]5CCN(CC5)C'
        molecules.append(smiles)
        self.molecules = molecules
        log_state_penalties[smiles] = 100.0 # this should have zero weight

        # Expand molecules without explicit stereochemistry and make canonical isomeric SMILES.
        molecules = sanitizeSMILES(self.molecules)

        # Create a system generator for desired forcefields
        # TODO: Debug why we can't ue pregenerated molecule ffxml parameters. This may be an openmoltools issue.
        molecules_xml_filename = resource_filename('perses', os.path.join(setup_path, 'Imatinib-epik-charged.ffxml'))

        print('Creating system generators...')
        from perses.rjmc.topology_proposal import SystemGenerator
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        barostat = MonteCarloBarostat(pressure, temperature)
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True, barostat=barostat)
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
            print(pdb_filename)
            pdbfile = PDBFile(pdb_filename)
            topologies[component] = pdbfile.topology
            positions[component] = pdbfile.positions

        # Construct positions and topologies for all solvent environments
        print('Constructing positions and topologies...')
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

                natoms = sum( 1 for atom in topologies[environment].atoms() )
                print("System '%s' has %d atoms" % (environment, natoms))

        # Set up the proposal engines.
        print('Initializing proposal engines...')
        from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
        proposal_metadata = { }
        proposal_engines = dict()
        for environment in environments:
            proposal_engines[environment] = SmallMoleculeSetProposalEngine(molecules, system_generators[environment], residue_name='MOL')

        # Generate systems
        print('Building systems...')
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        print('Defining thermodynamic states...')
        
        thermodynamic_states = dict()
        for component in components:
            for solvent in solvents:
                environment = solvent + '-' + component
                if solvent == 'explicit':
                    thermodynamic_states[environment] = states.ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                else:
                    thermodynamic_states[environment]   = states.ThermodynamicState(system=systems[environment], temperature=temperature)

        # Create SAMS samplers
        print('Creating SAMS samplers...')
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])

                storage = None
                if self.storage:
                    storage = NetCDFStorageView(self.storage, envname=environment)

                if solvent == 'explicit':
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                    sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
                else:
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature)
                    sampler_state = states.SamplerState(positions=positions[environment])

                mcmc_samplers[environment] = MCMCSampler(thermodynamic_state, sampler_state, copy.deepcopy(self._move))
                 # reduce number of steps for testing
                
                exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':self._ncmc_nsteps, 'mcmc_nsteps':self._mcmc_nsteps}, storage=storage)
                exen_samplers[environment].verbose = True
                sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
                sams_samplers[environment].verbose = True
                thermodynamic_states[environment] = thermodynamic_state

        # Create a constant-pH sampler
        from perses.samplers.samplers import ProtonationStateSampler
        designer = ProtonationStateSampler(complex_sampler=exen_samplers['explicit-complex'], solvent_sampler=sams_samplers['explicit-inhibitor'], log_state_penalties=log_state_penalties, storage=self.storage)
        designer.verbose = True

        # Store things.
        self.molecules = molecules
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.system_generators = system_generators
        self.systems = systems
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = designer

        # This system must currently be minimized.
        minimize(self)
        print('AblImatinibProtonationStateTestSystem initialized.')

class ImidazoleProtonationStateTestSystem(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for sampling protonation states of imidazole in water.

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

    >>> from perses.tests.testsystems import AblImatinibProtonationStateTestSystem
    >>> testsystem = AblImatinibProtonationStateTestSystem()
    # Build a system
    >>> system = testsystem.system_generators['explicit-inhibitor'].build_system(testsystem.topologies['explicit-inhibitor'])
    # Retrieve a SAMSSampler
    >>> sams_sampler = testsystem.sams_samplers['explicit-inhibitor']

    """
    def __init__(self, **kwargs):
        super(ImidazoleProtonationStateTestSystem, self).__init__(**kwargs)
        solvents = ['vacuum', 'explicit'] # TODO: Add 'implicit' once GBSA parameterization for small molecules is working
        components = ['imidazole']
        padding = 9.0*unit.angstrom
        explicit_solvent_model = 'tip3p'
        setup_path = 'data/constant-pH/imidazole/'
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres

        # Construct list of all environments
        environments = list()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                environments.append(environment)

        # Read mol2 file containing protonation states and extract canonical isomeric SMILES from this.
        from pkg_resources import resource_filename
        molecules = list()
        mol2_filename = resource_filename('perses', os.path.join(setup_path, 'imidazole/imidazole-epik-charged.mol2'))
        ifs = oechem.oemolistream(mol2_filename)
        mol = oechem.OEMol()
        while oechem.OEReadMolecule(ifs, mol):
            smiles = oechem.OEMolToSmiles(mol)
            molecules.append(smiles)
        # Read log probabilities
        log_state_penalties = dict()
        state_penalties_filename = resource_filename('perses', os.path.join(setup_path, 'imidazole/imidazole-state-penalties.out'))
        for (smiles, log_state_penalty) in zip(molecules, np.fromfile(state_penalties_filename, sep='\n')):
            log_state_penalties[smiles] = log_state_penalty

        # Add current molecule
        smiles = 'C1=CN=CN1'
        molecules.append(smiles)
        self.molecules = molecules
        log_state_penalties[smiles] = 0.0

        # Expand molecules without explicit stereochemistry and make canonical isomeric SMILES.
        molecules = sanitizeSMILES(self.molecules)

        # Create a system generator for desired forcefields
        print('Creating system generators...')
        from perses.rjmc.topology_proposal import SystemGenerator
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True, barostat=barostat)
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
            print(pdb_filename)
            pdbfile = PDBFile(pdb_filename)
            topologies[component] = pdbfile.topology
            positions[component] = pdbfile.positions

        # Construct positions and topologies for all solvent environments
        print('Constructing positions and topologies...')
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

                natoms = sum( 1 for atom in topologies[environment].atoms() )
                print("System '%s' has %d atoms" % (environment, natoms))

                # DEBUG: Write initial PDB file
                outfile = open(environment + '.initial.pdb', 'w')
                PDBFile.writeFile(topologies[environment], positions[environment], file=outfile)
                outfile.close()

        # Set up the proposal engines.
        print('Initializing proposal engines...')
        residue_name = 'UNL' # TODO: Figure out residue name automatically
        from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
        proposal_metadata = { }
        proposal_engines = dict()
        for environment in environments:
            storage = None
            if self.storage is not None:
                storage = NetCDFStorageView(self.storage, envname=environment)
            proposal_engines[environment] = SmallMoleculeSetProposalEngine(molecules, system_generators[environment], residue_name=residue_name, storage=storage)

        # Generate systems
        print('Building systems...')
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        print('Defining thermodynamic states...')
        
        thermodynamic_states = dict()
        for component in components:
            for solvent in solvents:
                environment = solvent + '-' + component
                if solvent == 'explicit':
                    thermodynamic_states[environment] = states.ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                else:
                    thermodynamic_states[environment] = states.ThermodynamicState(system=systems[environment], temperature=temperature)

        # Create SAMS samplers
        print('Creating SAMS samplers...')
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for solvent in solvents:
            for component in components:
                environment = solvent + '-' + component
                chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])

                storage = None
                if self.storage is not None:
                    storage = NetCDFStorageView(self.storage, envname=environment)

                if solvent == 'explicit':
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                    sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
                else:
                    thermodynamic_state = states.ThermodynamicState(system=systems[environment], temperature=temperature)
                    sampler_state = states.SamplerState(positions=positions[environment])

                mcmc_samplers[environment] = MCMCSampler(thermodynamic_state, sampler_state, copy.deepcopy(self._move))
                 # reduce number of steps for testing
                
                exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':self._ncmc_nsteps, 'mcmc_nsteps':self._mcmc_nsteps}, storage=storage)
                exen_samplers[environment].verbose = True
                sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
                sams_samplers[environment].verbose = True
                thermodynamic_states[environment] = thermodynamic_state

        # Store things.
        self.molecules = molecules
        self.environments = environments
        self.topologies = topologies
        self.positions = positions
        self.system_generators = system_generators
        self.systems = systems
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers
        self.sams_samplers = sams_samplers
        self.designer = None

        print('ImidazoleProtonationStateTestSystem initialized.')

def minimize(testsystem):
    """
    Minimize all structures in test system.

    TODO
    ----
    Use sampler thermodynamic states instead of testsystem.systems

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
    def __init__(self, constraints=app.HBonds, premapped_json_dict=None, **kwargs):
        super(SmallMoleculeLibraryTestSystem, self).__init__(**kwargs)
        # Expand molecules without explicit stereochemistry and make canonical isomeric SMILES.
        molecules = sanitizeSMILES(self.molecules)
        molecules = canonicalize_SMILES(molecules)
        environments = ['explicit', 'vacuum']
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : constraints }, barostat=barostat)
        system_generators['vacuum'] = SystemGenerator([gaff_xml_filename],
            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : constraints })

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
        from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine, PremappedSmallMoleculeSetProposalEngine, SmallMoleculeAtomMapper
        proposal_metadata = { }
        proposal_engines = dict()

        if not premapped_json_dict:
            for environment in environments:
                proposal_engines[environment] = SmallMoleculeSetProposalEngine(molecules, system_generators[environment])
        
        else:
            atom_mapper = SmallMoleculeAtomMapper.from_json(premapped_json_dict)
            for environment in environments:
                proposal_engines[environment] = PremappedSmallMoleculeSetProposalEngine(atom_mapper, system_generators[environment])

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.
        
        thermodynamic_states = dict()
        thermodynamic_states['explicit'] = states.ThermodynamicState(system=systems['explicit'], temperature=temperature, pressure=pressure)
        thermodynamic_states['vacuum']   = states.ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            storage = None
            if self.storage:
                storage = NetCDFStorageView(self.storage, envname=environment)

            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment == 'explicit':
                sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = states.SamplerState(positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state, copy.deepcopy(self._move))
             # reduce number of steps for testing
            
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':500}, storage=storage)
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['explicit'] : 1.0, sams_samplers['vacuum'] : -1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)

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
    def __init__(self, **kwargs):
        self.molecules = ['CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC']
        super(AlkanesTestSystem, self).__init__(**kwargs)

class KinaseInhibitorsTestSystem(SmallMoleculeLibraryTestSystem):
    """
    Library of clinical kinase inhibitors in various solvent environments.
    """
    def __init__(self, **kwargs):
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
        super(KinaseInhibitorsTestSystem, self).__init__(**kwargs)

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

    def __init__(self, **kwargs):
        # Read SMILES from CSV file of clinical kinase inhibitors.
        from pkg_resources import resource_filename
        molecules = list()
        molecules += self.read_smiles(resource_filename('perses', 'data/L99A-binders.txt'))
        molecules += self.read_smiles(resource_filename('perses', 'data/L99A-non-binders.txt'))
        self.molecules = molecules
        # Intialize
        super(T4LysozymeInhibitorsTestSystem, self).__init__(**kwargs)

class FusedRingsTestSystem(SmallMoleculeLibraryTestSystem):
    """
    Simple test system containing fused rings (benzene <--> naphtalene) in explicit solvent.
    """
    def __init__(self, **kwargs):
        self.molecules = ['c1ccccc1', 'c1ccc2ccccc2c1'] # benzene, naphthalene
        super(FusedRingsTestSystem, self).__init__(**kwargs)

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
    def __init__(self, **kwargs):
        super(ValenceSmallMoleculeLibraryTestSystem, self).__init__(**kwargs)
        initial_molecules = ['CCCCC','CC(C)CC', 'CCC(C)C', 'CCCCC', 'C(CC)CCC']
        molecules = self._canonicalize_smiles(initial_molecules)
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
        
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        thermodynamic_states['vacuum']   = states.ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            storage = None
            if self.storage:
                storage = NetCDFStorageView(self.storage, envname=environment)

            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment == 'explicit':
                sampler_state = states.SamplerState(positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = states.SamplerState(positions=positions[environment])
            mcmc_samplers[environment] = MCMCSampler(thermodynamic_states[environment], sampler_state, copy.deepcopy(self._move))
            00 # reduce number of steps for testing
            
            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], self.geometry_engine, options={'nsteps':0}, storage=storage)
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = { sams_samplers['vacuum'] : 1.0, sams_samplers['vacuum'] : -1.0 }
        designer = MultiTargetDesign(target_samplers, storage=self.storage)

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

    def _canonicalize_smiles(self, list_of_smiles):
        """
        Turn a list of smiles strings into openeye canonical
        isomeric smiles.

        Parameters
        ----------
        list_of_smiles : list of str
            input smiles

        Returns
        -------
        list_of_canonicalized_smiles : list of str
            canonical isomeric smiles
        """
        list_of_canonicalized_smiles = []
        ofs = oechem.oemolostream('current.mol2') # DEBUG
        for smiles in list_of_smiles:
            mol = oechem.OEMol()
            oechem.OESmilesToMol(mol, smiles)
            oechem.OEAddExplicitHydrogens(mol)
            can_smi = oechem.OECreateSmiString(mol, OESMILES_OPTIONS)
            list_of_canonicalized_smiles.append(can_smi)

        ofs.close() # DEBUG

        return list_of_canonicalized_smiles

class NullTestSystem(PersesTestSystem):
    """
    Test turning a small molecule into itself in vacuum
    Currently only trying to test ExpandedEnsemble sampler, therefore
    SAMS sampler and MultiTargetDesign are not implemented at this time

    Uses a custom ProposalEngine to only match subset of atoms, requiring
    geometry to build in the rest

    geometry_engine.write_proposal_pdb set to False

    Constructor:
    NullTestSystem(storage_filename="null.nc", exen_pdb_filename=None)

    Arguments:
        storage_filename, OPTIONAL, string
            Default is "null.nc"
            Storage must be provided in order to analyze testsystem acceptance rates
        exen_pdb_filename, OPTIONAL, string
            Default is None
            If value is not None, will write pdbfile after every ExpandedEnsemble
            iteration
        scheme, OPTIONAL, string
            Default is 'ncmc-geometry-ncmc'
            Scheme to be used by ExpandedEnsembleSampler
            Must be in ['geometry-ncmc-geometry','ncmc-geometry-ncmc','geometry-ncmc']
            Default will run NCMC on old and new system separately

    Only one environment ('vacuum') is currently implemented; however all
    samplers are saved in dictionaries for consistency with other testsystems

    """
    def __init__(self, storage_filename="null.nc", exen_pdb_filename=None, scheme='ncmc-geometry-ncmc', options=None):

        super(NullTestSystem, self).__init__(storage_filename=storage_filename)

        if options is None:
            options = {'nsteps':0}
        if 'nsteps' not in options.keys():
            options['nsteps'] = 0

        environments = ['vacuum', 'explicit']

#        self.geometry_engine.write_proposal_pdb = True

        system_generators = dict()
        topologies = dict()
        positions = dict()
        proposal_engines = dict()
        thermodynamic_states = dict()
        mcmc_samplers = dict()
        exen_samplers = dict()

        from perses.rjmc.topology_proposal import SystemGenerator
        from perses.tests.utils import oemol_to_omm_ff, get_data_filename, createOEMolFromIUPAC
        from perses.samplers.samplers import ExpandedEnsembleSampler

        for key in environments:
            gaff_xml_filename = get_data_filename('data/gaff.xml')
            if key == "vacuum":
                forcefield_kwargs = {'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : None, 'constraints' : None}
                ff_list = [gaff_xml_filename]
            if key == "explicit":
                ff_list = [gaff_xml_filename, 'tip3p.xml']
                forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : app.HBonds }
            system_generator = SystemGenerator(ff_list, forcefield_kwargs=forcefield_kwargs)
            system_generators[key] = system_generator

            proposal_engine = self.NullProposal(system_generator, residue_name=self.mol_name)
            initial_molecule = createOEMolFromIUPAC(iupac_name=self.mol_name)
            initial_system, initial_positions, initial_topology = oemol_to_omm_ff(initial_molecule, self.mol_name)

            if key == "explicit":
                modeller = app.Modeller(initial_topology, initial_positions)
                modeller.addSolvent(system_generators[key].getForceField(), model='tip3p', padding=9.0*unit.angstrom)
                initial_topology = modeller.getTopology()
                initial_positions = modeller.getPositions()
                initial_system = system_generators[key].build_system(initial_topology)

            initial_topology._state_key = proposal_engine._fake_states[0]

            temperature = 300*unit.kelvin
            thermodynamic_state = states.ThermodynamicState(system=initial_system, temperature=temperature)

            chemical_state_key = proposal_engine.compute_state_key(initial_topology)
            sampler_state = states.SamplerState(positions=initial_positions)

            mcmc_sampler = MCMCSampler(thermodynamic_state, sampler_state, copy.deepcopy(self._move))
            mcmc_sampler.nsteps = 500
            mcmc_sampler.timestep = 1.0*unit.femtosecond
            mcmc_sampler.verbose = True

            exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, initial_topology, chemical_state_key, proposal_engine, self.geometry_engine, options=options, storage=self.storage)
            exen_sampler.verbose = True
            if exen_pdb_filename is not None:
                exen_sampler.pdbfile = open(exen_pdb_filename,'w')

            topologies[key] = initial_topology
            positions[key] = initial_positions
            proposal_engines[key] = proposal_engine
            thermodynamic_states[key] = thermodynamic_state
            mcmc_samplers[key] = mcmc_sampler
            exen_samplers[key] = exen_sampler

        # save
        self.environments = environments
        self.storage_filename = storage_filename
        self.system_generators = system_generators
        self.topologies = topologies
        self.positions = positions
        self.proposal_engines = proposal_engines
        self.thermodynamic_states = thermodynamic_states
        self.mcmc_samplers = mcmc_samplers
        self.exen_samplers = exen_samplers


class NaphthaleneTestSystem(NullTestSystem):
    """
    Test turning Naphthalene into Naphthalene in vacuum
    Currently only trying to test ExpandedEnsemble sampler, therefore
    SAMS sampler and MultiTargetDesign are not implemented at this time

    Uses a custom ProposalEngine to only match one ring, requiring
    geometry to build in the other

    geometry_engine.write_proposal_pdb set to True

    Constructor:
    NaphthaleneTestSystem(storage_filename="naphthalene.nc", exen_pdb_filename=None)

    Arguments:
        storage_filename, OPTIONAL, string
            Default is "naphthalene.nc"
            Storage must be provided in order to analyze testsystem acceptance rates
        exen_pdb_filename, OPTIONAL, string
            Default is None
            If value is not None, will write pdbfile after every ExpandedEnsemble
            iteration
        scheme, OPTIONAL, string
            Default is 'geometry-ncmc-geometry'
            Scheme to be used by ExpandedEnsembleSampler
            Must be in ['geometry-ncmc-geometry','ncmc-geometry-ncmc','geometry-ncmc']
            Default will use a hybrid NCMC method

    Only one environment ('vacuum') is currently implemented; however all
    samplers are saved in dictionaries for consistency with other testsystems
    """

    def __init__(self, storage_filename="naphthalene.nc", exen_pdb_filename=None, scheme='geometry-ncmc-geometry', options=None):
        """
        __init__(self, storage_filename="naphthalene.nc", exen_pdb_filename=None, scheme='geometry-ncmc-geometry'):
        """
        from perses.rjmc.topology_proposal import NaphthaleneProposalEngine
        self.NullProposal = NaphthaleneProposalEngine
        self.mol_name = 'naphthalene'
        super(NaphthaleneTestSystem, self).__init__(storage_filename=storage_filename, exen_pdb_filename=exen_pdb_filename, scheme=scheme, options=options)

class ButaneTestSystem(NullTestSystem):
    """
    Test turning Butane into Butane in vacuum
    Currently only trying to test ExpandedEnsemble sampler, therefore
    SAMS sampler and MultiTargetDesign are not implemented at this time

    Uses a custom ProposalEngine to only match two carbons, have geometry
    engine choose positions for others

    geometry_engine.write_proposal_pdb set to True

    Constructor:
    ButaneTestSystem(storage_filename="butane.nc", exen_pdb_filename=None)

    Arguments:
        storage_filename, OPTIONAL, string
            Default is "butane.nc"
            Storage must be provided in order to analyze testsystem acceptance rates
        exen_pdb_filename, OPTIONAL, string
            Default is None
            If value is not None, will write pdbfile after every ExpandedEnsemble
            iteration
        scheme, OPTIONAL, string
            Default is 'geometry-ncmc-geometry'
            Scheme to be used by ExpandedEnsembleSampler
            Must be in ['geometry-ncmc-geometry','ncmc-geometry-ncmc','geometry-ncmc']
            Default will use a hybrid NCMC method

    Only one environment ('vacuum') is currently implemented; however all
    samplers are saved in dictionaries for consistency with other testsystems
    """

    def __init__(self, storage_filename="butane.nc", exen_pdb_filename=None, scheme='geometry-ncmc-geometry', options=None):
        """
        __init__(self, storage_filename="butane.nc", exen_pdb_filename=None, scheme='geometry-ncmc-geometry'):
        """
        from perses.rjmc.topology_proposal import ButaneProposalEngine
        self.NullProposal = ButaneProposalEngine
        self.mol_name = 'butane'
        super(ButaneTestSystem, self).__init__(storage_filename=storage_filename, exen_pdb_filename=exen_pdb_filename, scheme=scheme, options=options)

class PropaneTestSystem(NullTestSystem):
    """
    Test turning Propane into Propane in vacuum
    Currently only trying to test ExpandedEnsemble sampler, therefore
    SAMS sampler and MultiTargetDesign are not implemented at this time

    Uses a custom ProposalEngine to map CH3-CH2, have geometry build in the
    other CH3

    geometry_engine.write_proposal_pdb set to True

    Constructor:
    ButaneTestSystem(storage_filename="propane.nc", exen_pdb_filename=None)

    Arguments:
        storage_filename, OPTIONAL, string
            Default is "propane.nc"
            Storage must be provided in order to analyze testsystem acceptance rates
        exen_pdb_filename, OPTIONAL, string
            Default is None
            If value is not None, will write pdbfile after every ExpandedEnsemble
            iteration
        scheme, OPTIONAL, string
            Default is 'geometry-ncmc-geometry'
            Scheme to be used by ExpandedEnsembleSampler
            Must be in ['geometry-ncmc-geometry','ncmc-geometry-ncmc','geometry-ncmc']
            Default will use a hybrid NCMC method

    Only one environment ('vacuum') is currently implemented; however all
    samplers are saved in dictionaries for consistency with other testsystems
    """

    def __init__(self, storage_filename="propane.nc", exen_pdb_filename=None, scheme='geometry-ncmc-geometry', options=None):
        """
        __init__(self, storage_filename="propane.nc", exen_pdb_filename=None, scheme='geometry-ncmc-geometry'):
        """
        from perses.rjmc.topology_proposal import PropaneProposalEngine
        self.NullProposal = PropaneProposalEngine
        self.mol_name = 'propane'
        super(PropaneTestSystem, self).__init__(storage_filename=storage_filename, exen_pdb_filename=exen_pdb_filename, scheme=scheme, options=options)


def run_null_system(testsystem):
    """
    Intended for use with NullTestSystem subclasses ONLY

    Runs TestSystem ExpandedEnsemble sampler ONLY
    Uses BAR to check whether the free energies of the two states
    (both naphthalene) are within 6 sigma of 0
    Imports netCDF4 to read in storage file and access data

    Arguments:
    ----------
    testsystem : NaphthaleneTestSystem, ButantTestSystem, or PropaneTestSystem
        Only these three test systems have the proposal_engine._fake_states
        attribute, which differentiates between 2 states of a null proposal

    CURRENTLY:
    The expanded ensemble acceptance rate of naphthalene-A to naphthalene-B
    is very low.  This test will run 10 iterations of the ExpandedEnsemble
    sampler until a switch is accepted, and then run approximately that
    number of steps again, to ensure w_f and w_r have nonzero length. This
    should not be necessary if the acceptance rate is higher, and the
    number of exen_sampler iterations can be fixed.

    TODO:
        move netcdf import to analysis for general use
        move BAR import to analysis, define use of BAR to be generalized
    """
    if not issubclass(type(testsystem), NullTestSystem):
        raise(NotImplementedError("run_null_system is only compatible with NaphthaleneTestSystem, ButantTestSystem or PropaneTestSystem; given {0}".format(type(testsystem))))

    import netCDF4 as netcdf
    import pickle
    import codecs
    for key in testsystem.environments: # only one key: vacuum
        # run a single iteration to generate item in number_of_state_visits dict
        testsystem.exen_samplers[key].run(niterations=100)
        # until a switch is accepted, only the initial state will have an item
        # in the number_of_state_visits dict
        while len(testsystem.exen_samplers[key].number_of_state_visits.keys()) == 1:
            testsystem.exen_samplers[key].run(niterations=10)
        # after a switch has been accepted, run approximately the same number of
        # steps again, to end up with roughly equal number of proposals starting
        # from each state
        testsystem.exen_samplers[key].run(niterations=testsystem.exen_samplers[key].nrejected)
        print(testsystem.exen_samplers[key].number_of_state_visits)
        print("Acceptances in {0} iterations: {1}".format(testsystem.exen_samplers[key].iteration, testsystem.exen_samplers[key].naccepted))

        from perses.analysis import Analysis
        analysis = Analysis(testsystem.storage_filename)
        analysis.plot_exen_logp_components()

        ncfile = netcdf.Dataset(testsystem.storage_filename, 'r')
        ee_sam = ncfile.groups['ExpandedEnsembleSampler']
        niterations = ee_sam.variables['logp_accept'].shape[0]
        logps = np.zeros(niterations, np.float64)
        state_keys = list()
        for n in range(niterations):
            logps[n] = ee_sam.variables['logp_accept'][n]
            s_key = str(ee_sam.variables['proposed_state_key'][n])
            state_keys.append(pickle.loads(codecs.decode(s_key, "base64")))

        len_w_r = state_keys.count(testsystem.proposal_engines[key]._fake_states[0])
        len_w_f = state_keys.count(testsystem.proposal_engines[key]._fake_states[1])
        try:
            assert niterations == len_w_f + len_w_r
        except:
            print("{0} iterations, but {1} started from A and {2} started from B?".format(niterations, len_w_f, len_w_r))
        if len_w_f == 0 or len_w_r == 0:
            # test failure, but what to do?
            raise(Exception("Cannot run BAR because no transitions were made"))

        # after importing all logps, use proposed_state_key to split them into
        # separate arrays depending on the direction of the proposed switch
        w_f = np.zeros(len_w_f, np.float64)
        w_r = np.zeros(len_w_r, np.float64)
        w_f_count = 0
        w_r_count = 0
        for n in range(niterations):
            if state_keys[n] == testsystem.proposal_engines[key]._fake_states[1]:
                w_f[w_f_count] = logps[n]
                w_f_count += 1
            else:
                w_r[w_r_count] = logps[n]
                w_r_count += 1

        from pymbar import BAR
        [df, ddf] = BAR(w_f, w_r, method='self-consistent-iteration')
        print('%8.3f +- %.3f kT' % (df, ddf))
        NSIGMA_MAX = 6.0
        if (abs(df) > NSIGMA_MAX * ddf):
            msg = 'Delta F (%d proposals) = %f +- %f kT; should be within %f sigma of 0' % (niterations, df, ddf, NSIGMA_MAX)
            msg += '\n'
            msg += 'w_f = %s\n' % str(w_f)
            msg += 'w_r = %s\n' % str(w_r)
            raise Exception(msg)


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
    tmpfile = tempfile.NamedTemporaryFile()
    storage_filename = tmpfile.name
    testsystem = testsystem_class(storage_filename=storage_filename)
    # Check topologies
    check_topologies(testsystem)

def test_testsystems():
    """
    Test instantiation of all test systems.
    """
    testsystem_names = ['T4LysozymeInhibitorsTestSystem', 'KinaseInhibitorsTestSystem', 'AlkanesTestSystem', 'AlanineDipeptideTestSystem']
    niterations = 2 # number of iterations to run
    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        f = partial(checktestsystem, testsystem_class)
        f.description = "Testing %s" % (testsystem_name)
        yield f

def run_t4_inhibitors():
    """
    Run T4 lysozyme inhibitors in solvents test system.
    """
    testsystem = T4LysozymeInhibitorsTestSystem(storage_filename='output.nc', ncmc_nsteps=50, mcmc_nsteps=100)
    for environment in ['explicit', 'vacuum']:
        #testsystem.exen_samplers[environment].pdbfile = open('t4-' + component + '.pdb', 'w')
        #testsystem.exen_samplers[environment].options={'nsteps':50} # instantaneous MC
        testsystem.exen_samplers[environment].verbose = True
        testsystem.sams_samplers[environment].verbose = True
    testsystem.designer.verbose = True
    testsystem.designer.run(niterations=50)

    # Analyze data.
    #from perses.analysis import Analysis
    #analysis = Analysis(storage_filename='output.nc')
    #analysis.plot_sams_weights('sams.pdf')
    #analysis.plot_ncmc_work('ncmc.pdf')

def run_t4():
    """
    Run T4 lysozyme test system.
    """
    testsystem = T4LysozymeTestSystem(ncmc_nsteps=0)
    solvent = 'explicit'
    for component in ['complex', 'receptor']:
        testsystem.exen_samplers[solvent + '-' + component].pdbfile = open('t4-' + component + '.pdb', 'w')
        testsystem.sams_samplers[solvent + '-' + component].run(niterations=5)
    testsystem.designer.verbose = True
    testsystem.designer.run(niterations=5)

    # Analyze data.
    #from perses.analysis import Analysis
    #analysis = Analysis(storage_filename='output.nc')
    #analysis.plot_sams_weights('sams.pdf')
    #analysis.plot_ncmc_work('ncmc.pdf')

def run_myb():
    """
    Run myb test system.
    """
    testsystem = MybTestSystem(ncmc_nsteps=0, mcmc_nsteps=100)
    solvent = 'implicit'

    testsystem.exen_samplers[solvent + '-peptide'].pdbfile = open('myb-vacuum.pdb', 'w')
    testsystem.exen_samplers[solvent + '-complex'].pdbfile = open('myb-complex.pdb', 'w')
    testsystem.sams_samplers[solvent + '-complex'].run(niterations=5)
    #testsystem.designer.verbose = True
    #testsystem.designer.run(niterations=500)
    #testsystem.exen_samplers[solvent + '-peptide'].verbose=True
    #testsystem.exen_samplers[solvent + '-peptide'].run(niterations=100)

def run_abl_imatinib_resistance():
    """
    Run abl test system.
    """
    testsystem = AblImatinibResistanceTestSystem(ncmc_nsteps=20000, mcmc_nsteps=20000)
    #for environment in testsystem.environments:
    for environment in ['vacuum-complex']:
        print(environment)
        testsystem.exen_samplers[environment].pdbfile = open('abl-imatinib-%s.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].geometry_pdbfile = open('abl-imatinib-%s-geometry-proposals.pdb' % environment, 'w')
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
    testsystem = KinaseInhibitorsTestSystem(ncmc_nsteps=0, mcmc_nsteps=100)
    environment = 'vacuum'
    testsystem.exen_samplers[environment].pdbfile = open('kinase-inhibitors-vacuum.pdb', 'w')
    testsystem.exen_samplers[environment].geometry_pdbfile = open('kinase-inhibitors-%s-geometry-proposals.pdb' % environment, 'w')
    testsystem.exen_samplers[environment].geometry_engine.write_proposal_pdb = True # write proposal PDBs
    testsystem.sams_samplers[environment].run(niterations=100)

def run_valence_system():
    """
    Run valence molecules test system.

    This system only has one environment (vacuum), so SAMS is used.

    """
    testsystem = ValenceSmallMoleculeLibraryTestSystem(storage_filename='output.nc', ncmc_nsteps=0, mcmc_nsteps=10)
    environment = 'vacuum'
    testsystem.exen_samplers[environment].pdbfile = open('valence.pdb', 'w')
    testsystem.sams_samplers[environment].run(niterations=50)

def run_alanine_system(sterics=False):
    """
    Run alanine dipeptide in vacuum test system.

    If `sterics == True`, then sterics will be included.
    Otherwise, only valence terms are used.

    """
    if sterics:
        testsystem = AlanineDipeptideTestSystem(storage_filename='output.nc', ncmc_nsteps=0, mcmc_nsteps=100)
    else:
        testsystem = AlanineDipeptideValenceTestSystem(storage_filename='output.nc', ncmc_nsteps=0, mcmc_nsteps=100)
    environment = 'vacuum'
    print(testsystem.__class__.__name__)
    testsystem.exen_samplers[environment].pdbfile = open('valence.pdb', 'w')
    testsystem.sams_samplers[environment].update_method = 'two-stage'
    testsystem.sams_samplers[environment].second_stage_start = 100 # iteration to start second stage
    testsystem.sams_samplers[environment].run(niterations=200)

def test_valence_write_pdb_ncmc_switching():
    """
    Run abl test system.
    """
    testsystem = ValenceSmallMoleculeLibraryTestSystem(ncmc_nsteps=10, mcmc_nsteps=10)
    environment = 'vacuum'
    testsystem.exen_samplers[environment].run(niterations=1)

def run_abl_affinity_write_pdb_ncmc_switching():
    """
    Run abl test system.
    """
    testsystem = AblAffinityTestSystem(ncmc_nsteps=10000, mcmc_nsteps=10000)
    #for environment in testsystem.environments:
    for environment in ['vacuum-complex']:
        print(environment)
        testsystem.exen_samplers[environment].pdbfile = open('abl-imatinib-%s.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].geometry_pdbfile = open('abl-imatinib-%s-geometry-proposals.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].verbose = True
        testsystem.sams_samplers[environment].verbose = True
        #testsystem.mcmc_samplers[environment].run(niterations=5)
        testsystem.exen_samplers[environment].run(niterations=5)

        #testsystem.sams_samplers[environment].run(niterations=5)

    #testsystem.designer.verbose = True
    #testsystem.designer.run(niterations=500)
    #testsystem.exen_samplers[solvent + '-peptide'].verbose=True
    #testsystem.exen_samplers[solvent + '-peptide'].run(niterations=100)

def run_constph_abl():
    """
    Run Abl:imatinib constant-pH test system.
    """
    testsystem = AblImatinibProtonationStateTestSystem(ncmc_nsteps=50, mcmc_nsteps=2500)
    for environment in testsystem.environments:
    #for environment in ['explicit-inhibitor', 'explicit-complex']:
    #for environment in ['vacuum-inhibitor', 'vacuum-complex']:
        if environment not in testsystem.exen_samplers:
            print("Skipping '%s' for now..." % environment)
            continue

        print(environment)
        testsystem.exen_samplers[environment].pdbfile = open('abl-imatinib-constph-%s.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].geometry_pdbfile = open('abl-imatinib-constph-%s-geometry-proposals.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].verbose = True
        testsystem.exen_samplers[environment].proposal_engine.verbose = True
        testsystem.sams_samplers[environment].verbose = True
        #testsystem.mcmc_samplers[environment].run(niterations=5)
        #testsystem.exen_samplers[environment].run(niterations=5)

        #testsystem.sams_samplers[environment].run(niterations=5)

    # Run ligand in solvent constant-pH sampler calibration
    testsystem.sams_samplers['explicit-inhibitor'].verbose=True
    testsystem.sams_samplers['explicit-inhibitor'].run(niterations=100)
    #testsystem.exen_samplers['vacuum-inhibitor'].verbose=True
    #testsystem.exen_samplers['vacuum-inhibitor'].run(niterations=100)
    #testsystem.exen_samplers['explicit-complex'].verbose=True
    #testsystem.exen_samplers['explicit-complex'].run(niterations=100)

    # Run constant-pH sampler
    testsystem.designer.verbose = True
    testsystem.designer.update_target_probabilities() # update log weights from inhibitor in solvent calibration
    testsystem.designer.run(niterations=500)

def run_imidazole():
    """
    Run imidazole constant-pH test system.
    """
    testsystem = ImidazoleProtonationStateTestSystem(storage_filename='output.nc', ncmc_nsteps=500, mcmc_nsteps=1000)
    for environment in testsystem.environments:
        if environment not in testsystem.exen_samplers:
            print("Skipping '%s' for now..." % environment)
            continue

        print(environment)
        #testsystem.exen_samplers[environment].pdbfile = open('imidazole-constph-%s.pdb' % environment, 'w')
        #testsystem.exen_samplers[environment].geometry_pdbfile = open('imidazole-constph-%s-geometry-proposals.pdb' % environment, 'w')
        testsystem.exen_samplers[environment].verbose = True
        testsystem.exen_samplers[environment].proposal_engine.verbose = True
        testsystem.sams_samplers[environment].verbose = True

    # Run ligand in solvent constant-pH sampler calibration
    testsystem.sams_samplers['explicit-imidazole'].verbose=True
    testsystem.sams_samplers['explicit-imidazole'].run(niterations=100)

def run_fused_rings():
    """
    Run fused rings test system.
    Vary number of NCMC steps

    """
    #nsteps_to_try = [1, 10, 100, 1000, 10000, 100000] # number of NCMC steps
    nsteps_to_try = [10, 100, 1000, 10000, 100000] # number of NCMC steps
    for ncmc_steps in nsteps_to_try:
        storage_filename = 'output-%d.nc' % ncmc_steps
        testsystem = FusedRingsTestSystem(storage_filename=storage_filename, ncmc_nsteps=nsteps_to_try, mcmc_nsteps=100)
        for environment in ['explicit', 'vacuum']:
            testsystem.exen_samplers[environment].ncmc_engine.verbose = True # verbose output of work
            testsystem.sams_samplers[environment].verbose = True
        testsystem.designer.verbose = True
        testsystem.designer.run(niterations=100)

        # Analyze data.
        from perses.analysis import Analysis
        analysis = Analysis(storage_filename=storage_filename)
        #analysis.plot_sams_weights('sams.pdf')
        analysis.plot_ncmc_work('ncmc-%d.pdf' % ncmc_steps)

if __name__ == '__main__':
    #testsystem = PropaneTestSystem(scheme='geometry-ncmc-geometry', options = {'nsteps':10})
    #run_null_system(testsystem)
    #run_alanine_system(sterics=False)
    #run_fused_rings()
    #run_valence_system()
    run_t4_inhibitors()
    #run_imidazole()
    #run_constph_abl()
    #run_abl_affinity_write_pdb_ncmc_switching()
    #run_kinase_inhibitors()
    #run_abl_imatinib()
    #run_myb()

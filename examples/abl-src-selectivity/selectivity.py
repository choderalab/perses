from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
from perses.utils.smallmolecules import sanitizeSMILES
from perses.tests.testsystems import minimize
from perses.tests.testsystems import PersesTestSystem
import perses.rjmc.geometry as geometry
from perses.storage import NetCDFStorage, NetCDFStorageView

class Selectivity(PersesTestSystem):
    """
    Create a consistent set of SAMS samplers useful for optimizing kinase inhibitor selectivity.

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
        super(Selectivity, self).__init__(**kwargs)

        solvents = ['explicit'] # DEBUG
        components = ['src-imatinib', 'abl-imatinib'] # TODO: Add 'ATP:kinase' complex to enable resistance design
        padding = 9.0*unit.angstrom
        explicit_solvent_model = 'tip3p'
        setup_path = 'data/abl-src'
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        geometry_engine = geometry.FFAllAngleGeometryEngine()


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

        # Create a system generator for desired forcefields
        from perses.rjmc.topology_proposal import SystemGenerator
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        system_generators = dict()
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml'],
            forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : None },
            use_antechamber=True)
# NOTE implicit solvent not supported by this SystemGenerator
#        system_generators['implicit'] = SystemGenerator([gaff_xml_filename, 'amber99sbildn.xml', 'amber99_obc.xml'],
#            forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'implicitSolvent' : app.OBC2, 'constraints' : None },
#            use_antechamber=True)
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
        from perses.samplers.thermodynamics import ThermodynamicState
        thermodynamic_states = dict()
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmospheres
        for component in components:
            for solvent in solvents:
                environment = solvent + '-' + component
                if solvent == 'explicit':
                    thermodynamic_states[environment] = ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                else:
                    thermodynamic_states[environment] = ThermodynamicState(system=systems[environment], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
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
                    thermodynamic_state = ThermodynamicState(system=systems[environment], temperature=temperature, pressure=pressure)
                    sampler_state = SamplerState(system=systems[environment], positions=positions[environment], box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
                else:
                    thermodynamic_state = ThermodynamicState(system=systems[environment], temperature=temperature)
                    sampler_state = SamplerState(system=systems[environment], positions=positions[environment])

                mcmc_samplers[environment] = MCMCSampler(thermodynamic_state, sampler_state, storage=storage)
                mcmc_samplers[environment].nsteps = 5 # reduce number of steps for testing
                mcmc_samplers[environment].verbose = True
                exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment], chemical_state_key, proposal_engines[environment], options={'nsteps':10}, geometry_engine=geometry_engine, storage=storage)
                exen_samplers[environment].verbose = True
                sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
                sams_samplers[environment].verbose = True
                thermodynamic_states[environment] = thermodynamic_state

        # Create test MultiTargetDesign sampler.

        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = {sams_samplers['explicit-src-imatinib']: 1.0, sams_samplers['explicit-abl-imatinib']: -1.0}
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

def run_selectivity():

    storage_filename = 'output-src-abl.nc'
    system = Selectivity(storage_filename=storage_filename)
    system.designer.run(niterations=10)

    from perses.analysis import Analysis
    analysis = Analysis(storage_filename=storage_filename)
    analysis.plot_ncmc_work('ncmc-work.pdf')


if __name__ == '__main__':
    run_selectivity()

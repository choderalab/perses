import numpy as np
from perses.rjmc import geometry
from openmmtools import states, mcmc
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import cache
from perses.storage import NetCDFStorageView, NetCDFStorage
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
from typing import List, Dict

cache.global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")
cache.global_context_cache.capacity = 30


class HydrationPersesRun(object):
    """
    This class encapsulates the necessary objects to run RJ calculations for relative hydration free energies
    """

    def __init__(self, molecules: List[str], output_filename: str, ncmc_switching_times: Dict[str, int], equilibrium_steps: Dict[str, int], timestep: unit.Quantity, initial_molecule: str=None, geometry_options: Dict=None):
        self._molecules = [SmallMoleculeSetProposalEngine.canonicalize_smiles(molecule) for molecule in molecules]
        environments = ['explicit', 'vacuum']
        temperature = 300 * unit.kelvin
        pressure = 1.0 * unit.atmospheres
        constraints = app.HBonds
        self._storage = NetCDFStorage(output_filename)
        self._ncmc_switching_times = ncmc_switching_times
        self._n_equilibrium_steps = equilibrium_steps
        self._geometry_options = geometry_options

        # Create a system generator for our desired forcefields.
        from perses.rjmc.topology_proposal import SystemGenerator
        system_generators = dict()
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('perses', 'data/gaff.xml')
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        system_generators['explicit'] = SystemGenerator([gaff_xml_filename, 'tip3p.xml'],
                                                        forcefield_kwargs={'nonbondedMethod': app.PME,
                                                                           'nonbondedCutoff': 9.0 * unit.angstrom,
                                                                           'implicitSolvent': None,
                                                                           'constraints': constraints},
                                                        barostat=barostat)
        system_generators['vacuum'] = SystemGenerator([gaff_xml_filename],
                                                      forcefield_kwargs={'nonbondedMethod': app.NoCutoff,
                                                                         'implicitSolvent': None,
                                                                         'constraints': constraints})

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
        if initial_molecule:
            smiles = initial_molecule
        else:
            smiles = np.random.choice(molecules)
        molecule = createOEMolFromSMILES(smiles)
        topologies['vacuum'] = forcefield_generators.generateTopologyFromOEMol(molecule)
        positions['vacuum'] = extractPositionsFromOEMOL(molecule)

        # Create molecule in solvent.
        modeller = app.Modeller(topologies['vacuum'], positions['vacuum'])
        modeller.addSolvent(forcefield, model='tip3p', padding=9.0 * unit.angstrom)
        topologies['explicit'] = modeller.getTopology()
        positions['explicit'] = modeller.getPositions()

        # Set up the proposal engines.
        proposal_metadata = {}
        proposal_engines = dict()

        for environment in environments:
            proposal_engines[environment] = SmallMoleculeSetProposalEngine(self._molecules,
                                                                               system_generators[environment])

        # Generate systems
        systems = dict()
        for environment in environments:
            systems[environment] = system_generators[environment].build_system(topologies[environment])

        # Define thermodynamic state of interest.

        thermodynamic_states = dict()
        thermodynamic_states['explicit'] = states.ThermodynamicState(system=systems['explicit'],
                                                                     temperature=temperature, pressure=pressure)
        thermodynamic_states['vacuum'] = states.ThermodynamicState(system=systems['vacuum'], temperature=temperature)

        # Create SAMS samplers
        from perses.samplers.samplers import ExpandedEnsembleSampler, SAMSSampler
        mcmc_samplers = dict()
        exen_samplers = dict()
        sams_samplers = dict()
        for environment in environments:
            storage = NetCDFStorageView(self._storage, envname=environment)

            if self._geometry_options:
                n_torsion_divisions = self._geometry_options['n_torsion_divsions'][environment]
                use_sterics = self._geometry_options['use_sterics'][environment]

            else:
                n_torsion_divisions = 180
                use_sterics = False

            geometry_engine = geometry.FFAllAngleGeometryEngine(storage=storage, n_torsion_divisions=n_torsion_divisions, use_sterics=use_sterics)
            move = mcmc.LangevinSplittingDynamicsMove(timestep=timestep, splitting="V R O R V",
                                                       n_restart_attempts=10)
            chemical_state_key = proposal_engines[environment].compute_state_key(topologies[environment])
            if environment == 'explicit':
                sampler_state = states.SamplerState(positions=positions[environment],
                                                    box_vectors=systems[environment].getDefaultPeriodicBoxVectors())
            else:
                sampler_state = states.SamplerState(positions=positions[environment])
            mcmc_samplers[environment] = mcmc.MCMCSampler(thermodynamic_states[environment], sampler_state, move)


            exen_samplers[environment] = ExpandedEnsembleSampler(mcmc_samplers[environment], topologies[environment],
                                                                 chemical_state_key, proposal_engines[environment],
                                                                 geometry_engine,
                                                                 options={'nsteps': self._ncmc_switching_times[environment]}, storage=storage, ncmc_write_interval=self._ncmc_switching_times[environment])
            exen_samplers[environment].verbose = True
            sams_samplers[environment] = SAMSSampler(exen_samplers[environment], storage=storage)
            sams_samplers[environment].verbose = True

        # Create test MultiTargetDesign sampler.
        from perses.samplers.samplers import MultiTargetDesign
        target_samplers = {sams_samplers['explicit']: 1.0, sams_samplers['vacuum']: -1.0}
        designer = MultiTargetDesign(target_samplers, storage=self._storage)

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


if __name__=="__main__":
    import yaml
    with open('rj_hydration.yaml', 'r') as option_file:
        options_dictionary = yaml.load(option_file)

    #read the molecules into a list:
    molecule_file = options_dictionary['molecule_file']
    with open(molecule_file, 'r') as molecule_input_file:
        molecule_string = molecule_input_file.read()

    molecules = molecule_string.split("\n")

    hydration_run = HydrationPersesRun(molecules, options_dictionary['output_filename'],
                                       options_dictionary['ncmc_switching_times'],
                                       options_dictionary['equilibrium_steps'],
                                       options_dictionary['timestep'] * unit.femtoseconds)

    for environment in ['explicit', 'vacuum']:
        hydration_run.exen_samplers[environment].verbose = True
        hydration_run.sams_samplers[environment].verbose = True
        hydration_run.designer.verbose = True

    n_designer_iterations = options_dictionary['n_designer_iterations']
    hydration_run.designer.run(niterations=n_designer_iterations)

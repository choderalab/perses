import logging
import yaml
import numpy as np
import simtk.unit as unit
import os
from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
from perses.dispersed.smc import SequentialMonteCarlo
from openmmtools.multistate import MultiStateReporter, sams, replicaexchange

_logger = logging.getLogger("Simulation")
_logger.setLevel(logging.INFO)


class Simulation(object):
    """
    Create a Simulation object to handle staged equilibrium and nonequilibrium free energy sampling methods
    within a consistent API.
    """
    supported_sampler_parameters = {
                             'repex':{
                                      ##Hybrid Sampler##
                                      'timestep': 4 * unit.femtoseconds,
                                      'collision_rate': 5. / unit.picoseconds,
                                      'n_steps_per_move_application': 1,
                                      'reassign_velocities': False,
                                      'n_restart_attempts': 20,
                                      'splitting': "V R R R O R R R V",
                                      'constraint_tolerance' : 1e-6,
                                      'offline_freq': 10,

                                      ##Setup##
                                      'n_states': 13,
                                      #temperature is handled by proposal arguments/proposal_parameters
                                      'atom_selection': "not water",
                                      'checkpoint_interval': 100,
                                      'lambda_protocol': 'default',
                                      'trajectory_directory': 'repex_{index0}_to_{index1}',
                                      'trajectory_prefix': '{phase}',

                                      ##Equilibrate##
                                      "n_equilibration_iterations": 1,

                                      ##Extend##
                                      'n_cycles': 1000
                                      },

                             'sams': {'flatness_criteria': 'minimum-visits',
                                      'gamma0': 1.
                                      #the rest of the arguments are held by 'repex', which will be updated momentarily
                                     },

                             'smc': {
                                     ##__init__##
                                     'lambda_protocol': 'default',
                                     #temperature is handled by proposal arguments/proposal_parameters
                                     'trajectory_directory': 'neq_{index0}_to_{index1}',
                                     'trajectory_prefix': '{phase}',
                                     'atom_selection': "not water",
                                     'timestep': 4 * unit.femtoseconds,
                                     'collision_rate': 1. / unit.picoseconds,
                                     'eq_splitting_string': 'V R O R V',
                                     'neq_splitting_string': 'V R O R V',
                                     'ncmc_save_interval': None,
                                     'measure_shadow_work': False,
                                     'neq_integrator': 'langevin',
                                     'compute_endstate_correction': True,
                                     'external_parallelism': None,
                                     'internal_parallelism': {'library': ('dask', 'LSF'),
                                                              'num_processes': 1},

                                     ##sMC_anneal##
                                     'num_particles': 100,
                                     'protocols': {'forward': np.linspace(0,1, 1000),
                                                   'reverse': np.linspace(1,0,1000)},
                                     'directions': ['forward', 'reverse'],
                                     'num_integration_steps' : 1,
                                     'return_timer': False,
                                     'rethermalize': False,
                                     'trailblaze': None,
                                     'resample': None,

                                     ##equilibrate##
                                     'n_equilibration_iterations': 1,
                                     'n_steps_per_equilibration': 5000,
                                     'endstates': [0,1],
                                     'max_size': 1024*1e3,
                                     'decorrelate': True,
                                     'timer': False,
                                     'minimize': False
                                     }
                             }


    def __init__(self,
                 hybrid_factory,
                 sampler_type = 'repex',
                 sampler_arguments = None):
        """
        Initialization method to create simulation arguments and instantiate the samplers

        Arguments
        ---------
        hybrid_factory : perses.annihilation.relative.HybridTopologyFactory
            the hybrid topology factory containing the hybrid system to be sampled
        sampler_type : str
            the type of sampler to use
        sampler_arguments : dict or None, default None
            the non-default arguments of the sampler_type
        """
        #parse some arguments
        self.parse_arguments(sampler_type = sampler_type, sampler_arguments = sampler_arguments)
        self.hybrid_factory = hybrid_factory

        #instantiate the sampler (simulation object)
        self.create_sampler()

    def execute_sampler():
        """
        Execution method to conduct sampling (given some parameters, or default)
        """

















    def parse_arguments(sampler_type, sampler_arguments):
        """
        parse and validate init arguments

        Arguments
        ---------
        sampler_type : str
            the type of sampler to use
        sampler_arguments : dict or None, default None
            the non-default arguments of the sampler_type

        Instance Variables
        ------------------
        sampler_type : str
            the type of sampler to use
        sampler_parameters : dict
            updated arguments of the sampler type
        """

        #make some assertions
        assert sampler_type in list(self.supported_sampler_parameters.keys()), f"sampler type {sampler_type} is not supported.  Supported samplers are {list(self.supported_sampler_parameters.keys())}"
        if type(sampler_arguments) == dict:
            assert set(sampler_arguments.keys()).issubset(set(self.supported_sampler_parameters[sampler_type].keys())), f"There is at least one sampler argument that is not supported in {sampler_type} arguments"
            for keyword, _arg in sampler_arguments.items():
                assert type(_arg) == type(self.supported_sampler_parameters[sampler_type][keyword]), f"keyword '{keyword}' type '{type(_arg)}' is not the supported type '{type(self.supported_sampler_parameters[sampler_type][keyword])}'"
        elif sampler_arguments is None or sampler_arguments == {}:
            sampler_arguments = {}
            #there are no non-default arguments
            pass
        else:
            raise Exception(f"sampler_arguments must be of type 'dict' or 'NoneType'.")

        #now we can update the sampler_arguments
        self.sampler_type = sampler_type
        self.sampler_parameters = self.supported_sampler_parameters[self.sampler_type]
        self.sampler_parameters.update(sampler_arguments)



    def create_sampler():
        """
        wrapper method to instantiate the base sampler class to be used

        Instance Variables
        ------------------
        sampler : object
            the base sampler that is to be used
        """
        supported_samplers = list(self.supported_sampler_parameters.keys())

        if self.sampler_type in ['repex', 'sams']:
            mcmc_move = mcmc.LangevinSplittingDynamicsMove(timestep = self.sampler_parameters['timestep'],
                                                           collision_rate = self.sampler_parameters['collision_rate'],
                                                           n_steps = self.sampler_parameters['n_steps_per_move_application'],
                                                           reassign_velocities = self.sampler_parameters['reassign_velocities'],
                                                           n_restart_attempts = self.sampler_parameters['n_restart_attempts'],
                                                           splitting = self.sampler_parameters['splitting'],
                                                           constraint_tolerance = self.sampler_parameters['constraint_tolerance'])

            #create a multistate reporter
            storage_name = self.sampler_parameters['trajectory_directory'] + '/' + self.sampler_parameters['trajectory_prefix'] + '.nc'
            reporter = MultiStateReporter(storage_name,
                                          analysis_particle_indices = self.hybrid_factory.hybrid_topology.select(self.sampler_parameters['atom_selection']),
                                          checkpoint_interval = self.sampler_parameters['checkpoint_interval'])

            if self.sampler_type == 'repex':
                self.sampler = HybridRepexSampler(mcmc_moves = mcmc_move,
                                                  hybrid_factory = self.hybrid_factory,
                                                  online_analysis_interval = self.supported_sampler_parameters['offline_freq'])
            elif self.sampler_type == 'sams':
                self.sampler = HybridSAMSSampler(mcmc_moves = mcmc_move,
                                                 hybrid_factory = self.hybrid_factory,
                                                 online_analysis_interval = self.supported_sampler_parameters['offline_freq'],
                                                 online_analysis_minimum_iterations = 10, #perhaps this should be exposed?
                                                 flatness_criteria = self.supported_sampler_parameters['flatness_criteria'],
                                                 gamma0 = self.supported_sampler_parameters['gamma0'])
            else:
                raise Exception(f"sampler type {self.sampler_type} is not supported; this error should have been handled previously")

            #run the setup; with a check for which phase the simulation is being conducted in
            endstate_bool = False if self.sampler_parameters['trajectory_prefix'] == 'vacuum' else True
            self.sampler.setup(n_states = self.supported_sampler_parameters['n_states'],
                               temperature = temperature,
                               storage_file = reporter,
                               lambda_protocol = LambdaProtocol(functions = self.sampler_parameters['lambda_protocol']),
                               endstates = endstate_bool)

            #the self.sampler is primed for equilibrate and extend methods

        elif self.sampler_type == 'smc':
            self.sampler = SequentialMonteCarlo(factory = self.hybrid_factory,
                                                lambda_protocol = self.sampler_parameters['lambda_protocol'],
                                                temperature = self.sampler_parameters['temperature'],
                                                trajectory_directory = self.sampler_parameters['trajectory_directory'],
                                                trajectory_prefix = self.sampler_parameters['trajectory_prefix'],
                                                atom_selection = self.sampler_parameters['atom_selection'],
                                                timestep = self.sampler_parameters['timestep'],
                                                collision_rate = self.sampler_parameters['collision_rate'],
                                                eq_splitting_string = self.sampler_parameters['eq_splitting_string'],
                                                neq_splitting_string = self.sampler_parameters['neq_splitting_string'],
                                                ncmc_save_interval = self.sampler_parameters['ncmc_save_interval'],
                                                measure_shadow_work = self.sampler_parameters['measure_shadow_work'],
                                                neq_integrator = self.sampler_parameters['neq_integrator'],
                                                compute_endstate_correction = self.sampler_parameters['compute_endstate_correction'],
                                                external_parallelism = self.sampler_parameters['external_parallelism'],
                                                internal_parallelism = self.sampler_parameters['internal_parallelism'])

            self.sampler.minimize_sampler_states()

        else:
            if self.sampler_type not in supported_samplers:
                raise Exception(f"sampler type {self.sampler_type} is not a supported sampler (supported methods include {supported_samplers})")
            else:
                raise Exception(f"sampler type {self.sampler_type} is supported, but has not sampler creation method!")

















def simulation_from_yaml(filename: str):
    """ Returns a Simulation object from a yaml file

    Parameters
    ----------
    filename : str
        path to yaml filename containing simulation parameters

    Returns
    -------
    *Simulation object
        Will return a Simulation, RepexSimulation, SAMSSimulation
        or NEQSimulation object depending on the flag in the yaml file

    """
    with open(filename, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    simulations_dir = {'vanilla': Simulation, 'repex': RepexSimulation,
                       'sams': SAMSSimulation, 'neq': NEQSimulation}

    # assert that a simulation type has been specified
    # TODO add some sort of regonition if type-specific parmeters are defined
    # or default to vanilla?
    assert 'fe_type' in yaml_dict, 'FE simulation type must be specified'

    # validate the free energy simulation type that we can handle
    valid_fe_type = [sim for sim in simulations_dir.keys()]
    assert yaml_dict['fe_type'].lower() in valid_fe_type,\
        f"fe_type, {yaml_dict['fe_type']} is not recognised,'\
        ' valid fe_types are {valid_fe_type}"

    simulation = simulations_dir[yaml_dict['fe_type'].lower()](**yaml_dict)

    return simulation


class Simulation(object):
    """ Simulation object, holding parameters of basic simulations

    Parameters
    ----------
    phases : list, optional, default ['complex','solvent']
        List of phases to run: can be complex, solvent and/or vacuum
    steps : int, optional, default 1000
        Number of steps to run,
        where total simulation is steps*moves_per_step*timestep_fs
    moves_per_step : int, optional, default 250
        number of moves per simulation step
    timestep_fs : float, optional, default 4
        MD timestep in femto seconds
    eq_steps : int, optional, default 0
        number of equilibration timesteps
    pressure : float, optional, default 1.
        pressure in atm
    temperature : float, optional, default 300.
        temperature in Kelvin
    solvent_padding : float, optional, default 9.
        solvent padding in angstrom
    protein_pdb : str, optional, default None
        protein pdb filename
    ligand_file : str, optional, default None
        ligand filename
    forcefield_files : list, optional, default ['gaff.xml',
                                                'amber14/tip3p.xml',
                                                'amber14/protein.ff14SB.xml']
        filename of forcefields
    eq_splitting : str default 'V R O R V'
        equilibrium splitting string
    max_file_size : float, optional, default 10 * 1024e3
        maximum file size, default 10 MB
    checkpoint_interval : int, optional, default 100
        interval of steps at which to print checkpoint files
    output_directory : str, optional, default out
        directory for output
    output_prefix : str, optional, default out
        prefix of output files
    setup_pickle : str, optional, default setup.pkl
        filename for pickle of setup output
    atom_selection : str, optional, default all
        MDTraj selection syntax

    Attributes
    ----------
    phases
    steps
    moves_per_step
    timestep_fs
    eq_steps
    pressure
    temperature
    solvent_padding
    forcefield_files
    eq_splitting
    max_file_size
    output_directory
    output_prefix
    setup_pickle
    atom_selection

    """
    def __init__(self,
                 ligand_file: str,
                 protein_pdb: str = None,
                 phases: list = ['complex', 'solvent'],
                 steps: int = 1000,
                 moves_per_step: int = 250,
                 timestep_fs: float = 4.,
                 eq_steps: int = 0,
                 pressure: float = 1.,
                 temperature: float = 300.,
                 solvent_padding: float = 9.,
                 forcefield_files: list = [
                     'gaff.xml', 'amber14/tip3p.xml',
                     'amber14/protein.ff14SB.xml'],
                 eq_splitting: str = 'V R R R O R R R V',
                 max_file_size: float = 10 * 1024e3,
                 checkpoint_interval: int = 100,
                 output_directory: str = 'out',
                 output_prefix: str = 'out',
                 setup_pickle: str = 'setup.pkl',
                 atom_selection: str = 'all'):
        self.ligand = str(ligand_file)
        self.protein = protein_pdb
        # simulation basics
        self.phases = phases
        self._validate_phases()
        self.steps = int(steps)
        self.moves_per_step = int(moves_per_step)
        self.timestep_fs = float(timestep_fs)
        self._timestep = self.timestep_fs * unit.femtoseconds
        self.eq_steps = int(eq_steps)
        self.pressure = float(pressure) * unit.atmosphere
        self.temperature = float(temperature) * unit.kelvin
        self.solvent_padding = float(solvent_padding)
        self._solvent_padding_angstrom = self.solvent_padding * unit.angstrom
        self.forcefield_files = list(forcefield_files)
        self.eq_splitting = eq_splitting
        # details of outputs
        self.max_file_size = float(max_file_size)
        self.checkpoint_interval = int(checkpoint_interval)
        self.output_directory = str(output_directory)
        self._make_output_directory()
        self.output_prefix = str(output_prefix)
        self.setup_pickle = str(setup_pickle)
        self.atom_selection = atom_selection
        self._constraint_tolerance = 1e-06

    def _make_output_directory(self):
        assert (self._check_output_directory() is False), \
            'Output trajectory directory already exists. Refusing to overwrite'
        _logger.info(f'Making output directory: {self.output_directory}')
        os.makedirs(self.output_directory)

    # validation functions
    def _validate_phases(self):
        """Checks that phases passed to object are of recognised type

        Returns
        -------

        """
        assert isinstance(self.phases, list), \
            'Simulation.phases must be a list'

        valid_phases = ['complex', 'solvent', 'vacuum']
        for phase in self.phases:
            assert phase in valid_phases, \
                'phase {phase} not recognised, valid phases are {valid_phases}'
        if 'complex' in self.phases:
            assert self.protein is not None, \
                'Need a protein file to run a complex phase'
        return

    def _check_output_directory(self):
        """Checks to see if the output directory has been made

        Returns
        -------
        bool
            True or False if directory exists

        """
        import os
        return os.path.isdir(self.output_directory)

    def write_parameters(self, filename='config.yaml'):
        """Writes a .yaml file of all the parameters of the Simulation object
        this can be used to run a simulation with the same parameters.
        Private variables are ignored.

        Parameters
        ----------
        filename : str
            name of yaml file to save data to

        Returns
        -------

        """
        details = vars(self)
        details = {k: v for k, v in details.items() if not k.startswith('_')}
        with open(filename, 'w') as yaml_file:
            yaml.dump(details, yaml_file, default_flow_style=False)
        return


class AlchemicalSimulation(Simulation):
    """ Base class for alchemical simulation parameters.

    Parameters
    ----------
    old_ligand_index : int
        integer of old ligand in ligand file
    new_ligand_index : int
        integer of new ligand in ligand file
    lambda_functions : str or dict, optional, default = 'default'
        either a dictionary of functions to describe the perturbations of alchemical energies
        or a string refering to a dictionary known to perses ['default', 'namd', 'quarters']
    n_lambdas : int, optional, default 11
        number of lambda windows between 0 and 1
    softcore_v2 : bool, optional, default False
        softcore v2 will be used if True
    neglect_angles : bool, default False
        if True, angles will be neglected in FEPSetup
    atom_map_file : str default None
        file containing alchemical atom mapping
    mapping_strength : str, optional, default 'default'
        Strength of which to map one molecule onto the other
        can be weak (most atoms mapped), default or strong (least atoms mapped)
    anneal_14s : bool, default False
        Whether to anneal 1,4 interactions over the protocol
    topology_proposal : default None
        topology proposal object
    offline_freq : int, default 1
        frequency of offline FE evaluations
    Attributes
    ----------
    lambda_functions
    n_lambdas
    neglect_angles
    mapping_strength
    anneal_14s
    softcore_v2
    old_ligand_index
    new_ligand_index
    topology_proposal
    """
    def __init__(self,
                 *args,
                 old_ligand_index: int,
                 new_ligand_index: int,
                 lambda_functions='default',
                 n_lambdas: int = 11,
                 softcore_v2: bool = False,
                 neglect_angles: bool = False,
                 atom_map_file: str = None,
                 mapping_strength: str = 'default',
                 anneal_14s: bool = False,
                 topology_proposal: None = None,
                 offline_freq: int = 1,
                 **kwargs):
        super(AlchemicalSimulation, self).__init__(*args, **kwargs)
        from perses.annihilation.lambda_protocol import LambdaProtocol
        self.old_ligand_index = int(old_ligand_index)
        self.new_ligand_index = int(new_ligand_index)
        self.lambda_functions = lambda_functions
        self._lambda_protocol = LambdaProtocol(self.lambda_functions)
        self.n_lambdas = int(n_lambdas)
        self.neglect_angles = bool(neglect_angles)
        self.atom_map_file = atom_map_file
        self._validate_atom_map_file()
        self.mapping_strength = mapping_strength
        self._validate_mapping_strength()
        self.anneal_14s = bool(anneal_14s)
        self.softcore_v2 = bool(softcore_v2)
        self.old_ligand_index = int(old_ligand_index)
        self.new_ligand_index = int(new_ligand_index)
        self.offline_freq = int(offline_freq)
        self._topology_proposal = topology_proposal
        self._setup = False

    # validation functions
    def _validate_mapping_strength(self):
        """Checks that the mapping strength passed to object are
        of recognised type

        Returns
        -------

        """
        assert isinstance(self.mapping_strength, str), \
            'Simulation.phases must be a list'

        valid_mapping_strengths = ['default', 'weak', 'strong']
        assert self.mapping_strength in valid_mapping_strengths, \
            f'phase {self.mapping_strength} not recognised,'\
            ' valid phases are {valid_mapping_strengths}'
        return

    def _validate_atom_map_file(self):
        """Loads the atom mapping from file, if file is provided

        Returns
        -------

        """
        if self.atom_map_file is None:
            self._atom_map = None
        else:
            with open(self.atom_map_file, 'r') as f:
                self._atom_map = {int(x.split()[0]): int(x.split()[1])
                                  for x in f.readlines()}
        return

    def setup_fe(self):
        from perses.app.relative_setup import RelativeFEPSetup
        from perses.annihilation.relative import HybridTopologyFactory
        if self._topology_proposal is None:
            _logger.info(f'Running FEP setup')
            self._fe_setup = RelativeFEPSetup(self.ligand,
                                              self.old_ligand_index,
                                              self.new_ligand_index,
                                              self.forcefield_files,
                                              phases=self.phases,
                                              protein_pdb_filename=self.protein,
                                              pressure=self.pressure,
                                              temperature=self.temperature,
                                              solvent_padding=self._solvent_padding_angstrom,
                                              atom_map=self._atom_map,
                                              neglect_angles=self.neglect_angles,
                                              anneal_14s=self.anneal_14s)
            _logger.info(f'Finished FEP setup')
            self._save_setup()
            _logger.info(f'Setup saved')
        else:
            # loading the topology proposal from pickle file
            self._fe_setup = self._load_topology_proposal()
        self._render_atom_mapping()
        _logger.info(f"\tsetup is complete.  Writing proposals and '\
                     'positions for each phase to top_prop dict...")

        for phase in self.phases:
            # generate the HybridTopologyFactory
            hybrid_factory = HybridTopologyFactory(getattr(self._fe_setup, f'{phase}_topology_proposal'),
                                                   getattr(self._fe_setup, f'{phase}_old_positions'),
                                                   getattr(self._fe_setup, f'{phase}_new_positions'),
                                                   neglected_new_angle_terms=getattr(self._fe_setup, f'_{phase}_forward_neglected_angles'),
                                                   neglected_old_angle_terms=getattr(self._fe_setup, f'_{phase}_reverse_neglected_angles'),
                                                   softcore_LJ_v2=self.softcore_v2,
                                                   interpolate_old_and_new_14s=self.anneal_14s)
            setattr(self, f'{phase}_hybrid_factory', hybrid_factory)
        return

    def _save_setup(self):
        import pickle
        path = os.path.join(os.getcwd(), self.output_directory,
                            self.setup_pickle)
        with open(path, 'wb') as f:
            try:
                pickle.dump(self._fe_setup, f)
                _logger.info(f"\tsuccessfully dumped pickle to {path}.")
            except Exception as e:
                print(e)
                print("\tUnable to save setup object as a pickle")

    def _load_topology_proposal(self):
        import pickle
        path = os.path.join(os.getcwd(), self.output_directory,
                            self._topology_proposal)
        with open(path, 'rb') as f:
            try:
                fe_setup = pickle.load(f)
                _logger.info(f"\tsuccessfully loaded setup pickle.")
            except Exception as e:
                print(e)
                print("\tUnable to load setup object as a pickle")
        return fe_setup

    def _render_atom_mapping(self, outfile: str = 'mapping.png'):
        from perses.utils.smallmolecules import render_atom_mapping
        path = os.path.join(os.getcwd(), self.output_directory, outfile)
        render_atom_mapping(path, self._fe_setup._ligand_oemol_old,
                            self._fe_setup._ligand_oemol_new,
                            self._fe_setup.non_offset_new_to_old_atom_map)
        return

class SAMSSimulation(AlchemicalSimulation):
    """ Class for SAMS simulation

    Parameters
    ----------
    fe_type : str
        indentifies simulation as SAMS
    flatness_criteria : str, default 'minimum-visits'
        criteria to distinguish 0th and 1st stage of SAMS
    gamma0 : float, default 1.
        SAMS gamma0 parameter
    beta_factor : float, default 0.8
        SAMS beta factor

    Attributes
    ----------
    fe_type
    flatness_criteria
    offline_freq
    gamma0
    beta_factor

    """
    def __init__(self,
                 *args,
                 fe_type: str,
                 flatness_criteria: str = 'minimum-visits',
                 gamma0: float = 1.,
                 beta_factor: float = 0.8,
                 **kwargs):
        super(SAMSSimulation, self).__init__(*args, **kwargs)
        self.fe_type = fe_type
        self.flatness_criteria = flatness_criteria
        self._validate_flatness_criteria()
        self.gamma0 = float(gamma0)
        self.beta_factor = float(beta_factor)

    def _validate_flatness_criteria(self):
        """Checks that the flatness criteria passed to object are
        of recognised type

        Returns
        -------

        """
        assert isinstance(self.flatness_criteria, str), \
            'AlchemicalSimulation.phases must be a str'

        valid_flatness_criteria = ['minimum-visits',
                                   'histogram-flatness',
                                   'logZ-flatness']
        assert self.flatness_criteria in valid_flatness_criteria, \
            f'criteria {self.flatness_criteria} not recognised,'\
            ' valid phases are {valid_flatness_criteria}'
        return

    def setup(self):
        from perses.samplers.multistate import HybridSAMSSampler
        from openmmtools import mcmc
        from openmmtools.multistate import MultiStateReporter
        self.setup_fe()

        # this can go in Alchemical
        move = mcmc.LangevinSplittingDynamicsMove(timestep=self._timestep,
                                                  collision_rate=5.0 / unit.picosecond,
                                                  n_steps=self.steps,
                                                  reassign_velocities=False,
                                                  splitting=self.eq_splitting,
                                                  constraint_tolerance=self._constraint_tolerance)

        for phase in self.phases:
            # make a reporter
            storage_name = f'{self.output_directory}/{self.output_prefix}-{phase}.nc'
            _logger.info(f'Storing {phase} at {storage_name}')
            reporter = MultiStateReporter(storage_name,
                                          analysis_particle_indices=self.atom_selection,
                                          checkpoint_interval=self.checkpoint_interval)

            # make the sampler
            _logger.info(f'Generating SAMS sampler for {phase} phase')
            sampler = HybridSAMSSampler(mcmc_moves=move,
                                         hybrid_factory=getattr(self, f'{phase}_hybrid_factory'),
                                         online_analysis_interval=self.offline_freq)

            sampler.setup(n_states=self.n_lambdas,
                          temperature=self.temperature,
                          storage_file=reporter,
                          lambda_protocol=self._lambda_protocol)
            setattr(self, f'_{phase}_sampler', sampler)
        self._setup = True
        return

    def run(self):
        assert self._setup is True, \
            'Simulation is not set up. Run .setup() before .run()'
        for phase in self.phases:
            sampler = getattr(self,f'{phase}_sampler')
            _logger.info(f'Running {self.eq_steps} of equilibration for {phase} phase')
            sampler.equilibrate(self.eq_steps)
            _logger.info(f'Finished {self.eq_steps} of equilibration for {phase} phase')

            _logger.info(f'Running {self.steps} of production for {phase} phase')
            sampler.extend(self.steps)
            _logger.info(f'Finished {self.steps} of production for {phase} phase')


class RepexSimulation(AlchemicalSimulation):
    """ Class for REPEX simulation

    Parameters
    ----------
    fe_type : str
        indentifies simulation as REPEX

    Attributes
    ----------
    fe_type

    """
    def __init__(self, *args, fe_type, **kwargs):
        super(RepexSimulation, self).__init__(*args, **kwargs)
        self.fe_type = fe_type

    def setup(self):
        from perses.samplers.multistate import HybridRepexSampler
        from openmmtools import mcmc
        from openmmtools.multistate import MultiStateReporter
        self.setup_fe()

        # this can go in Alchemical
        move = mcmc.LangevinSplittingDynamicsMove(timestep=self._timestep,
                                                  collision_rate=5.0 / unit.picosecond,
                                                  n_steps=self.steps,
                                                  reassign_velocities=False,
                                                  splitting=self.eq_splitting,
                                                  constraint_tolerance=self._constraint_tolerance)

        for phase in self.phases:
            # make a reporter
            storage_name = f'{self.output_directory}/{self.output_prefix}-{phase}.nc'
            _logger.info(f'Storing {phase} at {storage_name}')
            reporter = MultiStateReporter(storage_name,
                                          analysis_particle_indices=self.atom_selection,
                                          checkpoint_interval=self.checkpoint_interval)

            # make the sampler
            _logger.info(f'Generating REPEX sampler for {phase} phase')
            sampler = HybridRepexSampler(mcmc_moves=move,
                                         hybrid_factory=getattr(self, f'{phase}_hybrid_factory'),
                                         online_analysis_interval=self.offline_freq)

            sampler.setup(n_states=self.n_lambdas,
                          temperature=self.temperature,
                          storage_file=reporter,
                          lambda_protocol=self._lambda_protocol)
            setattr(self, f'{phase}_sampler', sampler)
        self._setup = True
        return

    def run(self):
        assert self._setup is True, \
            'Simulation is not set up. Run .setup() before .run()'
        for phase in self.phases:
            sampler = getattr(self,f'{phase}_sampler')
            _logger.info(f'Running {self.eq_steps} of equilibration for {phase} phase')
            sampler.equilibrate(self.eq_steps)
            _logger.info(f'Finished {self.eq_steps} of equilibration for {phase} phase')

            _logger.info(f'Running {self.steps} of production for {phase} phase')
            sampler.extend(self.steps)
            _logger.info(f'Finished {self.steps} of production for {phase} phase')


class NEQSimulation(AlchemicalSimulation):
    """ Class for NEQ simulation

    Parameters
    ----------
    fe_type : str
        indentifies simulation as NEQ
    n_particles : int
        The number of times to run the entire sequence
    stage : str, default equilibrium
        Identify if NEQ is at equilibrium or annealing phase
    measure_shadow_work : bool default False
        wether to measure shadow work
    write_ncmc_configuration : bool False
        whether to write ncmc annealing perturbations
        if True, will write every ncmc_save_interval iterations
    direction : type
        Description of parameter `direction`.
    neq_splitting : str default 'V R O R V'
        NEQ splitting string
    ncmc_collision_rate_ps : type, default inf
        ncmc collision rate in picoseconds
    adapt : bool, default False
        wether to use an adaptive scheduler, if lsf is True
    lsf : bool default True
        If using LSF DASK client
    processes : int, default 100
        number of processes to use if lsf is True

    Attributes
    ----------
    fe_type
    stage
    measure_shadow_work
    write_ncmc_configuration
    direction
    neq_splitting
    ncmc_collision_rate_ps
    n_particles
    processes
    adapt
    lsf

    """
    def __init__(self,
                 *args,
                 fe_type: str,
                 n_particles: int,
                 stage: str = 'equilibrium',
                 measure_shadow_work: bool = False,
                 write_ncmc_configuration: bool = False,
                 direction=None,
                 neq_splitting: str = 'V R O R V',
                 ncmc_collision_rate_ps=np.inf,
                 relative_transform: bool = True,
                 observable: str = 'ESS',
                 trailblaze_observable_threshold: float = 0.,
                 resample_observable_threshold: float = 0.,
                 resampling_method: str = 'multinomial',
                 decorrelate: bool = True,
                 timer: bool = True,
                 minimize: bool = False,
                 online_protocol=None,
                 processes: int = 100,
                 adapt: bool = True,
                 lsf: bool = True,
                 **kwargs):
        super(NEQSimulation, self).__init__(*args, **kwargs)
        self.fe_type = fe_type
        self.n_particles = int(n_particles)
        self.stage = str(stage)
        self._validate_stage()
        self.measure_shadow_work = bool(measure_shadow_work)
        self.write_ncmc_configuration = bool(write_ncmc_configuration)
        self.direction = direction
        self._validate_direction()
        self.neq_splitting = neq_splitting
        self.ncmc_collision_rate_ps = ncmc_collision_rate_ps / unit.picoseconds
        self.relative_transform = bool(relative_transform)
        self.observable = str(observable).upper(),
        self._validate_observable()
        self.trailblaze_observable_threshold = float(trailblaze_observable_threshold),
        self.resample_observable_threshold = float(resample_observable_threshold),
        self._validate_observable_thresholds()
        self.resampling_method = str(resampling_method)
        self._validate_resampling_method()
        self.decorrelate = bool(decorrelate),
        self.timer = bool(timer),
        self.minimize = bool(minimize),
        self.online_protocol = online_protocol
        self._validate_online_protocol()
        # DASK
        self.processes = int(processes)
        self.adapt = bool(adapt)
        self.lsf = bool(lsf)
        self._make_output_directory()

    # validation functions
    def _validate_stage(self):
        """Checks that the stage passed to object is
        of recognised type

        Returns
        -------

        """
        import pickle

        valid_stages = ['equilibrium', 'anneal']
        assert isinstance(self.stage, str), \
            'NEQSimulation.stage must be a str'

        assert self.stage in valid_stages, \
            f'stage {self.stage} not recognised,'\
            f' valid phases are {valid_stages}'

        if self.stage == 'anneal':
            assert self.n_particles is not None, \
                'If running annealing, n_particles must be specified'

            # checking that the necessary equilibrium output exists
            for phase in self.phases:
                path = os.path.join(f'{self.output_directory}/{self.output_prefix}_{phase}_fep.eq.pkl')
                assert os.path.exists(path), f'Cannot find equilibrium output for {phase}'
                with open(path,'rb') as f:
                    fep = pickle.load(f)
                setattr(self, f'eq_{phase}_fep', fep)
        return

    def _validate_observable(self):
        valid_observables = ['ESS', 'CESS']
        assert self.observable in valid_observables, \
            f'observable {self.observable} not recognised'\
            f'valid observables are {valid_observables}'
        return

    def _validate_resampling_method(self):
        assert self.resampling_method == 'multinomial', \
            'only "multinomial" is currently the only supported resampling method'\
            f'{self.resampling_method} is not supported'


    def _validate_direction(self):
        """Checks that the direction passed to object is
        of recognised type

        Returns
        -------

        """
        valid_directions = [None, 'forward', 'reverse']
        assert self.direction in valid_directions, \
            f'stage {self.direction} not recognised, '\
            'valid phases are {valid_directions}'
        if self.direction is None:
            self._endstates = [0, 1]
        elif self.direction == 'forward':
            self._endstates = [0]
        elif self.direction == 'reverse':
            self._endstates = [1]
        return

    def _validate_online_protocol(self):
        if isinstance(self.online_protocol, int):
            self.online_protocol = {'forward',np.linspace(0., 1., self.online_protocol),
                                    'reverse',np.linspace(1., 0., self.online_protocol)}
        elif os.path.exists(self.online_protocol):
            self.online_protocol = np.load(self.online_protocol, allow_pickle=True)


    def _validate_observable_thresholds(self):
        assert 0. <= self.trailblaze_observable_threshold <= 1., \
            f'trailblaze_observable_threshold must be between 0. and 1'\
            f'current trailblaze_observable_threshold is {self.trailblaze_observable_threshold}'
        assert 0. <= self.resample_observable_threshold <= 1., \
            f'resample_observable_threshold must be between 0. and 1'\
            f'current resample_observable_threshold is {self.resample_observable_threshold}'

    def _make_output_directory(self):
        if self.stage == 'equilibrium':
            assert (self._check_output_directory() is False), \
                'Output trajectory directory already exists. Refusing to overwrite'
            _logger.info(f'Making output directory: {self.output_directory}')
            os.makedirs(self.output_directory)
        elif self.stage == 'anneal':
            assert self._check_output_directory() is True, \
                f'Cannot run annealling if output directory, {self.output_directory}, does not exist'

    def setup(self):
        self.setup_fe()

        for phase in self.phases:
            _logger.info(f'Generating NEQ sampler for {phase} phase')
            ne_fep = NonequilibriumSwitchingFEP(hybrid_factory=getattr(self, f'{phase}_hybrid_factory'),
                                                protocol=self._lambda_protocol,
                                                n_equilibrium_steps_per_iteration=self.n_steps,
                                                temperature = self.temperature,
                                                trajectory_directory=self.output_directory,
                                                trajectory_prefix=f'{self.output_prefix}_{phase}',
                                                atom_selection=self.atom_selection,
                                                eq_splitting_string=self.eq_splitting_string,
                                                neq_splitting_string=self.neq_splitting_string,
                                                measure_shadow_work=self.measure_shadow_work,
                                                timestep=self._timestep,
                                                ncmc_save_interval=self.checkpoint_interval,
                                                write_ncmc_configuration=self.write_ncmc_configuration,
                                                relative_transform=self.relative_transform)
            setattr(self, f'_{phase}_sampler', ne_fep)
        self._setup = True
        return

    def run(self):
        if self.stage == 'equilibrium':
            self.run_equilibrium()
        elif self.stage == 'anneal':
            self.run_anneal()
        return

    def run_equilibrium(self):
        _logger.info('Running equilibrium stage of NEQ simulation.')
        for phase in self.phases:
            _logger.info(f'Running {phase} phase')
            sampler = getattr(self, f'_{phase}_sampler')
            sampler.equilibrate(self.n_steps,
                                endstates=self._endstates,
                                max_size=self.max_file_size,
                                decorrelate=self.decorrelate,
                                timer=self.timer,
                                minimize=self.minimize,
                                LSF=self.lsf,
                                num_processes=2,  # currently always 2 for equilibrium
                                adapt=self.adapt)
            _logger.info(f'Finished {phase} phase')
            _logger.info(f'Storing output')
            path = os.path.join(f"{self.output_directory}/{self.output_prefix}_{phase}_fep.eq.pkl")
            with open(path, 'wb') as f:
                pickle.dump(ne_fep_run, f)
            _logger.info(f'Finished storing {phase} phase')

    # def run_anneal(self):
    #     _logger.info('Running annealing stage of NEQ simulation.')
    #     for phase in self.phases:

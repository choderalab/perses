import yaml
import numpy as np
import pickle
import os
import sys
import simtk.unit as unit
from simtk import openmm
import logging

from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
from perses.annihilation.relative import HybridTopologyFactory
from perses.app.relative_setup import NonequilibriumSwitchingFEP, RelativeFEPSetup
from perses.annihilation.lambda_protocol import LambdaProtocol

from openmmtools import mcmc, utils
from openmmtools.multistate import MultiStateReporter, sams, replicaexchange
from perses.utils.smallmolecules import render_atom_mapping
from perses.tests.utils import validate_endstate_energies
from perses.dispersed.smc import SequentialMonteCarlo

import datetime
class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated
        delta = datetime.datetime.fromtimestamp(record.relativeCreated/1000.0) - datetime.datetime.fromtimestamp(last/1000.0)
        record.relative = '{0:.2f}'.format(delta.seconds + delta.microseconds/1000000.0)
        self.last = record.relativeCreated
        return True

fmt = logging.Formatter(fmt="%(asctime)s:(%(relative)ss):%(name)s:%(message)s")
#logging.basicConfig(level = logging.NOTSET)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
[hndl.addFilter(TimeFilter()) for hndl in _logger.handlers]
[hndl.setFormatter(fmt) for hndl in _logger.handlers]

ENERGY_THRESHOLD = 1e-4
from openmmtools.constants import kB

def getSetupOptions(filename):
    """
    Reads input yaml file, makes output directory and returns setup options
    Parameter
    ---------
    filename : str
        .yaml file containing simulation parameters
    Returns
    -------
    setup_options :
        options provided in the yaml file
    phases : list of strings
        phases to simulate, can be 'complex', 'solvent' or 'vacuum'
    """
    yaml_file = open(filename, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()

    _logger.info("\tDetecting phases...")
    if 'phases' not in setup_options:
        setup_options['phases'] = ['complex','solvent']
        _logger.warning('\t\tNo phases provided - running complex and solvent as default.')
    else:
        _logger.info(f"\t\tphases detected: {setup_options['phases']}")

    if 'protocol-type' not in setup_options:
        setup_options['protocol-type'] = 'default'


    if 'small_molecule_forcefield' not in setup_options:
        setup_options['small_molecule_forcefield'] = 'gaff-2.11'

    if 'small_molecule_parameters_cache' not in setup_options:
        setup_options['small_molecule_parameters_cache'] = None

    if 'spectators' not in setup_options:
        setup_options['spectators'] = None

    # Not sure why these are needed
    # TODO: Revisit these?
    if 'neglect_angles' not in setup_options:
        setup_options['neglect_angles'] = False
    if 'anneal_1,4s' not in setup_options:
        setup_options['anneal_1,4s'] = False

    if 'nonbonded_method' not in setup_options:
        setup_options['nonbonded_method'] = 'PME'

    if 'run_type' not in setup_options:
        _logger.info(f"\t\t\trun_type is not specified; default to None")
        setup_options['run_type'] = None
    _logger.info(f"\tDetecting fe_type...")
    if setup_options['fe_type'] == 'sams':
        _logger.info(f"\t\tfe_type: sams")
        # check if some parameters are provided, otherwise use defaults
        if 'flatness-criteria' not in setup_options:
            setup_options['flatness-criteria'] = 'minimum-visits'
            _logger.info(f"\t\t\tflatness-criteria not specified: default to minimum-visits.")
        if 'offline-freq' not in setup_options:
            setup_options['offline-freq'] = 10
            _logger.info(f"\t\t\toffline-freq not specified: default to 10.")
        if 'gamma0' not in setup_options:
            setup_options['gamma0'] = 1.
            _logger.info(f"\t\t\tgamma0 not specified: default to 1.0.")
        if 'beta_factor' not in setup_options:
            setup_options['beta_factor'] = 0.8
            _logger.info(f"\t\t\tbeta_factor not specified: default to 0.8.")
        if 'n_replicas' not in setup_options:
            setup_options['n_replicas'] = 1
    elif setup_options['fe_type'] == 'repex':
        _logger.info(f"\t\tfe_type: repex")
        if 'offline-freq' not in setup_options:
            setup_options['offline-freq'] = 10
            _logger.info(f"\t\t\toffline-freq not specified: default to 10.")
    elif setup_options['fe_type'] == 'neq': #there are some neq attributes that are not used with the equilibrium samplers...
        _logger.info(f"\t\tfe_type: neq")
        if 'n_equilibrium_steps_per_iteration' not in setup_options:
            _logger.info(f"\t\t\tn_equilibrium_steps_per_iteration not specified: default to 1000.")
            setup_options['n_equilibrium_steps_per_iteration'] = 1000
        if 'measure_shadow_work' not in setup_options:
            _logger.info(f"\t\t\tmeasure_shadow_work not specified: default to False")
            setup_options['measure_shadow_work'] = False
        if 'write_ncmc_configuration' not in setup_options:
            _logger.info(f"\t\t\twrite_ncmc_configuration not specified: default to False")
            setup_options['write_ncmc_configuration'] = False
        if 'neq_integrator' not in setup_options:
            _logger.info(f"\t\t\tneq_integrator not specified; default to 'langevin'")
            setup_options['neq_integrator'] = 'langevin'

        #for dask implementation
        if 'processes' not in setup_options:
            _logger.info(f"\t\t\tprocesses is not specified; default to 0")
            setup_options['processes'] = 0
        if 'adapt' not in setup_options:
            _logger.info(f"\t\t\tadapt is not specified; default to True")
            setup_options['adapt'] = True
        if 'max_file_size' not in setup_options:
            _logger.info(f"\t\t\tmax_file_size is not specified; default to 10MB")
            setup_options['max_file_size'] = 10*1024e3
        if 'lambda_protocol' not in setup_options:
            _logger.info(f"\t\t\tlambda_protocol is not specified; default to 'default'")
            setup_options['lambda_protocol'] = 'default'
        if 'LSF' not in setup_options:
            _logger.info(f"\t\t\tLSF is not specified; default to False")
            setup_options['LSF'] = False

        if 'run_type' not in setup_options:
            _logger.info(f"\t\t\trun_type is not specified; default to None")
            setup_options['run_type'] = None
        elif setup_options['run_type'] == 'anneal':
            if 'out_trajectory_prefix' not in setup_options:
                raise Exception(f"'out_trajectory_prefix' must be defined if 'anneal' is called.  Aborting!")
            _logger.info(f"'run_type' was called as {setup_options['run_type']} attempting to detect file")
            for phase in setup_options['phases']:
                path = os.path.join(setup_options['trajectory_directory'], f"{setup_options['trajectory_prefix']}_{phase}_fep.eq.pkl")
                if os.path.exists(path):
                    _logger.info(f"\t\t\tfound {path}; loading and proceeding to anneal")
                else:
                    raise Exception(f"{path} could not be found.  Aborting!")
        elif setup_options['run_type'] == 'None':
            setup_options['run_type'] = None
        elif str(setup_options['run_type']) not in ['None', 'anneal', 'equilibrate']:
            raise Exception(f"'run_type' must be None, 'anneal', or 'equilibrate'; input was specified as {setup_options['run_type']} with type {type(setup_options['run_type'])}")

        #to instantiate the particles:

        if 'trailblaze' not in setup_options:
            assert 'lambdas' in setup_options, f"'lambdas' is not in setup_options, and 'trailblaze' is False. One must be specified.  Aborting!"
            assert type(setup_options['lambdas']) == int, f"lambdas is not an int.  Aborting!"
            setup_options['trailblaze'] = None
        else:
            assert type(setup_options['trailblaze']) == dict, f"trailblaze is specified, but is not a dict"

        if 'resample' in setup_options:
            assert type(setup_options['resample']) == dict, f"'resample' is not a dict"
            assert set(['criterion', 'method', 'threshold']).issubset(set(list(setup_options['resample'].keys()))), f"'resample' does not contain necessary keys"
        else:
            _logger.info(f"\t\tresample is not specified; defaulting to None")
            setup_options['resample'] = None

        if 'n_particles' not in setup_options:
            raise Exception(f"for particle annealing, 'n_particles' must be specified")
        if 'direction' not in setup_options:
            _logger.info(f"\t\t\tdirection is not specified; default to (running both forward and reverse)")
            setup_options['direction'] = ['forward', 'reverse']
        else:
            _logger.info(f"\t\t\tthe directions are as follows: {setup_options['direction']}")

        if 'ncmc_save_interval' not in setup_options:
            _logger.info(f"\t\t\tncmc_save_interval not specified: default to None.")
            setup_options['ncmc_save_interval'] = None
        if 'ncmc_collision_rate_ps' not in setup_options:
            _logger.info(f"\t\t\tcollision_rate not specified: default to np.inf.")
            setup_options['ncmc_collision_rate_ps'] = np.inf/unit.picoseconds
        else:
            setup_options['ncmc_collision_rate_ps'] /= unit.picoseconds
        if 'ncmc_rethermalize' not in setup_options:
            _logger.info(f"\t\t\tncmc_rethermalize not specified; default to False.")
            setup_options['ncmc_rethermalize'] = False

        #now lastly, for the algorithm_4 options:
        if 'observable' not in setup_options:
            _logger.info(f"\t\t\tobservable is not specified; default to ESS")
            setup_options['observable'] = 'ESS'
        if 'trailblaze_observable_threshold' not in setup_options:
            _logger.info(f"\t\t\ttrailblaze_observable_threshold is not specified; default to 0.0")
            setup_options['trailblaze_observable_threshold'] = None
        if 'resample_observable_threshold' not in setup_options:
            _logger.info(f"\t\t\tresample_observable_threshold is not specified; default to 0.0")
            setup_options['resample_observable_threshold'] = None
        if 'ncmc_num_integration_steps' not in setup_options:
            _logger.info(f"\t\t\tncmc_num_integration_steps is not specified; default to 1")
            setup_options['ncmc_num_integration_steps'] = 1
        if 'resampling_method' not in setup_options:
            _logger.info(f"\t\t\tresampling_method is not specified; default to 'multinomial'")
            setup_options['resampling_method'] = 'multinomial'
        if 'online_protocol' not in setup_options:
            _logger.info(f"\t\t\tonline_protocol is not specified; default to None")
            setup_options['online_protocol'] = None



        setup_options['n_steps_per_move_application'] = 1 #setting the writeout to 1 for now

    trajectory_directory = setup_options['trajectory_directory']

    # check if the neglect_angles is specified in yaml

    if 'neglect_angles' not in setup_options:
        setup_options['neglect_angles'] = False
        _logger.info(f"\t'neglect_angles' not specified: default to 'False'.")
    else:
        _logger.info(f"\t'neglect_angles' detected: {setup_options['neglect_angles']}.")

    if 'atom_expression' in setup_options:
        # need to convert the list to Integer
        from perses.utils.openeye import generate_expression
        setup_options['atom_expr'] = generate_expression(setup_options['atom_expression'])
    else:
        setup_options['atom_expr'] = None

    if 'bond_expression' in setup_options:
        # need to convert the list to Integer
        from perses.utils.openeye import generate_expression
        setup_options['bond_expr'] = generate_expression(setup_options['bond_expression'])
    else:
        setup_options['bond_expr'] = None

    if 'map_strength' not in setup_options:
        setup_options['map_strength'] = None 

    if 'anneal_1,4s' not in setup_options:
        setup_options['anneal_1,4s'] = False
        _logger.info(f"\t'anneal_1,4s' not specified: default to 'False' (i.e. since 1,4 interactions are not being annealed, they are being used to make new/old atom proposals in the geometry engine.)")

    if 'softcore_v2' not in setup_options:
        setup_options['softcore_v2'] = False
        _logger.info(f"\t'softcore_v2' not specified: default to 'False'")

    _logger.info(f"\tCreating '{trajectory_directory}'...")
    assert (not os.path.exists(trajectory_directory)), f'Output trajectory directory "{trajectory_directory}" already exists. Refusing to overwrite'
    os.makedirs(trajectory_directory)


    return setup_options

def run_setup(setup_options):
    """
    Run the setup pipeline and return the relevant setup objects based on a yaml input file.
    Parameters
    ----------
    setup_options : dict
        result of loading yaml input file
    Returns
    -------
    setup_dict: dict
        {'topology_proposals': top_prop, 'hybrid_topology_factories': htf, 'hybrid_samplers': hss}
        - 'topology_proposals':
    """
    phases = setup_options['phases']
    known_phases = ['complex','solvent','vacuum']
    for phase in phases:
        assert (phase in known_phases), f"Unknown phase, {phase} provided. run_setup() can be used with {known_phases}"

    if 'complex' in phases:
        _logger.info(f"\tPulling receptor (as pdb or mol2)...")
        # We'll need the protein PDB file (without missing atoms)
        try:
            protein_pdb_filename = setup_options['protein_pdb']
            receptor_mol2 = None
        except KeyError:
            try:
                receptor_mol2 = setup_options['receptor_mol2']
                protein_pdb_filename = None
            except KeyError as e:
                print("Either protein_pdb or receptor_mol2 must be specified if running a complex simulation")
                raise e
    else:
        protein_pdb_filename = None
        receptor_mol2 = None

    # And a ligand file containing the pair of ligands between which we will transform
    ligand_file = setup_options['ligand_file']
    _logger.info(f"\tdetected ligand file: {ligand_file}")

    # get the indices of ligands out of the file:
    old_ligand_index = setup_options['old_ligand_index']
    new_ligand_index = setup_options['new_ligand_index']
    _logger.info(f"\told ligand index: {old_ligand_index}; new ligand index: {new_ligand_index}")

    _logger.info(f"\tsetting up forcefield files...")
    forcefield_files = setup_options['forcefield_files']

    if "timestep" in setup_options:
        timestep = setup_options['timestep'] * unit.femtoseconds
        _logger.info(f"\ttimestep: {timestep}.")
    else:
        timestep = 1.0 * unit.femtoseconds
        _logger.info(f"\tno timestep detected: setting default as 1.0fs.")

    if "neq_splitting" in setup_options:
        neq_splitting = setup_options['neq_splitting']
        _logger.info(f"\tneq_splitting: {neq_splitting}")

        try:
            eq_splitting = setup_options['eq_splitting']
            _logger.info(f"\teq_splitting: {eq_splitting}")
        except KeyError as e:
            print("If you specify a nonequilibrium splitting string, you must also specify an equilibrium one.")
            raise e

    else:
        eq_splitting = "V R O R V"
        neq_splitting = "V R O R V"
        _logger.info(f"\tno splitting strings specified: defaulting to neq: {neq_splitting}, eq: {eq_splitting}.")

    if "measure_shadow_work" in setup_options:
        measure_shadow_work = setup_options['measure_shadow_work']
        _logger.info(f"\tmeasuring shadow work: {measure_shadow_work}.")
    else:
        measure_shadow_work = False
        _logger.info(f"\tno measure_shadow_work specified: defaulting to False.")

    pressure = setup_options['pressure'] * unit.atmosphere
    temperature = setup_options['temperature'] * unit.kelvin
    solvent_padding_angstroms = setup_options['solvent_padding'] * unit.angstrom
    _logger.info(f"\tsetting pressure: {pressure}.")
    _logger.info(f"\tsetting temperature: {temperature}.")
    _logger.info(f"\tsetting solvent padding: {solvent_padding_angstroms}A.")

    setup_pickle_file = setup_options['save_setup_pickle_as']
    _logger.info(f"\tsetup pickle file: {setup_pickle_file}")
    trajectory_directory = setup_options['trajectory_directory']
    _logger.info(f"\ttrajectory directory: {trajectory_directory}")
    try:
        atom_map_file = setup_options['atom_map']
        with open(atom_map_file, 'r') as f:
            atom_map = {int(x.split()[0]): int(x.split()[1]) for x in f.readlines()}
        _logger.info(f"\tsucceeded parsing atom map.")
    except Exception:
        atom_map=None
        _logger.info(f"\tno atom map specified: default to None.")

    if 'topology_proposal' not in setup_options:
        _logger.info(f"\tno topology_proposal specified; proceeding to RelativeFEPSetup...\n\n\n")
        fe_setup = RelativeFEPSetup(ligand_file, old_ligand_index, new_ligand_index, forcefield_files,phases=phases,
                                          protein_pdb_filename=protein_pdb_filename,
                                          receptor_mol2_filename=receptor_mol2, pressure=pressure,
                                          temperature=temperature, solvent_padding=solvent_padding_angstroms, spectator_filenames=setup_options['spectators'],
                                          map_strength=setup_options['map_strength'],
                                          atom_expr=setup_options['atom_expr'], bond_expr=setup_options['bond_expr'],
                                          atom_map=atom_map, neglect_angles = setup_options['neglect_angles'], anneal_14s = setup_options['anneal_1,4s'],
                                          small_molecule_forcefield=setup_options['small_molecule_forcefield'], small_molecule_parameters_cache=setup_options['small_molecule_parameters_cache'],
                                          trajectory_directory=trajectory_directory, trajectory_prefix=setup_options['trajectory_prefix'], nonbonded_method=setup_options['nonbonded_method'])

        _logger.info(f"\twriting pickle output...")
        with open(os.path.join(os.getcwd(), trajectory_directory, setup_pickle_file), 'wb') as f:
            try:
                pickle.dump(fe_setup, f)
                _logger.info(f"\tsuccessfully dumped pickle.")
            except Exception as e:
                print(e)
                print("\tUnable to save setup object as a pickle")

        _logger.info(f"\tsetup is complete.  Writing proposals and positions for each phase to top_prop dict...")

        top_prop = dict()
        for phase in phases:
            top_prop[f'{phase}_topology_proposal'] = getattr(fe_setup, f'{phase}_topology_proposal')
            top_prop[f'{phase}_geometry_engine'] = getattr(fe_setup, f'_{phase}_geometry_engine')
            top_prop[f'{phase}_old_positions'] = getattr(fe_setup, f'{phase}_old_positions')
            top_prop[f'{phase}_new_positions'] = getattr(fe_setup, f'{phase}_new_positions')
            top_prop[f'{phase}_added_valence_energy'] = getattr(fe_setup, f'_{phase}_added_valence_energy')
            top_prop[f'{phase}_subtracted_valence_energy'] = getattr(fe_setup, f'_{phase}_subtracted_valence_energy')
            top_prop[f'{phase}_logp_proposal'] = getattr(fe_setup, f'_{phase}_logp_proposal')
            top_prop[f'{phase}_logp_reverse'] = getattr(fe_setup, f'_{phase}_logp_reverse')
            top_prop[f'{phase}_forward_neglected_angles'] = getattr(fe_setup, f'_{phase}_forward_neglected_angles')
            top_prop[f'{phase}_reverse_neglected_angles'] = getattr(fe_setup, f'_{phase}_reverse_neglected_angles')

        _logger.info(f"\twriting atom_mapping.png")
        atom_map_outfile = os.path.join(os.getcwd(), trajectory_directory, 'atom_mapping.png')
        render_atom_mapping(atom_map_outfile, fe_setup._ligand_oemol_old, fe_setup._ligand_oemol_new, fe_setup.non_offset_new_to_old_atom_map)

    else:
        _logger.info(f"\tloading topology proposal from yaml setup options...")
        top_prop = np.load(setup_options['topology_proposal']).item()

    n_steps_per_move_application = setup_options['n_steps_per_move_application']
    _logger.info(f"\t steps per move application: {n_steps_per_move_application}")
    trajectory_directory = setup_options['trajectory_directory']

    trajectory_prefix = setup_options['trajectory_prefix']
    _logger.info(f"\ttrajectory prefix: {trajectory_prefix}")

    if 'atom_selection' in setup_options:
        atom_selection = setup_options['atom_selection']
        _logger.info(f"\tatom selection detected: {atom_selection}")
    else:
        _logger.info(f"\tno atom selection detected: default to all.")
        atom_selection = 'all'

    if setup_options['fe_type'] == 'neq':
        _logger.info(f"\tInstantiating nonequilibrium switching FEP")
        n_equilibrium_steps_per_iteration = setup_options['n_equilibrium_steps_per_iteration']
        ncmc_save_interval = setup_options['ncmc_save_interval']
        write_ncmc_configuration = setup_options['write_ncmc_configuration']
        if setup_options['LSF']:
            _internal_parallelism = {'library': ('dask', 'LSF'), 'num_processes': setup_options['processes']}
        else:
            _internal_parallelism = None


        ne_fep = dict()
        for phase in phases:
            _logger.info(f"\t\tphase: {phase}")
            hybrid_factory = HybridTopologyFactory(top_prop['%s_topology_proposal' % phase],
                                               top_prop['%s_old_positions' % phase],
                                               top_prop['%s_new_positions' % phase],
                                               neglected_new_angle_terms = top_prop[f"{phase}_forward_neglected_angles"],
                                               neglected_old_angle_terms = top_prop[f"{phase}_reverse_neglected_angles"],
                                               softcore_LJ_v2 = setup_options['softcore_v2'],
                                               interpolate_old_and_new_14s = setup_options['anneal_1,4s'])

            ne_fep[phase] = SequentialMonteCarlo(factory = hybrid_factory,
                                                 lambda_protocol = setup_options['lambda_protocol'],
                                                 temperature = temperature,
                                                 trajectory_directory = trajectory_directory,
                                                 trajectory_prefix = f"{trajectory_prefix}_{phase}",
                                                 atom_selection = atom_selection,
                                                 timestep = timestep,
                                                 eq_splitting_string = eq_splitting,
                                                 neq_splitting_string = neq_splitting,
                                                 collision_rate = setup_options['ncmc_collision_rate_ps'],
                                                 ncmc_save_interval = ncmc_save_interval,
                                                 internal_parallelism = _internal_parallelism)

        print("Nonequilibrium switching driver class constructed")

        return {'topology_proposals': top_prop, 'ne_fep': ne_fep}

    else:
        _logger.info(f"\tno nonequilibrium detected.")
        n_states = setup_options['n_states']
        _logger.info(f"\tn_states: {n_states}")
        if 'n_replicas' not in setup_options:
            n_replicas = n_states
        else:
            n_replicas = setup_options['n_replicas']
        _logger.info(f"\tn_replicas: {n_replicas}")
        checkpoint_interval = setup_options['checkpoint_interval']
        _logger.info(f"\tcheckpoint_interval: {checkpoint_interval}")
        htf = dict()
        hss = dict()
        _logger.info(f"\tcataloging HybridTopologyFactories...")

        for phase in phases:
            _logger.info(f"\t\tphase: {phase}:")
            #TODO write a SAMSFEP class that mirrors NonequilibriumSwitchingFEP
            _logger.info(f"\t\twriting HybridTopologyFactory for phase {phase}...")
            htf[phase] = HybridTopologyFactory(top_prop['%s_topology_proposal' % phase],
                                               top_prop['%s_old_positions' % phase],
                                               top_prop['%s_new_positions' % phase],
                                               neglected_new_angle_terms = top_prop[f"{phase}_forward_neglected_angles"],
                                               neglected_old_angle_terms = top_prop[f"{phase}_reverse_neglected_angles"],
                                               softcore_LJ_v2 = setup_options['softcore_v2'],
                                               interpolate_old_and_new_14s = setup_options['anneal_1,4s'])

        for phase in phases:
           # Define necessary vars to check energy bookkeeping
            _top_prop = top_prop['%s_topology_proposal' % phase]
            _htf = htf[phase]
            _forward_added_valence_energy = top_prop['%s_added_valence_energy' % phase]
            _reverse_subtracted_valence_energy = top_prop['%s_subtracted_valence_energy' % phase]

            zero_state_error, one_state_error = validate_endstate_energies(_top_prop, _htf, _forward_added_valence_energy, _reverse_subtracted_valence_energy, beta = 1.0/(kB*temperature), ENERGY_THRESHOLD = ENERGY_THRESHOLD, trajectory_directory=f'{setup_options["trajectory_directory"]}/{phase}')
            _logger.info(f"\t\terror in zero state: {zero_state_error}")
            _logger.info(f"\t\terror in one state: {one_state_error}")

            # generating lambda protocol
            lambda_protocol = LambdaProtocol(functions=setup_options['protocol-type'])
            _logger.info(f'Using lambda protocol : {setup_options["protocol-type"]}')


            if atom_selection:
                selection_indices = htf[phase].hybrid_topology.select(atom_selection)
            else:
                selection_indices = None

            storage_name = str(trajectory_directory)+'/'+str(trajectory_prefix)+'-'+str(phase)+'.nc'
            _logger.info(f'\tstorage_name: {storage_name}')
            _logger.info(f'\tselection_indices {selection_indices}')
            _logger.info(f'\tcheckpoint interval {checkpoint_interval}')
            reporter = MultiStateReporter(storage_name, analysis_particle_indices=selection_indices,
                                          checkpoint_interval=checkpoint_interval)

            if phase == 'vacuum':
                endstates = False
            else:
                endstates = True
            #TODO expose more of these options in input
            if setup_options['fe_type'] == 'sams':
                hss[phase] = HybridSAMSSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=timestep,
                                                                                    collision_rate=1.0 / unit.picosecond,
                                                                                    n_steps=n_steps_per_move_application,
                                                                                    reassign_velocities=False,
                                                                                    n_restart_attempts=20,constraint_tolerance=1e-06),
                                               hybrid_factory=htf[phase], online_analysis_interval=setup_options['offline-freq'],
                                               online_analysis_minimum_iterations=10,flatness_criteria=setup_options['flatness-criteria'],
                                               gamma0=setup_options['gamma0'])
                hss[phase].setup(n_states=n_states, n_replicas=n_replicas, temperature=temperature,storage_file=reporter,lambda_protocol=lambda_protocol,endstates=endstates)
            elif setup_options['fe_type'] == 'repex':
                hss[phase] = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=timestep,
                                                                                     collision_rate=1.0 / unit.picosecond,
                                                                                     n_steps=n_steps_per_move_application,
                                                                                     reassign_velocities=False,
                                                                                     n_restart_attempts=20,constraint_tolerance=1e-06),
                                                                                     hybrid_factory=htf[phase],online_analysis_interval=setup_options['offline-freq'])
                hss[phase].setup(n_states=n_states, temperature=temperature,storage_file=reporter,lambda_protocol=lambda_protocol,endstates=endstates)

            # save the systems and the states
            from simtk.openmm import XmlSerializer
            from perses.tests.utils import generate_endpoint_thermodynamic_states

            _logger.info('WRITING OUT XML FILES')
            #old_thermodynamic_state, new_thermodynamic_state, hybrid_thermodynamic_state, _ = generate_endpoint_thermodynamic_states(htf[phase].hybrid_system, _top_prop)
            # hybrid
            with open(f'{setup_options["trajectory_directory"]}/hybrid-{phase}-system.xml', 'w') as f:
                f.write(XmlSerializer.serialize(htf[phase].hybrid_system))
            #with open(f'{setup_options["trajectory_directory"]}/hybrid-{phase}-state.xml', 'w') as f:
            #    f.write(XmlSerializer.serialize(hybrid_thermodynamic_state))
            
            # old
            with open(f'{setup_options["trajectory_directory"]}/old-{phase}-system.xml', 'w') as f:
                f.write(XmlSerializer.serialize(htf[phase]._old_system))
            #with open(f'old-{phase}-state.xml', 'w') as f:
            #    f.write(XmlSerializer.serialize(old_thermodynamic_state))

            #new
            with open(f'{setup_options["trajectory_directory"]}/new-{phase}-system.xml', 'w') as f:
                f.write(XmlSerializer.serialize(htf[phase]._new_system))
            #with open(f'new-{phase}-state.xml', 'w') as f:
            #    f.write(XmlSerializer.serialize(new_thermodynamic_state))

        return {'topology_proposals': top_prop, 'hybrid_topology_factories': htf, 'hybrid_samplers': hss}


def run(yaml_filename=None):
    _logger.info("Beginning Setup...")
    if yaml_filename is None:
       try:
          yaml_filename = sys.argv[1]
          _logger.info(f"Detected yaml file: {yaml_filename}")
       except IndexError as e:
           _logger.critical(f"You must specify the setup yaml file as an argument to the script.")

    _logger.info(f"Getting setup options from {yaml_filename}")
    setup_options = getSetupOptions(yaml_filename)
    if 'lambdas' in setup_options:
        if type(setup_options['lambdas']) == int:
            lambdas = {}
            for _direction in setup_options['direction']:
                lims = (0,1) if _direction == 'forward' else (1,0)
                lambdas[_direction] = np.linspace(lims[0], lims[1], setup_options['lambdas'])
        else:
            lambdas = setup_options['lambdas']
    else:
        lambdas = None

    if setup_options['run_type'] == 'anneal':
        _logger.info(f"skipping setup and annealing...")
        trajectory_prefix = setup_options['trajectory_prefix']
        trajectory_directory = setup_options['trajectory_directory']
        out_trajectory_prefix = setup_options['out_trajectory_prefix']
        for phase in setup_options['phases']:
            ne_fep_run = pickle.load(open(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase)), 'rb'))
            #implement the appropriate parallelism, otherwise the default from the previous incarnation of the ne_fep_run will be used.
            if setup_options['LSF']:
                _internal_parallelism = {'library': ('dask', 'LSF'), 'num_processes': setup_options['processes']}
            else:
                _internal_parallelism = None
            ne_fep_run.implement_parallelism(external_parallelism = None, internal_parallelism = _internal_parallelism)
            ne_fep_run.neq_integrator = setup_options['neq_integrator']
            ne_fep_run.LSF = setup_options['LSF']
            ne_fep_run.AIS(num_particles = setup_options['n_particles'],
                           protocols = lambdas,
                           num_integration_steps = setup_options['ncmc_num_integration_steps'],
                           return_timer = False,
                           rethermalize = setup_options['ncmc_rethermalize'])

            # try to write out the ne_fep object as a pickle
            try:
                with open(os.path.join(trajectory_directory, "%s_%s_fep.neq.pkl" % (out_trajectory_prefix, phase)), 'wb') as f:
                    pickle.dump(ne_fep_run, f)
                    print("pickle save successful; terminating.")

            except Exception as e:
                print(e)
                print("Unable to save run object as a pickle; saving as npy")
                np.save(os.path.join(trajectory_directory, "%s_%s_fep.neq.npy" % (out_trajectory_prefix, phase)), ne_fep_run)

    else:
        _logger.info(f"Running setup...")
        setup_dict = run_setup(setup_options)

        trajectory_prefix = setup_options['trajectory_prefix']
        trajectory_directory = setup_options['trajectory_directory']

        #write out topology proposals
        try:
            _logger.info(f"Writing topology proposal {trajectory_prefix}_topology_proposals.pkl to {trajectory_directory}...")
            with open(os.path.join(trajectory_directory, "%s_topology_proposals.pkl" % (trajectory_prefix)), 'wb') as f:
                pickle.dump(setup_dict['topology_proposals'], f)
        except Exception as e:
            print(e)
            _logger.info("Unable to save run object as a pickle; saving as npy")
            np.save(os.path.join(trajectory_directory, "%s_topology_proposals.npy" % (trajectory_prefix)), setup_dict['topology_proposals'])

        n_equilibration_iterations = setup_options['n_equilibration_iterations'] #set this to 1 for neq_fep
        _logger.info(f"Equilibration iterations: {n_equilibration_iterations}.")

        if setup_options['fe_type'] == 'neq':
            temperature = setup_options['temperature'] * unit.kelvin
            max_file_size = setup_options['max_file_size']

            ne_fep = setup_dict['ne_fep']
            for phase in setup_options['phases']:
                ne_fep_run = ne_fep[phase]
                hybrid_factory = ne_fep_run.factory

                top_proposal = setup_dict['topology_proposals'][f"{phase}_topology_proposal"]
                _forward_added_valence_energy = setup_dict['topology_proposals'][f"{phase}_added_valence_energy"]
                _reverse_subtracted_valence_energy = setup_dict['topology_proposals'][f"{phase}_subtracted_valence_energy"]

                zero_state_error, one_state_error = validate_endstate_energies(hybrid_factory._topology_proposal, hybrid_factory, _forward_added_valence_energy, _reverse_subtracted_valence_energy, beta = 1.0/(kB*temperature), ENERGY_THRESHOLD = ENERGY_THRESHOLD, trajectory_directory=f'{setup_options["trajectory_directory"]}/{phase}')
                _logger.info(f"\t\terror in zero state: {zero_state_error}")
                _logger.info(f"\t\terror in one state: {one_state_error}")

                print("activating client...")
                processes = setup_options['processes']
                adapt = setup_options['adapt']
                LSF = setup_options['LSF']

                if setup_options['run_type'] == None or setup_options['run_type'] == 'equilibrate':
                    print("equilibrating...")
                    # Now we have to pull the files
                    if setup_options['direction'] == None:
                        endstates = [0,1]
                    else:
                        endstates = [0] if setup_options['direction'] == 'forward' else [1]
                    #ne_fep_run.activate_client(LSF = LSF, processes = 2, adapt = adapt) #we only need 2 processes for equilibration
                    ne_fep_run.minimize_sampler_states()
                    ne_fep_run.equilibrate(n_equilibration_iterations = setup_options['n_equilibration_iterations'],
                                           n_steps_per_equilibration = setup_options['n_equilibrium_steps_per_iteration'],
                                           endstates = [0,1],
                                           max_size = setup_options['max_file_size'],
                                           decorrelate = True,
                                           timer = True,
                                           minimize = False)
                    #ne_fep_run.deactivate_client()
                    with open(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase)), 'wb') as f:
                        pickle.dump(ne_fep_run, f)



                if setup_options['run_type'] == None:
                    print("annealing...")
                    if 'lambdas' in setup_options:
                        if type(setup_options['lambdas']) == int:
                            lambdas = {}
                            for _direction in setup_options['direction']:
                                lims = (0,1) if _direction == 'forward' else (1,0)
                                lambdas[_direction] = np.linspace(lims[0], lims[1], setup_options['lambdas'])
                        else:
                            lambdas = setup_options['lambdas']
                    else:
                        lambdas = None
                    ne_fep_run = pickle.load(open(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase)), 'rb'))
                    ne_fep_run.AIS(num_particles = setup_options['n_particles'],
                                   protocols = lambdas,
                                   num_integration_steps = setup_options['ncmc_num_integration_steps'],
                                   return_timer = False,
                                   rethermalize = setup_options['ncmc_rethermalize'])


                    print("calculation complete; deactivating client")
                    #ne_fep_run.deactivate_client()

                    # try to write out the ne_fep object as a pickle
                    try:
                        with open(os.path.join(trajectory_directory, "%s_%s_fep.neq.pkl" % (trajectory_prefix, phase)), 'wb') as f:
                            pickle.dump(ne_fep_run, f)
                            print("pickle save successful; terminating.")

                    except Exception as e:
                        print(e)
                        print("Unable to save run object as a pickle; saving as npy")
                        np.save(os.path.join(trajectory_directory, "%s_%s_fep.neq.npy" % (trajectory_prefix, phase)), ne_fep_run)

        elif setup_options['fe_type'] == 'sams':
            _logger.info(f"Detecting sams as fe_type...")
            _logger.info(f"Writing hybrid factory {trajectory_prefix}hybrid_factory.npy to {trajectory_directory}...")
            np.save(os.path.join(trajectory_directory, trajectory_prefix + "hybrid_factory.npy"),
                    setup_dict['hybrid_topology_factories'])

            hss = setup_dict['hybrid_samplers']
            logZ = dict()
            free_energies = dict()
            _logger.info(f"Iterating through phases for sams...")
            for phase in setup_options['phases']:
                _logger.info(f'\tRunning {phase} phase...')
                hss_run = hss[phase]

                _logger.info(f"\t\tequilibrating...\n\n")
                hss_run.equilibrate(n_equilibration_iterations)
                _logger.info(f"\n\n")

                _logger.info(f"\t\textending simulation...\n\n")
                hss_run.extend(setup_options['n_cycles'])
                _logger.info(f"\n\n")

                logZ[phase] = hss_run._logZ[-1] - hss_run._logZ[0]
                free_energies[phase] = hss_run._last_mbar_f_k[-1] - hss_run._last_mbar_f_k[0]
                _logger.info(f"\t\tFinished phase {phase}")

            for phase in free_energies:
                print(f"Comparing ligand {setup_options['old_ligand_index']} to {setup_options['new_ligand_index']}")
                print(f"{phase} phase has a free energy of {free_energies[phase]}")

        elif setup_options['fe_type'] == 'repex':
            _logger.info(f"Detecting repex as fe_type...")
            _logger.info(f"Writing hybrid factory {trajectory_prefix}hybrid_factory.npy to {trajectory_directory}...")
            np.save(os.path.join(trajectory_directory, trajectory_prefix + "hybrid_factory.npy"),
                    setup_dict['hybrid_topology_factories'])

            hss = setup_dict['hybrid_samplers']
            _logger.info(f"Iterating through phases for repex...")
            for phase in setup_options['phases']:
                print(f'Running {phase} phase')
                hss_run = hss[phase]

                _logger.info(f"\t\tequilibrating...\n\n")
                hss_run.equilibrate(n_equilibration_iterations)
                _logger.info(f"\n\n")

                _logger.info(f"\t\textending simulation...\n\n")
                hss_run.extend(setup_options['n_cycles'])
                _logger.info(f"\n\n")

                _logger.info(f"\t\tFinished phase {phase}")

if __name__ == "__main__":
    run()

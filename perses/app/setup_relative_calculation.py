import yaml
import numpy as np
import pickle
import os
import sys
import simtk.unit as unit
import logging
import warnings
from cloudpathlib import AnyPath
from pathlib import Path

from perses.annihilation.relative import HybridTopologyFactory, RESTCapableHybridTopologyFactory
from perses.app.relative_setup import RelativeFEPSetup
from perses.annihilation.lambda_protocol import LambdaProtocol

from openmmtools import mcmc, cache
from openmmtools.multistate import MultiStateReporter
from perses.utils.smallmolecules import render_atom_mapping
from perses.dispersed.utils import validate_endstate_energies, validate_endstate_energies_point
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

# TODO: We need to import these for logging to work, even if we don't use them. Why?
from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler

fmt = logging.Formatter(fmt="%(asctime)s:(%(relative)ss):%(name)s:%(message)s")
#logging.basicConfig(level = logging.NOTSET)
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=LOGLEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger()
_logger.setLevel(LOGLEVEL)
[hndl.addFilter(TimeFilter()) for hndl in _logger.handlers]
[hndl.setFormatter(fmt) for hndl in _logger.handlers]

ENERGY_THRESHOLD = 1e-4
from openmmtools.constants import kB

def getSetupOptions(filename, override_string=None):
    """
    Reads input yaml file, makes output directory and returns setup options

    Parameters
    ----------
    filename : str
        .yaml file containing simulation parameters

    override_string : List[str]
        List of strings in the form of key:value to override simulation
        parameters set in yaml file.
        Default: None

    Returns
    -------
    setup_options :
        options provided in the yaml file
    phases : list of strings
        phases to simulate, can be 'complex', 'solvent' or 'vacuum'
    """

    filename = AnyPath(filename)
    yaml_file = open(filename, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()
    if override_string:
        setup_options = _process_overrides(override_string, setup_options)
    _logger.info("\tDetecting phases...")
    if 'phases' not in setup_options:
        setup_options['phases'] = ['complex','solvent']
        _logger.warning('\t\tNo phases provided - running complex and solvent as default.')
    else:
        _logger.info(f"\t\tphases detected: {setup_options['phases']}")

    if 'protocol-type' not in setup_options:
        setup_options['protocol-type'] = 'default'


    if 'temperature' not in setup_options:
        setup_options['temperature'] = 300.
    if 'pressure' not in setup_options:
        setup_options['pressure'] = 1.
    if 'solvent_padding' not in setup_options:
        setup_options['solvent_padding'] = 9.
    if 'ionic_strength' not in setup_options:
        setup_options['ionic_strength'] = 0.15


    if 'small_molecule_forcefield' not in setup_options:
        setup_options['small_molecule_forcefield'] = None

    if 'small_molecule_parameters_cache' not in setup_options:
        setup_options['small_molecule_parameters_cache'] = None

    if 'remove_constraints' not in setup_options:
        setup_options['remove_constraints'] = False
        _logger.info('No constraints will be removed')
    # remove_constraints can be 'all' or 'not water'
    elif setup_options['remove_constraints'] not in ['all', 'not water', False]:
        _logger.warning("remove_constraints value of {setup_options['remove_constraints']} not understood. 'all', 'none' or 'not water' are valid options. NOT REMOVING ANY CONSTRAINTS")
        setup_options['remove_constraints'] = False

    if 'spectators' not in setup_options:
        _logger.info(f'No spectators')
        setup_options['spectators'] = None

    if 'complex_box_dimensions' not in setup_options:
        setup_options['complex_box_dimensions'] = None
    # If complex_box_dimensions is None, nothing to do
    elif setup_options['complex_box_dimensions'] is None:
        pass
    else:
        setup_options['complex_box_dimensions'] = tuple([float(x) for x in setup_options['complex_box_dimensions']])

    if 'solvent_box_dimensions' not in setup_options:
        setup_options['solvent_box_dimensions'] = None

    # Not sure why these are needed
    # TODO: Revisit these?
    if 'neglect_angles' not in setup_options:
        setup_options['neglect_angles'] = False
    if 'anneal_1,4s' not in setup_options:
        setup_options['anneal_1,4s'] = False

    if 'nonbonded_method' not in setup_options:
        setup_options['nonbonded_method'] = 'PME'

    if 'render_atom_map' not in setup_options:
        setup_options['render_atom_map'] = True

    if 'n_steps_per_move_application' not in setup_options:
        setup_options['n_steps_per_move_application'] = 1

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
                path = AnyPath(path)
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

    trajectory_directory = AnyPath(setup_options['trajectory_directory'])

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

    if not 'rmsd_restraint' in setup_options:
        setup_options['rmsd_restraint'] = False

    # Handling htf input parameter
    if 'hybrid_topology_factory' not in setup_options:
        default_htf_class_name = "HybridTopologyFactory"
        setup_options['hybrid_topology_factory'] = default_htf_class_name
        _logger.info(f"\t 'hybrid_topology_factory' not specified: default to {default_htf_class_name}")

    # Handling absence platform name input (backwards compatibility)
    if 'platform' not in setup_options:
        setup_options['platform'] = None  # defaults to choosing best platform

    # Handling counterion
    if 'transform_waters_into_ions_for_charge_changes' not in setup_options:
        setup_options['transform_waters_into_ions_for_charge_changes'] = True

    # Handling unsampled_endstates long range correction flag
    if 'unsampled_endstates' not in setup_options:
        setup_options['unsampled_endstates'] = True   # True by default (matches class default)

    os.makedirs(trajectory_directory, exist_ok=True)

    return setup_options


def get_openmm_platform(platform_name=None):
    """
    Return OpenMM's platform object based on given name. Setting to mixed precision if using CUDA or OpenCL.

    Parameters
    ----------
    platform_name : str, optional, default=None
        String with the platform name. If None, it will use the fastest platform supporting mixed precision.

    Returns
    -------
    platform : openmm.Platform
        OpenMM platform object.
    """
    if platform_name is None:
        # No platform is specified, so retrieve fastest platform that supports 'mixed' precision
        from openmmtools.utils import get_fastest_platform
        platform = get_fastest_platform(minimum_precision='mixed')
    else:
        from openmm import Platform
        platform = Platform.getPlatformByName(platform_name)
    # Set precision and properties
    name = platform.getName()  # get platform name to set properties
    if name in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
    if name in ['CUDA']:
        platform.setPropertyDefaultValue('DeterministicForces', 'true')

    return platform


def run_setup(setup_options, serialize_systems=True, build_samplers=True):
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
    from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
    phases = setup_options['phases']
    known_phases = ['complex', 'solvent', 'vacuum']
    for phase in phases:
        assert (phase in known_phases), f"Unknown phase, {phase} provided. run_setup() can be used with {known_phases}"

    # TODO: This is an overly complex way to specify defaults.
    #   We should replace this completely with a streamlined approach,
    #   such as deferring to defaults for modules we call unless the user
    #   chooses to override them.


    if 'use_given_geometries' not in list(setup_options.keys()):
        use_given_geometries = False
    else:
        assert type(setup_options['use_given_geometries']) == type(True)
        use_given_geometries = setup_options['use_given_geometries']

    if 'given_geometries_tolerance' not in list(setup_options.keys()):
        given_geometries_tolerance = 0.2 * unit.angstroms
    else:
        # Assume user input is in Angstroms
        given_geometries_tolerance = float(setup_options['given_geometries_tolerance']) * unit.angstroms

    if 'complex' in phases:
        _logger.info(f"\tPulling receptor (as pdb or mol2)...")
        # We'll need the protein PDB file (without missing atoms)
        try:
            protein_pdb_filename = setup_options['protein_pdb']
            assert protein_pdb_filename is not None
            receptor_mol2 = None
        except KeyError:
            try:
                receptor_mol2 = setup_options['receptor_mol2']
                assert receptor_mol2 is not None
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
        if isinstance(setup_options['timestep'], (float, int)):
            timestep = setup_options['timestep'] * unit.femtoseconds
        else:
            timestep = setup_options['timestep']
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
    if isinstance(setup_options['pressure'], (float, int)):
        pressure = setup_options['pressure'] * unit.atmosphere
    else:
        pressure = setup_options['pressure']
    if isinstance(setup_options['temperature'], (float, int)):
        temperature = setup_options['temperature'] * unit.kelvin
    else:
        temperature = setup_options['temperature']
    if isinstance(setup_options['solvent_padding'], (float, int)):
        solvent_padding_angstroms = setup_options['solvent_padding'] * unit.angstrom
    else:
        solvent_padding_angstroms = setup_options['solvent_padding']
    if isinstance(setup_options['ionic_strength'], (float, int)):
        ionic_strength = setup_options['ionic_strength'] * unit.molar
    else:
        ionic_strength = setup_options['ionic_strength']
    _logger.info(f"\tsetting pressure: {pressure}.")
    _logger.info(f"\tsetting temperature: {temperature}.")
    _logger.info(f"\tsetting solvent padding: {solvent_padding_angstroms}A.")
    _logger.info(f"\tsetting ionic strength: {ionic_strength}M.")

    setup_pickle_file = setup_options['save_setup_pickle_as'] if 'save_setup_pickle_as' in list(setup_options) else None
    _logger.info(f"\tsetup pickle file: {setup_pickle_file}")
    trajectory_directory = AnyPath(setup_options['trajectory_directory'])
    _logger.info(f"\ttrajectory directory: {trajectory_directory}")
    try:
        atom_map_file = setup_options['atom_map']
        atom_map_file = AnyPath(atom_map_file)
        with open(atom_map_file, 'r') as f:
            atom_map = {int(x.split()[0]): int(x.split()[1]) for x in f.readlines()}
        _logger.info(f"\tsucceeded parsing atom map.")
    except Exception:
        atom_map=None
        _logger.info(f"\tno atom map specified: default to None.")

    if 'topology_proposal' not in list(setup_options.keys()) or setup_options['topology_proposal'] is None:
        _logger.info(f"\tno topology_proposal specified; proceeding to RelativeFEPSetup...\n\n\n")
        if 'set_solvent_box_dims_to_complex' in list(setup_options.keys()) and setup_options['set_solvent_box_dims_to_complex']:
            set_solvent_box_dims_to_complex=True
        else:
            set_solvent_box_dims_to_complex=False

        _logger.info(f'Box dimensions: {setup_options["complex_box_dimensions"]} and {setup_options["solvent_box_dimensions"]}')
        fe_setup = RelativeFEPSetup(ligand_file, old_ligand_index, new_ligand_index, forcefield_files,phases=phases,
                                          protein_pdb_filename=protein_pdb_filename,
                                          receptor_mol2_filename=receptor_mol2, pressure=pressure,
                                          temperature=temperature, solvent_padding=solvent_padding_angstroms, spectator_filenames=setup_options['spectators'],
                                          map_strength=setup_options['map_strength'],
                                          atom_expr=setup_options['atom_expr'], bond_expr=setup_options['bond_expr'],
                                          neglect_angles = setup_options['neglect_angles'], anneal_14s = setup_options['anneal_1,4s'],
                                          small_molecule_forcefield=setup_options['small_molecule_forcefield'], small_molecule_parameters_cache=setup_options['small_molecule_parameters_cache'],
                                          trajectory_directory=trajectory_directory, trajectory_prefix=setup_options['trajectory_prefix'], nonbonded_method=setup_options['nonbonded_method'],
                                          complex_box_dimensions=setup_options['complex_box_dimensions'],solvent_box_dimensions=setup_options['solvent_box_dimensions'], ionic_strength=ionic_strength, remove_constraints=setup_options['remove_constraints'],
                                          use_given_geometries=use_given_geometries, given_geometries_tolerance=given_geometries_tolerance,
                                          transform_waters_into_ions_for_charge_changes = setup_options['transform_waters_into_ions_for_charge_changes'])


        _logger.info(f"\twriting pickle output...")
        if setup_pickle_file is not None:
            with open(AnyPath(os.path.join(trajectory_directory, setup_pickle_file)), 'wb') as f:
                try:
                    pickle.dump(fe_setup, f)
                    _logger.info(f"\tsuccessfully dumped pickle.")
                except Exception as e:
                    print(e)
                    print("\tUnable to save setup object as a pickle")

            _logger.info(f"\tsetup is complete.  Writing proposals and positions for each phase to top_prop dict...")
        else:
            _logger.info(f"\tsetup is complete.  Omitted writing proposals and positions for each phase to top_prop dict...")

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

        top_prop['ligand_oemol_old'] = fe_setup._ligand_oemol_old
        top_prop['ligand_oemol_new'] = fe_setup._ligand_oemol_new
        top_prop['non_offset_new_to_old_atom_map'] = fe_setup.non_offset_new_to_old_atom_map
        _logger.info(f"\twriting atom_mapping.png in {trajectory_directory}")
        atom_map_outfile = trajectory_directory / "atom_mapping.png"

        if 'render_atom_map' in list(setup_options.keys()) and setup_options['render_atom_map']:
            try:
                atom_map_outfile = str(atom_map_outfile)
                render_atom_mapping(atom_map_outfile, fe_setup._ligand_oemol_old, fe_setup._ligand_oemol_new, fe_setup.non_offset_new_to_old_atom_map)
            except TypeError:
                _logger.critical("COULD NOT WRITE ATOM MAPPING. \
                        YOU ARE PROBALLY WRITING TO A CLOUD FILE SYSTEM. \
                        CURRENTLY THIS IS NOT SUPPORTED FOR RENDERING ATOM MAPS")

    else:
        _logger.info(f"\tloading topology proposal from yaml setup options...")
        top_prop = np.load(setup_options['topology_proposal']).item()

    n_steps_per_move_application = setup_options['n_steps_per_move_application']
    _logger.info(f"\t steps per move application: {n_steps_per_move_application}")
    trajectory_directory = AnyPath(setup_options['trajectory_directory'])

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
            hybrid_factory = _generate_htf(phase, top_prop, setup_options)

            if build_samplers:
                ne_fep[phase] = SequentialMonteCarlo(factory=hybrid_factory,
                                                     lambda_protocol=setup_options['lambda_protocol'],
                                                     temperature=temperature,
                                                     trajectory_directory=trajectory_directory,
                                                     trajectory_prefix=f"{trajectory_prefix}_{phase}",
                                                     atom_selection=atom_selection,
                                                     timestep=timestep,
                                                     eq_splitting_string=eq_splitting,
                                                     neq_splitting_string=neq_splitting,
                                                     collision_rate=setup_options['ncmc_collision_rate_ps'],
                                                     ncmc_save_interval=ncmc_save_interval,
                                                     internal_parallelism=_internal_parallelism)

        print("Nonequilibrium switching driver class constructed")

        # TODO: Should this function return a single thing instead of two different objects for neq vs others?
        return {'topology_proposals': top_prop, 'ne_fep': ne_fep}

    else:
        _logger.info(f"\tno nonequilibrium detected.")
        htf = dict()
        hss = dict()
        _logger.info(f"\tcataloging HybridTopologyFactories...")

        for phase in phases:
            _logger.info(f"\t\tphase: {phase}:")
            #TODO write a SAMSFEP class that mirrors NonequilibriumSwitchingFEP
            _logger.info(f"\t\twriting HybridTopologyFactory for phase {phase}...")
            htf[phase] = _generate_htf(phase, top_prop, setup_options)

        for phase in phases:
            if not use_given_geometries:
                _validate_endstate_energies_for_htf(htf, top_prop, phase,
                                                    beta=1.0 / (kB * temperature),
                                                    ENERGY_THRESHOLD=ENERGY_THRESHOLD)
            else:
                _logger.info(f"'use_given_geometries' was passed to setup; skipping endstate validation")

            #TODO expose more of these options in input
            if build_samplers:

                n_states = setup_options['n_states']
                _logger.info(f"\tn_states: {n_states}")
                if 'n_replicas' not in setup_options:
                    n_replicas = n_states
                else:
                    n_replicas = setup_options['n_replicas']

                checkpoint_interval = setup_options['checkpoint_interval']

                # generating lambda protocol
                lambda_protocol = LambdaProtocol(functions=setup_options['protocol-type'])
                _logger.info(f'Using lambda protocol : {setup_options["protocol-type"]}')


                if atom_selection:
                    selection_indices = htf[phase].hybrid_topology.select(atom_selection)
                else:
                    selection_indices = None

                storage_name = AnyPath(trajectory_directory) / f"{trajectory_prefix}-{phase}.nc"
                _logger.info(f'\tstorage_name: {storage_name}')
                _logger.info(f'\tselection_indices {selection_indices}')
                _logger.info(f'\tcheckpoint interval {checkpoint_interval}')
                reporter = MultiStateReporter(storage_name, analysis_particle_indices=selection_indices,
                                              checkpoint_interval=checkpoint_interval)

                if phase == 'vacuum':
                    endstates = False
                else:
                    endstates = setup_options['unsampled_endstates']

                if setup_options['fe_type'] == 'fah':
                    _logger.info('SETUP FOR FAH DONE')
                    return {'topology_proposals': top_prop, 'hybrid_topology_factories': htf}

                # get platform
                platform = get_openmm_platform(platform_name=setup_options['platform'])
                # Setup context caches for multistate samplers
                energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
                sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)

                if setup_options['fe_type'] == 'sams':
                    hss[phase] = HybridSAMSSampler(mcmc_moves=mcmc.LangevinDynamicsMove(
                        timestep=timestep,
                        collision_rate=1.0 / unit.picosecond,
                        n_steps=n_steps_per_move_application,
                        reassign_velocities=False,
                        n_restart_attempts=20, constraint_tolerance=1e-06),
                        hybrid_factory=htf[phase], online_analysis_interval=setup_options['offline-freq'],
                        online_analysis_minimum_iterations=10, flatness_criteria=setup_options['flatness-criteria'],
                        gamma0=setup_options['gamma0']
                    )
                    hss[phase].setup(n_states=n_states, n_replicas=n_replicas, temperature=temperature,
                                     storage_file=reporter, lambda_protocol=lambda_protocol, endstates=endstates)
                    # We need to specify contexts AFTER setup
                    hss[phase].energy_context_cache = energy_context_cache
                    hss[phase].sampler_context_cache = sampler_context_cache
                elif setup_options['fe_type'] == 'repex':
                    hss[phase] = HybridRepexSampler(mcmc_moves=mcmc.LangevinDynamicsMove(
                        timestep=timestep,
                        collision_rate=1.0 / unit.picosecond,
                        n_steps=n_steps_per_move_application,
                        reassign_velocities=False,
                        n_restart_attempts=20, constraint_tolerance=1e-06),
                        hybrid_factory=htf[phase], online_analysis_interval=setup_options['offline-freq'],
                    )
                    hss[phase].setup(n_states=n_states, temperature=temperature, storage_file=reporter,
                                     endstates=endstates)
                    # We need to specify contexts AFTER setup
                    hss[phase].energy_context_cache = energy_context_cache
                    hss[phase].sampler_context_cache = sampler_context_cache
            else:
                _logger.info(f"omitting sampler construction")

            if serialize_systems:
                # save the systems and the states
                pass

                _logger.info('WRITING OUT XML FILES')
                #old_thermodynamic_state, new_thermodynamic_state, hybrid_thermodynamic_state, _ = generate_endpoint_thermodynamic_states(htf[phase].hybrid_system, _top_prop)

                xml_directory = AnyPath(setup_options["trajectory_directory"]) / "xml"
                if not os.path.exists(xml_directory):
                    os.makedirs(xml_directory)
                from perses.utils import data
                _logger.info('WRITING OUT XML FILES')
                _logger.info(f'Saving the hybrid, old and new system to disk')
                data.serialize(htf[phase].hybrid_system, trajectory_directory / "xml" /f"{phase}-hybrid-system.gz")
                data.serialize(htf[phase]._old_system, trajectory_directory / "xml" / f"{phase}-old-system.gz")
                data.serialize(htf[phase]._new_system, trajectory_directory / "xml" /f"{phase}-new-system.gz")

        return {'topology_proposals': top_prop, 'hybrid_topology_factories': htf, 'hybrid_samplers': hss}


def run(yaml_filename=None, override_string=None):
    cli_tool_name = sys.argv[0].split(os.sep)[-1]
    if cli_tool_name == "perses-relative":
        warnings.warn("perses-relative will be removed in 0.11, see https://github.com/choderalab/perses/tree/main/examples/new-cli for new CLI tool", FutureWarning)
    _logger.info("Beginning Setup...")
    if yaml_filename is None:
       try:
          yaml_filename = sys.argv[1]
          _logger.info(f"Detected yaml file: {yaml_filename}")
       except IndexError as e:
           _logger.critical(f"You must specify the setup yaml file as an argument to the script.")

    _logger.info(f"Getting setup options from {yaml_filename}")

    setup_options = getSetupOptions(yaml_filename, override_string=override_string)
    _logger.debug(f"Setup Options {setup_options}")

    # Generate yaml file with parsed setup options
    _generate_parsed_yaml(setup_options=setup_options, input_yaml_file_path=yaml_filename)

    # The name of the reporter file includes the phase name, so we need to check each
    # one
    for phase in setup_options['phases']:
        trajectory_directory = setup_options['trajectory_directory']
        trajectory_prefix = setup_options['trajectory_prefix']
        reporter_file = AnyPath(f"{trajectory_directory}/{trajectory_prefix}-{phase}.nc")
        # Once we find one, we are good to resume the simulation
        if os.path.isfile(reporter_file):
            _resume_run(setup_options)
            # There is a loop in _resume_run for each phase so once we extend each phase
            # we are done
            exit()

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
        trajectory_directory = AnyPath(setup_options['trajectory_directory'])
        out_trajectory_prefix = AnyPath(setup_options['out_trajectory_prefix'])
        for phase in setup_options['phases']:
            ne_fep_run = pickle.load(open(AnyPath(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase))), 'rb'))
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
                with open(AnyPath(os.path.join(trajectory_directory, "%s_%s_fep.neq.pkl" % (out_trajectory_prefix, phase))), 'wb') as f:
                    pickle.dump(ne_fep_run, f)
                    print("pickle save successful; terminating.")

            except Exception as e:
                print(e)
                print("Unable to save run object as a pickle; saving as npy")
                np.savez(AnyPath(os.path.join(trajectory_directory, "%s_%s_fep.neq.npy" % (out_trajectory_prefix, phase))), ne_fep_run)

    else:
        _logger.info(f"Running setup...")
        setup_dict = run_setup(setup_options)

        trajectory_prefix = setup_options['trajectory_prefix']
        trajectory_directory = setup_options['trajectory_directory']

        #write out topology proposals
        try:
            _logger.info(f"Writing topology proposal {trajectory_prefix}-topology_proposals.pkl to {trajectory_directory}...")
            with open(AnyPath(os.path.join(trajectory_directory, "%s-topology_proposals.pkl" % (trajectory_prefix))), 'wb') as f:
                pickle.dump(setup_dict['topology_proposals'], f)
        except Exception as e:
            print(e)
            _logger.info("Unable to save run object as a pickle; saving as npy")
            np.savez(AnyPath(os.path.join(trajectory_directory, "%s_topology_proposals.npy" % (trajectory_prefix))), setup_dict['topology_proposals'])

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

                # TODO: Validation here should be done with the same _validate_endstate_energies_for_htf function.
                zero_state_error, one_state_error = validate_endstate_energies(hybrid_factory._topology_proposal, hybrid_factory, _forward_added_valence_energy, _reverse_subtracted_valence_energy, beta = 1.0/(kB*temperature), ENERGY_THRESHOLD = ENERGY_THRESHOLD, trajectory_directory=AnyPath(f'{setup_options["trajectory_directory"]}/xml/{phase}'))
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
                    with open(AnyPath(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase))), 'wb') as f:
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
                    ne_fep_run = pickle.load(open(AnyPath(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase))), 'rb'))
                    ne_fep_run.AIS(num_particles = setup_options['n_particles'],
                                   protocols = lambdas,
                                   num_integration_steps = setup_options['ncmc_num_integration_steps'],
                                   return_timer = False,
                                   rethermalize = setup_options['ncmc_rethermalize'])


                    print("calculation complete; deactivating client")
                    #ne_fep_run.deactivate_client()

                    # try to write out the ne_fep object as a pickle
                    try:
                        with open(AnyPath(os.path.join(trajectory_directory, "%s_%s_fep.neq.pkl" % (trajectory_prefix, phase))), 'wb') as f:
                            pickle.dump(ne_fep_run, f)
                            print("pickle save successful; terminating.")

                    except Exception as e:
                        print(e)
                        print("Unable to save run object as a pickle; saving as npy")
                        np.savez(AnyPath(os.path.join(trajectory_directory, "%s_%s_fep.neq.npy" % (trajectory_prefix, phase))), ne_fep_run)

        elif setup_options['fe_type'] == 'sams':
            _logger.info(f"Detecting sams as fe_type...")
            _logger.info(f"Writing hybrid factory {trajectory_prefix}-hybrid_factory.npy to {trajectory_directory}...")
            np.savez(AnyPath(os.path.join(trajectory_directory, trajectory_prefix + "-hybrid_factory.npy")),
                    setup_dict['hybrid_topology_factories'])

            hss = setup_dict['hybrid_samplers']
            logZ = dict()
            free_energies = dict()
            _logger.info(f"Iterating through phases for sams...")
            for phase in setup_options['phases']:
                _logger.info(f'\tRunning {phase} phase...')
                hss_run = hss[phase]

                _logger.info(f"\t\tminimizing...\n\n")
                hss_run.minimize()
                _logger.info(f"\n\n")

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
            _logger.info(f"Writing hybrid factory {trajectory_prefix}-hybrid_factory.npy to {trajectory_directory}...")
            np.savez(AnyPath(os.path.join(trajectory_directory, trajectory_prefix + "-hybrid_factory.npy")),
                    setup_dict['hybrid_topology_factories'])

            hss = setup_dict['hybrid_samplers']
            _logger.info(f"Iterating through phases for repex...")
            for phase in setup_options['phases']:
                print(f'Running {phase} phase')
                hss_run = hss[phase]

                _logger.info(f"\t\tminimizing...\n\n")
                hss_run.minimize()
                _logger.info(f"\n\n")

                _logger.info(f"\t\tequilibrating...\n\n")
                hss_run.equilibrate(n_equilibration_iterations)
                _logger.info(f"\n\n")

                _logger.info(f"\t\textending simulation...\n\n")
                hss_run.extend(setup_options['n_cycles'])
                _logger.info(f"\n\n")

                _logger.info(f"\t\tFinished phase {phase}")


def _resume_run(setup_options):
    from openmmtools.cache import ContextCache
    from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
    # get platform
    platform = get_openmm_platform(platform_name=setup_options['platform'])
    # Setup context caches for multistate samplers
    energy_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)
    sampler_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)

    if setup_options['fe_type'] == 'sams':
        logZ = dict()
        free_energies = dict()

        _logger.info(f"Iterating through phases for sams...")
        for phase in setup_options['phases']:
            trajectory_directory = setup_options['trajectory_directory']
            trajectory_prefix = setup_options['trajectory_prefix']

            reporter_file = AnyPath(f"{trajectory_directory}/{trajectory_prefix}-{phase}.nc")
            reporter = MultiStateReporter(reporter_file)
            simulation = HybridSAMSSampler.from_storage(reporter)
            total_steps = setup_options['n_cycles']
            run_so_far = simulation.iteration
            left_to_do = total_steps - run_so_far
            # set context caches
            simulation.sampler_context_cache = sampler_context_cache
            simulation.energy_context_cache = energy_context_cache
            _logger.info(f"\t\textending simulation...\n\n")
            simulation.extend(n_iterations=left_to_do)
            logZ[phase] = simulation._logZ[-1] - simulation._logZ[0]
            free_energies[phase] = simulation._last_mbar_f_k[-1] - simulation._last_mbar_f_k[0]
            _logger.info(f"\t\tFinished phase {phase}")
        for phase in free_energies:
            print(f"Comparing ligand {setup_options['old_ligand_index']} to {setup_options['new_ligand_index']}")
            print(f"{phase} phase has a free energy of {free_energies[phase]}")

    elif setup_options['fe_type'] == 'repex':
        for phase in setup_options['phases']:
            print(f'Running {phase} phase')
            trajectory_directory = setup_options['trajectory_directory']
            trajectory_prefix = setup_options['trajectory_prefix']

            reporter_file = AnyPath(f"{trajectory_directory}/{trajectory_prefix}-{phase}.nc")
            reporter = MultiStateReporter(reporter_file)
            simulation = HybridRepexSampler.from_storage(reporter)
            total_steps = setup_options['n_cycles']
            run_so_far = simulation.iteration
            left_to_do = total_steps - run_so_far
            # set context caches
            simulation.sampler_context_cache = sampler_context_cache
            simulation.energy_context_cache = energy_context_cache
            _logger.info(f"\t\textending simulation...\n\n")
            simulation.extend(n_iterations=left_to_do)
            _logger.info(f"\n\n")
            _logger.info(f"\t\tFinished phase {phase}")
    else:
        raise("Can't resume")

def _process_overrides(overrides, yaml_options):
    """
    Takes in a string of overrides, converts them into a dict, then merges them with the
    yaml_options dict to override options set in the file
    """

    overrides_dict = {}
    for opt in overrides:
        key, val = opt.split(":", maxsplit=1)

        # Check for duplicates
        if key in overrides_dict:
            raise ValueError(
                f"There were duplicate override options, result will be ambiguous! Key {key} repeated!"
            )

        # I don't like this part, but I rather do this then to try and add type checking
        # and casting in setup_relative.py
        # We do int then float since slices might need a int, but if we can't make it an
        # int then it is probably a float, and if we can 't do that, then it is a str.

        # First lets see if we can make it a int:
        try:
            # First check if we have a number like 4.2 which python will convert to
            # 4 if you do int(4.2) but we can check if int(val) and float(val) cast to
            # the same object
            if int(val) != float(val):
                raise ValueError
            val = int(val)
        except ValueError:
            # Now try float
            try:
                val = float(val)
            except ValueError:
                # Just keep it a str
                pass

        overrides_dict[key] = val

    return {**yaml_options, **overrides_dict}


def _generate_htf(phase: str, topology_proposal_dictionary: dict, setup_options: dict):
    """
    Generates topology proposal for phase. Supports both HybridTopologyFactory and new RESTCapableHybridTopologyFactory
    """
    factory_name = setup_options['hybrid_topology_factory']
    htf_setup_dict = {
        "neglected_new_angle_terms": topology_proposal_dictionary[f"{phase}_forward_neglected_angles"],
        "neglected_old_angle_terms": topology_proposal_dictionary[f"{phase}_reverse_neglected_angles"],
        "softcore_LJ_v2": setup_options['softcore_v2'],
        "interpolate_old_and_new_14s": setup_options['anneal_1,4s'],
        "rmsd_restraint": setup_options['rmsd_restraint']
    }

    if factory_name == HybridTopologyFactory.__name__:
        factory = HybridTopologyFactory
    elif factory_name == RESTCapableHybridTopologyFactory.__name__:
        factory = RESTCapableHybridTopologyFactory
        # Add/use specified REST HTF parameters if present
        rest_specific_options = dict()
        try:
            rest_specific_options.update({'rest_radius': setup_options['rest_radius']})
        except KeyError:
            _logger.info("'rest_radius' not specified. Using default value.")
        try:
            rest_specific_options.update({'w_lifting': setup_options['w_lifting']})
        except KeyError:
            _logger.info("'w_lifting' not specified. Using default value.")

        # update htf_setup_dictionary with new parameters
        htf_setup_dict.update(rest_specific_options)
    else:
        raise ValueError(f"You specified an unsupported factory type: {factory_name}. Currently, the supported "
                         f"factories are: HybridTopologyFactory and RESTCapableHybridTopologyFactory.")
    htf = factory(topology_proposal_dictionary[f'{phase}_topology_proposal'],
                  topology_proposal_dictionary[f'{phase}_old_positions'],
                  topology_proposal_dictionary[f'{phase}_new_positions'],
                  **htf_setup_dict
                  )
    return htf


def _validate_endstate_energies_for_htf(hybrid_topology_factory_dict: dict, topology_proposal_dict: dict, phase: str,
                                        **kwargs):
    """
    Validates endstate energies according to different hybrid topology factories and phases.

    Parameters
    ----------
    hybrid_topology_factory: dict
        Dictionary with different hybrid topology factories for different phases. Phase as key, HTF as value.
    topology_proposal_dict: dict
        Dictionary with different topology proposals for different phases. Phase as key, top_pro as value.
    phase: str
        Name of the phase.
    """
    current_htf = hybrid_topology_factory_dict[phase]
    if isinstance(current_htf, HybridTopologyFactory):
        topology_proposal = topology_proposal_dict[f"{phase}_topology_proposal"]
        forward_added_valence_energy = topology_proposal_dict[f"{phase}_added_valence_energy"]
        reverse_substracted_valence_energy = topology_proposal_dict[f"{phase}_subtracted_valence_energy"]
        zero_state_error, one_state_error = validate_endstate_energies(topology_proposal,
                                                                       current_htf,
                                                                       forward_added_valence_energy,
                                                                       reverse_substracted_valence_energy,
                                                                       **kwargs)
        _logger.info(f"\t\terror in zero state: {zero_state_error}")
        _logger.info(f"\t\terror in one state: {one_state_error}")
    elif isinstance(current_htf, RESTCapableHybridTopologyFactory):
        for endstate in [0, 1]:
            validate_endstate_energies_point(current_htf, endstate=endstate, minimize=True)


def _generate_parsed_yaml(setup_options, input_yaml_file_path):
    """
    Creates YAML file with parsed setup options in the working directory of the simulation.

    It adds timestamp and ligands names information (old and new).

    Parameters
    ----------
    setup_options: dict
        Dictionary with perses setup options. Meant to be the returned dictionary from
        ``perses.app.setup_relative_calculation.getSetupOptions``,
    input_yaml_file_path: str or Path object
        Path to input yaml file with perses parameters

    Returns
    -------
    out_yaml_path: str
        String with the path to the generated parsed yaml file.
    """
    from openff.toolkit.topology import Molecule
    # The parsed yaml file will live in the experiment directory to avoid race conditions with other experiments
    yaml_path = AnyPath(setup_options['trajectory_directory'])
    yaml_name = AnyPath(input_yaml_file_path).name  # extract name from input/template yaml file.
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    yaml_parse_name = f"perses-{time}-{yaml_name}"
    # Add timestamp information
    setup_options["timestamp"] = time
    # Read input sdf file and save into list -- We don't check stereochemistry
    ligand_file_path = AnyPath(setup_options['ligand_file'])
    # Get the file format by getting the suffix and removing the "."
    ligand_file_format = ligand_file_path.suffix[1:]
    with open(ligand_file_path, "r") as ligand_file_object:
        ligands_list = Molecule.from_file(ligand_file_object, file_format=ligand_file_format, allow_undefined_stereo=True)
    # Get names according to indices in parsed setup options
    setup_options['old_ligand_name'] = ligands_list[setup_options['old_ligand_index']].name
    setup_options['new_ligand_name'] = ligands_list[setup_options['new_ligand_index']].name
    # Write parsed and added setup options into yaml file
    out_file_path = yaml_path / yaml_parse_name
    with open(out_file_path, "w") as outfile:
        yaml.dump(setup_options, outfile)

    return out_file_path



if __name__ == "__main__":
    run()

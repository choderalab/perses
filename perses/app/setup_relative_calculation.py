import yaml
import numpy as np
import pickle
import os
import sys
import simtk.unit as unit
import logging

from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
from perses.annihilation.relative import HybridTopologyFactory
from perses.app.relative_setup import NonequilibriumSwitchingFEP, RelativeFEPSetup
from perses.annihilation.lambda_protocol import LambdaProtocol

from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter, sams, replicaexchange
from perses.utils.smallmolecules import render_atom_mapping
from perses.tests.utils import validate_endstate_energies

logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("setup_relative_calculation")
_logger.setLevel(logging.INFO)

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
    setup_options = yaml.load(yaml_file)
    yaml_file.close()

    _logger.info("\tDetecting phases...")
    if 'phases' not in setup_options:
        setup_options['phases'] = ['complex','solvent']
        _logger.warning('\t\tNo phases provided - running complex and solvent as default.')
    else:
        _logger.info(f"\t\tphases detected: {setup_options['phases']}")

    if 'protocol-type' not in setup_options:
        setup_options['protocol-type'] = 'default'

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

        #for dask implementation
        if 'processes' not in setup_options:
            _logger.info(f"\t\t\tprocesses is not specified; default to 100")
            setup_options['processes'] = 100
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
            _logger.info(f"\t\t\tLSF is not specified; default to True")
            setup_options['LSF'] = True

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
        elif setup_options['run_type'] not in ['None', 'anneal', 'equilibrate']:
            raise Exception(f"'run_type' must be None, 'anneal', or 'equilibrate'; input was specified as {setup_options['run_type']} with type {type(setup_options['run_type'])}")

        #to instantiate the particles:
        if 'n_lambdas' not in setup_options:
            _logger.info(f"\t\t\tn_lambdas is not specified; default to None (an online protocol must be provided)")
            setup_options['n_lambdas'] = None
        if 'n_particles' not in setup_options:
            raise Exception(f"for particle annealing, 'n_particles' must be specified")
        if 'direction' not in setup_options:
            _logger.info(f"\t\t\tdirection is not specified; default to None (running both forward and reverse)")
            setup_options['direction'] = None
        if 'ncmc_save_interval' not in setup_options:
            _logger.info(f"\t\t\tncmc_save_interval not specified: default to None.")
            setup_options['ncmc_save_interval'] = None
        if 'ncmc_collision_rate_ps' not in setup_options:
            _logger.info(f"\t\t\tcollision_rate not specified: default to np.inf.")
            setup_options['ncmc_collision_rate_ps'] = np.inf/unit.picoseconds
        else:
            setup_options['ncmc_collision_rate_ps'] /= unit.picoseconds

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

    if 'mapping_strength' not in setup_options:
        setup_options['mapping_strength'] = 'default'

    if 'anneal_1,4s' not in setup_options:
        setup_options['anneal_1,4s'] = False
        _logger.info(f"\t'anneal_1,4s' not specified: default to 'False' (i.e. since 1,4 interactions are not being annealed, they are being used to make new/old atom proposals in the geometry engine.)")

    if 'softcore_v2' not in setup_options:
        setup_options['softcore_v2'] = False
        _logger.info(f"\t'softcore_v2' not specified: default to 'False'")

    _logger.info(f"\ttrajectory_directory detected: {trajectory_directory}.  making dir...")
    if setup_options['run_type'] != 'anneal':
        assert (os.path.exists(trajectory_directory) == False), 'Output trajectory directory already exists. Refusing to overwrite'
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
        _logger.info(f"\ttimestep: {timestep}fs.")
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
                                          temperature=temperature, solvent_padding=solvent_padding_angstroms,
                                          atom_map=atom_map, neglect_angles = setup_options['neglect_angles'], anneal_14s = setup_options['anneal_1,4s'])
        _logger.info(f"\n\n\n")

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
        if 'complex' in phases:
            top_prop['complex_topology_proposal'] = fe_setup.complex_topology_proposal
            top_prop['complex_geometry_engine'] = fe_setup._complex_geometry_engine
            top_prop['complex_old_positions'] = fe_setup.complex_old_positions
            top_prop['complex_new_positions'] = fe_setup.complex_new_positions
            top_prop['complex_added_valence_energy'] = fe_setup._complex_added_valence_energy
            top_prop['complex_subtracted_valence_energy'] = fe_setup._complex_subtracted_valence_energy
            top_prop['complex_logp_proposal'] = fe_setup._complex_logp_proposal
            top_prop['complex_logp_reverse'] = fe_setup._complex_logp_reverse
            top_prop['complex_forward_neglected_angles'] = fe_setup._complex_forward_neglected_angles
            top_prop['complex_reverse_neglected_angles'] = fe_setup._complex_reverse_neglected_angles

            _logger.info(f"\twriting complex render_atom_mapping...")
            atom_map_outfile = os.path.join(os.getcwd(), trajectory_directory, 'render_complex_mapping.png')
            render_atom_mapping(atom_map_outfile, fe_setup._ligand_oemol_old, fe_setup._ligand_oemol_new, fe_setup.non_offset_new_to_old_atom_map)

        if 'solvent' in phases:
            top_prop['solvent_topology_proposal'] = fe_setup.solvent_topology_proposal
            top_prop['solvent_geometry_engine'] = fe_setup._solvent_geometry_engine
            top_prop['solvent_old_positions'] = fe_setup.solvent_old_positions
            top_prop['solvent_new_positions'] = fe_setup.solvent_new_positions
            top_prop['solvent_added_valence_energy'] = fe_setup._solvated_added_valence_energy
            top_prop['solvent_subtracted_valence_energy'] = fe_setup._solvated_subtracted_valence_energy
            top_prop['solvent_logp_proposal'] = fe_setup._ligand_logp_proposal_solvated
            top_prop['solvent_logp_reverse'] = fe_setup._ligand_logp_reverse_solvated
            top_prop['solvent_forward_neglected_angles'] = fe_setup._solvated_forward_neglected_angles
            top_prop['solvent_reverse_neglected_angles'] = fe_setup._solvated_reverse_neglected_angles

            _logger.info(f"\twriting solvent render_atom_mapping...")
            atom_map_outfile = os.path.join(os.getcwd(), trajectory_directory, 'render_solvent_mapping.png')
            render_atom_mapping(atom_map_outfile, fe_setup._ligand_oemol_old, fe_setup._ligand_oemol_new, fe_setup.non_offset_new_to_old_atom_map)

        if 'vacuum' in phases:
            top_prop['vacuum_topology_proposal'] = fe_setup.vacuum_topology_proposal
            top_prop['vacuum_geometry_engine'] = fe_setup._vacuum_geometry_engine
            top_prop['vacuum_old_positions'] = fe_setup.vacuum_old_positions
            top_prop['vacuum_new_positions'] = fe_setup.vacuum_new_positions
            top_prop['vacuum_added_valence_energy'] = fe_setup._vacuum_added_valence_energy
            top_prop['vacuum_subtracted_valence_energy'] = fe_setup._vacuum_subtracted_valence_energy
            top_prop['vacuum_logp_proposal'] = fe_setup._vacuum_logp_proposal
            top_prop['vacuum_logp_reverse'] = fe_setup._vacuum_logp_reverse
            top_prop['vacuum_forward_neglected_angles'] = fe_setup._vacuum_forward_neglected_angles
            top_prop['vacuum_reverse_neglected_angles'] = fe_setup._vacuum_reverse_neglected_angles

            _logger.info(f"\twriting vacuum render_atom_mapping...")
            atom_map_outfile = os.path.join(os.getcwd(), trajectory_directory, 'render_vacuum_mapping.png')
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

            ne_fep[phase] = NonequilibriumSwitchingFEP(hybrid_factory = hybrid_factory,
                                                       protocol = setup_options['lambda_protocol'],
                                                       n_equilibrium_steps_per_iteration = n_equilibrium_steps_per_iteration,
                                                       temperature = temperature,
                                                       trajectory_directory=trajectory_directory,
                                                       trajectory_prefix=f"{trajectory_prefix}_{phase}",
                                                       atom_selection=atom_selection,
                                                       eq_splitting_string = eq_splitting,
                                                       neq_splitting_string = neq_splitting,
                                                       measure_shadow_work=measure_shadow_work,
                                                       timestep=timestep,
                                                       ncmc_save_interval = ncmc_save_interval,
                                                       write_ncmc_configuration = write_ncmc_configuration,
                                                       relative_transform = True)

        print("Nonequilibrium switching driver class constructed")

        return {'topology_proposals': top_prop, 'ne_fep': ne_fep}

    else:
        _logger.info(f"\tno nonequilibrium detected.")
        n_states = setup_options['n_states']
        _logger.info(f"\tn_states: {n_states}")
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

           # Define necessary vars to check energy bookkeeping
            _top_prop = top_prop['%s_topology_proposal' % phase]
            _htf = htf[phase]
            _forward_added_valence_energy = top_prop['%s_added_valence_energy' % phase]
            _reverse_subtracted_valence_energy = top_prop['%s_subtracted_valence_energy' % phase]

            zero_state_error, one_state_error = validate_endstate_energies(_top_prop, _htf, _forward_added_valence_energy, _reverse_subtracted_valence_energy, beta = 1.0/(kB*temperature), ENERGY_THRESHOLD = ENERGY_THRESHOLD)
            _logger.info(f"\t\terror in zero state: {zero_state_error}")
            _logger.info(f"\t\terror in one state: {one_state_error}")

            # generating lambda protocol
            lambda_protocol = LambdaProtocol(functions=setup_options['protocol-type'])


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

            #TODO expose more of these options in input
            if setup_options['fe_type'] == 'sams':
                _logger.warning(f'Relative free energies do not currently work with unsampled endstates')
                hss[phase] = HybridSAMSSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=timestep,
                                                                                             collision_rate=5.0 / unit.picosecond,
                                                                                             n_steps=n_steps_per_move_application,
                                                                                             reassign_velocities=False,
                                                                                             n_restart_attempts=20,
                                                                                             splitting="V R R R O R R R V",
                                                                                             constraint_tolerance=1e-06),
                                               hybrid_factory=htf[phase], online_analysis_interval=setup_options['offline-freq'],
                                               online_analysis_minimum_iterations=10,flatness_criteria=setup_options['flatness-criteria'],
                                               gamma0=setup_options['gamma0'])
                hss[phase].setup(n_states=n_states, temperature=temperature,storage_file=reporter,lambda_protocol=lambda_protocol)
            elif setup_options['fe_type'] == 'repex':
                hss[phase] = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=timestep,
                                                                                             collision_rate=5.0 / unit.picosecond,
                                                                                             n_steps=n_steps_per_move_application,
                                                                                             reassign_velocities=False,
                                                                                             n_restart_attempts=20,
                                                                                             splitting="V R R R O R R R V",
                                                                                             constraint_tolerance=1e-06),
                                                                                             hybrid_factory=htf[phase],online_analysis_interval=setup_options['offline-freq'])
                hss[phase].setup(n_states=n_states, temperature=temperature,storage_file=reporter,lambda_protocol=lambda_protocol)

        return {'topology_proposals': top_prop, 'hybrid_topology_factories': htf, 'hybrid_samplers': hss}

if __name__ == "__main__":
    _logger.info("Beginning Setup...")
    try:
       yaml_filename = sys.argv[1]
       _logger.info(f"Detected yaml file: {yaml_filename}")
    except IndexError as e:
        _logger.critical(f"You must specify the setup yaml file as an argument to the script.")

    _logger.info(f"Getting setup options from {yaml_filename}")
    setup_options = getSetupOptions(yaml_filename)
    if setup_options['run_type'] == 'anneal':
        _logger.info(f"skipping setup and annealing...")
        trajectory_prefix = setup_options['trajectory_prefix']
        trajectory_directory = setup_options['trajectory_directory']
        out_trajectory_prefix = setup_options['out_trajectory_prefix']
        for phase in setup_options['phases']:
            ne_fep_run = pickle.load(open(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase)), 'rb'))
            #ne_fep_run.activate_client(LSF = LSF, processes = processes, adapt = adapt) #now call n processes
            ne_fep_run.instantiate_particles(n_lambdas = None,
                                             n_particles = setup_options['n_particles'],
                                             direction = setup_options['direction'],
                                             ncmc_save_interval = setup_options['ncmc_save_interval'],
                                             collision_rate = setup_options['ncmc_collision_rate_ps']/unit.picoseconds,
                                             LSF = setup_options['LSF'],
                                             num_processes = setup_options['num_processes'],
                                             adapt = setup_options['adapt'])

            if setup_options['online_protocol'] is None:
                online_protocol = None
            elif type(setup_options['online_protocol']) == int:
                online_protocol = {'forward': np.linspace(0,1,setup_options['online_protocol']), 'reverse': np.linspace(1,0,setup_options['online_protocol'])}
            else:
                #attempt to load npy array
                online_protocol = np.load(setup_options['online_protocol'], allow_pickle = True)

            ne_fep_run.algorithm_4(observable = setup_options['observable'],
                                   trailblaze_observable_threshold = setup_options['trailblaze_observable_threshold'],
                                   resample_observable_threshold = setup_options['resample_observable_threshold'],
                                   num_integration_steps = setup_options['ncmc_num_integration_steps'],
                                   resampling_method = setup_options['resampling_method'],
                                   online_protocol = online_protocol)
            print("calculation complete; deactivating client")
            #ne_fep_run.deactivate_client()

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
                hybrid_factory = ne_fep_run._factory

                top_proposal = setup_dict['topology_proposals'][f"{phase}_topology_proposal"]
                _forward_added_valence_energy = setup_dict['topology_proposals'][f"{phase}_added_valence_energy"]
                _reverse_subtracted_valence_energy = setup_dict['topology_proposals'][f"{phase}_subtracted_valence_energy"]

                zero_state_error, one_state_error = validate_endstate_energies(hybrid_factory._topology_proposal, hybrid_factory, _forward_added_valence_energy, _reverse_subtracted_valence_energy, beta = 1.0/(kB*temperature), ENERGY_THRESHOLD = ENERGY_THRESHOLD)
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
                    ne_fep_run.equilibrate(n_equilibration_iterations,
                                           endstates = endstates,
                                           max_size = max_file_size,
                                           decorrelate = True,
                                           timer = True,
                                           minimize = False,
                                           LSF = setup_options['LSF'],
                                           num_processes = 2,
                                           adapt = setup_options['adapt'])
                    #ne_fep_run.deactivate_client()
                    with open(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase)), 'wb') as f:
                        pickle.dump(ne_fep_run, f)



                if setup_options['run_type'] == None:
                    print("annealing...")
                    ne_fep_run = pickle.load(open(os.path.join(trajectory_directory, "%s_%s_fep.eq.pkl" % (trajectory_prefix, phase)), 'rb'))
                    #ne_fep_run.activate_client(LSF = LSF, processes = processes, adapt = adapt) #now call n processes
                    ne_fep_run.instantiate_particles(n_lambdas = None,
                                                     n_particles = setup_options['n_particles'],
                                                     direction = setup_options['direction'],
                                                     ncmc_save_interval = setup_options['ncmc_save_interval'],
                                                     collision_rate = setup_options['ncmc_collision_rate_ps']/unit.picoseconds,
                                                     LSF = setup_options['LSF'],
                                                     num_processes = setup_options['num_processes'],
                                                     adapt = setup_options['adapt'])

                    if setup_options['online_protocol'] is None:
                        online_protocol = None
                    elif type(setup_options['online_protocol']) == int:
                        online_protocol = {'forward': np.linspace(0,1,setup_options['online_protocol']), 'reverse': np.linspace(1,0,setup_options['online_protocol'])}
                    else:
                        #attempt to load npy array
                        online_protocol = np.load(setup_options['online_protocol'], allow_pickle = True)

                    ne_fep_run.algorithm_4(observable = setup_options['observable'],
                                           trailblaze_observable_threshold = setup_options['trailblaze_observable_threshold'],
                                           resample_observable_threshold = setup_options['resample_observable_threshold'],
                                           num_integration_steps = setup_options['ncmc_num_integration_steps'],
                                           resampling_method = setup_options['resampling_method'],
                                           online_protocol = online_protocol)

                    # ne_fep_run.instantiate_particles(n_particles = setup_options['n_particles'],
                    #                                  direction = setup_options['direction'])
                    # ne_fep_run.algorithm_4(observable = setup_options['observable'],
                    #                        trailblaze_observable_threshold = setup_options['trailblaze_observable_threshold'],
                    #                        resample_observable_threshold = setup_options['resample_observable_threshold'],
                    #                        check_interval = setup_options['check_interval'],
                    #                        resampling_method = setup_options['resampling_method'])
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
            _logger.info(f"Writing hybrid factory {trajectory_prefix}hybrid_factory.npy to {trajectory_directory}...")
            np.save(os.path.join(trajectory_directory, trajectory_prefix + "hybrid_factory.npy"),
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

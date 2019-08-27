import yaml
import numpy as np
import pickle
import os
import sys
import simtk.unit as unit
import logging

from perses.samplers.samplers import HybridSAMSSampler, HybridRepexSampler
from perses.annihilation.relative import HybridTopologyFactory
from perses.app.relative_setup import NonequilibriumSwitchingFEP, RelativeFEPSetup
from perses.annihilation.lambda_protocol import LambdaProtocol

from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter, sams, replicaexchange
from perses.utils.smallmolecules import render_atom_mapping

logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("setup_relative_calculation")
_logger.setLevel(logging.INFO)

ENERGY_THRESHOLD = 1e-1
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

    if 'lambda-protocol' not in setup_options:
        setup_options['lambda-protocol'] = None
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

    trajectory_directory = setup_options['trajectory_directory']

    # check if the neglect_angles is specified in yaml

    if 'neglect_angles' not in setup_options:
        setup_options['neglect_angles'] = False
        _logger.info(f"\t'neglect_angles' not specified: default to 'False'.")
    else:
        _logger.info(f"\t'neglect_angles' detected: {setup_options['neglect_angles']}.")


    _logger.info(f"\ttrajectory_directory detected: {trajectory_directory}.  making dir...")
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
    fe_setup : NonequilibriumFEPSetup
        The setup class for this calculation
    ne_fep : NonequilibriumSwitchingFEP
        The nonequilibrium driver class
    """
    from perses.tests.utils import validate_endstate_energies
    phases = setup_options['phases']

    if len(phases) != 2:
        _logger.info(f"\tnumber of phases is NOT 2...complex and solvent will be provided...")
        phases = ['complex', 'solvent']

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
        neq_splitting = "V R O H R V"
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
                                          atom_map=atom_map, neglect_angles = setup_options['neglect_angles'])
        _logger.info(f"\n\n\n")

        _logger.info(f"\twriting pickle output...")
        pickle_outfile = open(os.path.join(os.getcwd(), trajectory_directory, setup_pickle_file), 'wb')

        try:
            pickle.dump(fe_setup, pickle_outfile)
            _logger.info(f"\tsuccessfully dumped pickle.")
        except Exception as e:
            print(e)
            print("\tUnable to save setup object as a pickle")
        finally:
            pickle_outfile.close()

        _logger.info(f"\tsetup is complete.  Writing proposals and positions for each phase to top_prop dict...")

        top_prop = dict()
        if 'complex' in phases:
            top_prop['complex_topology_proposal'] = fe_setup.complex_topology_proposal
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
        _logger.info(f"\tno atom selection detected: default to None.")
        atom_selection = None

    if setup_options['fe_type'] == 'nonequilibrium':
        n_equilibrium_steps_per_iteration = setup_options['n_equilibrium_steps_per_iteration']

        n_steps_ncmc_protocol = setup_options['n_steps_ncmc_protocol']
        scheduler_address = setup_options['scheduler_address']

        ne_fep = dict()
        for phase in phases:
            ne_fep[phase] = NonequilibriumSwitchingFEP(top_prop['%s_topology_proposal' % phase],
                                                       top_prop['%s_old_positions' % phase],
                                                       top_prop['%s_new_positions' % phase],
                                                       n_equil_steps=n_equilibrium_steps_per_iteration,
                                                       ncmc_nsteps=n_steps_ncmc_protocol,
                                                       nsteps_per_iteration=n_steps_per_move_application,
                                                       temperature=temperature,
                                                       trajectory_directory=trajectory_directory,
                                                       trajectory_prefix='-'.join([trajectory_prefix, '%s' % phase]),
                                                       atom_selection=atom_selection,
                                                       scheduler_address=scheduler_address, eq_splitting_string=eq_splitting,
                                                       neq_splitting_string=neq_splitting,
                                                       timestep=timestep,
                                                       measure_shadow_work=measure_shadow_work)

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
                                               neglected_old_angle_terms = top_prop[f"{phase}_reverse_neglected_angles"])

           # Define necessary vars to check energy bookkeeping
            _top_prop = top_prop['%s_topology_proposal' % phase]
            _htf = htf[phase]
            _forward_added_valence_energy = top_prop['%s_added_valence_energy' % phase]
            _reverse_subtracted_valence_energy = top_prop['%s_subtracted_valence_energy' % phase]

            zero_state_error, one_state_error = validate_endstate_energies(_top_prop, _htf, _forward_added_valence_energy, _reverse_subtracted_valence_energy, beta = 1.0/(kB*temperature))
            _logger.info(f"\t\terror in zero state: {zero_state_error}")
            _logger.info(f"\t\terror in one state: {one_state_error}")

            # generating lambda protocol
            lambda_protocol = LambdaProtocol(type=setup_options['protocol-type'],
                                             functions=setup_options['lambda_protocol'])

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
                hss[phase] = HybridSAMSSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=timestep,
                                                                                             collision_rate=5.0 / unit.picosecond,
                                                                                             n_steps=n_steps_per_move_application,
                                                                                             reassign_velocities=False,
                                                                                             n_restart_attempts=20,
                                                                                             splitting="V R R R O R R R V"),
                                               hybrid_factory=htf[phase], online_analysis_interval=setup_options['offline-freq'],
                                               online_analysis_minimum_iterations=10,flatness_criteria=setup_options['flatness-criteria'],
                                               gamma0=setup_options['gamma0'],lambda_protocol=lambda_protocol)
                hss[phase].setup(n_states=n_states, temperature=temperature,storage_file=reporter)
            elif setup_options['fe_type'] == 'repex':
                hss[phase] = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=timestep,
                                                                                             collision_rate=5.0 / unit.picosecond,
                                                                                             n_steps=n_steps_per_move_application,
                                                                                             reassign_velocities=False,
                                                                                             n_restart_attempts=20,
                                                                                             splitting="V R R R O R R R V"),
                                                                                             hybrid_factory=htf[phase],online_analysis_interval=setup_options['offline-freq'],lambda_protocol=lambda_protocol)
                hss[phase].setup(n_states=n_states, temperature=temperature,storage_file=reporter)

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

    _logger.info(f"Running setup...")
    setup_dict = run_setup(setup_options)

    trajectory_prefix = setup_options['trajectory_prefix']
    trajectory_directory = setup_options['trajectory_directory']
    #write out topology proposals
    _logger.info(f"Writing topology proposal {trajectory_prefix}topology_proposals.npy to {trajectory_directory}...")
    np.save(os.path.join(setup_options['trajectory_directory'], trajectory_prefix+"topology_proposals.npy"),
            setup_dict['topology_proposals'])


    n_equilibration_iterations = setup_options['n_equilibration_iterations']
    _logger.info(f"Equilibration iterations: {n_equilibration_iterations}.")
    if setup_options['fe_type'] == 'nonequilibrium':
        _logger.info(f"Detecting nonequilibrium as fe_type...(this is neither logged nor debugged properly).")
        n_cycles = setup_options['n_cycles']
        n_iterations_per_cycle = setup_options['n_iterations_per_cycle']
        total_iterations = n_cycles*n_iterations_per_cycle

        ne_fep = setup_dict['ne_fep']
        for phase in setup_options['phases']:
            ne_fep_run = ne_fep[phase]
            hybrid_factory = ne_fep_run._factory
            np.save(os.path.join(trajectory_directory, "%s_%s_hybrid_factory.npy" % (trajectory_prefix, phase)),
                    hybrid_factory)

            print("equilibrating")
            ne_fep_run.equilibrate(n_iterations=n_equilibration_iterations)

            print("equilibration complete")
            for i in range(n_cycles):
                ne_fep_run.run(n_iterations=n_iterations_per_cycle)
                print(i)

            print("calculation complete")
            df, ddf = ne_fep_run.current_free_energy_estimate

            print("The free energy estimate is %f +/- %f" % (df, ddf))

            endpoint_file_prefix = os.path.join(trajectory_directory, "%s_%s_endpoint{endpoint_idx}.npy" %
                                                (trajectory_prefix, phase))

            endpoint_work_paths = [endpoint_file_prefix.format(endpoint_idx=lambda_state) for lambda_state in [0, 1]]

            # try to write out the ne_fep object as a pickle
            try:
                pickle_outfile = open(os.path.join(trajectory_directory, "%s_%s_ne_fep.pkl" %
                                                   (trajectory_prefix, phase)), 'wb')
            except Exception as e:
                pass

            try:
                pickle.dump(ne_fep, pickle_outfile)
            except Exception as e:
                print(e)
                print("Unable to save run object as a pickle")
            finally:
                pickle_outfile.close()

            # save the endpoint perturbations
            for lambda_state, reduced_potential_difference in ne_fep._reduced_potential_differences.items():
                np.save(endpoint_work_paths[lambda_state], np.array(reduced_potential_difference))

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
        logZ = dict()
        free_energies = dict()
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

import yaml
import numpy as np
import pickle
import os
import sys
import logging
import simtk.unit as unit

from perses.samplers.samplers import HybridSAMSSampler
from perses.annihilation.new_relative import HybridTopologyFactory
from perses.app.relative_setup import NonequilibriumSwitchingFEP, RelativeFEPSetup

from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter, sams, replicaexchange


logging.basicConfig(level=logging.DEBUG)

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

    if 'phases' not in setup_options:
        setup_options['phases'] = ['complex','solvent']
        print('No phases provided - running complex and solvent as default')

    trajectory_directory = setup_options['trajectory_directory']
    assert (os.path.exists(trajectory_directory) == False), 'Output directory already exists. Refusing to overwrite'
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
    phases = setup_options['phases']

    if len(phases) == 0:
        phases = ['complex','solvent']
        print('No phases defined. Running complex and solvent.')

    if len(phases) != 2:
        print("Relative free energy simulations generally require two phases, either solvent and "
              "complex for binding free energies or vacuum and solvent for solvation free energies")
        print(f"{len(phases)} phases have been provided.")
        print("Continuing anyway...")

    known_phases = ['complex','solvent','vacuum']
    for phase in phases:
        assert (phase in known_phases), f"Unknown phase, {phase} provided. run_setup() can be used with {known_phases}"

    if 'complex' in phases:
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

    # get the indices of ligands out of the file:
    old_ligand_index = setup_options['old_ligand_index']
    new_ligand_index = setup_options['new_ligand_index']

    forcefield_files = setup_options['forcefield_files']

    if "timestep" in setup_options:
        timestep = setup_options['timestep'] * unit.femtoseconds
    else:
        timestep = 1.0 * unit.femtoseconds

    if "neq_splitting" in setup_options:
        neq_splitting = setup_options['neq_splitting']

        try:
            eq_splitting = setup_options['eq_splitting']
        except KeyError as e:
            print("If you specify a nonequilibrium splitting string, you must also specify an equilibrium one.")
            raise e

    else:
        eq_splitting = "V R O R V"
        neq_splitting = "V R O H R V"

    if "measure_shadow_work" in setup_options:
        measure_shadow_work = setup_options['measure_shadow_work']
    else:
        measure_shadow_work = False

    pressure = setup_options['pressure'] * unit.atmosphere
    temperature = setup_options['temperature'] * unit.kelvin
    solvent_padding_angstroms = setup_options['solvent_padding'] * unit.angstrom

    setup_pickle_file = setup_options['save_setup_pickle_as']
    trajectory_directory = setup_options['trajectory_directory']
    try:
        atom_map_file = setup_options['atom_map']
        with open(atom_map_file, 'r') as f:
            atom_map = {int(x.split()[0]): int(x.split()[1]) for x in f.readlines()}
    except Exception:
        atom_map=None

    if 'topology_proposal' not in setup_options:
        fe_setup = RelativeFEPSetup(ligand_file, old_ligand_index, new_ligand_index, forcefield_files,phases=phases,
                                          protein_pdb_filename=protein_pdb_filename,
                                          receptor_mol2_filename=receptor_mol2, pressure=pressure,
                                          temperature=temperature, solvent_padding=solvent_padding_angstroms,
                                          atom_map=atom_map)

        pickle_outfile = open(os.path.join(os.getcwd(), trajectory_directory, setup_pickle_file), 'wb')

        try:
            pickle.dump(fe_setup, pickle_outfile)
        except Exception as e:
            print(e)
            print("Unable to save setup object as a pickle")
        finally:
            pickle_outfile.close()

        print("Setup object has been created.")

        top_prop = dict()
        if 'complex' in phases:
            top_prop['complex_topology_proposal'] = fe_setup.complex_topology_proposal
            top_prop['complex_old_positions'] = fe_setup.complex_old_positions
            top_prop['complex_new_positions'] = fe_setup.complex_new_positions
        if 'solvent' in phases:
            top_prop['solvent_topology_proposal'] = fe_setup.solvent_topology_proposal
            top_prop['solvent_old_positions'] = fe_setup.solvent_old_positions
            top_prop['solvent_new_positions'] = fe_setup.solvent_new_positions
        if 'vacuum' in phases:
            top_prop['vacuum_topology_proposal'] = fe_setup.vacuum_topology_proposal
            top_prop['vacuum_old_positions'] = fe_setup.vacuum_old_positions
            top_prop['vacuum_new_positions'] = fe_setup.vacuum_new_positions

    else:
        top_prop = np.load(setup_options['topology_proposal']).item()

    n_steps_per_move_application = setup_options['n_steps_per_move_application']
    trajectory_directory = setup_options['trajectory_directory']
    trajectory_prefix = setup_options['trajectory_prefix']

    if 'atom_selection' in setup_options:
        atom_selection = setup_options['atom_selection']
    else:
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
        n_states = setup_options['n_states']
        checkpoint_interval = setup_options['checkpoint_interval']
        htf = dict()
        hss = dict()
        for phase in phases:
            print(f"HERE {phase}")
            #TODO write a SAMSFEP class that mirrors NonequilibriumSwitchingFEP
            htf[phase] = HybridTopologyFactory(top_prop['%s_topology_proposal' % phase],
                                               top_prop['%s_old_positions' % phase],
                                               top_prop['%s_new_positions' % phase])

            if atom_selection:
                selection_indices = htf[phase].hybrid_topology.select(atom_selection)
            else:
                selection_indices = None

            storage_name = str(trajectory_directory)+'/'+str(trajectory_prefix)+'-'+str(phase)+'.nc'
            print(f'storage_name {storage_name}')
            print(f'selection_indices {selection_indices}')
            print(f'checkpoint interval {checkpoint_interval}')
            reporter = MultiStateReporter(storage_name, analysis_particle_indices=selection_indices,
                                          checkpoint_interval=checkpoint_interval)

            #TODO expose more of these options in input
            hss[phase] = HybridSAMSSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep=timestep,
                                                                                         collision_rate=5.0 / unit.picosecond,
                                                                                         n_steps=n_steps_per_move_application,
                                                                                         reassign_velocities=False,
                                                                                         n_restart_attempts=6,
                                                                                         splitting="V R R R O R R R V"),
                                           hybrid_factory=htf[phase], online_analysis_interval=10,
                                           online_analysis_target_error=0.2, online_analysis_minimum_iterations=10)
            hss[phase].setup(n_states=n_states, temperature=temperature,storage_file=reporter)

        return {'topology_proposals': top_prop, 'hybrid_topology_factories': htf, 'hybrid_sams_samplers': hss}

if __name__ == "__main__":
    try:
       yaml_filename = sys.argv[1]
    except IndexError as e:
        print("You need to specify the setup yaml file as an argument to the script.")

    setup_options = getSetupOptions(yaml_filename)
    setup_dict = run_setup(setup_options)

    trajectory_prefix = setup_options['trajectory_prefix']
    trajectory_directory = setup_options['trajectory_directory']
    #write out topology proposals
    np.save(os.path.join(setup_options['trajectory_directory'], trajectory_prefix+"topology_proposals.npy"),
            setup_dict['topology_proposals'])


    n_equilibration_iterations = setup_options['n_equilibration_iterations']
    if setup_options['fe_type'] == 'nonequilibrium':
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

    else:
        np.save(os.path.join(trajectory_directory, trajectory_prefix + "hybrid_factory.npy"),
                setup_dict['hybrid_topology_factories'])

        hss = setup_dict['hybrid_sams_samplers']
        logZ = dict()
        free_energies = dict()
        for phase in setup_options['phases']:
            print(f'Running {phase} phase')
            hss_run = hss[phase]
            hss_run.minimize()
            hss_run.equilibrate(n_equilibration_iterations)
            hss_run.extend(setup_options['n_cycles'])
            logZ[phase] = hss_run._logZ[-1] - hss_run._logZ[0]
            free_energies[phase] = hss_run._last_mbar_f_k[-1] - hss_run._last_mbar_f_k[0]
            print(f"Finished phase {phase}")
        for phase in free_energies:
            print(f"{phase} phase has a free energy of {free_energies[phase]}")




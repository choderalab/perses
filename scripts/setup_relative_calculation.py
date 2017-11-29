import openeye.oechem as oechem
import sys
import progressbar
import yaml
from perses.rjmc import topology_proposal, geometry
from perses.dispersed import relative_setup
from perses.tests import utils
import simtk.unit as unit
from openmmtools.constants import kB
import simtk.openmm.app as app
from openmoltools import forcefield_generators
import copy
import numpy as np
import pickle
import progressbar

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
    #We'll need the protein PDB file (without missing atoms)
    protein_pdb_filename = setup_options['protein_pdb']

    #And a ligand file containing the pair of ligands between which we will transform
    ligand_file = setup_options['ligand_file']

    #get the indices of ligands out of the file:
    old_ligand_index = setup_options['old_ligand_index']
    new_ligand_index = setup_options['new_ligand_index']

    forcefield_files = setup_options['forcefield_files']

    #get the simulation parameters
    pressure = setup_options['pressure'] * unit.atmosphere
    temperature = setup_options['temperature'] * unit.kelvin
    solvent_padding_angstroms = setup_options['solvent_padding'] * unit.angstrom

    setup_pickle_file = setup_options['save_setup_pickle_as']

    fe_setup = relative_setup.NonequilibriumFEPSetup(protein_pdb_filename, ligand_file, old_ligand_index, new_ligand_index, forcefield_files, pressure=pressure, temperature=temperature, solvent_padding=solvent_padding_angstroms)

    pickle_outfile = open(setup_pickle_file, 'wb')

    try:
        pickle.dump(fe_setup, pickle_outfile)
    except Exception as e:
        print(e)
        print("Unable to save setup object as a pickle")
    finally:
        pickle_outfile.close()

    print("Setup object has been created.")

    phase = setup_options['phase']

    if phase == "complex":
        topology_proposal = fe_setup.complex_topology_proposal
        old_positions = fe_setup.complex_old_positions
        new_positions = fe_setup.complex_old_positions
    elif phase == "solvent":
        topology_proposal = fe_setup.solvent_topology_proposal
        old_positions = fe_setup.solvent_old_positions
        new_positions = fe_setup.solvent_new_positions
    else:
        raise ValueError("Phase must be either complex or solvent.")

    forward_functions = setup_options['forward_functions']

    n_equilibrium_steps_per_iteration = setup_options['n_equilibrium_steps_per_iteration']
    n_steps_ncmc_protocol = setup_options['n_steps_ncmc_protocol']
    n_steps_per_move_application = setup_options['n_steps_per_move_application']

    trajectory_directory = setup_options['trajectory_directory']
    trajectory_prefix = setup_options['trajectory_prefix']
    atom_selection = setup_options['atom_selection']

    scheduler_address = setup_options['scheduler_address']

    ne_fep = relative_setup.NonequilibriumSwitchingFEP(topology_proposal, old_positions, new_positions,
                                                       forward_functions=forward_functions,
                                                       n_equil_steps=n_equilibrium_steps_per_iteration,
                                                       ncmc_nsteps=n_steps_ncmc_protocol,
                                                       nsteps_per_iteration=n_steps_per_move_application,
                                                       temperature=temperature,
                                                       trajectory_directory=trajectory_directory,
                                                       trajectory_prefix=trajectory_prefix,
                                                       atom_selection=atom_selection,
                                                       scheduler_address=scheduler_address)

    print("Nonequilibrium switching driver class constructed")

    return fe_setup, ne_fep

def get_topology_proposals(fe_setup):
    """
    Get a dictionary of the various TopologyProposal objects in a NonequilibriumFEPSetup object

    Parameters
    ----------
    fe_setup : NonequilibriumSwitchingFEP

    Returns
    -------
    topology_proposals: dict
        dictionary of topology_proposal_name: topology_proposal, as well as initial positions
    """
    topology_proposals = {}

    topology_proposals['solvent_topology_proposal'] = fe_setup.solvent_topology_proposal
    topology_proposals['complex_topology_proposal'] = fe_setup.complex_topology_proposal

    topology_proposals['solvent_old_positions'] = fe_setup.solvent_old_positions
    topology_proposals['solvent_new_positions'] = fe_setup.solvent_new_positions

    topology_proposals['complex_old_positions'] = fe_setup.complex_old_positions
    topology_proposals['complex_new_positions'] = fe_setup.complex_new_positions

    return topology_proposals

if __name__=="__main__":
    import os
    yaml_filename = "basic_setup.yaml"
    yaml_file = open(yaml_filename, 'r')
    setup_options = yaml.load(yaml_file)
    yaml_file.close()

    fe_setup, ne_fep = run_setup(setup_options)
    print("setup complete")

    n_cycles = setup_options['n_cycles']
    n_iterations_per_cycle = setup_options['n_iterations_per_cycle']

    total_iterations = n_cycles*n_iterations_per_cycle

    trajectory_directory = setup_options['trajectory_directory']
    trajectory_prefix = setup_options['trajectory_prefix']

    #write out topology proposals
    topology_proposals_setup = get_topology_proposals(fe_setup)
    np.save(os.path.join(trajectory_directory, trajectory_prefix+"topology_proposals.npy"), topology_proposals_setup)

    #write out hybrid factory
    hybrid_factory = ne_fep._factory
    np.save(os.path.join(trajectory_directory, trajectory_prefix+"hybrid_factory.npy"), hybrid_factory)

    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=total_iterations)
    for i in range(n_cycles):
        ne_fep.run(n_iterations=n_iterations_per_cycle)
        bar.update((i+1)*n_iterations_per_cycle)

    df, ddf = ne_fep.current_free_energy_estimate

    print("The free energy estimate is %f +/- %f" % (df, ddf))

    endpoint_file_prefix = os.path.join(trajectory_directory, trajectory_prefix + "endpoint{endpoint_idx}.npy")

    endpoint_work_paths = [endpoint_file_prefix.format(endpoint_idx=lambda_state) for lambda_state in [0, 1]]

    #save the endpoint perturbations
    for lambda_state, reduced_potential_diference in ne_fep._reduced_potential_differences:
        np.save(endpoint_work_paths[lambda_state], np.array(reduced_potential_diference))

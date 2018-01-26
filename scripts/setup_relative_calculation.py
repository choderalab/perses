import yaml
from perses.dispersed import relative_setup
import numpy as np
import pickle
import progressbar
import os
import sys

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
    try:
       yaml_filename = sys.argv[1]
    except IndexError as e:
        print("You need to specify the setup yaml file as an argument to the script.")
        raise e

    yaml_file = open(yaml_filename, 'r')
    setup_options = yaml.load(yaml_file)
    yaml_file.close()

    fe_setup, ne_fep = relative_setup.run_setup(setup_options)
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

    print("equilibrating")
    ne_fep.equilibrate(n_iterations=100)

    print("equilibration complete")
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=total_iterations)
    bar.update(0)
    for i in range(n_cycles):
        ne_fep.run(n_iterations=n_iterations_per_cycle)
        print(i)
        #bar.update((i+1)*n_iterations_per_cycle)

    print("calculation complete")
    df, ddf = ne_fep.current_free_energy_estimate

    print("The free energy estimate is %f +/- %f" % (df, ddf))

    endpoint_file_prefix = os.path.join(trajectory_directory, trajectory_prefix + "endpoint{endpoint_idx}.npy")

    endpoint_work_paths = [endpoint_file_prefix.format(endpoint_idx=lambda_state) for lambda_state in [0, 1]]

    #try to write out the ne_fep object as a pickle
    try:
        pickle_outfile = open(os.path.join(trajectory_directory, trajectory_prefix+ "ne_fep.pkl"), 'wb')
    except Exception as e:
        pass

    try:
        pickle.dump(ne_fep, pickle_outfile)
    except Exception as e:
        print(e)
        print("Unable to save run object as a pickle")
    finally:
        pickle_outfile.close()

    #save the endpoint perturbations
    for lambda_state, reduced_potential_diference in ne_fep._reduced_potential_differences.items():
        np.save(endpoint_work_paths[lambda_state], np.array(reduced_potential_diference))
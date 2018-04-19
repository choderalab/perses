import yaml
from perses.dispersed import relative_setup
import numpy as np
import pickle
import progressbar
import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    try:
       yaml_filename = sys.argv[1]
    except IndexError as e:
        print("You need to specify the setup yaml file as an argument to the script.")
        raise e

    yaml_file = open(yaml_filename, 'r')
    setup_options = yaml.load(yaml_file)
    yaml_file.close()

    setup_dict = relative_setup.run_setup(setup_options)
    print("setup complete")

    trajectory_directory = setup_options['trajectory_directory']

    if not os.path.exists(trajectory_directory):
        os.makedirs(trajectory_directory)

    trajectory_prefix = setup_options['trajectory_prefix']
    #write out topology proposals
    np.save(os.path.join(trajectory_directory, trajectory_prefix+"topology_proposals.npy"),
            setup_dict['topology_proposals'])

    if setup_options['fe_type'] == 'nonequilibrium':
        n_cycles = setup_options['n_cycles']
        n_iterations_per_cycle = setup_options['n_iterations_per_cycle']
        total_iterations = n_cycles*n_iterations_per_cycle

        ne_fep = setup_dict['ne_fep']
        for phase in ['complex', 'solvent']:
            ne_fep_run = ne_fep[phase]
            hybrid_factory = ne_fep_run._factory
            np.save(os.path.join(trajectory_directory, "%s_%s_hybrid_factory.npy" % (trajectory_prefix, phase)),
                    hybrid_factory)

            print("equilibrating")
            ne_fep_run.equilibrate(n_iterations=100)

            print("equilibration complete")
            bar = progressbar.ProgressBar(redirect_stdout=True, max_value=total_iterations)
            bar.update(0)
            for i in range(n_cycles):
                ne_fep_run.run(n_iterations=n_iterations_per_cycle)
                print(i)
                # bar.update((i+1)*n_iterations_per_cycle)

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
        for phase in ['complex', 'solvent']:
            hss_run = hss[phase]
            hss_run.minimize()
            hss_run.equilibrate(10)
            hss_run.extend(1000)
            free_energies = dict()
            free_energies[phase] = hss_run._logZ[-1] - hss_run._logZ[0]
            print("Finished phase %s with dG estimated as %.4f kT" % (phase, free_energies[phase]))

        print("Total ddG is estimated as %.4f kT" % (free_energies['complex'] - free_energies['solvent']))
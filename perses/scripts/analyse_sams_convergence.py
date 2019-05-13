import matplotlib.pyplot as plt
import os, sys
from glob import glob
from perses.scripts import utils

if __name__ == '__main__':
    directory = sys.argv[1]
    files = sorted(glob(os.path.join(os.getcwd(), directory, '*.nc')))
    files = [x for x in files if 'checkpoint' not in x]
    f, axarr = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(16, 8))
    for i, filename in enumerate(files):
        phase = filename.split('-')[1].rstrip('.nc')
        ncfile = utils.open_netcdf(filename)
        logZ = ncfile.groups['online_analysis'].variables['logZ_history']
        n_iterations, n_states = logZ.shape
        axarr[i, 0].plot(logZ, '.')
        axarr[i, 0].set_xlabel('iteration')
        axarr[i, 0].set_ylabel('logZ / kT')
        axarr[i, 0].set_title('%s_%s' % (phase, directory))
        states = ncfile.variables['states']
        n_iterations, n_replicas = states.shape
        axarr[i, 1].plot(states, '.')
        axarr[i, 1].set_xlabel('iteration')
        axarr[i, 1].set_ylabel('thermodynamic state')
        axarr[i, 1].axis([0, n_iterations, 0, n_states])
        gamma = ncfile.groups['online_analysis'].variables['gamma_history']
        axarr[i, 2].plot(gamma, '.')
        axarr[i, 2].set_xlabel('iteration')
        axarr[i, 2].set_ylabel('gamma')

    f.tight_layout()
    f.savefig('%s.png' % directory, dpi=300)

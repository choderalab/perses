import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys
from netCDF4 import Dataset



if __name__ == '__main__':
    filename, plot_name = sys.argv[1:]
    ncfile = Dataset(filename, 'r')
    logZ = ncfile.groups['online_analysis'].variables['logZ_history']
    n_iterations, n_states = logZ.shape
    print(n_iterations, n_states)
    f, axarr = plt.subplots(4, sharex=False, sharey=False, figsize=(8, 20))
    axarr[0].plot(logZ, '.')
    axarr[0].set_xlabel('iteration')
    axarr[0].set_ylabel('logZ / kT')
    axarr[0].set_title(plot_name)
    axarr[1].plot(np.transpose(np.transpose(logZ[:-2]) - logZ[:-2, 0]), '.')
    axarr[1].set_xlabel('iteration')
    axarr[1].set_ylabel('logZ / kT')
    states = ncfile.variables['states']
    n_iterations, n_replicas = states.shape
    axarr[2].plot(states, '.')
    axarr[2].set_xlabel('iteration')
    axarr[2].set_ylabel('thermodynamic state')
    axarr[2].axis([0, n_iterations, 0, n_states])
    gamma = ncfile.groups['online_analysis'].variables['gamma_history']
    axarr[3].plot(gamma, '.')
    axarr[3].set_xlabel('iteration')
    axarr[3].set_ylabel('gamma')
    f.tight_layout()
    f.savefig('%s.pdf' % plot_name)
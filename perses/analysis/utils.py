import numpy as np
import logging

_logger = logging.getLogger("analysis/utils")

def open_netcdf(filename):
    from netCDF4 import Dataset
    import os

    # checking that the file exists
    exists = os.path.isfile(filename)

    if exists:
        ncfile = Dataset(filename, 'r')

    return ncfile

def get_offline_free_energies(filename,offline_frequency=10):

    ncfile = open_netcdf(filename)

    all_fe = ncfile.groups['online_analysis'].variables['free_energy_history']

    offline_fe = all_fe[0::offline_frequency]

    fe = np.zeros(len(offline_fe))
    err = np.zeros(len(offline_fe))
    for i, s in enumerate(offline_fe):
        fe[i] = s[0]
        err[i] = s[1]

    return fe, err

def get_t0(filename):

    ncfile = open_netcdf(filename)
    t0 = np.asarray(ncfile.groups['online_analysis'].variables['t0'])[0]
    # todo need a huge error warning if t0 = 0 as that means sams hasn't moved from first to second stage
    if t0 == 0:
        _logger.warning(f"t0 for this file is zero, which means that stage 2 was not reached within the simulation")
    return t0


def plot_replica_mixing(ncfile_name, title='',filename='replicas.png'):
    """
    Plots the path of each replica through the states, with marginal distribution shown

    Parameters
    ----------
    ncfile_name : str
        path to nc file to analyse
    title : str, default=''
        Title to add to plot
    filename : str, default='replicas.png'
        path where to save the output plot

    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    ncfile = open_netcdf(ncfile_name)
    
    n_iter, n_states = ncfile.variables['states'].shape
    cmaps = plt.cm.get_cmap('gist_rainbow')
    colours = [cmaps(i) for i in np.linspace(0.,1.,n_states)]
    fig, axes = plt.subplots(nrows=n_states, ncols=2, sharex='col',sharey=True,figsize=(15,2*n_states), squeeze=True, gridspec_kw={'width_ratios': [5, 1]})

    for rep in range(n_states):
        ax = axes[rep,0]
        y = ncfile.variables['states'][:,rep]
        ax.plot(y,marker='.', linewidth=0, markersize=2,color=colours[rep])
        ax.set_xlim(-1,n_iter+1)
        hist_plot = axes[rep,1]
        hist_plot.hist(y, bins=n_states, orientation='horizontal',histtype='step',color=colours[rep],linewidth=3)

        ax.set_ylabel('State')
        hist_plot.yaxis.set_label_position("right")
        hist_plot.set_ylabel(f'Replica {rep}',rotation=270, labelpad=10)  

        # just plotting for the bottom plot
        if rep == n_states-1:
            ax.set_xlabel('Iteration')
            hist_plot.set_xlabel('State count')

    fig.tight_layout()
    plt.title(title)
    plt.savefig(filename)
    

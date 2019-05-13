import numpy as np

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

def plot_relative(filenameA, filenameB,offline_frequency=10):
    import matplotlib.pyplot as plt

    feA , errA = get_offline_free_energies(filenameA)
    feB , errB = get_offline_free_energies(filenameB)

    dG = feA - feB
    ddG = (errA**2 + errB**2)**0.5

    # plotting
    plt.plot(fe, linewidth=0.75)
    plt.fill_between(range(len(fe)), fe - err, fe + err, alpha=0.2)
    plt.xlabel('iteration')
    plt.ylabel('free energy / units??')




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

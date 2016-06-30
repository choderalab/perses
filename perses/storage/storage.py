"""
Storage layer for perses automated molecular design.

TODO
----
* Add write_sampler_state(modname, sampler_state, iteration)
* Generalize write_quantity to handle units

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

import os, os.path
import sys, math
import numpy as np
import copy
import time
import netCDF4 as netcdf

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# STORAGE
################################################################################

class NetCDFStorage(object):

    def __init__(self, filename, mode='a'):
        self._filename = filename

        # Open NetCDF file here...
        self._ncfile = netcdf.Dataset(self._filename, mode=mode)

    def write_configuration(self, modname, varname, positions, topology, iteration=None, frame=None, nframes=None):
        """Write a configuration (or one of a sequence of configurations) to be stored as a native NetCDF array

        Parameters
        ---
        modname : str
            The name of the module in the code writing the variable
        varname : str
            The variable name to be stored
        positions : simtk.unit.Quantity of size [natoms,3] with units compatible with angstroms
            The positions to be written
        topology : simtk.openmm.Topology object
            The corresponding Topology object
        iteration : int, optional, default=None
            The local iteration for the module, or `None` if this is a singleton
        frame : int, optional, default=None
            If these coordinates are part of multiple frames in a sequence, the frame number
        nframes : int, optional, default=None
            If these coordinates are part of multiple frames in a sequence, the total number of frames in the sequence

        """
        pass

    def write_object(self, modname, varname, object, iteration=None):
        """Serialize a Python object

        Parameters
        ---
        modname : str
            The name of the module in the code writing the variable
        varname : str
            The variable name to be stored
        object : object
            The object to be serialized
        iteration : int, optional, default=None
            The local iteration for the module, or `None` if this is a singleton
        """
        pass

    def write_quantity(self, modname, varname, value, iteration=None):
        """Write a floating-point number

        Parameters
        ---
        modname : str
            The name of the module in the code writing the variable
        varname : str
            The variable name to be stored
        value : float
            The floating-point value to be written
        iteration : int, optional, default=None
            The local iteration for the module, or `None` if this is a singleton
        """
        pass

    def write_array(self, modname, varname, array, iteration=None):
        """Write a numpy array as a native NetCDF array

        Parameters
        ---
        modname : str
            The name of the module in the code writing the variable
        varname : str
            The variable name to be stored
        array : numpy.array of arbitrary dimension
            The numpy array to be written
        iteration : int, optional, default=None
            The local iteration for the module, or `None` if this is a singleton
        """
        pass

    def sync(self):
        """Flush write buffer.
        """
        self._ncfile.sync()

    def close(self):
        """Close the storage layer.
        """
        self._ncfile.close()

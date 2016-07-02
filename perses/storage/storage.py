"""
Storage layer for perses automated molecular design.

TODO
----
* Add write_sampler_state(modname, sampler_state, iteration)
* Generalize write_quantity to handle units
* Add data access routines for reading to isolate low-level storage layer

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
import pickle

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# STORAGE
################################################################################

class NetCDFStorage(object):
    """NetCDF storage layer.
    """

    def __init__(self, filename, mode='w'):
        """Create NetCDF storage layer, creating or appending to an existing file.

        Parameters
        ----------
        filename : str
           Name of storage file to bind to.
        mode : str, optional, default='w'
           File open mode, 'w' for (over)write, 'a' for append.

        """
        self._filename = filename
        self._ncfile = netcdf.Dataset(self._filename, mode=mode)
        self._envname = None
        self._modname = None

        # Create standard dimensions.
        if 'iterations' not in self._ncfile.dimensions:
            self._ncfile.createDimension('iterations', size=None)

    def _find_group(self):
        """Retrieve the specified group, creating it if it does not exist.

        """
        groupname = '/'
        if self._envname is not None:
            groupname += self._envname + '/'
        if self._modname is not None:
            groupname += self._modname + '/'
        ncgrp = self._ncfile.createGroup(groupname)
        return ncgrp

    def sync(self):
        """Flush write buffer.
        """
        self._ncfile.sync()

    def close(self):
        """Close the storage layer.
        """
        self._ncfile.close()

    def write_configuration(self, varname, positions, topology, iteration=None, frame=None, nframes=None):
        """Write a configuration (or one of a sequence of configurations) to be stored as a native NetCDF array

        Parameters
        ---
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

    def write_object(self, varname, obj, iteration=None):
        """Serialize a Python object

        Parameters
        ---
        varname : str
            The variable name to be stored
        obj : object
            The object to be serialized
        iteration : int, optional, default=None
            The local iteration for the module, or `None` if this is a singleton
        """
        ncgrp = self._find_group()

        if varname not in ncgrp.variables:
            if iteration is not None:
                ncgrp.createVariable(varname, str, dimensions=('iterations',), chunksizes=(1,))
            else:
                ncgrp.createVariable(varname, str, dimensions=(), chunksizes=(1,))

        value = pickle.dumps(obj)
        if iteration is not None:
            ncgrp.variables[varname][iteration] = value
        else:
            ncgrp.variables[varname] = value

    def write_quantity(self, varname, value, iteration=None):
        """Write a floating-point number

        Parameters
        ---
        varname : str
            The variable name to be stored
        value : float
            The floating-point value to be written
        iteration : int, optional, default=None
            The local iteration for the module, or `None` if this is a singleton
        """
        ncgrp = self._find_group()

        if varname not in ncgrp.variables:
            if iteration is not None:
                ncgrp.createVariable(varname, 'f8', dimensions=('iterations',), chunksizes=(1,))
            else:
                ncgrp.createVariable(varname, 'f8', dimensions=(), chunksizes=(1,))

        if iteration is not None:
            ncgrp.variables[varname][iteration] = value
        else:
            ncgrp.variables[varname] = value

    def write_array(self, varname, array, iteration=None):
        """Write a numpy array as a native NetCDF array

        Parameters
        ----------
        varname : str
            The variable name to be stored
        array : numpy.array of arbitrary dimension
            The numpy array to be written
        iteration : int, optional, default=None
            The local iteration for the module, or `None` if this is a singleton
        """
        ncgrp = self._find_group()

        def dimension_name(dimension_index):
            dimension_name = ''
            if self._envname: dimension_name += self._envname + '_'
            if self._modname: dimension_name += self._modname + '_'
            dimension_name += varname + '_' + str(dimension_index)
            return dimension_name

        if varname not in ncgrp.variables:
            # Create dimensions
            dimensions = list()
            if iteration is not None:
                dimensions.append('iterations')
            for (dimension_index, size) in enumerate(array.shape):
                ncdim = self._ncfile.createDimension(dimension_name(dimension_index), size)
                dimensions.append(dimension_name(dimension_index))
            dimensions = tuple(dimensions)

            # Create variables
            if iteration is not None:
                ncgrp.createVariable(varname, array.dtype, dimensions=dimensions, chunksizes=((1,) + array.shape))
            else:
                ncgrp.createVariable(varname, array.dtype, dimensions=dimensions, chunksizes=array.shape)

        # Check dimensions
        expected_shape = list()
        for (dimension_index, size) in enumerate(array.shape):
            expected_shape.append(self._ncfile.dimensions[dimension_name(dimension_index)].size)
        expected_shape = tuple(expected_shape)
        if expected_shape != array.shape:
            raise Exception("write_array called for /%s/%s/%s with different dimension (%s) than initially called (%s); dimension must stay constant." % (envname, modname, varname, str(array.shape), str(expected_shape)))

        if iteration is not None:
            ncgrp.variables[varname][iteration] = array
        else:
            ncgrp.variables[varname] = array

################################################################################
# BOUND STORAGE VIEWS THAT ENCAPSULATE ENVIRONMENT NAMES AND MODULE NAMES
################################################################################

class NetCDFStorageView(NetCDFStorage):
    """NetCDF storage view with bound environment and module names.
    """
    def __init__(self, storage, envname=None, modname=None):
        """Initialize a view of the storage with a specific environment and module name.

        Parameters
        ----------
        envname : str, optional, default=None
            Set the name of the environment this module is attached to.
        modname : str, optional, default=None
            Set the name of the module in the code writing the variable
        """
        self._filename = storage._filename
        self._ncfile = storage._ncfile
        self._envname = storage._envname
        self._modname = storage._modname

        if envname: self._envname = envname
        if modname: self._modname = modname

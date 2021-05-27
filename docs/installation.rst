.. _installation:

Installation
************

Installing via `conda`
======================

The simplest way to install ``perses`` is via the `conda <http://www.continuum.io/blog/conda>`_  package manager.
Packages are provided on the `conda-forge Anaconda Cloud channel <http://anaconda.org/conda-forge>`_ for Linux, OS X, and Win platforms.
The `perses Anaconda Cloud page <https://anaconda.org/conda-forge/perses>`_ has useful instructions and `download statistics <https://anaconda.org/conda-forge/perses/files>`_.

If you are using the `anaconda <https://www.continuum.io/downloads/>`_ scientific Python distribution, you already have the ``conda`` package manager installed.
If not, the quickest way to get started is to install the `miniconda <http://conda.pydata.org/miniconda.html>`_ distribution, a lightweight minimal installation of Anaconda Python.

|

Release build
-------------

You can install the latest stable release build of perses via the ``conda`` package with

.. code-block:: none

   $ conda config --add channels conda-forge openeye
   $ conda install perses

This version is recommended for all users not actively developing new algorithms for alchemical free energy calculations.

.. note:: ``conda`` will automatically dependencies from binary packages automatically, including difficult-to-install packages such as OpenMM, numpy, and scipy. This is really the easiest way to get started.

|

Upgrading your installation
---------------------------

To update an earlier ``conda`` installation of perses to the latest release version, you can use ``conda update``:

.. code-block:: bash

   $ conda update perses

|

.. perses documentation master file, created by
   sphinx-quickstart on Sun May 14 17:09:48 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. caution::

    This is module is undergoing heavy development. None of the API calls are final.
    This software is provided without any guarantees of correctness, you will likely encounter bugs.

    If you are interested in this code, please wait for the official release to use it.
    In the mean time, to stay informed of development progress you are encouraged to:

    * Follow `this feed`_ for updates on releases.
    * Check out the `github repository`_ .

.. _this feed: https://github.com/choderalab/perses/releases.atom
.. _github repository: https://github.com/choderalab/perses

perses
======

A Python framework for automated small molecule free energy driven design.

``perses`` is a Python framework that uses `OpenMM <http://openmm.org>`_ for GPU-accelerated molecular design driven by alchemical free energy calculations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   changelog

Modules
-------

.. toctree::
  :maxdepth: 2

  annihilation
  bias
  rjmc
  samplers
  storage
  analysis

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

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

PERSES
======

A Python framework for automated small molecule free energy driven design.

``PERSES`` is a Python framework that uses `OpenMM <http://openmm.org>`_ for GPU-accelerated molecular design driven by alchemical free energy calculations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   changelog
   examples

Background
----------

Perses performs relative free energy calculations using a single topology method. Single topology methods are those where the two 'things' that are being compared are done so by generating a single object whose parameters are perturbed between a representation of thing A to thing B. Perses supports perturbations between small molecules (for relative binding or relative hydration free energy calculations) and protein residues (resistance mutations).

Setting up and running a perses calculations involves three main stages.

Determining the atom-mapping of ligand A onto ligand B to work out the 2D topology of the single-topology alchemical object. Herein, alchemical topologies, systems, geometries etc. will be referred to as *hybrid*. This is handled by a ``ProposalEngine``.

From this 2D hybrid topology, we then generate a 3D hybrid system. We use the input topology and coordinates for ligand A and the system and use RJMC to build in the atoms of ligand B. This is handled by a ``GeometryEngine``.

With the hybrid system and hybrid topology, it's possible to perform free energy calculations. Equilibrium methods such as REPEX and SAMS or non-equilibrium switching can be used. The method in which ligand A and ligand B are perturbed is handled by the ``LambdaProtocol``, and sampled using samplers such as ``HybridSAMSSampler`` and ``HybridRepexSampler``.

Modules
-------

.. toctree::
  :maxdepth: 2

  annihilation
  bias
  rjmc
  samplers
  dispersed
  storage
  analysis
  utils

API Reference
-------------

.. toctree::
   :maxdepth: 1

   api/generated

Developers
----------

* Patrick B. Grinaway
* Julie M. Behr
* Hannah E. Bruce Macdonald
* Dominic A. Rufa
* Jaime Rodríguez-Guerra
* Ivy Zhang
* Mike Henry
* Iván Pulido 
* John D. Chodera

Indices and tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

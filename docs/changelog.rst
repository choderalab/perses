.. _changelog:

***************
Release history
***************

This section lists features and improvements of note in each release.

The full release history can be viewed `at the GitHub perses releases page <https://github.com/choderalab/perses/releases>`_.

0.9.2 - Bugfix release
-----------------------
Bugfixes
^^^^^^^^
- (PR `#835 <https://github.com/choderalab/perses/pull/835>`_)
  Write out YAML file after all options are parsed and set. Saved as YAML original file name + date + time. Resolves
  `#817 <https://github.com/choderalab/perses/issues/817>`_.
- (PR `#840 <https://github.com/choderalab/perses/pull/840>`_)
  Minor improvements to point mutation executor. Make sure reverse geometry proposal is directly after forward proposal.
  Fixes formatting problem for complex positions.
- (PR `#841 <https://github.com/choderalab/perses/pull/841>`_)
  Minor improvements to PolymerProposalEngine.
- (PR `#844 <https://github.com/choderalab/perses/pull/844>`_)
  Minimal examples of amino acid (small molecule), protein-ligand and ligand mutations, with automated testing.
- (PR `#849 <https://github.com/choderalab/perses/pull/849>`_)
  Use an instance of ContextCache instead of the default global instance.
  More info at `#613 (comment) <https://github.com/choderalab/perses/issues/613#issuecomment-899746348>`_.

New features
^^^^^^^^
- (PR `#708 <https://github.com/choderalab/perses/pull/708>`_)
  Create visualization module for generating PyMOL movies.
- (PR `#834 <https://github.com/choderalab/perses/pull/834>`_)
  Enable protein mutation transformations involving nonstandard amino acids, specifically: ASH, GLH, HID, HIE, HIP, LYN.
- (PR `#838 <https://github.com/choderalab/perses/pull/838>`_)
  Official Docker image hosted on docker hub ``docker pull choderalab/perses:0.9.1``.
  Resolves `#832 <https://github.com/choderalab/perses/pull/832>`_.

0.9.1 - Bugfix release
-----------------------

Bugfixes
^^^^^^^^
- (PR `#830 <https://github.com/choderalab/perses/pull/830>`_)
  Added limited support for resuming simulations from the CLI. 
  Assumes simulations are only going to be resumed from the production step and not equilibration step.
  To extend the simulation, change ``n_cycles`` to a larger number and re-run the CLI tool.
  ``LOGLEVEL`` can now be set with an environmental variable when using the CLI tool.
- (PR `#821 <https://github.com/choderalab/perses/pull/821>`_)
  Added tests for the resume simulation functionality.
- (PR `#828 <https://github.com/choderalab/perses/pull/828>`_)
  Addresses (`issue #815 <https://github.com/choderalab/perses/issues/815>`_) by checking the potential energy of the proposed positions before generating the ``RepartitonedHybridTopologyFactory``.
- (PR `#809 <https://github.com/choderalab/perses/pull/809>`_) 
  The atom mapping facility was overhauled to address a bug in mapping rings (`#805 <https://github.com/choderalab/perses/issues/805>`_).
  Atom mapping is now handled via an ``AtomMapper`` factory that generates an ``AtomMapping`` class that localizes all relevant functionality.
- (PR `#824 <https://github.com/choderalab/perses/pull/824>`_)
  The default timestep is now 4 fs (was 1 fs) and the minimum openMM version is now 7.5.0
- (PR `#812 <https://github.com/choderalab/perses/pull/812>`_)
  Automatically set package version by ``git tag`` using versioneer
- (PR `#804 <https://github.com/choderalab/perses/pull/804>`_)
  Set the default temperature back to 300 K for ``relative_point_mutation_setup.py``.
- (PR `#796 <https://github.com/choderalab/perses/pull/796>`_)
  Removed defunct ``atom_map`` argument from FEP constructor.

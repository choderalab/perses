.. _changelog:

***************
Release history
***************

This section lists features and improvements of note in each release.

The full release history can be viewed `at the GitHub yank releases page <https://github.com/choderalab/perses/releases>`_.


0.9.1 - Bugfix release
-----------------------

Bugfixes
^^^^^^^^
- (PR `#830 <https://github.com/choderalab/perses/pull/830>`_)
  Added support for resuming simulations from the CLI
- (PR `#821 <https://github.com/choderalab/perses/pull/821>`_)
  Added tests for the resume simulation functionality.
- (PR `#828 <https://github.com/choderalab/perses/pull/828>`_)
  Addresses (`issue #815 <https://github.com/choderalab/perses/issues/815>`_) by checking the potential energy of the proposed positions before generating the ``RepartitonedHybridTopologyFactory``.
- (PR `#809 <https://github.com/choderalab/perses/pull/809>`_) 
  The atom mapping facility was overhauled to address a bug in mapping rings (`#805 <https://github.com/choderalab/perses/issues/805>`_).
  Atom mapping is now handled via an ``AtomMapper`` factory that generates an ``AtomMapping`` class that localizes all relevant functionality.

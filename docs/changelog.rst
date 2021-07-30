.. _changelog:

***************
Release history
***************

This section lists features and improvements of note in each release.

The full release history can be viewed `at the GitHub perses releases page <https://github.com/choderalab/perses/releases>`_.


0.9.1 - Bugfix release
-----------------------

Bugfixes
^^^^^^^^
- (PR `#809 <https://github.com/choderalab/perses/pull/809>`_) 
  The atom mapping facility was overhauled to address a bug in mapping rings (`#805 <https://github.com/choderalab/perses/issues/805>`_).
  Atom mapping is now handled via an ``AtomMapper`` factory that generates an ``AtomMapping`` class that localizes all relevant functionality.
- (PR `#824 <https://github.com/choderalab/perses/pull/824>`_)
  The default timestep is now 4 fs (was 1 fs) and the minimum openMM version is now 7.5.0
- (PR `#812 <https://github.com/choderalab/perses/pull/812>`_)
  Automatically set package version by ``git tag`` using versioneer

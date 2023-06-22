.. _changelog:

***************
Release history
***************

This section lists features and improvements of note in each release.

The full release history can be viewed `at the GitHub perses releases page <https://github.com/choderalab/perses/releases>`_.

0.10.2 - Release
----------------

Bugfix release.

Enhancements
^^^^^^^^^^^^

* Added upport for reading input files (ex yaml, sdf, pdbs) from AWS, GCP, and Azure. See the documentation for [cloudpathlib](https://cloudpathlib.drivendata.org/stable/authentication/) for how to setup authentication. Currently only reading the yaml from S3 is unit tested (ie `perses-cli --yaml s3://perses-testing/template_s3.yaml`), but other cloud providers and input files should work. Please report any issues on our issue tracker!  by @mikemhenry in https://github.com/choderalab/perses/pull/1073
* ``CUDA`` platform was hardcoded in geometry engine, generating performance issues by not clearing openmm contexts correctly. Fixed by defaulting to using the faster ``CPU`` platform (for the geometry engine) and explicitly deleting context variables after they are used. by @ijpulidos in https://github.com/choderalab/perses/pull/1091
* Set aromatic draw style to `OEAromaticStyle_Circle` in atom mapper rendering by @mikemhenry in https://github.com/choderalab/perses/pull/1103
* Ligands with atoms changing constrained status were not being handled by mapping proposal. Atoms in bonds that change constrained/unconstrained to unconstrained/constrained during the transformation are now _demapped_. by @ijpulidos in https://github.com/choderalab/perses/pull/1125
* Now that upstream ``openmmtools`` is storing velocities on checkpoint, the small molecule transformation pipeline does not reassign velocities on resume ny defualt. Instead, the velocities are read from the checkpoint file. by @ijpulidos in https://github.com/choderalab/perses/pull/1133
* CLI workflow for replica exchange now uses the faster ``LangevinMiddleIntegrator`` via the ``LangevinDynamicsMove``. Tests were updated to reflect the changes.  by @ijpulidos in https://github.com/choderalab/perses/pull/1138
* Add opencontainers image-spec to `Dockerfile` by @SauravMaheshkar in https://github.com/choderalab/perses/pull/1139
* Updated to support openff-toolkit 0.11, which included API-breaking changes. by @jchodera in https://github.com/choderalab/perses/pull/1128
* Make solute-only trajectory writing optional @mikemhenry in https://github.com/choderalab/perses/pull/1185
* Users can now specify solvent model for simulations using the ``solvent_model`` field in the input YAML file. by @ijpulidos in https://github.com/choderalab/perses/pull/1202

Documentation
^^^^^^^^^^^^^
* Document setting `ionic_strength` in `examples/new-cli/template.yaml` by @mikemhenry in https://github.com/choderalab/perses/pull/1104
* Add reproducible version of COVID Moonshot example anyone can run as an example by @jchodera in https://github.com/choderalab/perses/pull/1145
* Speed up RTD env generation by @mikemhenry in https://github.com/choderalab/perses/pull/1105
* Fix documentation string for `ProteinMutationExecutor`, using `False` for `reassign_velocities` parameter. by @ijpulidos in https://github.com/choderalab/perses/pull/1169
* Document how to control the log level in the CLI by @mikemhenry in https://github.com/choderalab/perses/pull/1198

Bug Fixes
^^^^^^^^^

* Fixes bug where if a `:` was in a key, we could not override the arguement in our perses-cli.  @mikemhenry in https://github.com/choderalab/perses/pull/1062
* Resolves #1157 objects serialized with `utils.data.serialize` now will be compressed with `bzip2` or `gzip` depending on file name (`.gz` and `.bz2`, respectively) by @mikemhenry in https://github.com/choderalab/perses/pull/1163
* Fixes for new openmmtools 0.23.0 by @mikemhenry in https://github.com/choderalab/perses/pull/1203

Testing/CI/Packaging 
^^^^^^^^^^^^^^^^^^^^
* add python 3.10 to CI by @mikemhenry in https://github.com/choderalab/perses/pull/1080
* skip broken tests by @mikemhenry in https://github.com/choderalab/perses/pull/1074
* Feat/fix ssl gpu error by @mikemhenry in https://github.com/choderalab/perses/pull/1095
* Previously the keyword argument `save_freq` in `validate_endstate_energies_md` was not functional and the value of `250` steps was hard coded. Now, `save_freq` works and has a default value of `250` steps. by @mikemhenry in https://github.com/choderalab/perses/pull/1101
* fix issue with test asset name by @mikemhenry in https://github.com/choderalab/perses/pull/1102
* Examples and benchmarks template input files now run vacuum, solvent and complex phases in order. by @ijpulidos in https://github.com/choderalab/perses/pull/1122
* Add openmm 8  to testing matrix by @mikemhenry in https://github.com/choderalab/perses/pull/1124
* Fix resume tests  to use new CLI (resolves issue #1150) by @mikemhenry in https://github.com/choderalab/perses/pull/1151
* Avoid testing non-examples in CI by @ijpulidos in https://github.com/choderalab/perses/pull/1164
* only run dev with newest python version by @mikemhenry in https://github.com/choderalab/perses/pull/1165
* remove outdated recipe by @mikemhenry in https://github.com/choderalab/perses/pull/1159
* update release process by @mikemhenry in https://github.com/choderalab/perses/pull/1162
* add env caching to CI by @mikemhenry in https://github.com/choderalab/perses/pull/1178
* Skip failing openmm 8 tests by @mikemhenry in https://github.com/choderalab/perses/pull/1186
* Add small molecule repex consistency tests by @zhang-ivy in https://github.com/choderalab/perses/pull/1065
* Fix RESTTopologyFactory test by @zhang-ivy in https://github.com/choderalab/perses/pull/1188
* enable merge queue by @mikemhenry in https://github.com/choderalab/perses/pull/1206
* Using default online analysis interval in GPU repex tests by @ijpulidos in https://github.com/choderalab/perses/pull/1207

New Contributors
^^^^^^^^^^^^^^^^

* @SauravMaheshkar made their first contribution in https://github.com/choderalab/perses/pull/1139

**Full Changelog**: https://github.com/choderalab/perses/compare/0.10.1...0.10.2


0.10.1 - Release
----------------

Bugfix release.

Bugfixes
^^^^^^^^
- Bug when trying to use the new ``RESTCapableHybridTopologyFactory`` in the small molecule pipeline -- fixed by not specifying a protocol, hence allowing the ``HybridCompatibilityMixin`` object to automatically handle it. Issue `#1039 <https://github.com/choderalab/perses/issues/1039>`_ (PR `#1045 <https://github.com/choderalab/perses/pull/1045>`_)
- Bug in ``create_endstates_from_real_systems()`` -- fixed by setting the global parameters for valence forces to the appropriate endstate. Also added tyk2 transformation test. Issue `#1041 <https://github.com/choderalab/perses/issues/1041>`_ (PR `#1050 <https://github.com/choderalab/perses/pull/1050>`_).
- Bug in the ``RESTCapableHybridTopologyFactory`` lifting expression -- fixed by separating the cutoff distance from the lifting distance. (PR `#1046 <https://github.com/choderalab/perses/pull/1046>`_)
- Fix bug in ``RelativeFEPSetup`` that prevents user from controlling the padding when solvating for solvent phase calculations. (PR `#1053 <https://github.com/choderalab/perses/pull/1053>`_)
- Bug in ``test_unsampled_endstate_energies_GPU`` and ``test_RESTCapableHybridTopologyFactory_energies_GPU`` -- fixed by removing unit-less ``rest_radius`` argument and using default instead. (PR `#1057 <https://github.com/choderalab/perses/pull/1057>`_)

Enhancements
^^^^^^^^^^^^
- Add flag ``transform_waters_into_ions_for_charge_changes`` for disabling the introduction of a counterion for charge changing transformations. Issue `#1004 <https://github.com/choderalab/perses/issues/1004>`_ (PR `#1030 <https://github.com/choderalab/perses/pull/1030>`_)
- Perses output yaml file now adds timestamp and ligands names information (for old and new ligands). Issue `#998 <https://github.com/choderalab/perses/issues/998>`_ (PR `#1052 <https://github.com/choderalab/perses/pull/1052>`_).
- Protein mutation repex internal consistency tests to ensure convergence. So far only testing neutral transformations. Issue `#1044 <https://github.com/choderalab/perses/issues/1044>`_ (PR `#1054 <https://github.com/choderalab/perses/pull/1054>`_).

0.10.0 - Release
----------------

New command line interface (CLI), many enhancements for the API (especially the ``PointMutationExecutor``) and improved testing.

Bugfixes
^^^^^^^^
- Bug in geometry engine's ``_determine_extra_torsions``: when ``topology_index_map``, which contains the atom indices involved in a particular torsion, is None -- fixed by not trying to add that torsion when ``topology_index_map`` is ``None``. (`#855 <https://github.com/choderalab/perses/pull/855>`_)
- Bug generated by changes upstream in the ``openmm`` package -- default method for calculating solvent padding changed, which resulted in smaller boxes. Fixed by adding more padding to the simulation box, it is now 1.1 nm. Openmm issue `#3502 <https://github.com/openmm/openmm/issues/3502>`_. Perses issue `#949 <https://github.com/choderalab/perses/issues/949>`_ (`#953 <https://github.com/choderalab/perses/pull/953>`_)
- Fixed energy bookkeeping in test of ``HybridTopologyFactory`` when a ring amino acid is involved in transformation. (`#969 <https://github.com/choderalab/perses/pull/969>`_)
- Avoid changing the global context cache behavior on module imports. Issue `#968 <https://github.com/choderalab/perses/issues/968>`_ (`#977 <https://github.com/choderalab/perses/pull/977>`_).
- Benchmark free energy plots now shift data to experimental mean. (`#981 <https://github.com/choderalab/perses/pull/981>`_)
- Skip introduction of counterion for charge changing mutations in vacuum and fix typo in the phase name in ``test_resume_protein_mutation_no_checkpoint`` (`#991 <https://github.com/choderalab/perses/pull/991>`_).
- Recovered logging capabilities respecting the ``LOGLEVEL`` environment variable. Issue `#1018 <https://github.com/choderalab/perses/issues/1018>`_ (`#1032 <https://github.com/choderalab/perses/pull/1032>`_).


Enhancements
^^^^^^^^^^^^
- Improved continuous integration (CI) performance. (`#961 <https://github.com/choderalab/perses/pull/961>`_)
- ``PointMutationExecutor`` now accepts both solute and solvated PDBs (previously only accepted solute PDBs). (`#967 <https://github.com/choderalab/perses/pull/967>`_)
- Tests and examples are now using ``openff-2.0.0`` force field instead of ``openff-1.0.0``. (`#971 <https://github.com/choderalab/perses/pull/971>`_)
- Use names (instead of indices) for fetching the force components of a system, avoiding issues with force reordering upstream in ``openmm``. Issue `#993 <https://github.com/choderalab/perses/issues/993>`_ (`#976 <https://github.com/choderalab/perses/pull/976>`_ and `#1007 <https://github.com/choderalab/perses/pull/1007>`_)
- Increase stability of simulations by decreasing the default hydrogen mass to 3 amu in the ``PointMutationExecutor``. Issue `#982 <https://github.com/choderalab/perses/issues/982>`_ (`#983 <https://github.com/choderalab/perses/pull/983>`_).
- Improved CI tests on both CPU and GPU. Better handling of temporary directories, closing opened reporter files when tests are finished, and using same environments for CPU and GPU (`#985 <https://github.com/choderalab/perses/pull/985>`_ `#989 <https://github.com/choderalab/perses/pull/989>`_ `#1012 <https://github.com/choderalab/perses/pull/1012>`_)
- Performance increase when retrieving the old or new positions from the hybrid positions. Issue `#1005 <https://github.com/choderalab/perses/issues/1005>`_ (`#1020 <https://github.com/choderalab/perses/pull/1020>`_)
- Use of unique names for force components in ``HybridTopologyFactory`` (`#1022 <https://github.com/choderalab/perses/pull/1022>`_).
- New function ``create_endstates_from_real_systems()`` for creating unsampled endstates for currently supported hybrid topology factories (`#1023 <https://github.com/choderalab/perses/pull/1023>`_).
- Improve the readability and usability of ``PointMutationExecutor`` and updates how parameters are specified for solvation (`#1024 <https://github.com/choderalab/perses/pull/1024>`_).

New features
^^^^^^^^^^^^
- Introduce ``RESTCapableHybridTopologyFactory``. Hybrid factory that allows for REST scaling, alchemical scaling, and 4th dimension softcore. So far, only working for protein mutations (`#848 <https://github.com/choderalab/perses/pull/848>`_ `#992 <https://github.com/choderalab/perses/pull/992>`_).
- New perses command line interface (CLI) ``perses-cli`` using ``click``. Allowing a more friendly interface for users. It tests the running environment, sets the platform for the simulation and allows interactive overriding arbitrary options in the input YAML file. Former ``perses-relative`` CLI entry point is now deprecated (`#972 <https://github.com/choderalab/perses/pull/972>`_ `#1021 <https://github.com/choderalab/perses/pull/1021>`_ `#1027 <https://github.com/choderalab/perses/pull/1027>`_).
- Support for handling charge changes (by transforming a water into a counterion) for both protein mutations and ligands transformations. `#862 <https://github.com/choderalab/perses/issues/862>`_ (`#973 <https://github.com/choderalab/perses/pull/973>`_).
- Hybrid topology factory class name can now be specified using the ``hybrid_topology_factory`` parameter in the input YAML file (`#988 <https://github.com/choderalab/perses/pull/988>`_).
- Introduce ``unsampled_endstates`` boolean option in input YAML file, which enables/disables creation of unsampled endstates with long-range sterics correction. Issue `#1033 <https://github.com/choderalab/perses/issues/1033>`_ (`#1037 <https://github.com/choderalab/perses/pull/1037>`_).

0.9.5 - Release
---------------

Enhancements
---------------
- (PR `#948 <https://github.com/choderalab/perses/pull/948>`_ & PR `#952 <https://github.com/choderalab/perses/pull/952>`_) Add citation file, which should enhance the citation generated by Zenodo.

0.9.4 - Release
---------------

Performance optimizations:
^^^^^^^^^^^^^^^^^^^^^^^^^
- (PR `#938 <https://github.com/choderalab/perses/pull/938>`_) Separate ContextCache objects are now created for propagation and energy computation in replica exchange calculations to avoid periodic cycling behavior.

Bugfixes
^^^^^^^^
- (PR `#938 <https://github.com/choderalab/perses/pull/938>`_) Mixed precision and deterministic forces are used by default.
- (PR `#938 <https://github.com/choderalab/perses/pull/938>`_) Velocities are refreshed from the Maxwell-Boltzmann distribution each iteration to avoid sudden cooling when simulations are resumed.
- (PR `#944 <https://github.com/choderalab/perses/pull/944>`_) Fixes to visualization module.

Enhancements
---------------
- (PR `#909 <https://github.com/choderalab/perses/pull/909>`_) Overhaul of Folding@home setup pipeline
- (PR `#909 <https://github.com/choderalab/perses/pull/909>`_) `use_given_geometries` is now `True` by default
- (PR `#934 <https://github.com/choderalab/perses/pull/934>`_) Preview of new CLI tool perses-cli that takes in a yaml file and creates dummy output. Work in progress. CLI/API still subject to changes.

0.9.3 - Release
---------------

Bugfixes
^^^^^^^^

- (PR `#894 <https://github.com/choderalab/perses/pull/894>`_)
  Remove unused argument 'implicitSolvent' from SystemGenerator in tests.

- (PR `#893 <https://github.com/choderalab/perses/pull/893>`_)
  Add installation instructions to readme.

- (PR `#892 <https://github.com/choderalab/perses/pull/892>`_)
  Allow `generate_dipeptide_top_pos_sys` to accept `demap_CBs`.

- (PR `#878 <https://github.com/choderalab/perses/pull/878>`_)
  Fix stochastic failures in RepartitionedHybridTopologyFactory test.

- (PR `#877 <https://github.com/choderalab/perses/pull/877>`_)
  Fix naked charge padding (sigmas for hydroxyl hydrogens are changed from 1.0 nm to 0.06 nm).

- (PR `#874 <https://github.com/choderalab/perses/pull/874>`_)
  Added readme instructions on how to run perses examples using the docker container with GPUs/CUDA.

- (PR `#866 <https://github.com/choderalab/perses/pull/866>`_)
  Fix endstate validation handling in PointMutationExecutor.

- (PR `#860 <https://github.com/choderalab/perses/pull/860>`_)
  Simplify `_construct_atom_map` for protein mutations.

- Various CI fixes
  * PR `#787 <https://github.com/choderalab/perses/pull/787>`_
  * PR `#850 <https://github.com/choderalab/perses/pull/850>`_
  * PR `#858 <https://github.com/choderalab/perses/pull/858>`_
  * PR `#868 <https://github.com/choderalab/perses/pull/868>`_
  * PR `#871 <https://github.com/choderalab/perses/pull/871>`_
  * PR `#880 <https://github.com/choderalab/perses/pull/880>`_
  * PR `#887 <https://github.com/choderalab/perses/pull/887>`_
  * PR `#898 <https://github.com/choderalab/perses/pull/898>`_

New features
^^^^^^^^^^^^

- (PR `#896 <https://github.com/choderalab/perses/pull/896>`_)
  Drop support for older OpenMM versions.
  We now only support versions >= 7.6.0.

- (PR `#924 <https://github.com/choderalab/perses/pull/924>`_)
  Command line utility to automatically run and analyze benchmarks using the data set found in https://github.com/openforcefield/protein-ligand-benchmark/

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
^^^^^^^^^^^^

- (PR `#708 <https://github.com/choderalab/perses/pull/708>`_)
  Create visualization module for generating PyMOL movies.
- (PR `#834 <https://github.com/choderalab/perses/pull/834>`_)
  Enable protein mutation transformations involving nonstandard amino acids, specifically: ASH, GLH, HID, HIE, HIP, LYN.
- (PR `#838 <https://github.com/choderalab/perses/pull/838>`_)
  Official Docker image hosted on docker hub ``docker pull choderalab/perses:0.9.2``.
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

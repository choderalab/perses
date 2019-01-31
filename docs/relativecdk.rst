.. _relativeCDK2:

Relative free energies
**********************

Example script to calculate the relative free energies of ligands to CDK2. The input is available in ``perses/examples/cdk2-example``


Input
-----

* ``CDK2_fixed_nohet.pdb``: PDB file for CDK2 structure from JACS benchmark (cited below) run through PDBFixer and pdb4amber.

* ``CDK2_ligands.sdf``: SDF file of CDK2 ligands from JACS structure

* ``cdk2_nonequilibrium.yaml``: Example configuration file for running CDK2 relative free energy calculations with nonequilibrium switching

* ``cdk2_sams.yaml``: Example configuration file for running CDK2 relative free energy calculations with SAMS.

Running
-------

The example can be run by calling ``python ../../scripts/setup_relative_calculation.py cdk2_sams.yaml``. By default, the included yaml file directs the code to set up and run the solvent phase. 

The options for the run are contained in the ``yaml`` file passed to the ``setup_relative_calculation.py`` script. There are two examples provided,
``cdk2_sams.yaml`` and ``cdk2_nonequilibrium.yaml``, which allow the user to run either a SAMS or nonequilibrium switching-based free energy calculation, respectively.

Analysis
--------
#TODO

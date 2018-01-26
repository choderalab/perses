# CDK2 protein-ligand example

This folder contains an example setup for a protein-ligand relative free energy calculation.

## Origin of input files

The input files in this directory are from the Schrodinger JACS dataset, with the protein pdb fixed using [PDBFixer](https://github.com/pandegroup/pdbfixer).

## Running example

The example can be run by calling `python setup_relative_calculation.py cdk2_setup.yaml`. By default, the included yaml file directs the code to set up and run the solvent phase. 
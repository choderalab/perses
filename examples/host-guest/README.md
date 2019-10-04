# Host-guest example system

This folder contains an example system for preparing a host-guest relative free energy calculation in vacuum. This is primarily for testing and evaluation purposes. 

## Source of input files

The input files were taken from the [SAMPL6 SAMPLing](https://github.com/MobleyLab/SAMPL6) challenge. We start with `complex.pdb` from [CB8-G3-0](https://github.com/MobleyLab/SAMPL6/tree/master/host_guest/SAMPLing/CB8-G3-0/PDB). The git revision used for this input file was `93f642ef9e2c478985cd14f3084d899ae3ee1c6d`. We additionally generate a second input molecule for the relative calculation by its IUPAC name,
`(5-methoxy-4-quinolyl)-(5-vinylquinuclidin-1-ium-2-yl)methanol`. This molecule was chosen to be a small perturbation to the `G3-0` ligand already in the complex.


## Preparation of input files

We first take the solvated complex and extract the host and guest molecules, without solvent. We convert both to OpenEye `OEMol`s and write the host to a separate `mol2` file. We generate the alternate guest molecule by IUPAC name, and use OpenEye Omega to generate an initial conformation for it. Both guests are written to a `mol2` file, with the original molecule from the complex first.

## Running example

To run the example, call `python setup_relative_calculation.py hg_setup.yaml`.
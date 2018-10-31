# FreeSolv example

This folder contains an example of optimizing a subset of FreeSolv molecules for hydration free energy
using Reversible Jump MCMC. The `MultiTargetDesigner` that gets instantiated is intended to focus on the states
whose relative hydration free energy is most favorable.

## Components

### Database
The database originates from the Mobley Lab [FreeSolv repository](https://github.com/MobleyLab/FreeSolv).

### Preparation
In this example, we restrict ourselves to molecules that contain a benzene ring. To prepare
the included database, run `bash prepare_freesolv.sh`. This will generate several output files:

* `filtered_database.smi`: A set of SMILES strings for the molecules to use in the calculation
* `filtered_database.pdf`: A pdf depiction of the molecules selected
* `database.smi`: All molecules in FreeSolv in SMILES format
* `full_database.pdf`: Depiction of the full FreeSolv database

## Reference

David L. Mobley, Michael Shirts, Nathan Lim, John Chodera, Kyle Beauchamp, & Lee-Ping. (2018, January 26). MobleyLab/FreeSolv: Version 0.52 (Version v0.52). Zenodo. http://doi.org/10.5281/zenodo.1161245
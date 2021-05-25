# CDK2 protein-ligand example

This folder contains an example setup for a protein-ligand relative free energy calculation.

## Origin of input files

The input files in this directory are from the Schrodinger JACS dataset, with the protein pdb fixed using [PDBFixer](https://github.com/pandegroup/pdbfixer).

## File description

* CDK2_protein.pdb: PDB file for CDK2 structure from JACS benchmark (cited below) run through PDBFixer and pdb4amber.

* CDK2_ligands_shifted.sdf: SDF file of CDK2 ligands from JACS structure

* cdk2_nonequilibrium.yaml: Example configuration file for running CDK2 relative free energy calculations with nonequilibrium switching

* cdk2_sams.yaml: Example configuration file for running CDK2 relative free energy calculations with SAMS.

* cdk2_repex.yaml: Example configuration file for running CDK2 relative free energy calculations with REPEX.

* run.py: Helper script to run a range of pairs in the same job submission script

* submit_ligpairs.sh: LSF job submission script for running a range of ligand pairs simultaneously

## Running example

The example can be run by calling `perses-relative cdk2_sams.yaml`. By default, the included yaml file directs the code to set up and run the solvent phase. 

The options for the run are contained in the `yaml` file passed to the `perses.app.setup_relative_calculation.py` script. There are three examples provided,
`cdk2_sams.yaml`, `cdk2_nonequilibrium.yaml` and `cdk2_repex.yaml`, which allow the user to run either a SAMS, REPEX or nonequilibrium switching-based free energy calculation.

To run a range of ligands, change the ligand indexed in `run.py` to those of interest. In `submit-ligpairs.sh`, change the jobarray so that it is the same length of the list of ligand pairs.
To run the job on lilac, or a similar cluster, use `bsub < submit-ligpairs.sh`.
## References

Accurate and Reliable Prediction of Relative Ligand Binding Potency in Prospective Drug Discovery by Way of a Modern Free-Energy Calculation Protocol and Force Field
Lingle Wang et al. JACS 137:2695, 2017.
DOI: http://doi.org/10.1021/ja512751q
SI info: http://pubs.acs.org/doi/suppl/10.1021/ja512751q/suppl_file/ja512751q_si_001.pdf
SI datasets: https://drive.google.com/drive/u/1/folders/0BylmDElgu6QLTnJ2WGMzNXBENkk

This dataset is used for testing relative transformations.

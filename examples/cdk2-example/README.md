# CDK2 protein-ligand example

This folder contains an example setup for a protein-ligand relative free energy calculation.

## Origin of input files

The input files in this directory are from the Schrodinger JACS dataset, with the protein pdb fixed using [PDBFixer](https://github.com/pandegroup/pdbfixer).

## Running example

The example can be run by calling `python ../../scripts/setup_relative_calculation.py cdk2_sams.yaml`. By default, the included yaml file directs the code to set up and run the solvent phase. 

The options for the run are contained in the `yaml` file passed to the `setup_relative_calculation.py` script. There are two examples provided,
`cdk2_sams.yaml` and `cdk2_nonequilibrium.yaml`, which allow the user to run either a SAMS or nonequilibrium switching-based free energy calculation, respectively.
## References

Accurate and Reliable Prediction of Relative Ligand Binding Potency in Prospective Drug Discovery by Way of a Modern Free-Energy Calculation Protocol and Force Field
Lingle Wang et al. JACS 137:2695, 2017.
DOI: http://doi.org/10.1021/ja512751q
SI info: http://pubs.acs.org/doi/suppl/10.1021/ja512751q/suppl_file/ja512751q_si_001.pdf
SI datasets: https://drive.google.com/drive/u/1/folders/0BylmDElgu6QLTnJ2WGMzNXBENkk

This dataset is used for testing relative transformations.
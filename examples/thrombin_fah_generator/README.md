# Example: Setting up thrombin system from JACS benchmark set on Folding@home

## Manifest
* `fah_all.yaml` : perses YAML template for 0 -> 1 transformation
* `perses.yml` : conda environment file (OUTDATED)
* `run.py` : driver Python script to set up individual transformations as PROJ#####/RUNS/RUN# using template `fah_all.yaml`
* `submit-ligpairs` : LSF batch script for setting up all ligand pairs
* `Thrombin_protein.pdb` : thrombin receptor PDB file
* `Thrombin_ligands_shifted.pdb` : ligands SDF file containing all ligands with coordinates for common substructure aligned

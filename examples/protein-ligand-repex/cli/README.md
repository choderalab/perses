# Run tyk2 protein-ligand example
The files in this directory are the input files needed to run a single transformation (edge) for an
alchemical relative binding free energy simulation using the perses command line interface (CLI).

It runs an alchemical relative binding free energy calculation using replica exchange with `5 ns/replica`.

To run the example:
```bash
perses-cli --yaml protein-ligand.yaml
```

Please run `perses-cli --help` for more details and options to run the simulation.

The file `protein-ligand.yaml` has information on the different parameters to set up a replica exchange (repex)
simulation.

## Scripts
The directory `scripts_utils` has different shell scripts that can help automating typical runs of a complete
alchemical network of transformations. These are only meant for illustrative purposes, adapt to your needs.

* `run_star_map.sh`: Example script to run a star map network in serial.
* `submit-star-map-serial.sh`: Example script to submit an LSF scheduler job of a star map network in an HPC environment.
* `submit-dense-map.sh`: Example script to submit an LSF scheduler job of a dense map (all-to-all) network in an HPC environment.
* `cleanup.sh`: Example script to clean up results from a simulation, useful for restarting simulations from scratch.
# New CLI

## Setup and Overview

NOTE: This CLI tool is under active development and the API is expected to change!

After installing `perses` check to see if the CLI tool works with `perses-cli --help`, it should look something like this:

```bash
$ perses-cli --help
Usage: perses-cli [OPTIONS]

  test

Options:
  --yaml FILE           [required]
  --platform-name TEXT
  --override TEXT
  --help                Show this message and exit.
```
The `--yaml` argument is the path to the yaml file.

If `--platform-name` is used e.g. `--platform-name CUDA` then an error will be raised if the requested platform is unavailable.
This is useful since by default, we attempt to use the fastest platform available.
If there is misconfiguration issue on a GPU node for example, we will fall back to using the CPU which is likely undesirable.
Hence, it is recommended to use `--platform-name CUDA` when running on a system with GPU resources.
Options for `--platform-name` are `Reference`, `CPU`, `CUDA`, and `OpenCL`.

The `--overrride` option is used to specify and option that will override an option set in the yaml.
For example, if your yaml file contained

```yaml
old_ligand_index: 0
new_ligand_index: 1
trajectory_directory: lig0to1
```

and you wanted to instead use ligand at index 2 for the old and ligand 3 for the new, you would run:

```bash
$ perses-cli --yaml template.yaml --override old_ligand_index:2 --override new_ligand_index:3 --override trajectory_directory:lig2to3
```
This will override the options in the yaml (be sure to change the `trajectory_directory` so you don't overwrite your previous simulation.
Currently only `key:value` parts of the yaml can be overridden i.e. not sequences or lists.

To view all options ultimately used in the simulation, a file named `perses-$date-$yaml_name.yaml` is created under the simulation/experiment directory.

## Example Use Case

In this example folder we have a protein: [2ZFF](https://www.rcsb.org/structure/2zff) and some ligands which we will use for a series of free energy calculations.
We will use a geometric based atom mapping.
Our yaml file `template.yaml` is setup to do an alchemical transformation from ligand 0 to ligand 1 for the solvent, and complex phase.
We will use a bash loop (see `run_star_map.sh`) to do a star map with 6 different ligands.
We will put the ligand at index 0 at the center of star map with the following bash script:

```bash
#!/usr/bin/env bash

set -xeuo pipefail

old_ligand_idx=0
for new_ligand_idx in $(seq 1 10)
do 
	perses-cli --yaml template.yaml --override old_ligand_index:"$old_ligand_idx" --override new_ligand_index:"$new_ligand_idx" --override trajectory_directory:lig"$old_ligand_idx"to"$new_ligand_idx"
done
```

This is equivalent to running all of these commands manually:

```bash
$ perses-cli --yaml template.yaml --override old_ligand_index:0 --override new_ligand_index:1 --override trajectory_directory:lig0to1
$ perses-cli --yaml template.yaml --override old_ligand_index:0 --override new_ligand_index:2 --override trajectory_directory:lig0to2
$ perses-cli --yaml template.yaml --override old_ligand_index:0 --override new_ligand_index:3 --override trajectory_directory:lig0to3
$ perses-cli --yaml template.yaml --override old_ligand_index:0 --override new_ligand_index:4 --override trajectory_directory:lig0to4
$ perses-cli --yaml template.yaml --override old_ligand_index:0 --override new_ligand_index:5 --override trajectory_directory:lig0to5
```

## Analysis 
To analyze the data, we provide a command line tool `analyze-benchmark.py` in the examples directory that can be run as follows:

```bash
python analyze-benchmark.py
```
This requires the `ligands.sdf` file to have the experimental data in `<EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL>` and `<EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR>` tags.

So far, both the `ligands.sdf` file and subdirectories with the results must live in the same base directory where this script is run. This will be improved in the future to allow custom paths and filenames.

## Docker Example

First, grab the dev image of perses that has the new CLI tool with `docker pull choderalab/perses:dev`.
General docker instructions for using perses and docker can be found [here](https://github.com/choderalab/perses/tree/main/docker#readme).

The following examples expect a local copy of the perses source tree.

### Running single edge simulation
To perform a single edge simulation using the provided container, use the following docker command, for example for the `0` to `5` ligand transformation:

```bash
docker run --rm --gpus device=0 --mount type=bind,source=$HOME/.OpenEye/,target=/openeye/,readonly --mount type=bind,source=$HOME/workdir/repos/perses/examples/,target=/mnt/ -w /mnt/new-cli choderalab/perses:dev perses-cli --yaml template.yaml --override old_ligand_index:0 --override new_ligand_index:5 --override trajectory_directory:lig0to5
```
Of importance there are the paths to the OpenEye license file (in this example is `$HOME/.OpenEye/`), path to the examples directory in perses (`$HOME/Projects/perses/examples/`).

### Running serial star map

To run the star map in serial using the same container you can use the following docker command (to make it easier to read the command is split across multiple lines but this is not necessary).
```bash
docker run -it --rm --gpus device=0 --mount type=bind,source=$HOME/.OpenEye/,target=/openeye/,readonly \
                                    --mount type=bind,source=$HOME/repos/perses/examples/,target=/mnt/ \
                                    -w /mnt/new-cli choderalab/perses:dev bash ./run_star_map.sh
```



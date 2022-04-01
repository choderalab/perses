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

The `--overrride` option is used to specify and option that will override an option set in the yaml.
For example, if your yaml file contained

```yaml
old_ligand_index: 0
new_ligand_index: 1
trajectory_directory: lig0to1
```

and you wanted to instead use ligand at index 2 for the old and ligand 3 for the new, you would run:

```bash
$ perses-cli --yaml my.yaml --override old_ligand_index:2 --override new_ligand_index:3 --override trajectory_directory:lig2to3
```
This will override the options in the yaml (be sure to change the `trajectory_directory` so you don't overwrite your previous simulation.
Currently only `key:value` parts of the yaml can be overridden i.e. not sequences or lists.

## Example Use Case

In this example folder we have a protein: [2ZFF](https://www.rcsb.org/structure/2zff) and some ligands which we will use for a series of free energy calculations.
We will use a geometric based atom mapping.
Our yaml file `my.yaml` is setup to do an alchemical transformation from ligand 0 to ligand 1 for the solvent, vacuum, and complex phase.
We will use a bash loop to do a star map with 6 different ligands.
We will put the ligand at index 0 at the center of star map with the following bash script:
```bash
#!/usr/bin/env bash

for new_ligand_idx in $(seq 1 5)
do 
	perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:"$new_ligand_idx" --override trajectory_directory:lig0to"$new_ligand_idx"
done
```
This is equivalent to running all of these commands manually:

```bash
$ perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:1 --override trajectory_directory:lig0to1
$ perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:2 --override trajectory_directory:lig0to2
$ perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:3 --override trajectory_directory:lig0to3
$ perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:4 --override trajectory_directory:lig0to4
$ perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:5 --override trajectory_directory:lig0to5
```

## Analysis 
TODO

## Docker Example

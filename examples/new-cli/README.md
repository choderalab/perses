# New CLI

## Setup and Overview

NOTE: This CLI tool is under active development and the API is expected to chage!

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

If `--platform-name` is used e.g. `--platform-name CUDA` then an error will be raised if the requested platform is unavialable.
This is useful since by defualt, we attempt to use the fastest platfrom aviable.
If there is misconfiguration issue on a GPU node for example, we will fall back to using the CPU which is likely undiserable.
Hence, it is recomended to use `--platform-name CUDA` when running on a system with GPU resources. 

The `--overrride` option is used to specify and option that will override an option set in the yaml.
For example, if your yaml file contained

```yaml
old_ligand_index: 0
new_ligand_index: 1
```

and you wanted to instead use ligand at index 2 for the old and ligand 3 for the new, you would run:

```bash
$ perses-cli --yaml my.yaml --override old_ligand_index:2 --override new_ligand_index:3
```
Currently only key:value parts of the yaml can be overrided i.e. not sequences or lists.

## Example Use Case

In this example folder we have a protein: [2ZFF](https://www.rcsb.org/structure/2zff) and some ligands which we will use for a series of free energy calculations.
Our yaml file `my.yaml` is setup to do an alchemical transformation from ligand 0 to ligand 1 for the solvent, vaccume, and complex phase. 


## Analysis 
TODO

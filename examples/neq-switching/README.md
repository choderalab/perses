# MCL1 transdimensional nonequilibrium switching example

## Setup

### Forcefield

Before running any simulations, first generate the requisite forcefield file:
```bash
python generate_forcefields_from_molfile.py MCL1_ligands.sdf MCL1_ligands.xml
```

### Complex system parameterization

To parameterize the MCL1 systems for equilibrium simulation (which precedes the nonequilibrium switching experiments),
run:
```bash
python run_equilibrium_setup.py input_options.yaml
```

where `input_options.yaml` contains the options (such as output directories) for simulation. See the example file for more details

This script will generate stored numpy arrays containing initial positions, an `mdtraj.Topology` object and an `openmm.System` object.
All systems are solvated to the same number of waters. 

## Run equilibrium simulations

Before running the transdimensional nonequilibrium switching calculations, we generate a cache of equilibrium samples.
To do this, run:
```bash
python run_equilibrium.py input_options.yaml [index]
```

where `input_options.yaml` is the same as the one used in the setup, and `index` is the index (starting at 1)
of the system to simulate. Having the index as a command-line parameter makes it straightforward to
run as an array job (whose indices start at 1 for LSF).

## Run transdimensional nonequilibrium switching

After generating the equilibrium cache, run the transdimensional nonequilibrium switching experiments by:

```bash
python run_nonequilibrium.py input_options.yaml [pair_index]
```

Similarly to the equilibrium run command, we give the same `input_options.yaml` file that we've been using.
We also give an index, but note that this is an index for the pair `(molecule_a, molecule_b)` (note that this is not equivalent to `(molecule_b, molecule_a)`, which has a different index).

The output files will be placed in the directory specified in the configuration file. The output is a numpy array that is named:
`project_prefix_initial_ligand_proposal_ligand.npy`, and contains an array of size: `[n_iterations, 7]`

The components of each iteration are:

```pythonstub
        results[i, 0] = initial_logP
        results[i, 1] = logP_reverse
        results[i, 2] = final_logP
        results[i, 3] = logP_work
        results[i, 4] = initial_hybrid_logP
        results[i, 5] = final_hybrid_logP
        results[i, 6] = logP_geometry_forward
```
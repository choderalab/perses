# Barnase-Barstar protein-protein interaction example

This example is based on the tools and work by Ivy Zhang in https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00333

More info: https://github.com/choderalab/perses-barnase-barstar-paper

## How to run

This example assumes that you will be running on an HPC infrastructure using MPI, running with
multiple GPUs.

### Why MPI?

Sampling protein-protein interactions for Free Energy calculations is a very computationally costly task, by
parallelizing the replica propagation (lambda windows) using MPI and multiple GPUs a higher throughput can be
achieved.

### Running environment

Make sure your environment has a `mpiplus` and `mpi4py` installed in your environment, besides of `perses`.

### Pipeline

Please note that in order to run in an MPI environment we have to separate the generation of the Hybrid Topology
with respect to the actual computation parts of the repex algorithm.

1. Generate HTFs with
```bash
python generate_htfs.py ala_vacuum.pdb 2 THR results_dir
```
Input pdb file can be found in https://github.com/choderalab/perses/blob/main/perses/data/ala_vacuum.pdb .
Please download the files from the mentioned URL to run this command.



2. Run with MPI (same host with 2 different GPUs) using
```bash
mpiexec -f hostfile -configfile configfile
```
you may need to adapt the command and the `hostfile` and `configfile` to your MPI environment. We have
previously prepared the config files for practical purposes. How to set up an MPI environment and files
is beyond the scope of this example.

# Perses Container

The container can be downloaded with `docker pull choderalab/perses:0.9.2`

## OpenEye License

The open eye license file is not included and must be passed in as a mount point.
The container expects `oe_license.txt` in `/openeye/` because we set `ENV OE_LICENSE=/openeye/oe_license.txt` in the Dockerfile.

To do that use the `--mount` option.
For example, if you have an `oe_license.txt` file in `$HOME/.openeye/`.

```
$ docker run --rm --mount type=bind,source=$HOME/.openeye/,target=/perses/,readonly choderalab/perses:0.9.2 python -c "import openeye; assert openeye.oechem.OEChemIsLicensed(), 'OpenEye license checks failed!'"
```

## GPU Support

Pass the option `--gpus device=0` to use the host's GPU:

```
$ docker run -it --rm --gpus device=0 --mount type=bind,source=$HOME/.openeye/,target=/openeye/,readonly choderalab/perses:0.9.2 python -m simtk.testInstallation
OpenMM Version: 7.5.1
Git Revision: a9cfd7fb9343e21c3dbb76e377c721328830a3ee

There are 3 Platforms available:

1 Reference - Successfully computed forces
2 CPU - Successfully computed forces
3 CUDA - Successfully computed forces

Median difference in forces between platforms:

Reference vs. CPU: 6.29536e-06
Reference vs. CUDA: 6.73195e-06
CPU vs. CUDA: 7.32829e-07

All differences are within tolerance.
```
Note: `perses` currently works best on a single GPU. 
See the documentation [here](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu) for how to specify a single GPU on a multi-GPU system.

## Running perses examples from the container in GPUs using CUDA

If you plan to use our docker container and CUDA/GPUs, quick instructions to get the examples running are as follows.

We have our examples with minimalistic setups to satisfy our current continuous integration (CI) workflow. To get more "realistic" simulations we recommend making the following modifications.

1) Clone the repository (say it lives in `$HOME/repos/perses`) with `git clone https://github.com/choderalab/perses.git`

2) Modify the `run_neq_distrib_flattened.py` inside each of the examples you want to run:
- Change the platform name to 'CUDA '(it is CPU by default to satisfy our continuous integration). Should be in line 21 in the script.
- (recommended) You might want to change the nsteps_eq and nsteps_neq (number of steps for equilibrium and nonequilibrium, respectively) to a larger value. Something around 250000 steps (1ns in time) for each should be okay. These should be lines 17 and 18 in the scripts.
- (recommended) Please change the save frequency or else you would be storing A LOT! I guess the actual value depends on what you actually want to see, but please do change it if you increase the number of steps as in the previous item.

3) Run our latest docker container and execute an example.
    For example for the kinase-neq-switching example, you would do something like:
```bash
docker run -it --rm --gpus device=0 --mount type=bind,source=$HOME/.OpenEye/,target=/openeye/,readonly --mount type=bind,source=$HOME/repos/perses/examples/,target=/mnt/ -w /mnt/kinase-neq-switching choderalab/perses:0.9.2  python run_neq_distrib_flattened.py
 ```
Of importance there are the paths to the OpenEye license file (in this example is `$HOME/.OpenEye/`), path to the examples directory in perses (`$HOME/repos/perses/examples/`) and the actual example subdirectory example you want to run (`/mnt/kinase-neq-switching`)

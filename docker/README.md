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

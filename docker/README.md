
The open eye license file is not included and must be passed in as a mount point.
I have an `oe_license.txt` file in `/home/mmh/.openeye/`.
The container expects `oe_license.txt` in `/perses/` because we set `ENV OE_LICENSE=/perses/oe_license.txt` in the Dockerfile.

To do that use the `--mount` option.

docker run --mount type=bind,source=/home/mmh/.openeye/,target=/perses/,readonly oe-license-test:0.1  python -c "import openeye; assert openeye.oechem.OEChemIsLicensed(), 'OpenEye license checks failed!'"

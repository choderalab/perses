Developer Notes / Tools
=======================

Assorted notes for developers.

How to do a release
-------------------

First make sure everything is working:

- Make sure CI is passing
- Run GPU tests manually on HPC or the self-hosted EC2 runner on AWS
- Run the PLB (https://github.com/openforcefield/protein-ligand-benchmark)

Then create a new release on github: https://github.com/choderalab/perses/releases/new
Either make a new tag and push it, or select the create a new tag on release option.
Make sure that the change log is detailed and communicates Bugfixes, Enhancements, and New features.
Also attach plots from the PLB in a "Benchmark data" section.

Once the release is posted on github, the conda-forge bot should pick up the release and make a PR to the perses feedstock repository: https://github.com/conda-forge/perses-feedstock
Be sure to update the meta.yaml file with any new dependencies or version pins.

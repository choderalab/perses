<!--- [![Travis Build Status](https://travis-ci.org/choderalab/perses.svg?branch=master)](https://travis-ci.org/choderalab/perses/branches) --->
[![GH Actions Status](https://github.com/choderalab/perses/workflows/CI/badge.svg)](https://github.com/choderalab/perses/actions?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/choderalab/perses/branch/main/graph/badge.svg)](https://codecov.io/gh/choderalab/perses/branch/main)
[![Documentation Status](https://readthedocs.org/projects/perses/badge/?version=latest)](http://perses.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/27087846.svg)](https://zenodo.org/badge/latestdoi/27087846)

# Perses

Experiments with expanded ensemble simulation to explore chemical and mutational space.

## License
This software is licensed under the [MIT license](https://opensource.org/licenses/MIT), a permissive open source license.

## Notice

Please be aware that this code is made available in the spirit of open science, but is currently pre-alpha--that is,
**it is not guaranteed to be completely tested or provide the correct results**, and the API can change at any time
without warning. If you do use this code, do so at your own risk. We appreciate your input, including raising issues
about potential problems with the code, but may not be able to address your issue until other development activities
have concluded.

## Install

See our installation instructions [here](https://perses.readthedocs.io/en/latest/installation.html).

### Quick Start

In a fresh conda environment:

```
$ conda config --add channels conda-forge openeye
$ conda install perses openeye-toolkits
```

## Manifest

* `perses/` - Package containing code for performing expanded ensemble simulations
* `examples/` - Contains examples for various systems and methods of simulation
* `attic/` - some old code that may be useful as part of the new setup
* `devtools/` - Continuous integration and packaging utilities
* `notes/` - LaTeX notes deriving acceptance criteria and stochastic approximation methods

## Contributors

A complete list of contributors can be found at [GitHub Insights](https://github.com/choderalab/perses/graphs/contributors).

Major contributors include:
* Julie M. Behr
* Hannah E. Bruce Macdonald
* John D. Chodera
* Patrick B. Grinaway
* Mike M. Henry
* Iván J. Pulido
* Jaime Rodríguez-Guerra
* Dominic A. Rufa
* Ivy Zhang

## Cite
Please consider citing:
* **Software**:

Rufa, D. A., Zhang, I., Bruce Macdonald, H. E., Grinaway, P. B., Pulido, I., Henry, M. M., Rodríguez-Guerra, J., Wittmann, M., Albanese, S. K., Glass, W. G., Silveira, A., Schaller, D., Naden, L. N., & Chodera, J. D. (2023). Perses (0.10.3). Zenodo. https://doi.org/10.5281/zenodo.8350218

* **Protein mutations**:

Zhang, I., Rufa, D. A., Pulido, I., Henry, M. M., Rosen, L. E., Hauser, K., Singh, S., & Chodera, J. D. (2023). Identifying and Overcoming the Sampling Challenges in Relative Binding Free Energy Calculations of a Model Protein:Protein Complex. Journal of chemical theory and computation, 19(15), 4863–4882. https://doi.org/10.1021/acs.jctc.3c00333

* **Small molecule transformations**:

Rufa, D. A., Bruce Macdonald, H. E., Fass, J., Wieder, M., Grinaway, P. B., Roitberg, A. E., Isayev, O., & Chodera, J. D. (2020). Towards chemical accuracy for alchemical free energy calculations with hybrid physics-based machine learning / molecular mechanics potentials. In bioRxiv (p. 2020.07.29.227959). https://doi.org/10.1101/2020.07.29.227959


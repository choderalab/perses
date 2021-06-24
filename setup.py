# -*- coding: utf-8 -*-
"""Perses: Tools for expanded-ensemble simulations with OpenMM
"""

from __future__ import absolute_import, print_function

DOCLINES = __doc__.split("\n")

import os
import sys
from os.path import join as pjoin
from os.path import relpath

from setuptools import setup

import versioneer

try:
    sys.dont_write_bytecode = True
    sys.path.insert(0, ".")
finally:
    sys.dont_write_bytecode = False


if "--debug" in sys.argv:
    sys.argv.remove("--debug")
    DEBUG = True
else:
    DEBUG = False

# Useful function
def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(pjoin(root, fn), package_root))
    return files


CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: C++
Programming Language :: Python
Development Status :: 4 - Beta
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
"""

extensions = []

setup(
    name="perses",
    author="John D. Chodera",
    author_email="john.chodera@choderalab.org",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url="https://github.com/choderalab/perses",
    platforms=["Linux", "Mac OS-X", "Unix"],
    classifiers=CLASSIFIERS.splitlines(),
    packages=[
        "perses",
        "perses.storage",
        "perses.analysis",
        "perses.samplers",
        "perses.rjmc",
        "perses.annihilation",
        "perses.bias",
        "perses.tests",
        "perses.dispersed",
        "perses.app",
        "perses.utils",
    ],
    # package_data={'perses' : find_package_data('perses','examples') + find_package_data('perses','data')}, # I don't think this works
    package_data={
        "perses": find_package_data("perses/data", "perses")
    },  # I think this is fixed
    zip_safe=False,
    entry_points={
        "openmm.forcefielddir": ["perses=perses:get_datadir"],
        "console_scripts": [
            "perses-relative = perses.app.setup_relative_calculation:run",
            "perses-fah = perses.app.fah_generator:run",
        ],
    },
    ext_modules=extensions,
    install_requires=[],
)

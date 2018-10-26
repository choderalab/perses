# -*- coding: utf-8 -*-
"""Perses: Tools for expanded-ensemble simulations with OpenMM
"""

from __future__ import print_function, absolute_import

DOCLINES = __doc__.split("\n")

import os
import sys
import glob
import traceback
import numpy as np
from os.path import join as pjoin
from os.path import relpath
from setuptools import setup, Extension, find_packages
try:
    sys.dont_write_bytecode = True
    sys.path.insert(0, '.')
    from basesetup import write_version_py, CompilerDetection, check_dependencies
finally:
    sys.dont_write_bytecode = False


if '--debug' in sys.argv:
    sys.argv.remove('--debug')
    DEBUG = True
else:
    DEBUG = False

#Useful function
def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(pjoin(root, fn), package_root))
    return files


# #########################
VERSION = '0.1'
ISRELEASED = False
__version__ = VERSION
# #########################

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)
Programming Language :: C++
Programming Language :: Python
Development Status :: 4 - Beta
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
"""

extensions = []

setup(name='perses',
      author='Patrick Grinaway',
      author_email='patrick.grinaway@choderalab.org',
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      version=__version__,
      url='https://github.com/choderalab/perses',
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      classifiers=CLASSIFIERS.splitlines(),
      packages=['perses', 'perses.storage', 'perses.analysis', 'perses.samplers', 'perses.rjmc', 'perses.annihilation', 'perses.bias', 'perses.tests', 'perses.dispersed'],
      #package_data={'perses' : find_package_data('perses','examples') + find_package_data('perses','data')}, # I don't think this works
      package_data={'perses' : find_package_data('examples', 'perses') + find_package_data('perses/data', 'perses')}, # I think this is fixed
      zip_safe=False,
      entry_points={
       'openmm.forcefielddir' : ['perses=perses:get_datadir']
      },
      ext_modules=extensions,
      install_requires=[],
      )

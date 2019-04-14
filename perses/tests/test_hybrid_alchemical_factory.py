"""
Unit tests for the hybrid relative alchemical factory.

"""

__author__ = 'John D. Chodera'

################################################################################
# GLOBAL IMPORTS
################################################################################

from nose.plugins.attrib import attr
from simtk import unit, openmm

################################################################################
# Suppress matplotlib logging
################################################################################

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

################################################################################
# CONSTANTS
################################################################################

from openmmtools.constants import kB
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

################################################################################
# TESTS
################################################################################

@attr('travis')
def test_vacuum_hybrid_system():
    """Test the creation of vacuum hybrid systems."""
    iupac_name_pairs = [
        ('propane', '1-chloropropane'),
        ('benzene', 'phenol'),
        ('styrene', '2-phenylethanol'),
    ]

    from perses.tests.utils import create_vacuum_hybrid_system
    for old_iupac_name, new_iupac_name in iupac_name_pairs:
        topology_factory, hybrid_factory = create_vacuum_hybrid_system(old_iupac_name=old_iupac_name, new_iupac_name=new_iupac_name)

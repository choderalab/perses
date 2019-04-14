import unittest
import tempfile
import os, sys
import numpy as np
if sys.version_info >= (3, 0):
    from io import StringIO
else:
    from cStringIO import StringIO

ISTRAVIS = os.environ.get('TRAVIS', None) == 'true'

from perses.tests.utils import get_data_filename, Timer
from perses.forcefields import OEGAFFTemplateGenerator, generateTopologyFromOEMol

################################################################################
# Suppress matplotlib logging
################################################################################

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

################################################################################
# Tests
################################################################################

# TODO: Add tests for SystemGenerator

class TestOEGAFFTemplateGenerator(unittest.TestCase):
    def setUp(self):
        from openeye import oechem
        nmolecules = 5 # number of molecules to read
        ifs = oechem.oemolistream(get_data_filename("minidrugbank/MiniDrugBank_tripos.mol2"))
        self.oemols = list() # list of molecules to use
        for index in range(nmolecules):
            oemol = oechem.OEMol()
            oechem.OEReadMolecule(ifs, oemol)
            self.oemols.append(oemol)
        ifs.close()

        # Suppress DEBUG logging from various packages
        import logging
        for name in ['parmed', 'matplotlib']:
            logging.getLogger(name).setLevel(logging.WARNING)

    def test_create(self):
        """Test creation of an OEGAFFTemplateGenerator"""
        # Create an empty generator
        generator = OEGAFFTemplateGenerator()
        # Create a generator that knows about a few molecules
        generator = OEGAFFTemplateGenerator(oemols=self.oemols)
        # Create a generator that also has a database cache
        with tempfile.TemporaryDirectory() as tmpdirname:
            cache = os.path.join(tmpdirname, 'db.json')
            # Create a new database file
            generator = OEGAFFTemplateGenerator(oemols=self.oemols, cache=cache)
            del generator
            # Reopen it (with cache still empty)
            generator = OEGAFFTemplateGenerator(oemols=self.oemols, cache=cache)
            del generator

    def test_parameterize(self):
        """Test parameterizing molecules with OEGAFFTemplateGenerator"""
        for gaff_version in ['gaff', 'gaff2']:
            # Create a generator that knows about a few molecules
            generator = OEGAFFTemplateGenerator(oemols=self.oemols)
            # Create a ForceField
            from simtk.openmm.app import ForceField
            gaff_xml_filename = get_data_filename("{}.xml".format(gaff_version))
            forcefield = ForceField(gaff_xml_filename)
            # Register the template generator
            forcefield.registerTemplateGenerator(generator.generator)
            # Parameterize some molecules
            from simtk.openmm.app import NoCutoff
            from openmoltools.forcefield_generators import generateTopologyFromOEMol
            for oemol in self.oemols:
                topology = generateTopologyFromOEMol(oemol)
                with Timer() as t1:
                    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff)
                assert system.getNumParticles() == sum([1 for atom in oemol.GetAtoms()])
                # Molecule should now be cached
                with Timer() as t2:
                    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff)
                assert system.getNumParticles() == sum([1 for atom in oemol.GetAtoms()])
                assert (t2.interval < t1.interval)

    def test_add_oemols(self):
        """Test that OEMols can be added to OEGAFFTemplateGenerator later"""
        gaff_version = 'gaff'
        # Create a generator that does not know about any molecules
        generator = OEGAFFTemplateGenerator()
        # Create a ForceField
        from simtk.openmm.app import ForceField
        gaff_xml_filename = get_data_filename("{}.xml".format(gaff_version))
        forcefield = ForceField(gaff_xml_filename)
        # Register the template generator
        forcefield.registerTemplateGenerator(generator.generator)

        # Check that parameterizing a molecule fails
        oemol = self.oemols[0]
        from simtk.openmm.app import NoCutoff
        from openmoltools.forcefield_generators import generateTopologyFromOEMol
        try:
            # This should fail with an exception
            system = forcefield.createSystem(generateTopologyFromOEMol(oemol), nonbondedMethod=NoCutoff)
        except ValueError as e:
            # Exception 'No template found...' is expected
            assert str(e).startswith('No template found')

        # Now add the molecule to the generator and ensure parameterization passes
        generator.add_oemols(oemol)
        system = forcefield.createSystem(generateTopologyFromOEMol(oemol), nonbondedMethod=NoCutoff)
        assert system.getNumParticles() == sum([1 for atom in oemol.GetAtoms()])

        # Add multiple molecules, including repeats
        generator.add_oemols(self.oemols)

        # Ensure all molecules can be parameterized
        for oemol in self.oemols:
            system = forcefield.createSystem(generateTopologyFromOEMol(oemol), nonbondedMethod=NoCutoff)
            assert system.getNumParticles() == sum([1 for atom in oemol.GetAtoms()])

    def test_cache(self):
        """Test cache capability of OEGAFFTemplateGenerator"""
        from simtk.openmm.app import ForceField, NoCutoff
        gaff_version = 'gaff'
        gaff_xml_filename = get_data_filename("{}.xml".format(gaff_version))
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a generator that also has a database cache
            cache = os.path.join(tmpdirname, 'db.json')
            generator = OEGAFFTemplateGenerator(oemols=self.oemols, cache=cache)
            # Create a ForceField
            forcefield = ForceField(gaff_xml_filename)
            # Register the template generator
            forcefield.registerTemplateGenerator(generator.generator)
            # Parameterize the molecules
            for oemol in self.oemols:
                forcefield.createSystem(generateTopologyFromOEMol(oemol), nonbondedMethod=NoCutoff)

            # Check database contents
            def check_database(generator):
                db_entries = generator._db.all()
                nentries = len(db_entries)
                nmolecules = len(self.oemols)
                assert (nmolecules == nentries), \
                    "Expected {} entries but database only has {}\n db contents: {}".format(nmolecules, nentries, db_entries)

            check_database(generator)
            # Clean up, forcing closure of database
            del forcefield, generator

            # Create a generator that also uses the database cache but has no molecules
            print('Creating new generator with just cache...')
            generator = OEGAFFTemplateGenerator(cache=cache)
            # Check database contents
            check_database(generator)
            # Create a ForceField
            forcefield = ForceField(gaff_xml_filename)
            # Register the template generator
            forcefield.registerTemplateGenerator(generator.generator)
            # Parameterize the molecules; this should succeed
            for oemol in self.oemols:
                from openeye import oechem
                forcefield.createSystem(generateTopologyFromOEMol(oemol), nonbondedMethod=NoCutoff)

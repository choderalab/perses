"""
System generators that build an OpenMM System object from a Topology object.

"""

################################################################################
# LOGGER
################################################################################

import logging
_logger = logging.getLogger("perses.forcefields.system_generator")

################################################################################
# System generators
################################################################################

class SystemGenerator(object):
    """
    This is a utility class to generate OpenMM Systems from
    topology objects.

    Parameters
    ----------
    forcefields_to_use : list of string
        List of the names of ffxml files that will be used in system creation.
    forcefield_kwargs : dict of arguments to createSystem, optional
        Allows specification of various aspects of system creation.
    metadata : dict, optional
        Metadata associated with the SystemGenerator.
    barostat : MonteCarloBarostat, optional, default=None
        If provided, a matching barostat will be added to the generated system.
    oemols : list of openeye.oechem OEMol
        Additional molecules that should be parameterized on the fly by OEGAFFTemplateGenerator
    cache : filename or TinyDB instance
        JSON filename or TinyDB instance that can be used to cache parameterized small molecules by OEGAFFTemplateGenerator
    """

    def __init__(self, forcefields_to_use, forcefield_kwargs=None, metadata=None, barostat=None, oemols=None, cache=None):
        # Cache force fields and settings to use
        self._forcefield_xmls = forcefields_to_use
        self._forcefield_kwargs = forcefield_kwargs if forcefield_kwargs is not None else {}

        # Create and cache a ForceField object
        from simtk.openmm import app
        self._forcefield = app.ForceField(*self._forcefield_xmls)

        # Create and cache a residue template generator
        self._generator = None
        if oemols or cache:
            from .openmm_forcefield import OEGAFFTemplateGenerator
            self._generator = OEGAFFTemplateGenerator(oemols=oemols)
            self._forcefield.registerTemplateGenerator(self._generator.generator)

        # Ensure that center-of-mass motion removal is not added
        if 'removeCMMotion' not in self._forcefield_kwargs:
            self._forcefield_kwargs['removeCMMotion'] = False

        # Cache barostat information
        # TODO: Is this necessary?
        self._barostat = None
        if barostat is not None:
            pressure = barostat.getDefaultPressure()
            if hasattr(barostat, 'getDefaultTemperature'):
                temperature = barostat.getDefaultTemperature()
            else:
                temperature = barostat.getTemperature()
            frequency = barostat.getFrequency()
            self._barostat = (pressure, temperature, frequency)

    def get_forcefield(self):
        """
        Return the associated ForceField object.

        Returns
        -------
        forcefield : simtk.openmm.app.ForceField
            The current ForceField object.
        """
        return self._forcefield

    def build_system(self, new_topology):
        """
        Build a system from the new_topology, adding templates
        for the molecules in oemol_list

        Parameters
        ----------
        new_topology : simtk.openmm.app.Topology object
            The topology of the system

        Returns
        -------
        new_system : openmm.System
            A system object generated from the topology
        """
        import time
        _logger.info('Generating System...')
        timer_start = time.time()

        try:
            system = self._forcefield.createSystem(new_topology, **self._forcefield_kwargs)
        except Exception as e:
            # Capture information about failure to parameterize the system
            from simtk import unit
            import numpy as np
            nparticles = sum([1 for atom in new_topology.atoms()])
            positions = unit.Quantity(np.zeros([nparticles,3], np.float32), unit.angstroms)
            # Write PDB file of failed topology
            from simtk.openmm.app import PDBFile
            outfile = open('BuildSystem-failure.pdb', 'w')
            pdbfile = PDBFile.writeFile(new_topology, positions, outfile)
            outfile.close()
            msg = str(e)
            import traceback
            msg += traceback.format_exc()
            msg += "\n"
            msg += "PDB file written as 'BuildSystem-failure.pdb'"
            raise Exception(msg)

        # Add barostat if requested.
        if self._barostat is not None:
            from simtk import openmm
            import numpy as np
            MAXINT = np.iinfo(np.int32).max
            barostat = openmm.MonteCarloBarostat(*self._barostat)
            seed = np.random.randint(MAXINT)
            barostat.setRandomNumberSeed(seed)
            system.addForce(barostat)

        _logger.info('System generation took %.3f s' % (time.time() - timer_start))

        return system

    @property
    def ffxmls(self):
        return self._forcefield_xmls

    @property
    def forcefield(self):
        return self._forcefield

    @property
    def generator(self):
        return self._generator

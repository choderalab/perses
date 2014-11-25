__author__ = 'Patrick B. Grinaway'

import openeye.oechem as oechem
import gaff2xml
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as units
import joblib

class Calibrator(object):

    """
    Given a ligand system, generate new sets of charges
    representing fluorination states, as well as biases for the expanded ensemble simulation in implicit solvent.
    """

    def __init__(self, ligand_oemol, ligand_prmtop, ligand_inpcrd, openmm_platform='CPU'):
        """
        Initialize a Calibrator object with a ligand system and oemol. Set up the system, context, etc.
        using the specified platform. Does not perform any calibration

        Arguments
        ---------
        ligand_oemol : openeye.oechem.OEmol
            An OEMol object representing the unmodified ligand
        ligand_prmtop : simtk.openmm.app.AmberPrmtopFile
            An object representing the Amber topology of the ligand
        ligand_inpcrd : simtk.openmm.app.AmberInpcrdFile
            An object representing the Amber coordinates of the ligand
        openmm_platform : String
            The platform to use to calculate energies
        """
        self._ligand_oemol = ligand_oemol
        ligand_prmtop = app.AmberPrmtopFile("hoobajoo")
        self._ligand_system = ligand_prmtop.createSystem(constraints=app.HBonds,implicitSolvent=app.OBC2)
        temperature = 300.0 * units.kelvin
        timestep = 1.0 * units.femtoseconds
        collision_rate = 9.1 / units.picoseconds
        self._integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        platform = openmm.Platform.getPlatformByName(openmm_platform)
        self._context = openmm.Context(self._ligand_system, self._integrator, platform)



    def _get_charges(self):


    @property
    def potential_energy(self):
        state = self._context.getState(getEnergy=True)
        return state.getPotentialEnergy()






__author__ = 'Patrick B. Grinaway'

import simtk.openmm as openmm
import simtk.openmm.app as app
import gaff2xml
import openeye.oechem as oechem
from copy import deepcopy
import sklearn.externals.joblib.memory as memory

class FluorineScan(object):

    """
    Class for performing Fluorine scans, in which a ligand interacting with a receptor can change its hydrogens to fluorines. The probability
    of being in a particular fluorination state is given by a weight, set to maximize sampling of the tightest-binding variant.

    """


    def __init__(self, complex_system, complex_prmtop, fluorinatable_residue, ligand_oemol):
        """
        Initialize a FluorineScan object.

        Arguments
        ---------
        complex_system : simtk.openmm.System
            Object containing the complex system (ligand and receptor) to simulate
        complex_prmtop : simtk.openmm.app.AmberPrmtopFile
            Object useful for getting information about exceptions and topology
        fluorinatable_residue : String
            Name of residue (ligand) that can be fluorinated
        ligand_oemol : openeye.oechem.OEMol
            OEMol of ligand to be fluorine-scanned. Will be used to generate charges of variants.



        """

        self._reference_oemol = ligand_oemol
        self._fluorine_atomic_num = 9



    def _calibrate(self, atoms_to_fluorinate):
        """
        Generate the charges and weights for the variant with the requested atoms fluorinated.
        This method creates a copy of the reference_oemol to avoid changing it.

        Arguments
        ---------
        atoms_to_fluorinate : list of ints
            A list of the atoms to change to fluorines

        """
        #mol_to_change = deepcopy(self._reference_oemol)
        mol_to_change = oechem.OEMol()
        for atom_index in atoms_to_fluorinate:
            mol_to_change.GetAtom(atom_index).SetAtomicNum(self._fluorine_atomic_num)
        charges = 




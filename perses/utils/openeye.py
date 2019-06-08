"""

Utility functions for simulations using openeye toolkits

"""

__author__ = 'John D. Chodera'


from openeye import oechem
import simtk.unit as unit
import numpy as np

def extractPositionsFromOEMol(molecule,units=unit.angstrom):
    """
    Get a molecules coordinates from an openeye.oemol

    Parameters
    ----------
    molecule : openeye.oechem.OEMol object
    units : simtk.unit, default angstrom

    Returns
    -------
    positions : np.array
    """
    positions = unit.Quantity(np.zeros([molecule.NumAtoms(), 3], np.float32), units)
    coords = molecule.GetCoords()
    for index in range(molecule.NumAtoms()):
        positions[index,:] = unit.Quantity(coords[index], units)

    return positions

def giveOpenmmPositionsToOEMol(positions, molecule):
    """
    Replace OEMol positions with openmm format positions

    Parameters
    ----------
    positions : openmm.topology.positions
    molecule : openeye.oechem.OEMol object

    Returns
    -------
    molecule : openeye.oechem.OEMol
        molecule with updated positions

    """
    assert molecule.NumAtoms() == len(positions), "Number of openmm positions does not match number of atoms in OEMol object"
    coords = molecule.GetCoords()
    for key in coords.keys(): # openmm in nm, openeye in A
        coords[key] = (positions[key][0]/unit.angstrom,positions[key][1]/unit.angstrom,positions[key][2]/unit.angstrom)
    molecule.SetCoords(coords)

    return molecule


def OEMol_to_omm_ff(molecule, data_filename='data/gaff2.xml'):
    """
    Convert an openeye.oechem.OEMol to a openmm system, positions and topology

    Parameters
    ----------
    oemol : openeye.oechem.OEMol object
        input molecule to convert
    data_filename : str, default 'data/gaff2.xml'
        path to .xml forcefield file, default is gaff2.xml in perses package

    Return
    ------
    system : openmm.system
    positions : openmm.positions
    topology : openmm.topology

    """
    from perses.rjmc import topology_proposal
    from openmoltools import forcefield_generators
    from perses.utils.data import get_data_filename

    gaff_xml_filename = get_data_filename(data_filename)
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])
    topology = forcefield_generators.generateTopologyFromOEMol(molecule)
    system = system_generator.build_system(topology)
    positions = extractPositionsFromOEMol(molecule)

    return system, positions, topology

def createSystemFromIUPAC(iupac_name):
    """
    Create an openmm system out of an oemol

    Parameters
    ----------
    iupac_name : str
        IUPAC name

    Returns
    -------
    molecule : openeye.oechem.OEMol
        OEMol molecule
    system : openmm.System object
        OpenMM system
    positions : [n,3] np.array of floats
        Positions
    topology : openmm.app.Topology object
        Topology
    """

    # Create OEMol
    molecule = iupac_to_oemol(iupac_name)

    # generate openmm system, positions and topology
    system, positions, topology = OEMol_to_omm_ff(molecule)

    return (molecule, system, positions, topology)

def createSystemFromSMILES(smiles,title='MOL'):
    """
    Create an openmm system from a smiles string

    Parameters
    ----------
    smiles : str
        smiles string of molecule

    Returns
    -------
    molecule : openeye.oechem.OEMol
        OEMol molecule
    system : openmm.System object
        OpenMM system
    positions : [n,3] np.array of floats
        Positions
    topology : openmm.app.Topology object
        Topology
    """
    # clean up smiles string
    from perses.utils.smallmolecules import sanitizeSMILES
    smiles = sanitizeSMILES([smiles])
    smiles = smiles[0]

    # Create OEMol
    molecule = smiles_to_oemol(smiles,title=title)

    # generate openmm system, positions and topology
    system, positions, topology = OEMol_to_omm_ff(molecule)

    return (molecule, system, positions, topology)

def describe_oemol(mol):
    """
    Render the contents of an OEMol to a string.

    Parameters
    ----------
    mol : OEMol
        Molecule to describe

    Returns
    -------
    description : str
        The description
    """
    #TODO this needs a test
    description = ""
    description += "ATOMS:\n"
    for atom in mol.GetAtoms():
        description += "%8d %5s %5d\n" % (atom.GetIdx(), atom.GetName(), atom.GetAtomicNum())
    description += "BONDS:\n"
    for bond in mol.GetBonds():
        description += "%8d %8d\n" % (bond.GetBgnIdx(), bond.GetEndIdx())
    return description

def createOEMolFromSDF(sdf_filename, index=0):
    """
    Load an SDF file into an OEMol. Since SDF files can contain multiple molecules, an index can be provided as well.

    Parameters
    ----------
    sdf_filename : str
        The name of the SDF file
    index : int, default 0
        The index of the molecule in the SDF file

    Returns
    -------
    mol : openeye.oechem.OEMol object
        The loaded oemol object
    """
    #TODO this needs a test
    ifs = oechem.oemolistream()
    ifs.open(sdf_filename)
    # get the list of molecules
    mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]
    # we'll always take the first for now
    mol_to_return = mol_list[index]
    return mol_to_return

def calculate_mol_similarity(molA, molB):
    """
    Function to calculate the similarity between two oemol objects
    should be used to utils/openeye.py or openmoltools
    :param molA: oemol object of molecule A
    :param molB: oemol object of molecule B
    :return: float, tanimoto score of the two molecules, between 0 and 1
    """
    fpA = oegraphsim.OEFingerPrint()
    fpB = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeFP(fpA, molA, oegraphsim.OEFPType_MACCS166)
    oegraphsim.OEMakeFP(fpB, molB, oegraphsim.OEFPType_MACCS166)

    return oegraphsim.OETanimoto(fpA, fpB)

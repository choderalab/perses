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

def createOEMolFromIUPAC(iupac_name,max_confs=1):
    """
    Generate an OEMol object using an IUPAC code

    Parameters
    ----------
    iupac_name : str
        standard IUPAC name of a molecule
    max_confs : int, default 1
        maximum number of conformers to generate

    Returns
    -------
    molecule : openeye.oechem.OEMol
        OEMol object of the molecule
    """
    from openeye import oeiupac, oeomega

    # Create molecule.
    molecule = oechem.OEMol()
    oeiupac.OEParseIUPACName(molecule, iupac_name)

    # Set title.
    molecule.SetTitle(iupac_name)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(molecule)

    # Create atom names.
    oechem.OETriposAtomNames(molecule)

    # Create bond types
    oechem.OETriposBondTypeNames(molecule)

    # Assign geometry
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(max_confs)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)
    omega(molecule)

    return molecule

def createOEMolFromSMILES(smiles, title='MOL',max_confs=1):
    """
    Generate an oemol from a SMILES string

    Parameters
    ----------
    smiles : str
        SMILES string of molecule
    title : str, default 'MOL'
        title of OEMol molecule
    max_confs : int, default 1
        maximum number of conformers to generate

    Returns
    -------
    molecule : openeye.oechem.OEMol
        OEMol object of the molecule
    """
    from openeye import oeiupac, oeomega

    # Create molecule
    molecule = oechem.OEMol()
    oechem.OESmilesToMol(molecule, smiles)

    # Set title.
    molecule.SetTitle(title)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(molecule)

    # Create atom names.
    oechem.OETriposAtomNames(molecule)

    # Assign geometry
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(max_confs)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)
    omega(molecule)

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

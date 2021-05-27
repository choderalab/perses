"""

Utility functions for simulations using openeye toolkits

"""

__author__ = 'John D. Chodera'


from simtk import unit
from simtk.openmm import app
import simtk.unit as unit
import numpy as np
import logging

logging.basicConfig(level=logging.NOTSET)
_logger = logging.getLogger("utils.openeye")
_logger.setLevel(logging.INFO)

def system_generator_wrapper(oemols,
                            barostat = None,
                            forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                            forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus},
                            nonperiodic_forcefield_kwargs = {'nonbondedMethod': app.NoCutoff},
                            small_molecule_forcefield = 'gaff-2.11',
                            **kwargs
                            ):
    """
    make a system generator (vacuum) for a small molecule

    Parameters
    ----------
    oemols : list of openeye.oechem.OEMol
        oemols
    barostat : openmm.MonteCarloBarostat, default None
        barostat
    forcefield_files : list of str
        pointers to protein forcefields and solvent
    forcefield_kwargs : dict
        dict of forcefield_kwargs
    nonperiodic_forcefield_kwargs : dict
        dict of args for non-periodic system
    small_molecule_forcefield : str
        pointer to small molecule forcefield to use

    Returns
    -------
    system_generator : openmmforcefields.generators.SystemGenerator
    """
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import SystemGenerator
    from openeye import oechem

    system_generator = SystemGenerator(forcefields = forcefield_files, barostat=barostat, forcefield_kwargs=forcefield_kwargs,nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                                         small_molecule_forcefield = small_molecule_forcefield, molecules=[Molecule.from_openeye(oemol) for oemol in oemols], cache=None)
    return system_generator

def smiles_to_oemol(smiles, title='MOL', max_confs=1):
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
    from openeye import oeomega, oechem

    # Create molecule
    molecule = oechem.OEMol()
    oechem.OESmilesToMol(molecule, smiles)

    # create unique atom names
    if len([atom.GetName() for atom in molecule.GetAtoms()]) > len(set([atom.GetName() for atom in molecule.GetAtoms()])):
        # the atom names are not unique
        molecule = generate_unique_atom_names(molecule)
    else:
        pass

    # Set title.
    molecule.SetTitle(title)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)
    oechem.OEAssignHybridization(molecule)
    oechem.OEAddExplicitHydrogens(molecule)
    oechem.OEPerceiveChiral(molecule)

    # Create atom names.
    oechem.OETriposAtomNames(molecule)
    oechem.OETriposBondTypeNames(molecule)

    # perceive chirality before attempting omega geometry proposal
    assert oechem.OEPerceiveChiral(molecule), f"chirality perception failed"

    # Assign geometry
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(max_confs)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)

    omega(molecule)
    return molecule


def iupac_to_oemol(iupac, title='MOL', max_confs=1):
    """
    Generate an oemol from an IUPAC name
    Parameters
    ----------
    iupac : str
        iupac name of molecule
    title : str, default 'MOL'
        title of OEMol molecule
    max_confs : int, default 1
        maximum number of conformers to generate
    Returns
    -------
    molecule : openeye.oechem.OEMol
        OEMol object of the molecule
    """
    from openeye import oeiupac, oeomega, oechem

    # Create molecule
    molecule = oechem.OEMol()
    oeiupac.OEParseIUPACName(molecule, iupac)

    # Set title.
    molecule.SetTitle(title)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)
    oechem.OEAssignHybridization(molecule)
    oechem.OEAddExplicitHydrogens(molecule)
    oechem.OEPerceiveChiral(molecule)

    # Create atom names.
    oechem.OETriposAtomNames(molecule)
    oechem.OETriposBondTypeNames(molecule)

    # Assign geometry
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(max_confs)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)
    omega(molecule)
    return molecule


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


def OEMol_to_omm_ff(molecule, system_generator):
    """
    Convert an openeye.oechem.OEMol to a openmm system, positions and topology

    Parameters
    ----------
    oemol : openeye.oechem.OEMol object
        input molecule to convert
    system_generator : openmmforcefields.generators.SystemGenerator

    Returns
    -------
    system : openmm.system
    positions : openmm.positions
    topology : openmm.topology

    """
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    topology = generateTopologyFromOEMol(molecule)
    system = system_generator.create_system(topology)
    positions = extractPositionsFromOEMol(molecule)

    return system, positions, topology

def createSystemFromIUPAC(iupac_name, title="MOL", **system_generator_kwargs):
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
    from openeye import oechem
    from openmoltools.openeye import generate_conformers

    # Create OEMol
    # TODO write our own of this function so we can be
    # sure of the oe flags that are being used
    molecule = iupac_to_oemol(iupac_name, title=title)

    molecule = generate_conformers(molecule, max_confs=1)

    system_generator = system_generator_wrapper([molecule], **system_generator_kwargs)

    # generate openmm system, positions and topology
    system, positions, topology = OEMol_to_omm_ff(molecule, system_generator)

    return (molecule, system, positions, topology)


def createSystemFromSMILES(smiles,title='MOL', **system_generator_kwargs):
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
    molecule = smiles_to_oemol(smiles, title=title)
    system_generator = system_generator_wrapper([molecule], **system_generator_kwargs)

    # generate openmm system, positions and topology
    system, positions, topology = OEMol_to_omm_ff(molecule, system_generator)

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


def createOEMolFromSDF(sdf_filename, index=0, add_hydrogens=True, allow_undefined_stereo=False):
    """
    # TODO change this to return a list of all the mols if required
    Load an SDF file into an OEMol. Since SDF files can contain multiple
    molecules, an index can be provided as well.

    Parameters
    ----------
    sdf_filename : str
        The name of the SDF file
    index : int, default 0
        The index of the molecule in the SDF file
    allow_undefined_stereo : bool, default=False
        wether to skip stereo perception

    Returns
    -------
    mol : openeye.oechem.OEMol object
        The loaded oemol object
    """
    from openeye import oechem

    # TODO this needs a test
    ifs = oechem.oemolistream()
    ifs.open(sdf_filename)
    # get the list of molecules
    mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]
    # we'll always take the first for now

    # pick out molecule of interest
    molecule = mol_list[index]

    # Generate unique atom names
    if len([atom.GetName() for atom in molecule.GetAtoms()]) > len(set([atom.GetName() for atom in molecule.GetAtoms()])):
        molecule = generate_unique_atom_names(molecule)

    # Assign aromaticity and hydrogens.
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)
    oechem.OEAssignHybridization(molecule)
    if add_hydrogens:
        oechem.OEAddExplicitHydrogens(molecule)
    oechem.OEPerceiveChiral(molecule)

    # perceive chirality
    if not allow_undefined_stereo:
        assert oechem.OE3DToInternalStereo(molecule), f"the stereochemistry perception from 3D coordinates failed"
        assert not has_undefined_stereocenters(molecule), f"there is an atom with an undefined stereochemistry"

    return molecule


def calculate_mol_similarity(molA, molB):
    """
    Function to calculate the similarity between two oemol objects
    should be used to utils/openeye.py or openmoltools
    :param molA: oemol object of molecule A
    :param molB: oemol object of molecule B
    :return: float, tanimoto score of the two molecules, between 0 and 1
    """
    from openeye import oegraphsim

    fpA = oegraphsim.OEFingerPrint()
    fpB = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeFP(fpA, molA, oegraphsim.OEFPType_MACCS166)
    oegraphsim.OEMakeFP(fpB, molB, oegraphsim.OEFPType_MACCS166)

    return oegraphsim.OETanimoto(fpA, fpB)


def createSMILESfromOEMol(molecule):
    from openeye import oechem

    smiles = oechem.OECreateSmiString(molecule,
                                      oechem.OESMILESFlag_DEFAULT |
                                      oechem.OESMILESFlag_Hydrogens)
    return smiles


def generate_unique_atom_names(molecule):
    """
    Check if an oemol has unique atom names, and if not, then assigns them

    Parameters
    ----------
    molecule : openeye.oechem.OEMol object
        oemol object to check
    Returns
    -------
    molecule : openeye.oechem.OEMol object
        oemol, either unchanged if atom names are
        already unique, or newly generated atom names
    """
    from openeye import oechem

    atom_names = []

    atom_count = 0
    for atom in molecule.GetAtoms():
        atom_names.append(atom.GetName())
        atom_count += 1

    if len(set(atom_names)) == atom_count:
        # one name per atom therefore unique
        _logger.info(f'molecule {molecule.GetTitle()} \
                     has unique atom names already')
        return molecule
    else:
        # generating new atom names
        from collections import defaultdict
        try:
            from openmm.app.element import Element
        except ModuleNotFoundError:  # <=7.5.0
            from simtk.openmm.app import Element
        _logger.info(f'molecule {molecule.GetTitle()} \
                     does not have unique atom names. Generating now...')
        element_counts = defaultdict(int)
        for atom in molecule.GetAtoms():
            element = Element.getByAtomicNumber(atom.GetAtomicNum())
            element_counts[element._symbol] += 1
            name = element._symbol + str(element_counts[element._symbol])
            atom.SetName(name)
        return molecule


def has_undefined_stereocenters(mol):
    """
    Check that _if_ a molecule has a stereocenter,
    the stereochemistry is defined
    if no stereocenter then will return False too

    Parameters
    ----------
    molecule : openeye.oechem.OEMol object
        oemol object to check

    Returns
    -------
    bool : True if undefined Stereochemistry
           False if no stereochemistry or all stereocenter's are labelled
    """
    from openeye import oechem

    assert oechem.OEPerceiveChiral(mol), f"chirality perception failed"

    for atom in mol.GetAtoms():
        if atom.IsChiral():
            if not atom.HasStereoSpecified():
                return True  # we have a stereocenter with no stereochemistry!
    for bond in mol.GetBonds():
        if bond.IsChiral():
            if not bond.HasStereoSpecified():
                return True  # we have a geometric isomer that isn't specified!
    return False  # nothing bad found


def generate_expression(list):
    """Turns a list of strings into an oechem atom or bond expression
    This allows us to pass in matching expressions in the input .yaml
    Note: strings are case sensitive

    >>> atom_expr = generate_expression("Hybridization", "IntType")

    Parameters
    ----------
    list : list of strings
        List of strings

    Returns
    -------
    integer
        Integer that openeye magically understands for matching expressions

    """
    from openeye import oechem

    total_expr = 0

    for string in list:
        try:
            expr = getattr(oechem, f'OEExprOpts_{string}')
        except AttributeError:
            raise Exception(f'{string} not recognised, no expression of oechem.OEExprOpts_{string}.\
            This is case sensitive, so please check carefully and see , \
            https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemConstants/OEExprOpts.html\
            for options')
        # doing bitwise OR check
        total_expr = total_expr | expr

    return total_expr


def get_scaffold(molecule, adjustHcount=False):
    """
    Takes an openeye.oechem.oemol and returns
    an openeye.oechem.oemol of the scaffold

    The scaffold is a molecule where all the atoms that are not in rings, and are not linkers between rings.
    double bonded atoms exo to a ring are included as ring atoms

    This function has been completely taken from openeye's extractscaffold.py script
    https://docs.eyesopen.com/toolkits/python/oechemtk/oechem_examples/oechem_example_extractscaffold.html#section-example-oechem-extractscaffold
    Parameters
    ----------
    mol : openeye.oechem.oemol
        entire molecule to get the scaffold of
    adjustHcount : bool, default=False
        add/remove hydrogens to satisfy valence of scaffold


    Returns
    -------
    openeye.oechem.oemol
        scaffold oemol of the input mol. New oemol.
    """
    from openeye import oechem

    def TraverseForRing(visited, atom):
        visited.add(atom.GetIdx())

        for nbor in atom.GetAtoms():
            if nbor.GetIdx() not in visited:
                if nbor.IsInRing():
                    return True

                if TraverseForRing(visited, nbor):
                    return True

        return False

    def DepthFirstSearchForRing(root, nbor):
        visited = set()
        visited.add(root.GetIdx())

        return TraverseForRing(visited, nbor)

    class IsInScaffold(oechem.OEUnaryAtomPred):
        def __call__(self, atom):
            if atom.IsInRing():
                return True

            count = 0
            for nbor in atom.GetAtoms():
                if DepthFirstSearchForRing(atom, nbor):
                    count += 1

            return count > 1

    dst = oechem.OEMol()
    pred = IsInScaffold()

    oechem.OESubsetMol(dst, molecule, pred, adjustHcount)
    return dst

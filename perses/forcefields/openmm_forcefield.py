import json
from mdtraj.utils.delay_import import import_
from mdtraj.utils import enter_temp_directory

################################################################################
# LOGGER
################################################################################

import logging
_logger = logging.getLogger("perses.forcefields.openmm_forcefield")

################################################################################
# Force field generators
################################################################################

def getoutput(cmd):
    """Compatibility function to substitute deprecated commands.getoutput in Python2.7"""
    import subprocess
    try:
        out = subprocess.getoutput(cmd)
    except AttributeError:
        out = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT,
                               stdout=subprocess.PIPE).communicate()[0]
    try:
        return str(out.decode())
    except:
        return str(out)

def generateTopologyFromOEMol(molecule):
    """
    Generate an OpenMM Topology object from an OEMol molecule.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule from which a Topology object is to be generated.

    Returns
    -------
    topology : simtk.openmm.app.Topology
        The Topology object generated from `molecule`.

    """
    # Create a Topology object with one Chain and one Residue.
    from simtk.openmm.app import Topology
    topology = Topology()
    chain = topology.addChain()
    resname = molecule.GetTitle()
    residue = topology.addResidue(resname, chain)

    # Create atoms in the residue.
    from simtk.openmm.app.element import Element
    for atom in molecule.GetAtoms():
        name = atom.GetName()
        element = Element.getByAtomicNumber(atom.GetAtomicNum())
        atom = topology.addAtom(name, element, residue)

    # Create bonds.
    atoms = { atom.name : atom for atom in topology.atoms() }
    for bond in molecule.GetBonds():
        topology.addBond(atoms[bond.GetBgn().GetName()], atoms[bond.GetEnd().GetName()])

    return topology

def _writeMolecule(molecule, output_filename, standardize=True):
    """
    Write the molecule to a file.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to write (will be modified by writer).
    output_filename : str
        The filename of file to be written; type is autodetected by extension.
    standardize : bool, optional, default=True
        Standardize molecular properties such as atom names in the output file.

    """
    from openmoltools.openeye import molecule_to_mol2
    molecule_to_mol2(molecule, tripos_mol2_filename=output_filename, conformer=0, residue_name=molecule.GetTitle(), standardize=standardize)

def generateResidueTemplate(molecule, residue_atoms=None, normalize=True, gaff_version='gaff'):
    """
    Generate an residue template for simtk.openmm.app.ForceField using GAFF and AM1-BCC ELF10 charges.

    This requires the OpenEye toolkit.

    Parameters
    ----------
    molecule : openeye.oechem.OEMol
        The molecule to be parameterized.
        The molecule must have explicit hydrogens.
        Net charge will be inferred from the net formal charge on each molecule.
        Partial charges will be determined automatically using oequacpac and canonical AM1-BCC charging rules.
    residue_atomset : set of OEAtom, optional, default=None
        If not None, only the atoms in this set will be used to construct the residue template
    normalize : bool, optional, default=True
        If True, normalize the molecule by checking aromaticity, adding
        explicit hydrogens, and renaming by IUPAC name.
    gaff_version : str, default = 'gaff'
        One of ['gaff', 'gaff2']; selects which atom types to use.

    Returns
    -------
    template : simtk.openmm.app.forcefield._TemplateData
        Residue template for ForceField using atom types and parameters from `gaff.xml` or `gaff2.xml`.
    additional_parameters_ffxml : str
        Contents of ForceField `ffxml` file defining additional parameters from parmchk(2).

    Notes
    -----
    The residue template will be named after the molecule title.
    This method preserves stereochemistry during AM1-BCC charge parameterization.
    Atom names in molecules will be assigned Tripos atom names if any are blank or not unique.

    """
    from openeye import oechem

    # Make a copy of the molecule so we don't modify the original
    molecule = oechem.OEMol(molecule)

    # OpenMM requires every residue have a globally unique (but arbitrary) name, so generate one.
    from uuid import uuid4
    template_name = molecule.GetTitle() + '-' + str(uuid4())

    # Perform soem normalization on the molecule.
    oechem.OEFindRingAtomsAndBonds(molecule)
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)

    # Generate unique atom names
    oechem.OETriposAtomNames(molecule)

    # Compute net formal charge.
    oechem.OEAssignFormalCharges(molecule)
    net_charge = sum([ atom.GetFormalCharge() for atom in molecule.GetAtoms() ])

    # Generate canonical AM1-BCC ELF10 charges
    from openeye import oequacpac
    smiles = oechem.OEMolToSmiles(molecule)
    oequacpac.OEAssignCharges(molecule, oequacpac.OEAM1BCCELF10Charges())

    # Set title to something that antechamber can handle
    molecule.SetTitle('MOL')

    # Geneate a single conformation
    from openeye import oeomega
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(True)
    omega(molecule)

    # Create temporary directory for running antechamber.
    import tempfile
    import os
    tmpdir = tempfile.mkdtemp()
    prefix = 'molecule'
    input_mol2_filename = os.path.join(tmpdir, prefix + '.tripos.mol2')
    gaff_mol2_filename = os.path.join(tmpdir, prefix + '.gaff.mol2')
    frcmod_filename = os.path.join(tmpdir, prefix + '.frcmod')

    # Write Tripos mol2 file for input into antechamber
    with oechem.oemolostream(input_mol2_filename) as ofs:
        oechem.OEWriteMolecule(ofs, molecule)

    # Parameterize the molecule with antechamber.
    run_antechamber(template_name, input_mol2_filename, charge_method=None, net_charge=net_charge, gaff_mol2_filename=gaff_mol2_filename, frcmod_filename=frcmod_filename, gaff_version=gaff_version)

    # Read the resulting GAFF mol2 file as a ParmEd structure
    from openeye import oechem
    ifs = oechem.oemolistream(gaff_mol2_filename)
    ifs.SetFlavor(oechem.OEFormat_MOL2, oechem.OEIFlavor_MOL2_DEFAULT | oechem.OEIFlavor_MOL2_M2H | oechem.OEIFlavor_MOL2_Forcefield)
    m2h = True
    oechem.OEReadMolecule(ifs, molecule)
    ifs.close()

    # If residue_atoms = None, add all atoms to the residues
    if residue_atoms == None:
        residue_atoms = [ atom for atom in molecule.GetAtoms() ]

    # Modify partial charges so that charge on residue atoms is integral.
    residue_charge = 0.0
    sum_of_absolute_charge = 0.0
    for atom in residue_atoms:
        charge = atom.GetPartialCharge()
        residue_charge += charge
        sum_of_absolute_charge += abs(charge)
    excess_charge = residue_charge - net_charge
    if sum_of_absolute_charge == 0.0:
        sum_of_absolute_charge = 1.0
    for atom in residue_atoms:
        charge = atom.GetPartialCharge()
        atom.SetPartialCharge( charge + excess_charge * (abs(charge) / sum_of_absolute_charge) )

    # Create residue template.
    from simtk.openmm.app import ForceField, Element
    template = ForceField._TemplateData(template_name)
    for (index, atom) in enumerate(molecule.GetAtoms()):
        atomname = atom.GetName()
        typename = atom.GetType()
        element = Element.getByAtomicNumber(atom.GetAtomicNum())
        charge = atom.GetPartialCharge()
        parameters = { 'charge' : charge }
        atom_template = ForceField._TemplateAtomData(atomname, typename, element, parameters)
        template.atoms.append(atom_template)
    for bond in molecule.GetBonds():
        if (bond.GetBgn() in residue_atoms) and (bond.GetEnd() in residue_atoms):
            template.addBondByName(bond.GetBgn().GetName(), bond.GetEnd().GetName())
        elif (bond.GetBgn() in residue_atoms) and (bond.GetEnd() not in residue_atoms):
            template.addExternalBondByName(bond.GetBgn().GetName())
        elif (bond.GetBgn() not in residue_atoms) and (bond.GetEnd() in residue_atoms):
            template.addExternalBondByName(bond.GetEnd().GetName())

    # Generate additional parameters, if needed
    # TODO: Do we have to make sure that we don't duplicate existing parameters already loaded in the forcefield?
    from inspect import signature # use introspection to support multiple parmed versions
    from io import StringIO
    leaprc = StringIO('parm = loadamberparams %s' % frcmod_filename)
    import parmed
    params = parmed.amber.AmberParameterSet.from_leaprc(leaprc)
    kwargs = {}
    if 'remediate_residues' in signature(parmed.openmm.OpenMMParameterSet.from_parameterset).parameters:
        kwargs['remediate_residues'] = False
    params = parmed.openmm.OpenMMParameterSet.from_parameterset(params, **kwargs)
    ffxml = StringIO()
    kwargs = {}
    if 'write_unused' in signature(params.write).parameters:
        kwargs['write_unused'] = True
    params.write(ffxml, **kwargs)
    additional_parameters_ffxml = ffxml.getvalue()

    return template, additional_parameters_ffxml


def run_antechamber(molecule_name, input_filename, charge_method="bcc", net_charge=None, gaff_mol2_filename=None, frcmod_filename=None,
    input_format='mol2', resname=False, log_debug_output=False, gaff_version = 'gaff'):
    """Run AmberTools antechamber and parmchk2 to create GAFF mol2 and frcmod files.

    Parameters
    ----------
    molecule_name : str
        Name of the molecule to be parameterized, will be used in output filenames.
    ligand_filename : str
        The molecule to be parameterized.  Must be tripos mol2 format.
    charge_method : str, optional
        If not None, the charge method string will be passed to Antechamber.
    net_charge : int, optional
        If not None, net charge of the molecule to be parameterized.
        If None, Antechamber sums up partial charges from the input file.
    gaff_mol2_filename : str, optional, default=None
        Name of GAFF mol2 filename to output.  If None, uses local directory
        and molecule_name
    frcmod_filename : str, optional, default=None
        Name of GAFF frcmod filename to output.  If None, uses local directory
        and molecule_name
    input_format : str, optional, default='mol2'
        Format specifier for input file to pass to antechamber.
    resname : bool, optional, default=False
        Set the residue name used within output files to molecule_name
    log_debug_output : bool, optional, default=False
        If true, will send output of tleap to logger.
    gaff_version : str, default = 'gaff'
        One of ['gaff', 'gaff2']; selects which atom types to use.

    Returns
    -------
    gaff_mol2_filename : str
        GAFF format mol2 filename produced by antechamber
    frcmod_filename : str
        Amber frcmod file produced by prmchk
    """
    utils = import_("openmoltools.utils")
    ext = utils.parse_ligand_filename(input_filename)[1]

    if not gaff_version in ['gaff', 'gaff2']:
        raise Exception("Error: gaff_version must be one of 'gaff' or 'gaff2'")

    if gaff_mol2_filename is None:
        gaff_mol2_filename = molecule_name + '.gaff.mol2'
    if frcmod_filename is None:
        frcmod_filename = molecule_name + '.frcmod'

    #Build absolute paths for input and output files
    import os
    gaff_mol2_filename = os.path.abspath( gaff_mol2_filename )
    frcmod_filename = os.path.abspath( frcmod_filename )
    input_filename = os.path.abspath( input_filename )

    def read_file_contents(filename):
        infile = open(filename, 'r')
        contents = infile.read()
        infile.close()
        return contents

    #Use temporary directory context to do this to avoid issues with spaces in filenames, etc.
    with enter_temp_directory():
        local_input_filename = 'in.' + input_format
        import shutil
        shutil.copy( input_filename, local_input_filename )

        # Run antechamber.
        #verbosity = 2 # verbose
        verbosity = 0 # brief
        cmd = "antechamber -i %(local_input_filename)s -fi %(input_format)s -o out.mol2 -fo mol2 -s %(verbosity)d -at %(gaff_version)s" % vars()
        if charge_method is not None:
            cmd += ' -c %s' % charge_method
        if net_charge is not None:
            cmd += ' -nc %d' % net_charge
        if resname:
            cmd += ' -rn %s' % molecule_name

        if log_debug_output: logger.debug(cmd)
        output = getoutput(cmd)
        import os
        if not os.path.exists('out.mol2'):
            msg  = "antechamber failed to produce output mol2 file\n"
            msg += "command: %s\n" % cmd
            msg += "output:\n"
            msg += 8 * "----------" + '\n'
            msg += output
            msg += 8 * "----------" + '\n'
            msg += "input mol2:\n"
            msg += 8 * "----------" + '\n'
            msg += read_file_contents(local_input_filename)
            msg += 8 * "----------" + '\n'
            raise Exception(msg)
        if log_debug_output: logger.debug(output)

        # Run parmchk.
        cmd = "parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s %s" % gaff_version
        if log_debug_output: logger.debug(cmd)
        output = getoutput(cmd)
        if not os.path.exists('out.frcmod'):
            msg  = "parmchk2 failed to produce output frcmod file\n"
            msg += "command: %s\n" % cmd
            msg += "output:\n"
            msg += 8 * "----------" + '\n'
            msg += output
            msg += 8 * "----------" + '\n'
            msg += "input mol2:\n"
            msg += 8 * "----------" + '\n'
            msg += read_file_contents('out.mol2')
            msg += 8 * "----------" + '\n'
            raise Exception(msg)
        if log_debug_output: logger.debug(output)
        check_for_errors(output)

        #Copy back
        shutil.copy( 'out.mol2', gaff_mol2_filename )
        shutil.copy( 'out.frcmod', frcmod_filename )

    return gaff_mol2_filename, frcmod_filename

def check_for_errors( outputtext, other_errors = None, ignore_errors = None ):
    """Check AMBER package output for the string 'ERROR' (upper or lowercase) and (optionally) specified other strings and raise an exception if it is found (to avoid silent failures which might be noted to log but otherwise ignored).

    Parameters
    ----------
    outputtext : str
        String listing output text from an (AMBER) command which should be checked for errors.
    other_errors : list(str), default None
        If specified, provide strings for other errors which will be chcked for, such as "improper number of arguments", etc.
    ignore_errors: list(str), default None
        If specified, AMBER output lines containing errors but also containing any of the specified strings will be ignored (because, for example, AMBER issues an "ERROR" for non-integer charges in some cases when only a warning is needed).

    Notes
    -----
    If error(s) are found, raise a RuntimeError and attept to print the appropriate errors from the processed text."""
    lines = outputtext.split('\n')
    error_lines = []
    for line in lines:
        if 'ERROR' in line.upper():
            error_lines.append( line )
        if not other_errors == None:
            for err in other_errors:
                if err.upper() in line.upper():
                    error_lines.append( line )

    if not ignore_errors == None and len(error_lines)>0:
        new_error_lines = []
        for ign in ignore_errors:
            ignore = False
            for err in error_lines:
                if ign in err:
                    ignore = True
            if not ignore:
                new_error_lines.append( err )
        error_lines = new_error_lines

    if len(error_lines) > 0:
        print("Unexpected errors encountered running AMBER tool. Offending output:")
        for line in error_lines: print(line)
        raise(RuntimeError("Error encountered running AMBER tool. Exiting."))

    return

def run_tleap(molecule_name, gaff_mol2_filename, frcmod_filename, prmtop_filename=None, inpcrd_filename=None, log_debug_output=False, leaprc='leaprc.gaff'):
    """Run AmberTools tleap to create simulation files for AMBER

    Parameters
    ----------
    molecule_name : str
        The name of the molecule
    gaff_mol2_filename : str
        GAFF format mol2 filename produced by antechamber
    frcmod_filename : str
        Amber frcmod file produced by prmchk
    prmtop_filename : str, optional, default=None
        Amber prmtop file produced by tleap, defaults to molecule_name
    inpcrd_filename : str, optional, default=None
        Amber inpcrd file produced by tleap, defaults to molecule_name
    log_debug_output : bool, optional, default=False
        If true, will send output of tleap to logger.
    leaprc : str, optional, default = 'leaprc.gaff'
        Optionally, specify alternate leaprc to use, such as `leaprc.gaff2`

    Returns
    -------
    prmtop_filename : str
        Amber prmtop file produced by tleap
    inpcrd_filename : str
        Amber inpcrd file produced by tleap
    """
    if prmtop_filename is None:
        prmtop_filename = "%s.prmtop" % molecule_name
    if inpcrd_filename is None:
        inpcrd_filename = "%s.inpcrd" % molecule_name

    #Get absolute paths for input/output
    import os
    gaff_mol2_filename = os.path.abspath( gaff_mol2_filename )
    frcmod_filename = os.path.abspath( frcmod_filename )
    prmtop_filename = os.path.abspath( prmtop_filename )
    inpcrd_filename = os.path.abspath( inpcrd_filename )

    #Work in a temporary directory, on hard coded filenames, to avoid any issues AMBER may have with spaces and other special characters in filenames
    with enter_temp_directory():
        import shutil
        shutil.copy( gaff_mol2_filename, 'file.mol2' )
        shutil.copy( frcmod_filename, 'file.frcmod' )

        tleap_input = """
    source oldff/leaprc.ff99SB
    source %s
    LIG = loadmol2 file.mol2
    check LIG
    loadamberparams file.frcmod
    saveamberparm LIG out.prmtop out.inpcrd
    quit

""" % leaprc

        file_handle = open('tleap_commands', 'w')
        file_handle.writelines(tleap_input)
        file_handle.close()

        cmd = "tleap -f %s " % file_handle.name
        if log_debug_output: logger.debug(cmd)

        output = getoutput(cmd)
        if log_debug_output: logger.debug(output)

        check_for_errors( output, other_errors = ['Improper number of arguments'] )

        #Copy back target files
        shutil.copy( 'out.prmtop', prmtop_filename )
        shutil.copy( 'out.inpcrd', inpcrd_filename )

    return prmtop_filename, inpcrd_filename

class OEGAFFTemplateGenerator(object):
    """
    OpenMM ForceField residue template generator for GAFF/AM1-BCC using pre-cached OpenEye toolkit OEMols.

    Examples
    --------

    Create a template generator for GAFF for a single OEMol and register it with ForceField:

    >>> from openmoltools.forcefield_generators import OEGAFFTemplateGenerator
    >>> template_generator = OEGAFFTemplateGenerator(oemols=oemol)
    >>> from simtk.openmm.app import ForceField
    >>> forcefield = ForceField('gaff.xml', 'amber14-all.xml', 'tip3p.xml')
    >>> forcefield.registerTemplateGenerator(template_generator.generator)

    Create a template generator for GAFF2 for multiple OEMols:

    >>> template_generator = OEGAFFTemplateGenerator(oemols=[oemol1, oemol2], gaff_version='gaff2')

    You can also some OEMols later on after the generator has been registered:

    >>> forcefield.add_oemols(oemol)
    >>> forcefield.add_oemols([oemol1, oemol2])

    You can optionally create or use a tiny database cache of pre-parameterized molecules:

    >>> template_generator = OEGAFFTemplateGenerator(cache='gaff-molecules.json')

    Newly parameterized molecules will be written to the cache, saving time next time!

    """
    def __init__(self, oemols=None, cache=None, gaff_version='gaff'):
        """
        Create an OEGAFFTemplateGenerator with some OpenEye toolkit OEMols

        Requies the OpenEye Toolkit.

        Parameters
        ----------
        oemols : OEMol or list of OEMol, optional, default=None
            If specified, these molecules will be recognized and parameterized with antechamber as needed.
            The parameters will be cached in case they are encountered again the future.
        cache : str, optional, default=None
            Filename for global caching of parameters.
            If specified, parameterized molecules will be stored in a TinyDB instance.
            Note that no checking is done to determine this cache was created with the same GAFF version.
        gaff_version : str, default = 'gaff'
            One of ['gaff', 'gaff2']; selects which GAFF major version to use.

        .. todo :: Should we support SMILES instead of OEMols?

        Examples
        --------

        Create a template generator for GAFF for a single OEMol and register it with ForceField:

        >>> from openmoltools.forcefield_generators import OEGAFFTemplateGenerator
        >>> template_generator = OEGAFFTemplateGenerator(oemols=oemol)

        Create a template generator for GAFF2 for multiple OEMols:

        >>> template_generator = OEGAFFTemplateGenerator(oemols=[oemol1, oemol2], gaff_version='gaff2')

        You can optionally create or use a tiny database cache of pre-parameterized molecules:

        >>> template_generator = OEGAFFTemplateGenerator(cache='gaff-molecules.json')

        """
        from openeye import oechem

        self._gaff_version = gaff_version

        # Add oemols to the dictionary
        self._oemols = dict()
        self.add_oemols(oemols)

        self._cache = cache
        self._smiles_added_to_db = set() # set of SMILES added to the database this session

    # TODO: Replace this encoder/decoder logic when openmm objects are properly serializable
    class _JSONEncoder(json.JSONEncoder):
        def default(self, o):
            from simtk.openmm.app import ForceField, Element
            if isinstance(o, ForceField._TemplateData):
                s = {'_type' : '_TemplateData'}
                s.update(o.__dict__)
                return s
            elif isinstance(o, ForceField._TemplateAtomData):
                s = {'_type' : '_TemplateAtomData'}
                s.update(o.__dict__)
                return s
            elif isinstance(o, Element):
                return {'_type' : 'Element', 'atomic_number' : o.atomic_number}
            else:
                return super(OEGAFFTemplateGenerator._JSONEncoder, self).default(o)

    class _JSONDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

        def object_hook(self, obj):
            from simtk.openmm.app import ForceField, Element
            if '_type' not in obj:
                return obj
            type = obj['_type']
            if type == '_TemplateData':
                template = ForceField._TemplateData.__new__(ForceField._TemplateData)
                del obj['_type']
                template.__dict__ = obj
                return template
            if type == '_TemplateAtomData':
                atom = ForceField._TemplateAtomData.__new__(ForceField._TemplateAtomData)
                del obj['_type']
                atom.__dict__ = obj
                return atom
            elif type == 'Element':
                return Element.getByAtomicNumber(obj['atomic_number'])
            return obj

    def add_oemols(self, oemols=None):
        """
        Add specified list of OEMol objects to cached molecules that will be recognized.

        Parameters
        ----------
        oemols : OEMol or list of OEMol, optional, default=None
            If specified, these molecules will be recognized and parameterized with antechamber as needed.
            The parameters will be cached in case they are encountered again the future.

        Examples
        --------
        Add some OEMols later on after the generator has been registered:

        >>> forcefield.add_oemols(oemol)
        >>> forcefield.add_oemols([oemol1, oemol2])

        """
        # Return if empty
        if not oemols:
            return

        # Ensure oemols is iterable
        try:
            iterator = iter(oemols)
        except TypeError as te:
            oemols = [ oemols ]

        # Create copies
        oemols = [ oemol.CreateCopy() for oemol in oemols ]

        # Cache OEMols
        from openeye import oechem
        self._oemols.update( { oechem.OEMolToSmiles(oemol) : oemol for oemol in oemols } )

    @staticmethod
    def _match_residue(residue, oemol_template):
        """Determine whether a residue matches an OEMol template and return a list of corresponding atoms.

        This implementation uses NetworkX for graph isomorphism determination.

        Parameters
        ----------
        residue : simtk.openmm.app.topology.Residue
            The residue to check
        oemol_template : openeye.oechem.OEMol
            The OEMol template to compare it to

        Returns
        -------
        matches : dict of int : int
            matches[residue_atom_index] is the corresponding OEMol template atom index
            or None if it does not match the template

        """
        import networkx as nx
        from openeye import oechem

        # Make a copy of the template
        oemol_template = oemol_template.CreateCopy()

        # Ensure atom names are unique
        oechem.OETriposAtomNames(oemol_template)

        # Build list of external bonds for residue
        number_of_external_bonds = { atom : 0 for atom in residue.atoms() }
        for bond in residue.external_bonds():
            if bond[0] in number_of_external_bonds: number_of_external_bonds[bond[0]] += 1
            if bond[1] in number_of_external_bonds: number_of_external_bonds[bond[1]] += 1

        # Residue graph
        residue_graph = nx.Graph()
        for atom in residue.atoms():
            residue_graph.add_node(atom, element=atom.element.atomic_number, number_of_external_bonds=number_of_external_bonds[atom])
        for bond in residue.internal_bonds():
            residue_graph.add_edge(bond[0], bond[1])

        # Template graph
        # TODO: We can support templates with "external" bonds or atoms using attached string data in future
        # See https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemClasses/OEAtomBase.html
        template_graph = nx.Graph()
        for oeatom in oemol_template.GetAtoms():
            template_graph.add_node(oeatom.GetName(), element=oeatom.GetAtomicNum(), number_of_external_bonds=0)
        for oebond in oemol_template.GetBonds():
            template_graph.add_edge(oebond.GetBgn().GetName(), oebond.GetEnd().GetName())

        # Determine graph isomorphism
        from networkx.algorithms import isomorphism
        graph_matcher = isomorphism.GraphMatcher(residue_graph, template_graph)
        if graph_matcher.is_isomorphic() == False:
            return None

        # Translate to local residue atom indices
        atom_index_within_residue = { atom : index for (index, atom) in enumerate(residue.atoms()) }
        atom_index_within_template = { oeatom.GetName() : index for (index, oeatom) in enumerate(oemol_template.GetAtoms()) }
        matches = { atom_index_within_residue[residue_atom] : atom_index_within_template[template_atom] for (residue_atom, template_atom) in graph_matcher.mapping.items() }

        return matches

    def generator(self, forcefield, residue, structure=None):
        """
        Residue template generator method to register with simtk.openmm.app.ForceField

        Parameters
        ----------
        forcefield : simtk.openmm.app.ForceField
            The ForceField object to which residue templates and/or parameters are to be added.
        residue : simtk.openmm.app.Topology.Residue
            The residue topology for which a template is to be generated.

        Returns
        -------
        success : bool
            If the generator is able to successfully parameterize the residue, `True` is returned.
            If the generator cannot parameterize the residue, it should return `False` and not modify `forcefield`.

        """
        from openeye import oechem
        from io import StringIO

        # If a database is specified, check against molecules in the database
        if self._cache is not None:
            from tinydb import TinyDB
            db = TinyDB(self._cache)
            for entry in db:
                # Skip any molecules we've added to the database this session
                if entry['smiles'] in self._smiles_added_to_db:
                    continue

                # See if the template matches
                oemol_template = oechem.OEMol()
                oechem.OESmilesToMol(oemol_template, entry['smiles'])
                oechem.OEAddExplicitHydrogens(oemol_template)
                if self._match_residue(residue, oemol_template):
                    # Register the template
                    template = self._JSONDecoder().decode(entry['template'])
                    forcefield.registerResidueTemplate(template)
                    # Add the parameters
                    # TODO: Do we have to worry about parameter collisions?
                    forcefield.loadFile(StringIO(entry['ffxml']))
                    # Signal success
                    return True

        # Check against the molecules we know about
        for smiles, oemol_template in self._oemols.items():
            # See if the template matches
            if self._match_residue(residue, oemol_template):
                # Generate template and parameters.
                [template, ffxml] = generateResidueTemplate(oemol_template, gaff_version=self._gaff_version)
                # Register the template
                forcefield.registerResidueTemplate(template)
                # Add the parameters
                # TODO: Do we have to worry about parameter collisions?
                forcefield.loadFile(StringIO(ffxml))
                # If a cache is specified, add this molecule
                if self._cache is not None:
                    print('Writing {} to cache'.format(smiles))
                    db.insert({'smiles' : smiles, 'template' : self._JSONEncoder().encode(template), 'ffxml' : ffxml})
                    self._smiles_added_to_db.add(smiles)
                    db.close()

                # Signal success
                return True

        # Report that we have failed to parameterize the residue
        print("Didn't know how to parameterize residue {}".format(residue.name))
        return False

"""
Contains base utility class to generate an openmm System
object from the topology proposal
"""
import simtk.openmm as openmm
import os
import logging
import openmoltools
try:
    from subprocess import getoutput  # If python 3
except ImportError:
    from commands import getoutput  # If python 2
import simtk.openmm.app as app

class SystemGenerator(object):
    """
    This is the base class for utility functions that generate a System
    object from TopologyProposal namedtuple

    Arguments
    ---------
    metadata : dict
        contains metadata (such as forcefield) for system creation
    """

    def __init__(self, metadata):
        pass

    def new_system(self, top_proposal):
        """
        Create a system with ligand and protein

        Arguments
        ---------
        top_proposal : TopologyProposal namedtuple
            Contains the topology of the new system to be made

        Returns
        -------
        system : simtk.openmm.System object
            openmm System containing the protein and ligand(s)
        """
        return openmm.System()


class ComplexSystemGenerator(SystemGenerator):
    """
    This subclass of SystemGenerator creates a new system for a given receptor
    + small molecule pair. It currently works in implicit solvent (OBC2) only and
    "remembers" systems that it already generated.

    Arguments
    ---------
    metadata : dict
        Should contain the filename (pdb) of the receptor in a simulation-worthy state
    """

    def __init__(self, metadata):
        self._receptor_filename = metadata['receptor_filename']
        self._systems_generated = dict()
        self._topologies_generated = dict()

    def new_system(self, top_proposal):
        """
        Call this method to get a new system, with the receptor associated with this instance
        and the small molecule specified in the proposal.

        Arguments
        ---------
        top_proposal : TopologyProposal namedtuple
            Topology proposal namedtuple containing metadata

        Returns
        -------
        new_system : openmm.System object
            the new system
        new_topology : simtk.openmm.app.Topology object
            The topology of the system (possibly will be deprecated)
        """
        #get the smiles of the molecule out of the proposal:
        mol_smiles = top_proposal.metadata['molecule_smiles']

        if mol_smiles in self._systems_generated.keys():
            return self._systems_generated['mol_smiles'], self._topologies_generated['mol_smiles']

        mol = top_proposal.metadata['oemol']

        #get the prmtop--naming everything ligand for now.
        #TODO: Decide on a better naming scheme
        prmtop = self._run_tleap("ligand", mol)

        #add the topology to the generated tops, create the system and do the same for it
        self._topologies_generated['mol_smiles'] = prmtop.topology
        system = prmtop.createSystem(implicitSolvent=app.OBC2)
        self._systems_generated['mol_smiles'] = system

        return system, prmtop.topology


    def _run_tleap(self, ligand_name, ligand_oemol):
        """
        Run tleap to create a prmtop with both receptor and ligand files.

        Arguments
        ---------
        ligand_name : str
            The name of the ligand
        ligand_oemol : openeye.oechem.OEMol object
            The oemol object representing the ligand

        Returns
        -------
        prmtop : simtk.openmm.app.AmberPrmtopFile
            The prmtop object that can be used to create a topology and system
        """
        #change into a temporary directory, remembering where we came from
        cwd = os.getcwd()
        temp_dir = os.mkdtemp()
        os.chdir(temp_dir)

        #run antechamber to get parameters for molecule
        _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(ligand_oemol, tripos_mol2_filename=ligand_name + '.tripos.mol2', conformer=0, residue_name=ligand_name)
        gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(ligand_name, tripos_mol2_filename)

        #now get ready to run tleap to generate prmtop
        tleap_input = self._gen_tleap_input(gaff_mol2, frcmod, "complex")
        tleap_file = open('tleap_commands', 'w')
        tleap_file.writelines(tleap_input)
        tleap_file.close()
        tleap_cmd_str = "tleap -f %s " % tleap_file.name

        #call tleap, log output to logger
        output = getoutput(tleap_cmd_str)
        logging.debug(output)

        #read in the prmtop file
        prmtop = app.AmberPrmtopFile("complex.prmtop")

        #return to where we were
        os.chdir(cwd)

        #delete temporary directory
        os.unlink(temp_dir)

        return prmtop



    def _gen_tleap_input(self, ligand_gaff_mol2, ligand_frcmod, complex_name):
        """
        This is a utility function to generate the input string necessary to run tleap
        """

        tleapstr = """
        # Load AMBER '96 forcefield for protein.
        source oldff/leaprc.ff99SBildn

        # Load GAFF parameters.
        source leaprc.gaff

        # Set GB radii to recommended values for OBC.
        set default PBRadii mbondi2

        # Load in protein.
        receptor = loadPdb {receptor_filename}.pdb

        # Load parameters for ligand.
        loadAmberParams {ligand_frcmod}

        # Load ligand.
        ligand = loadMol2 {ligand_gaf_fmol2}

        # Create complex.
        complex = combine {{ receptor ligand }}

        # Check complex.
        check complex

        # Report on net charge.
        charge complex

        # Write parameters.
        saveAmberParm complex {complex_name}.prmtop {complex_name}.inpcrd

        # Exit
        quit
        """

        tleap_input = tleapstr.format(ligand_frcmod=ligand_frcmod, ligand_gaff_mol2=ligand_gaff_mol2, receptor_filename=self._receptor_filename, complex_name=complex_name)
        return tleap_input
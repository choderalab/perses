import openeye.oechem as oechem
import rdkit.Chem as chem
import yaml
from perses.rjmc import topology_proposal, geometry
from perses.annihilation import relative
from perses.tests import utils
import simtk.unit as unit
from openmmtools.constants import kB



def load_sd_file_rdkit(mol_filename):
    mol_file = open(mol_filename, 'r')
    suppl = chem.SDMolSupplier(mol_file)
    mols = [mol for mol in suppl]
    return mols[0]

def load_mol_file(mol_filename):
    """
    Utility function to load
    Parameters
    ----------
    mol_filename : str
        The name of the molecule file. Must be supported by openeye.

    Returns
    -------
    mol : oechem.OEMol
        the molecule as an OEMol
    """
    ifs = oechem.oemolistream()
    ifs.open(mol_filename)
    #get the list of molecules
    mol_list = [mol for mol in ifs.GetOEMols()]
    #we'll always take the first for now
    return mol_list[0]



class RelativeFreeEnergySetup(object):
    """
    This is a class to perform relative free energy calculation setups. It assumes you already have a complex with the
    first molecule, and a SMILES string for the second.
    """

    def __init__(self, complex_topology, complex_positions, ligand_name, initial_molecule_SMILES, final_molecule_SMILES, temperature=300.0*unit.kelvin):
        """
        Create a RelativeFreeEnergySetup tool using the perses infrastructure.

        Parameters
        ----------
        complex_topology : openmm.app.topology
            The topology of the complex of protein with initial ligand
        complex_positions : [n, 3] ndarray
            The positions of the atoms in the complex.
        ligand_name : str
            The name of the ligand residue in the complex
        initial_molecule_SMILES : str
            The molecule in the complex already
        final_molecule_SMILES : str
            The molecule to transform the initial molecule into

        """
        self._temperature = temperature
        kT = kB * self._temperature
        self._beta = 1.0 / kT
        self._complex_topology = complex_topology
        self._complex_positions = complex_positions
        self._ligand_name = ligand_name

        gaff_filename = utils.get_data_filename('data/gaff.xml')
        system_generator = topology_proposal.SystemGenerator([gaff_filename, 'ff99sbildn.xml', 'tip3p.xml'])
        current_system = system_generator.build_system(complex_topology)
        self._geometry_engine = geometry.FFAllAngleGeometryEngine()
        self._proposal_engine = topology_proposal.SmallMoleculeSetProposalEngine(
            [initial_molecule_SMILES, final_molecule_SMILES], system_generator, residue_name=ligand_name)

        #generate topology proposal
        self._topology_proposal = self._proposal_engine.propose(current_system, complex_topology)

        #generate new positions with geometry engine
        new_positions, _ = self._geometry_engine.propose(self._topology_proposal, complex_positions, self._beta)

        #generate a hybrid topology
        self._hybrid_factory = relative.HybridTopologyFactory(current_system, self._topology_proposal.new_system,
                                                              complex_topology, self._topology_proposal.new_topology,
                                                              complex_positions, new_positions,
                                                              self._topology_proposal.old_to_new_atom_map)

        #generate the materials we need to run a simulation:
        self._hybrid_system, self._hybrid_topology, self._hybrid_positions, self._sys2_indices_in_system, self._sys1_indices_in_system = self._hybrid_factory.createPerturbedSystem()



    @property
    def hybrid_system(self):
        return self._hybrid_system

    @property
    def hybrid_positions(self):
        return self._hybrid_positions

    @property
    def hybrid_topology(self):
        return self._hybrid_topology

    @property
    def complex_topology(self):
        return self._complex_topology

    @property
    def complex_positions(self):
        return self._complex_positions

    @property
    def topology_proposal(self):
        return self._topology_proposal

    @property
    def ligand_name(self):
        return self._ligand_name

class SolvatedLigandSystemFactory(object):
    """
    This class is a helper class that can create a solvent hybrid system with the same atom map as a
    corresponding protein-ligand hybrid system. This is necessary so that the endpoints of the two calculations are
    exactly the same.
    """

    def __init__(self, relative_free_energy_setup):
        self._hybrid_topology = relative_free_energy_setup.hybrid_topology
        self._ligand_name = relative_free_energy_setup.ligand_name

        #get the list of atoms that pertain to the ligand:
        self._ligand_atoms_in_hybrid = None




if __name__=="__main__":
    import sys
    #yaml input is the only input
    input_filename = sys.argv[1]
    input_file = open(input_filename, 'r')
    input_data = yaml.load(input_file)
    input_file.close()

    #load the molecules that will form this calculation
    initial_molecule = load_mol_file(input_data['initial_molecule'])
    final_molecule = load_mol_file(input_data['final_molecule'])


"""
This is the base class for generating a biasing potential
for expanded ensemble simulation
"""
import openeye.oechem as oechem
import openeye.oeomega as oeomega
import openmoltools
import openeye.oeiupac as oeiupac
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as units

class BiasEngine(object):
    """
    Generates the bias for expanded ensemble simulations
    
    Arguments
    ---------
    metadata : dict
        Dictionary containing metadata relevant to the implementation
    """
    
    def __init__(self, metadata):
        pass

    def g_k(self, topology_proposal):
        """
        Generate a biasing weight g_k for the state indicated.
        
        Arguments
        --------
        topology_proposal : namedtuple of type TopologyProposal
            Contains information about the state for which g_k should be generated

        Returns
        -------
        g_k : float
            Bias for the given state
        """
        return 0

class MinimizedPotentialBias(BiasEngine):
    """
    This class calculates the bias potential for expanded ensemble simulations,
    using a minimized potential energy as the bias.

    Arguments
    ---------
    smiles_list : list of str
        list of smiles strings corresponding to molecules
    """

    def __init__(self, smiles_list):
        self._smiles_list = smiles_list
        self._mol_dict = self._create_molecule_list(smiles_list)
        self._gk = dict()

    def _create_molecule_list(self, smiles_list):
        """
        Utility function to create oemols out of smiles strings

        Returns
        -------
        oemol_list : dict of smiles : oemol
            smile : oemols for the simulation
        """
        oemol_dict ={}
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(1)
        for smiles in smiles_list:
            mol = oechem.OEMol()
            oechem.OESmilesToMol(mol, smiles)
            oechem.OEAddExplicitHydrogens(mol)
            omega(mol)
            oemol_dict[smiles] = mol
        return oemol_dict

    def _create_implicit_solvent_openmm(self, mol):
        """
        Take a list of oemols, and generate openmm systems
        and positions for each.

        Arguments
        ---------
        mol : oemol
            oemol to be turned into system, positions

        Returns
        --------
        system : simtk.openmm.System
            openmm system corresponding to molecule
        positions : np.array, Quantity nm
           array of atomic positions
        """
        molecule_name = oeiupac.OECreateIUPACName(mol)
        openmoltools.openeye.enter_temp_directory()
        _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(mol, tripos_mol2_filename=molecule_name + '.tripos.mol2', conformer=0, residue_name='MOL')
        gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(molecule_name, tripos_mol2_filename)
        prmtop_file, inpcrd_file = openmoltools.utils.run_tleap(molecule_name, gaff_mol2, frcmod)
        prmtop = app.AmberPrmtopFile(prmtop_file)
        crd = app.AmberInpcrdFile(inpcrd_file)
        system = prmtop.createSystem(implicitSolvent=app.OBC2, constraints=app.HBonds)
        positions = crd.getPositions(asNumpy=True)
        return system, positions

    def g_k(self, molecule_smiles):
        """
        Retrieve or compute the g_k for the given molecule

        Arguments
        ---------
        molecule_smiles : string
            SMILES representation of the molecule

        Returns
        -------
        g_k : float
           Bias weight
        """
        if molecule_smiles in self._gk.keys():
            return self._gk[molecule_smiles]
        else:
            system, positions = self._create_implicit_solvent_openmm(self._mol_dict[molecule_smiles])
            timestep = 2*units.femtoseconds
            integrator = openmm.VerletIntegrator(timestep)
            platform = openmm.Platform.getPlatformByName("CPU")
            context = openmm.Context(system, integrator, platform)
            context.setPositions(positions)
            openmm.LocalEnergyMinimizer.minimize(context)
            state = context.getState(getEnergy=True)
            g_k = state.getPotentialEnergy()
            self._gk[molecule_smiles] = g_k
            return g_k

    def precompute_gk(self):
        """
        A utility function to compute all the g_ks
        and return them as a {smiles : g_k} dict

        Returns
        ------
        gks : dict of type {string : float}
            dict of {smiles : g_k}
        """
        gks = dict()
        for smiles in self._smiles_list:
           gks[smiles] = self.g_k(smiles)
        return gks

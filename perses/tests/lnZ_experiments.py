# In[0]
import numpy as np
import seaborn as sns
import mdtraj as md
from perses.tests import utils
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import testsystems, states, mcmc, integrators

from openmmtools.constants import kB
import tqdm
from tqdm import tqdm_notebook, trange
from openeye import oechem

temperature = 300.0 * unit.kelvin
beta = 1.0 / (temperature*kB)
PATH = "/home/dominic/Downloads/"
import logging
logging.basicConfig(filename = "{}vacRJMC.log".format(PATH), level = logging.INFO)
REFERENCE_PLATFORM = openmm.Platform.getPlatformByName("Reference")

def convert_to_md(openmm_positions):
    """
    Convert openmm position objects into numpy ndarrays

    Arguments
    ---------
    openmm_positions: openmm unit.Quantity object
        Positions generated from openmm simulation

    Returns
    -------
    md_positions_stacked: np.ndarray
        Positions in md_unit_system (nanometers)
    """
    _openmm_positions_no_units = [_posits.value_in_unit_system(unit.md_unit_system) for _posits in openmm_positions]
    md_positions_stacked = np.stack(_openmm_positions_no_units)

    return md_positions_stacked

def compute_rp(system, positions):
    """
    Utility function to compute the reduced potential

    Arguments
    ---------
    system: openmm system object
    positions: openmm unit.Quantity object
        openmm position (single frame)

    Returns
    -------
    rp: float
        reduced potential
    """
    from simtk.unit.quantity import is_dimensionless
    _i = openmm.VerletIntegrator(1.0)
    _ctx = openmm.Context(system, _i, REFERENCE_PLATFORM)
    _ctx.setPositions(positions)
    rp = beta*_ctx.getState(getEnergy=True).getPotentialEnergy()
    assert is_dimensionless(rp), "reduced potential is not dimensionless"
    del _ctx
    return rp

#create iid bead system and save
def create_iid_systems(system_attributes, mol, num_iterations):
    """
    Function to simulate i.i.d conformations of the initial molecule

    Arguments
    ---------
    system_attributes: dict
        dict of molecule A and B with (oemol, sys, pos, top)
    mol: str
        molecule to simulate

    Returns
    -------
    iid_positions_A: numpy.ndarray
        num_iterations of independent initial molecule conformations
    """
    from openmmtools import integrators
    import tqdm
    _oemolA, _sysA, _posA, _topA = system_attributes[mol]
    _platform = openmm.Platform.getPlatformByName("CPU")
    _integrator = integrators.LangevinIntegrator(temperature, 1./unit.picoseconds, 0.002*unit.picoseconds)
    _ctx = openmm.Context(_sysA, _integrator)
    _ctx.setPositions(_posA)
    openmm.LocalEnergyMinimizer.minimize(_ctx)

    _iid_positions_A=[]
    for _iteration in tqdm.trange(num_iterations):
        _integrator.step(1000)
        _state=_ctx.getState(getPositions=True)
        _iid_positions_A.append(_state.getPositions(asNumpy=True))

    _iid_positions_A = convert_to_md(_iid_positions_A)

    return _iid_positions_A
from perses.tests.utils import createOEMolFromSMILES
def create_system_from_mol(molecule):
    from perses.tests.utils import get_data_filename, extractPositionsFromOEMOL
    from perses.tests.utils import createOEMolFromSMILES
    from io import StringIO

    # Generate a topology.
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    topology = generateTopologyFromOEMol(molecule)

    # Initialize a forcefield with GAFF.
    # TODO: Fix path for `gaff.xml` since it is not yet distributed with OpenMM
    from simtk.openmm.app import ForceField
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    forcefield = ForceField(gaff_xml_filename)

    # Generate template and parameters.
    from openmoltools.forcefield_generators import generateResidueTemplate
    [template, ffxml] = generateResidueTemplate(molecule)

    # Register the template.
    forcefield.registerResidueTemplate(template)

    # Add the parameters.
    forcefield.loadFile(StringIO(ffxml))

    # Create the system.
    system = forcefield.createSystem(topology, removeCMMotion=False)


    # Extract positions
    positions = extractPositionsFromOEMOL(molecule)



    return (molecule, system, positions, topology)
#define the topology proposal function
def generate_vacuum_topology_proposal(current_system, current_top, current_mol, proposed_mol, atom_expr=None, bond_expr=None):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

    Parameters
    ----------
    current_system : openmm.System
        name of the first molecule
    current_mol : oechem.OEMol
         OEMol of the first molecule
    proposed_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : perses.rjmc.topology_proposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    from openmoltools import forcefield_generators
    from perses.rjmc import topology_proposal as tp
    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename
    from openeye import oechem


    proposed_mol.SetTitle("MOL")
    current_mol_name = current_mol.GetTitle()

    initial_smiles = oechem.OEMolToSmiles(current_mol)
    final_smiles = oechem.OEMolToSmiles(proposed_mol)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = tp.SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.NoCutoff})
    proposal_engine = tp.SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name="MOL", atom_expr=atom_expr, bond_expr=bond_expr)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(current_system, current_top, current_mol=current_mol, proposed_mol=proposed_mol)

    return topology_proposal

def generate_small_molecule_library(IUPAC = True, SMILES = False, mol1 = 'benzene', mol2 = 'toluene', nb_removed = True):
    """
    Generate mol_attributes dictionary for running eq. simulations, topology proposal, and hybrid proposal.

    Parameters
    ----------
    IUPAC : bool
        Whether mol1 and mol2 are IUPACS
    SMILES : bool
        Whether mol1 and mol2 are IUPACS
    mol1 : str
        molecule 1.  IUPAC if IUPAC == True; else, SMILES
    mol2 : str
        molecule 2.  IUPAC if IUPAC == True; else, SMILES
    nb_removed : bool
        Whether to make a copy of the resulting dictionary and remove nonbonded forces

    Returns
    -------
    mol_attributes : dict
        mol_attributes[mol] = (molecule, system, positions, topology).
        molecule: OpenEyeMOL; system, positions, and topology are openmm objects

    mol_attributes_no_nb : dict
        mol_attributes[mol] = (molecule, system, positions, topology).
        molecule: OpenEyeMOL; system, positions, and topology are openmm objects

    """
    from perses.tests.utils import createOEMolFromSMILES, createOEMolFromIUPAC
    mol_attributes = dict()
    if IUPAC == True and SMILES == False:
        molecule1 = createOEMolFromIUPAC(mol1)
        molecule1.SetTitle("MOL")

        molecule2 = createOEMolFromIUPAC(mol2)
        molecule1.SetTitle("MOL")

    elif IUPAC == False and SMILES == True:
        molecule1 = createOEMolFromSMILES(mol1, "MOL")
        molecule2 = createOEMolFromSMILES(mol2, "MOL")
    else:
        raise Exception("Either IUPAC or SMILES must be turned on (exclusively)")

    #now we actually create attributes
    for molecule, label in zip([molecule1, molecule2], ["mol1", "mol2"]):
        mol_attributes[label] = create_system_from_mol(molecule)

    #make copy and remove nb if specified
    if nb_removed == True:
        import copy
        mol_attributes_no_nb = copy.deepcopy(mol_attributes)
        for mol in mol_attributes_no_nb:
            mol_attributes_no_nb[mol][1].removeForce(3)
            #print("{} nonbonded copy forces: {}".format(mol, mol_attributes_no_nb[mol][1].getForces()))
        return mol_attributes, mol_attributes_no_nb


    return mol_attributes

def organize_lnZ_distributions(OrderedDict_list, positions, topology, indices_of_interest = None, filename = "{}_lnZ.png".format(PATH)):
    """
    Histogram plots for lnZ_phi.

    Parameters
    ----------
    OrderedDict_list : list(OrderedDict)
        a list of ordered dicts, each of which contains keys of atom torsions and values of lnZ_phi
    topology : openmm.topology object
        hybrid topology
    indices of interest : list
        list of ints, each of which corresponding to the index of a newly proposed atom
    Returns
    -------
    sns.distplot
    atom_dict : dictionary of atoms and corresponding lnZ_phi list
    torsion_dict : dictionary of torsions and corresponding lnZ_phi list
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    import mdtraj as md
    sns.set(color_codes = True)

    #pull the first ordered dict to get the possible indices
    sample_dict = OrderedDict_list[0]
    possible_indices = list(set([torsion[0] for torsion in sample_dict.keys()]))
    possible_torsions_unflat = [dict.keys() for dict in OrderedDict_list]
    possible_torsions = list(set([y for x in possible_torsions_unflat for y in x]))

    #ensure indices of interest are all in the set of possible indices
    if indices_of_interest != None:
        assert all(index in possible_indices for index in indices_of_interest), "Found index of interest not in possible indices"
    else:
        indices_of_interest = possible_indices

    #initialize atom dict and then place
    posits = convert_to_md(positions)
    torsion_dict = {torsion: list() for torsion in possible_torsions}
    theta_angle_lnZ = []
    for index, dict in enumerate(OrderedDict_list):
        t = md.Trajectory(posits[index], topology)
        for torsion in dict.keys():
            theta = list(torsion[0:3])
            angle = md.compute_angles(t, np.array([theta]))[0]
            lnZ = dict[torsion]
            theta_angle_lnZ.append((theta, angle, lnZ))
            torsion_dict[torsion].append(lnZ)

    atom_dict = {atom_index: list() for atom_index in possible_indices}
    for atom_index in possible_indices:
        for key in torsion_dict.keys():
            if key[0] == atom_index:
                atom_dict[atom_index] += torsion_dict[key]



    #pull atom list for labels
    atoms = list(topology.atoms())
    for index in indices_of_interest:
        lbl = "{}: {}".format(index, atoms[index].name)
        ax1 = sns.distplot(np.array(atom_dict[index]), label = lbl, rug = True, norm_hist = True)
    ax1.legend(loc = 'best')
    ax1.set(xlabel = 'lnZ_phi', ylabel = 'p(lnZ_phi)')
    plt.savefig(filename)

    return atom_dict, theta_angle_lnZ


def plot_theta_phi(positions, topology, theta_indices, phi_indices, filename = "{}theta_phi.png".format(PATH)):
    """
    Plot the phi-theta scatter plot from an equilibrium simulation (of mol2)

    Parameters
    ----------
    positions : np.ndarray
        frames of equilibrium simulation
    topology : openmm.topology object
        topology
    theta_indices : tuple of size 3
        tuple of atom indices to define

    """
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    sns.set(color_codes = True)
    import mdtraj as md
    t = md.Trajectory(positions, topology)
    theta = md.compute_angles(t, np.array([theta_indices]))
    phi = md.compute_dihedrals(t, np.array([phi_indices]))

    ax = sns.scatterplot(phi[:,0], theta[:,0])
    ax.set(xlabel = 'phi', ylabel = 'theta')
    plt.savefig(filename)

def plot_theta_lnZ(theta_angle_lnZ, angle_of_interest, filename = "{}theta_lnZ.png".format(PATH)):
    """
    plot the theta, lnZ scatter

    """
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    sns.set(color_codes = True)

    angle = []; lnZs = []
    #make a set of angles
    for item in theta_angle_lnZ:
        if item[0] == angle_of_interest:
            angle.append(item[1][0])
            lnZs.append(item[2])


    ax = sns.scatterplot(angle, lnZs)
    ax.set(xlabel = 'theta', ylabel = 'lnZs')
    plt.savefig(filename)









class SmallMoleculeVacuumRJMC(object):
    """
    This class proposed a jump from arbitrary A --> A' and back (where A is real system and A' is the Alchemical system in the
    thermodynamic cycle in A --RJMC--> A' --lambda_perturbation--> B' --RJMC--> B)
    """

    def __init__(self, num_iterations = 1, mol1 = 'propane', mol2 = 'chloropropane', IUPAC = True, SMILES = False, nb_removed = True):
        """
        Parameters
        ----------
        num_iterations = int
            Number of forward proposals to conduct
        mol1 : str
            molecule 1.  IUPAC if IUPAC == True; else, SMILES
        mol2 : str
            molecule 2.  IUPAC if IUPAC == True; else, SMILES
        IUPAC : bool
            Whether mol1 and mol2 are IUPACS
        SMILES : bool
            Whether mol1 and mol2 are IUPACS
        nb_removed : bool
            Whether to make a copy of the resulting dictionary and remove nonbonded forces

        Returns
        -------
        mol_attributes : dict
            mol_attributes[mol] = (molecule, system, positions, topology).
            molecule: OpenEyeMOL; system, positions, and topology are openmm objects

        mol_attributes_no_nb : dict
            mol_attributes[mol] = (molecule, system, positions, topology).
            molecule: OpenEyeMOL; system, positions, and topology are openmm objects
        """
        self.num_iterations = num_iterations
        self.mol1 = mol1
        self.mol2 = mol2
        self.IUPAC = IUPAC
        self.SMILES = SMILES
        self.nb_removed = nb_removed

        #equilibrium filenames to write to disk
        self.mol1_eq_filename = '{}_eq.npy'.format(self.mol1)
        self.mol2_eq_filename = '{}_eq.npy'.format(self.mol2)

        if self.nb_removed:
            self.mol_attributes, self.mol_attributes_no_nb = generate_small_molecule_library(mol1 = mol1, mol2 = mol2, nb_removed = self.nb_removed)
            #verify the nonbonded forces are removed
            logging.info("mol_attributes_no_nb forces:")
            logging.info("      mol1: {}".format(self.mol_attributes_no_nb['mol1'][1].getForces()))
            logging.info("      mol2: {}".format(self.mol_attributes_no_nb['mol2'][1].getForces()))

        else:
            self.mol_attributes = generate_small_molecule_library(mol1 = self.mol1, mol2 = self.mol2, nb_removed = self.nb_removed)

        #generate pdb files of mol1 and mol2
        app.pdbfile.PDBFile.writeFile(self.mol_attributes['mol1'][3], self.mol_attributes['mol1'][2], open('{}{}_vacRJMC.pdb'.format(PATH, self.mol1), 'w'))
        app.pdbfile.PDBFile.writeFile(self.mol_attributes['mol2'][3], self.mol_attributes['mol2'][2], open('{}{}_vacRJMC.pdb'.format(PATH, self.mol2), 'w'))
        logging.info("generated pdbs for both molecules")




    def eq_simulations(self):
        """
        Generate iid distributions for mol1 and mol2
        """
        self.iid1, self.iid2 = create_iid_systems(self.mol_attributes_no_nb, "mol1", self.num_iterations), create_iid_systems(self.mol_attributes_no_nb, "mol2", self.num_iterations)
        self.iid1_quantity, self.iid2_quantity = unit.Quantity(self.iid1, unit = unit.nanometers), unit.Quantity(self.iid2, unit = unit.nanometers)
        logging.info("generated iid samples for both molecules.  proceeding to save to disk...")
        self.mol1_eq_filename, self.mol2_eq_filename = '{}{}_eq_vacRJMC.npy'.format(PATH, self.mol1), '{}{}_eq_vacRJMC.npy'.format(PATH, self.mol2)
        np.save(self.mol1_eq_filename, self.iid1)
        np.save(self.mol2_eq_filename, self.iid2)
        logging.info("saved {}_eq_vacRJMC.npy and {}_eq_vacRJMC.npy to disk".format(self.mol1, self.mol2))

    def initial_topology_proposal(self):
        """
        Generate an initial topology proposal.  Note we do not use the nb version in this case since the hybrid factory cannot handle
        the proposal when there are no nonbonded forces.
        """
        self.initial_topology_proposal = generate_vacuum_topology_proposal(current_system = self.mol_attributes['mol1'][1],
                                                                           current_top = self.mol_attributes['mol1'][3],
                                                                           current_mol = self.mol_attributes['mol1'][0],
                                                                           proposed_mol = self.mol_attributes['mol2'][0],
                                                                           atom_expr=None,
                                                                           bond_expr=None)
        logging.info("Completed initial_topology_proposal.")

    def create_hybrid_factory(self):
        """
        Generate a hybrid factory from the initial_topology_proposal
        """
        from perses.annihilation.new_relative import HybridTopologyFactory

        self.hybrid_factory = HybridTopologyFactory(self.initial_topology_proposal, self.mol_attributes['mol1'][2], self.mol_attributes['mol2'][2])
        logging.info("Completed hybrid_factory")


    def create_new_hybrid_sys_top(self):
        """
        Instantiates a new topology from the hybrid system that removed all nonbonded forces and places all custom bonded forces (at lambda=0)
        into normal force objects whilst deleting all other forces.
        """

        #generate copies of old system and hybrid systems
        import copy

        old_sys = copy.deepcopy(self.mol_attributes_no_nb['mol1'][1])
        hybrid_sys = copy.deepcopy(self.hybrid_factory.hybrid_system)

        #remove force indices of nonbonded forces
        hybrid_sys.removeForce(6)
        hybrid_sys.removeForce(6)

        #print the forces once more to the logger
        logging.info("Printing force of the old system copy after nb force removal")
        for force in old_sys.getForces():
            logging.info("      {}".format(force))

        logging.info("Printing force of the hybrid system copy after nb force removal")
        for force in old_sys.getForces():
            logging.info("      {}".format(force))

        #next, we make another copy for reference
        old_copy, hybrid_copy = copy.deepcopy(old_sys), copy.deepcopy(hybrid_sys)

        #remove custom forces
        hybrid_copy.removeForce(4)
        hybrid_copy.removeForce(2)
        hybrid_copy.removeForce(0)

        #define old and hybrid_topologies
        old_topology = self.initial_topology_proposal.old_topology
        hybrid_topology = self.hybrid_factory._hybrid_topology

        #get indices
        old_indices = [atom.index for atom in old_topology.atoms()]
        hybrid_indices = [atom.index for atom in hybrid_topology.atoms()]
        new_indices = [hybrid_index for hybrid_index in hybrid_indices if hybrid_index not in old_indices]

        #get bond parameters
        num_bonds = hybrid_sys.getForce(1).getNumBonds()
        num_indices = 2
        bond_params =  [hybrid_sys.getForce(1).getBondParameters(i) for i in range(num_bonds)]
        new_bond_params = [tuple(param_set) for param_set in bond_params if any(item in param_set[:num_indices] for item in new_indices )]

        #get angle parameters
        num_angles = hybrid_sys.getForce(3).getNumAngles()
        num_indices = 3
        angle_params = [hybrid_sys.getForce(3).getAngleParameters(i) for i in range(num_angles)]
        new_angle_params = [tuple(param_set) for param_set in angle_params if any(item in param_set[:num_indices] for item in new_indices)]

        #get torsion parameters
        num_torsions = hybrid_sys.getForce(5).getNumTorsions()
        num_indices = 4
        torsion_params =  [hybrid_sys.getForce(5).getTorsionParameters(i) for i in range(num_torsions)]
        new_torsion_params = [tuple(param_set) for param_set in torsion_params if any(item in param_set[:num_indices] for item in new_indices)]

        #now to remove existing valence force objects
        hybrid_copy.removeForce(2)
        hybrid_copy.removeForce(1)
        hybrid_copy.removeForce(0)

        #now to remove forces in non-custom valence forces that don't include new atom
        bond_force = openmm.HarmonicBondForce()
        hybrid_copy.addForce(bond_force)
        for parameter_set in new_bond_params:
            bond_force.addBond(*parameter_set)

        angle_force = openmm.HarmonicAngleForce()
        hybrid_copy.addForce(angle_force)
        for parameter_set in new_angle_params:
            angle_force.addAngle(*parameter_set)

        torsion_force = openmm.PeriodicTorsionForce()
        hybrid_copy.addForce(torsion_force)
        for parameter_set in new_torsion_params:
            torsion_force.addTorsion(*parameter_set)

        #now to add old params from old_sys to non_custom valence forces in hybrid_copy
        num_bonds = old_sys.getForce(0).getNumBonds()
        num_indices = 2
        bond_params =  [old_sys.getForce(0).getBondParameters(i) for i in range(num_bonds)]

        num_angles = old_sys.getForce(1).getNumAngles()
        num_indices = 3
        angle_params =  [old_sys.getForce(1).getAngleParameters(i) for i in range(num_angles)]

        num_torsions = old_sys.getForce(2).getNumTorsions()
        num_indices = 4
        torsion_params = [old_sys.getForce(2).getTorsionParameters(i) for i in range(num_torsions)]

        bond_force = hybrid_copy.getForce(0)
        for parameter_set in bond_params:
            bond_force.addBond(*parameter_set)

        angle_force = hybrid_copy.getForce(1)
        for parameter_set in angle_params:
            angle_force.addAngle(*parameter_set)

        torsion_force = hybrid_copy.getForce(2)
        for parameter_set in torsion_params:
            torsion_force.addTorsion(*parameter_set)

        logging.info("Printing the forces from the hybrid_copy...")
        num_bonds = hybrid_copy.getForce(0).getNumBonds()
        num_angles = hybrid_copy.getForce(1).getNumAngles()
        num_torsions = hybrid_copy.getForce(2).getNumTorsions()
        logging.info("      Printing bond forces...")
        for force in [hybrid_copy.getForce(0).getBondParameters(i) for i in range(num_bonds)]:
            logging.info("          {}".format(force))
        logging.info("      Printing angle forces...")
        for force in [hybrid_copy.getForce(1).getAngleParameters(i) for i in range(num_angles)]:
            logging.info("          {}".format(force))
        logging.info("      Printing torsion forces...")
        for force in [hybrid_copy.getForce(2).getTorsionParameters(i) for i in range(num_torsions)]:
            logging.info("          {}".format(force))

        #start a new hybrid topology and add all atoms
        self.new_hybrid_topology = app.Topology()
        new_chain = self.new_hybrid_topology.addChain("0")
        new_res = self.new_hybrid_topology.addResidue("MOL", new_chain)
        for index, atom in enumerate([atom for atom in self.hybrid_factory._hybrid_topology.atoms()]):
            self.new_hybrid_topology.addAtom(atom.name, atom.element, new_res, index)
        logging.info("All atoms have been added to new_hybrid_topology")
        for atom in list(self.new_hybrid_topology.atoms()):
            logging.info("      {}".format(atom))

        #get a list of bonds and add to topology
        bond_pairs = [hybrid_copy.getForce(0).getBondParameters(i)[:2] for i in range(num_bonds)]
        atoms = list(self.new_hybrid_topology.atoms())
        for pair in bond_pairs:
            self.new_hybrid_topology.addBond(atoms[pair[0]], atoms[pair[1]])
        logging.info("All bonds have been added to new_hybrid_topology")
        for bond in bond_pairs:
            logging.info("      {}".format(bond))

        self.hybrid_system = hybrid_copy
        app.pdbfile.PDBFile.writeFile(self.new_hybrid_topology, self.hybrid_factory._hybrid_positions, open('{}{}_{}_hybrid_vacRJMC.pdb'.format(PATH, self.mol1, self.mol2), 'w'))


    def conduct_old_hybrid_top_proposal(self):
        from perses.rjmc.topology_proposal import TopologyProposal
        self.final_topology_proposal = TopologyProposal(new_topology = self.new_hybrid_topology, new_system = self.hybrid_system,
                                                        old_topology = self.mol_attributes_no_nb['mol1'][3], old_system = self.mol_attributes_no_nb['mol1'][1],
                                                        logp_proposal = 0.0,
                                                        new_to_old_atom_map = {value: key for key, value in self.hybrid_factory._old_to_hybrid_map.items()},
                                                        old_chemical_state_key = 'old', new_chemical_state_key = 'new', metadata = None)


    def run_rj_simple_system_lnZ(self, configurations_initial, topology_proposal, n_replicates):
        """
        Function to execute reversibje jump MC

        Arguments
        ---------
        configurations_initial: openmm.Quantity
            n_replicate frames of equilibrium simulation of initial system
        topology_proposal: dict
            perses.topology_proposal object
        n_replicates: int
            number of replicates to simulate

        Returns
        -------
        final_positions: list
            list of openmm position objects for final molecule proposal
        """
        import tqdm
        from perses.rjmc.geometry_lnZ import FFAllAngleGeometryEngine
        final_positions = []
        _geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0)
        lnZs = list()
        for _replicate_idx in tqdm.trange(n_replicates):
            _old_positions = configurations_initial[_replicate_idx, :, :]
            _new_positions, _lp, lnZ = _geometry_engine.propose(topology_proposal, _old_positions, beta)
            final_positions.append(_new_positions)
            lnZs.append(lnZ)

        return final_positions, lnZs

    def run_rj_simple_system(self, configurations_initial, topology_proposal, n_replicates):
        """
        Function to execute reversibje jump MC

        Arguments
        ---------
        configurations_initial: openmm.Quantity
            n_replicate frames of equilibrium simulation of initial system
        topology_proposal: dict
            perses.topology_proposal object
        n_replicates: int
            number of replicates to simulate

        Returns
        -------
        logPs: numpy ndarray
            shape = (n_replicates, 4) where logPs[i] = (reduced potential of initial molecule, log proposal probability, reversed log proposal probability, reduced potential of proposed molecule)
        final_positions: list
            list of openmm position objects for final molecule proposal
        """
        import tqdm
        from perses.rjmc.geometry import FFAllAngleGeometryEngine
        final_positions = []
        logPs = np.zeros([n_replicates, 4])
        logp_forward_list = []
        _geometry_engine = FFAllAngleGeometryEngine(metadata=None, use_sterics=False, n_bond_divisions=1000, n_angle_divisions=180, n_torsion_divisions=360, verbose=True, storage=None, bond_softening_constant=1.0, angle_softening_constant=1.0)
        for _replicate_idx in tqdm.trange(n_replicates):
            _old_positions = configurations_initial[_replicate_idx, :, :]
            _new_positions, _lp, logp_forwards = _geometry_engine.propose(topology_proposal, _old_positions, beta)
            _lp_reverse = _geometry_engine.logp_reverse(topology_proposal, _new_positions, _old_positions, beta)
            _initial_rp = compute_rp(topology_proposal.old_system, _old_positions)
            logPs[_replicate_idx, 0] = _initial_rp
            logPs[_replicate_idx, 1] = _lp
            logPs[_replicate_idx, 2] = _lp_reverse
            final_rp = compute_rp(topology_proposal.new_system, _new_positions)
            logPs[_replicate_idx, 3] = final_rp
            final_positions.append(_new_positions)
            logp_forward_list.append(logp_forwards)
        return logPs, final_positions, logp_forward_list

    def forward_transformation(self, iid_positions_A, _topology_proposal, printer=False):
        """
        Function to conduct run_rj_simple_system RJMC on each conformation of initial molecule (i.e. A --> B)

        Arguments
        ---------
        iid_positions_A: openmm.Quantity
            ndarray of iid conformations of molecule A
        printer: boolean
            whether to print the stacked positions of the proposed molecule

        Returns
        -------
        proposed positions: openmm.Quantity
            num_iterations of final proposal molecule
        self.work_forward: ndarray
            numpy array of forward works
        """
        _data_forward, _proposed_positions, logp_forwards = self.run_rj_simple_system(iid_positions_A, _topology_proposal, self.num_iterations)
        self.work_forward = _data_forward[:, 3] - _data_forward[:, 0] - _data_forward[:, 2] + _data_forward[:, 1]

        _proposed_positions_stacked=convert_to_md(_proposed_positions)
        proposed_positions=unit.Quantity(_proposed_positions_stacked, unit=unit.nanometers)

        if printer:
            print('data_forward: ')
            print(_data_forward)


        return proposed_positions, logp_forwards

# In[1]
test = SmallMoleculeVacuumRJMC(num_iterations = 1000, mol1 = 'propane', mol2 = 'butane', IUPAC = True, SMILES = False, nb_removed = True)
test.eq_simulations()
test.initial_topology_proposal()
test.create_hybrid_factory()
test.create_new_hybrid_sys_top()
test.conduct_old_hybrid_top_proposal()

# In[2]
#proposed_positions, lnZs = test.run_rj_simple_system_lnZ(test.iid1_quantity, test.final_topology_proposal, test.num_iterations)
#lnZs

# # In[3]
# bonds = list(test.new_hybrid_topology.bonds())
# for bond in bonds:
#     print(bond)

# In[3]

# In[4]
#plot_theta_phi(test.iid2, test.mol_attributes_no_nb['mol2'][3], theta_indices = [8,1,3], phi_indices = [8,1,3,2], filename = "{}{}_{}theta_phi.png".format(PATH, test.mol1, test.mol2))
# atom_dict, theta_angle_lnZ = organize_lnZ_distributions(lnZs, proposed_positions, test.new_hybrid_topology, indices_of_interest = None, filename = "{}_lnZ.png".format(PATH))
# theta_angle_lnZ

# In[5]
# plot_theta_lnZ(theta_angle_lnZ, [1])

proposed_posits, logp_forwards = test.forward_transformation(test.iid1_quantity, test.final_topology_proposal, False)

# In[6]
full = []
for dict in logp_forwards:
    lst = []
    for key, value in dict.items():
        lst.append(value)
    full.append(lst)

np_lp = np.array(full)

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set(color_codes = True)

sns.distplot(np_lp[:,0])
sns.distplot(np_lp[:,1])
sns.distplot(np_lp[:,2])
sns.distplot(np_lp[:,3])

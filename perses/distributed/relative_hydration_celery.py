from perses.distributed import feptasks
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator, LangevinIntegrator
from openmmtools.states import ThermodynamicState
import pymbar
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
from perses.tests.utils import createSystemFromIUPAC, get_data_filename
from perses.annihilation.new_relative import HybridTopologyFactory
from perses.rjmc.topology_proposal import TopologyProposal, SmallMoleculeSetProposalEngine, SystemGenerator
from perses.rjmc.geometry import FFAllAngleGeometryEngine
import openeye.oechem as oechem
import celery
from openmoltools import forcefield_generators
import copy


kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

def append_topology(destination_topology, source_topology, exclude_residue_name=None):
    """
    Add the source OpenMM Topology to the destination Topology.

    Parameters
    ----------
    destination_topology : simtk.openmm.app.Topology
        The Topology to which the contents of `source_topology` are to be added.
    source_topology : simtk.openmm.app.Topology
        The Topology to be added.
    exclude_residue_name : str, optional, default=None
        If specified, any residues matching this name are excluded.

    """
    newAtoms = {}
    for chain in source_topology.chains():
        newChain = destination_topology.addChain(chain.id)
        for residue in chain.residues():
            if (residue.name == exclude_residue_name):
                continue
            newResidue = destination_topology.addResidue(residue.name, newChain, residue.id)
            for atom in residue.atoms():
                newAtom = destination_topology.addAtom(atom.name, atom.element, newResidue, atom.id)
                newAtoms[atom] = newAtom
    for bond in source_topology.bonds():
        if (bond[0].residue.name==exclude_residue_name) or (bond[1].residue.name==exclude_residue_name):
            continue
        # TODO: Preserve bond order info using extended OpenMM API
        destination_topology.addBond(newAtoms[bond[0]], newAtoms[bond[1]])

def _build_new_topology(self, current_receptor_topology, oemol_proposed):
    """
    Construct a new topology
    Parameters
    ----------
    oemol_proposed : oechem.OEMol object
        the proposed OEMol object
    current_receptor_topology : app.Topology object
        The current topology without the small molecule

    Returns
    -------
    new_topology : app.Topology object
        A topology with the receptor and the proposed oemol
    mol_start_index : int
        The first index of the small molecule
    """
    oemol_proposed.SetTitle(self._residue_name)
    mol_topology = forcefield_generators.generateTopologyFromOEMol(oemol_proposed)
    new_topology = app.Topology()
    append_topology(new_topology, current_receptor_topology)
    append_topology(new_topology, mol_topology)
    # Copy periodic box vectors.
    if current_receptor_topology._periodicBoxVectors != None:
        new_topology._periodicBoxVectors = copy.deepcopy(current_receptor_topology._periodicBoxVectors)

    return new_topology

def generate_vacuum_hybrid_topology(mol_name="propane", ref_mol_name="butane"):
    from topology_proposal import SmallMoleculeSetProposalEngine, TopologyProposal
    import simtk.openmm.app as app
    from openmoltools import forcefield_generators

    from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC, get_data_filename

    m, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(mol_name)
    refmol = createOEMolFromIUPAC(ref_mol_name)

    initial_smiles = oechem.OEMolToSmiles(m)
    final_smiles = oechem.OEMolToSmiles(refmol)

    gaff_xml_filename = get_data_filename("data/gaff.xml")
    forcefield = app.ForceField(gaff_xml_filename, 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    gaff_filename = get_data_filename('data/gaff.xml')
    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'])
    geometry_engine = FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, pos_old, beta)

    return topology_proposal, pos_old, new_positions


class NonequilibriumFEPSetup(object):
    """
    This class is a helper class for nonequilibrium FEP. It generates the input objects that are necessary for the two
    legs of a relative FEP calculation. For each leg, that is a TopologyProposal, old_positions, and new_positions.
    Importantly, it ensures that the atom maps in the solvent and complex phases match correctly.
    """

    def __init__(self, complex_pdb_filename, new_ligand_smiles, forcefield_files, pressure=1.0*unit.atmosphere, temperature=300.0*unit.kelvin):
        """
        Initialize a NonequilibriumFEPSetup object

        Parameters
        ----------
        complex_pdb_filename : str
            The name of the protein-ligand complex pdb filename
        new_ligand_smiles : str
            The SMILES string representing the other ligand.
        forcefield_files : list of str
            The list of ffxml files that contain the forcefields that will be used
        """
        self._complex_pdb_filename = complex_pdb_filename
        self._new_ligand_smiles = new_ligand_smiles
        self._pressure = pressure
        self._temperature = temperature
        self._barostat_period = 50

        complex_pdbfile = open(self._complex_pdb_filename, 'r')
        pdb_file = app.PDBFile(complex_pdbfile)
        complex_pdbfile.close()

        self._complex_topology_old = pdb_file.topology
        self._complex_positions_old = pdb_file.positions
        self._forcefield = app.ForceField(*forcefield_files)

        if pressure is not None:
            barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
            self._system_generator = SystemGenerator(forcefield_files, barostat=barostat)
        else:
            self._system_generator = SystemGenerator(forcefield_files)

        self._complex_proposal_engine = SmallMoleculeSetProposalEngine([new_ligand_smiles], self._system_generator)
        self._geometry_engine = FFAllAngleGeometryEngine()

        self._complex_topology_old_solvated, self._complex_positions_old_solvated, self._complex_system_old_solvated = self._solvate_system(self._complex_topology_old, self._complex_positions_old)

        self._complex_topology_proposal = self._complex_proposal_engine.propose(self._complex_system_old_solvated, self._complex_topology_old_solvated)
        self._complex_positions_new_solvated, _ = self._geometry_engine.propose(self._complex_topology_proposal, self._complex_positions_old_solvated)
        
    def _solvate_system(self, topology, positions, padding=9.0*unit.angstrom, model='tip3p'):
        """
        Generate a solvated topology, positions, and system for a given input topology and positions.
        For generating the system, the forcefield files provided in the constructor will be used.

        Parameters
        ----------
        topology : app.Topology
            Topology of the system to solvate
        positions : [n, 3] ndarray of Quantity nm

        Returns
        -------
        solvated_topology : app.Topology
            Topology of the system with added waters
        solvated_positions : [n + 3(n_waters), 3] ndarray of Quantity nm
            Solvated positions
        solvated_system : openmm.System
            The parameterized system, containing a barostat if one was specified.
        """
        modeller = app.Modeller(topology, positions)
        modeller.addSolvent(self._forcefield, model=model, padding=padding)
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()
        solvated_system = self._system_generator.build_system(solvated_topology)
        if self._pressure is not None:
            barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
            solvated_system.addForce(barostat)

        return solvated_topology, solvated_positions, solvated_system


class NonequilibriumSwitchingFEP(object):
    """
    This class manages Nonequilibrium switching based relative free energy calculations, carried out on a distributed computing framework.
    """

    default_forward_functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }

    def __init__(self, topology_proposal, pos_old, new_positions, use_dispersion_correction=False, forward_functions=None, ncmc_nsteps=100, concurrency=4, platform_name="OpenCL", temperature=300.0*unit.kelvin):
        self._factory = HybridTopologyFactory(topology_proposal, pos_old, new_positions, use_dispersion_correction=use_dispersion_correction)
        if forward_functions == None:
            self._forward_functions = self.default_forward_functions
        else:
            self._forward_functions = forward_functions
        self._reverse_functions = {param : param_formula.replace("lambda", "(1-lambda)") for param, param_formula in self._forward_functions.items()}

        self._hybrid_system = self._factory.hybrid_system
        self._initial_hybrid_positions = self._factory.hybrid_positions
        self._concurrency = concurrency

        self._ncmc_nsteps = ncmc_nsteps
        self._thermodynamic_state = ThermodynamicState(self._hybrid_system, temperature=temperature)
        self._forward_integrator = AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=self._forward_functions, nsteps_neq=ncmc_nsteps, temperature=temperature)
        self._reverse_integrator = AlchemicalNonequilibriumLangevinIntegrator(alchemical_functions=self._reverse_functions, nsteps_neq=ncmc_nsteps, temperature=temperature)
        self._equilibrium_integrator = LangevinIntegrator(temperature=temperature)


        self._current_positions_forward_result = None
        self._current_positions_reverse_result = None
        self._current_nonequilibrium_work_result = []

        self._platform_name = platform_name
        self._temperature = temperature
        self._forward_work = []
        self._reverse_work = []

    def run_equilibrium(self, n_steps=500):
        """
        Run equilibrium for both end states.
        """
        current_positions_forward = self._current_positions_forward_result.get()
        current_positions_reverse = self._current_positions_reverse_result.get()

        equilibrated_result_forward = feptasks.run_equilibrium.delay(current_positions_forward, n_steps, 0.0, self._forward_functions, self._thermodynamic_state, self._equilibrium_integrator)
        equilibrated_result_reverse = feptasks.run_equilibrium.delay(current_positions_reverse, n_steps, 1.0, self._reverse_functions, self._thermodynamic_state, self._equilibrium_integrator)

        self._current_positions_forward_result = equilibrated_result_forward
        self._current_positions_reverse_result = equilibrated_result_reverse

    def minimize(self, max_steps=50):
        """
        Minimize both end states
        Parameters
        ----------
        max_steps : int, default 50
            max number of steps for openmm minimizer.
        """
        if not self._current_positions_forward_result:
            current_positions_forward = self._initial_hybrid_positions
            current_positions_reverse = self._initial_hybrid_positions
        else:
            current_positions_forward = self._current_positions_forward_result.get()
            current_positions_reverse = self._current_positions_reverse_result.get()

        minimized_forward_result = feptasks.minimize.delay(current_positions_forward, max_steps, 0.0, self._forward_functions, self._thermodynamic_state, self._forward_integrator)
        minimized_reverse_result = feptasks.minimize.delay(current_positions_reverse, max_steps, 0.0, self._reverse_functions, self._thermodynamic_state, self._reverse_integrator)

        self._current_positions_forward_result = minimized_forward_result
        self._current_positions_reverse_result = minimized_reverse_result

    def run_nonequilibrium_task(self, async=True):
        """
        Run nonequilibrium trajectories in both
        Parameters
        ----------
        ncmc_nsteps : int, default 100
            number of steps to take in NCMC protocol
        """
        tasks = []
        current_forward_positions = self._current_positions_forward_result.get()
        current_reverse_positions = self._current_positions_reverse_result.get()

        for i in range(self._concurrency):
            tasks.append(feptasks.run_protocol.s(current_forward_positions, self._ncmc_nsteps, self._thermodynamic_state, self._forward_integrator))
        for i in range(self._concurrency):
            tasks.append(feptasks.run_protocol.s(current_reverse_positions, self._ncmc_nsteps, self._thermodynamic_state, self._reverse_integrator))

        self._current_nonequilibrium_work_result = celery.group(tasks).apply_async()

        if not async:
            work_values = self._current_nonequilibrium_work_result.join()
            self._forward_work.append(work_values[:self._concurrency])
            self._reverse_work.append(work_values[self._concurrency:])

    def collect_ne_work(self):
        """
        Collect the nonequilibrium work, if using async mode.
        """
        if not self._current_nonequilibrium_work_result:
            return

        work_values = self._current_nonequilibrium_work_result.join()
        self._forward_work.append(work_values[:self._concurrency])
        self._reverse_work.append(work_values[self._concurrency:])
        self._current_nonequilibrium_work_result = None

    @property
    def forward_work(self):
        return self._forward_work

    @property
    def reverse_work(self):
        return self._reverse_work

if __name__=="__main__":
    topology_proposal, pos_old, new_positions = generate_vacuum_hybrid_topology()
    ne_fep = NonequilibriumSwitchingFEP(topology_proposal, pos_old, new_positions)
    ne_fep.minimize()
    ne_fep.run_equilibrium()
    ne_fep.run_nonequilibrium_task()
    ne_fep.run_equilibrium()
    ne_fep.collect_ne_work()




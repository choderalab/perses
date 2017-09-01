from perses.distributed import feptasks
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator, LangevinIntegrator
from openmmtools.states import ThermodynamicState
import pymbar
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
from perses.tests.utils import createSystemFromIUPAC, get_data_filename, extractPositionsFromOEMOL
from perses.annihilation.new_relative import HybridTopologyFactory
from perses.rjmc.topology_proposal import TopologyProposal, TwoMoleculeSetProposalEngine, SystemGenerator, SmallMoleculeSetProposalEngine
from perses.rjmc.geometry import FFAllAngleGeometryEngine
import openeye.oechem as oechem
import celery
from openmoltools import forcefield_generators
import copy
import mdtraj as md
from io import StringIO
from openmmtools.constants import kB

class NonequilibriumFEPSetup(object):
    """
    This class is a helper class for nonequilibrium FEP. It generates the input objects that are necessary for the two
    legs of a relative FEP calculation. For each leg, that is a TopologyProposal, old_positions, and new_positions.
    Importantly, it ensures that the atom maps in the solvent and complex phases match correctly.
    """

    def __init__(self, protein_pdb_filename, ligand_file, old_ligand_index, new_ligand_index, forcefield_files, pressure=1.0*unit.atmosphere, temperature=300.0*unit.kelvin):
        """
        Initialize a NonequilibriumFEPSetup object

        Parameters
        ----------
        protein_pdb_filename : str
            The name of the protein pdb file
        ligand_file : str
            the name of the ligand file (any openeye supported format)
        ligand_smiles : list of two str
            The SMILES strings representing the two ligands
        forcefield_files : list of str
            The list of ffxml files that contain the forcefields that will be used
        """
        self._protein_pdb_filename = protein_pdb_filename
        self._pressure = pressure
        self._temperature = temperature
        self._barostat_period = 50

        self._ligand_file = ligand_file
        self._old_ligand_index = old_ligand_index
        self._new_ligand_index = new_ligand_index

        self._old_ligand_oemol = self.load_sdf(self._ligand_file, index=self._old_ligand_index)
        self._new_ligand_oemol = self.load_sdf(self._ligand_file, index=self._new_ligand_index)

        self._old_ligand_positions = extractPositionsFromOEMOL(self._old_ligand_oemol)

        ffxml=forcefield_generators.generateForceFieldFromMolecules([self._old_ligand_oemol, self._new_ligand_oemol])

        self._old_ligand_oemol.SetTitle("MOL")
        self._new_ligand_oemol.SetTitle("MOL")

        self._new_ligand_smiles = oechem.OECreateSmiString(self._new_ligand_oemol, oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)
        #self._old_ligand_smiles = '[H]c1c(c(c(c(c1N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])nc(n3[H])[H])[H])[H])S(=O)(=O)C([H])([H])[H])[H]'
        self._old_ligand_smiles = oechem.OECreateSmiString(self._old_ligand_oemol, oechem.OESMILESFlag_DEFAULT | oechem.OESMILESFlag_Hydrogens)

        print(self._new_ligand_smiles)
        print(self._old_ligand_smiles)

        self._old_ligand_topology = forcefield_generators.generateTopologyFromOEMol(self._old_ligand_oemol)
        self._old_ligand_md_topology = md.Topology.from_openmm(self._old_ligand_topology)
        self._new_ligand_topology = forcefield_generators.generateTopologyFromOEMol(self._new_ligand_oemol)
        self._new_liands_md_topology = md.Topology.from_openmm(self._new_ligand_topology)


        protein_pdbfile = open(self._protein_pdb_filename, 'r')
        pdb_file = app.PDBFile(protein_pdbfile)
        protein_pdbfile.close()

        self._protein_topology_old = pdb_file.topology
        self._protein_md_topology_old = md.Topology.from_openmm(self._protein_topology_old)
        self._protein_positions_old = pdb_file.positions
        self._forcefield = app.ForceField(*forcefield_files)
        self._forcefield.loadFile(StringIO(ffxml))

        print("Generated forcefield")

        self._complex_md_topology_old = self._protein_md_topology_old.join(self._old_ligand_md_topology)
        self._complex_topology_old = self._complex_md_topology_old.to_openmm()

        n_atoms_complex_old = self._complex_topology_old.getNumAtoms()
        n_atoms_protein_old = self._protein_topology_old.getNumAtoms()

        self._complex_positions_old = unit.Quantity(np.zeros([n_atoms_complex_old, 3]), unit=unit.nanometers)
        self._complex_positions_old[:n_atoms_protein_old, :] = self._protein_positions_old
        self._complex_positions_old[n_atoms_protein_old:, :] = self._old_ligand_positions

        if pressure is not None:
            barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
            self._system_generator = SystemGenerator(forcefield_files, barostat=barostat, forcefield_kwargs={'nonbondedMethod' : app.PME})
        else:
            self._system_generator = SystemGenerator(forcefield_files)

        #self._complex_proposal_engine = TwoMoleculeSetProposalEngine(self._old_ligand_smiles, self._new_ligand_smiles, self._system_generator, residue_name="MOL")
        self._complex_proposal_engine = TwoMoleculeSetProposalEngine(self._old_ligand_oemol, self._new_ligand_oemol, self._system_generator, residue_name="MOL")
        self._geometry_engine = FFAllAngleGeometryEngine()

        self._complex_topology_old_solvated, self._complex_positions_old_solvated, self._complex_system_old_solvated = self._solvate_system(self._complex_topology_old, self._complex_positions_old)
        self._complex_md_topology_old_solvated = md.Topology.from_openmm(self._complex_topology_old_solvated)
        print(self._complex_proposal_engine._smiles_list)

        beta = 1.0 / (kB * temperature)

        self._complex_topology_proposal = self._complex_proposal_engine.propose(self._complex_system_old_solvated, self._complex_topology_old_solvated)
        self._complex_positions_new_solvated, _ = self._geometry_engine.propose(self._complex_topology_proposal, self._complex_positions_old_solvated, beta)

        #now generate the equivalent objects for the solvent phase. First, generate the ligand-only topologies and atom map
        self._solvent_topology_proposal, self._old_solvent_positions = self._generate_ligand_only_topologies(self._complex_positions_old_solvated, self._complex_positions_new_solvated)
        self._new_solvent_positions, _ = self._geometry_engine.propose(self._solvent_topology_proposal, self._old_solvent_positions, beta)

    def load_sdf(self, sdf_filename, index=0):
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
        ifs = oechem.oemolistream()
        ifs.open(sdf_filename)
        #get the list of molecules
        mol_list = [oechem.OEMol(mol) for mol in ifs.GetOEMols()]
        #we'll always take the first for now
        mol_to_return = mol_list[index]
        return mol_to_return

    def _solvate_system(self, topology, positions, padding=4.0*unit.angstrom, model='tip3p'):
        """
        Generate a solvated topology, positions, and system for a given input topology and positions.
        For generating the system, the forcefield files provided in the constructor will be used.

        Parameters
        ----------
        topology : app.Topology
            Topology of the system to solvate
        positions : [n, 3] ndarray of Quantity nm
            the positions of the unsolvated system

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
        hs = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H'] and atom.residue.name != "MOL"]
        modeller.delete(hs)
        modeller.addHydrogens(forcefield=self._forcefield)
        print("preparing to add solvent")
        modeller.addSolvent(self._forcefield, model=model, padding=padding)
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()
        print("solvent added, parameterizing")
        solvated_system = self._system_generator.build_system(solvated_topology)
        print("System parameterized")
        if self._pressure is not None:
            barostat = openmm.MonteCarloBarostat(self._pressure, self._temperature, self._barostat_period)
            solvated_system.addForce(barostat)

        return solvated_topology, solvated_positions, solvated_system

    def _generate_ligand_only_topologies(self, old_positions, new_positions):
        """
        This method generates ligand-only topologies and positions from a TopologyProposal containing a solvated complex.
        The output of this method is then used when building the solvent-phase simulation with the same atom map.

        Parameters
        ----------
        topology_proposal : perses.rjmc.TopologyProposal
             TopologyProposal representing the solvated complex transformation

        Returns
        -------
        old_ligand_topology : app.Topology
            The old topology without the receptor or solvent
        new_ligand_topology : app.Topology
            The new topology without the receptor or solvent
        old_ligand_positions : [m, 3] ndarray of Quantity nm
            The positions of the old ligand without receptor or solvent
        new_ligand_positions : [n, 3] ndarray of Quantity nm
            The positions of the new ligand without receptor or solvent
        atom_map : dict of int: it
            The mapping between the two topologies without ligand or solvent.
        """
        old_complex = md.Topology.from_openmm(self._complex_topology_proposal.old_topology)
        new_complex = md.Topology.from_openmm(self._complex_topology_proposal.new_topology)

        complex_atom_map = self._complex_topology_proposal.old_to_new_atom_map

        old_mol_start_index, old_mol_len = self._complex_proposal_engine._find_mol_start_index(old_complex.to_openmm())
        new_mol_start_index, new_mol_len = self._complex_proposal_engine._find_mol_start_index(new_complex.to_openmm())

        old_pos = unit.Quantity(np.zeros([len(old_positions), 3]), unit=unit.nanometers)
        old_pos[:,:] = old_positions
        old_ligand_positions = old_pos[old_mol_start_index:(old_mol_start_index+old_mol_len), :]
        new_ligand_positions = new_positions[new_mol_start_index:(new_mol_start_index+new_mol_len), :]

        #atom_map_adjusted = {}

        #loop through the atoms in the map. If the old index is creater than the old_mol_start_index but less than that
        #plus the old mol length, then it is valid to include its adjusted value in the map.
        #for old_idx, new_idx in complex_atom_map.items():
        #    if old_idx > old_mol_start_index and old_idx < old_mol_len + old_mol_start_index:
        #        atom_map_adjusted[old_idx - old_mol_len] = new_idx - new_mol_start_index

        #subset the topologies:

        old_ligand_topology = old_complex.subset(old_complex.select("resname == 'MOL' "))
        new_ligand_topology = new_complex.subset(new_complex.select("resname == 'MOL' "))

        #solvate the old ligand topology:
        old_solvated_topology, old_solvated_positions, old_solvated_system = self._solvate_system(old_ligand_topology.to_openmm(), old_ligand_positions)

        old_solvated_md_topology = md.Topology.from_openmm(old_solvated_topology)

        #now remove the old ligand, leaving only the solvent
        solvent_only_topology = old_solvated_md_topology.subset(old_solvated_md_topology.select("water"))

        #append the solvent to the new ligand-only topology:
        new_solvated_ligand_md_topology = new_ligand_topology.join(solvent_only_topology)
        nsl,b = new_solvated_ligand_md_topology.to_dataframe()
        #dirty hack because new_solvated_ligand_md_topology.to_openmm() was throwing bond topology error
        new_solvated_ligand_md_topology = md.Topology.from_dataframe(nsl,b)

        new_solvated_ligand_omm_topology = new_solvated_ligand_md_topology.to_openmm()
        new_solvated_ligand_omm_topology.setPeriodicBoxVectors(old_solvated_topology.getPeriodicBoxVectors())

        #create the new ligand system:
        new_solvated_system = self._system_generator.build_system(new_solvated_ligand_omm_topology)

        new_to_old_atom_map = {complex_atom_map[x]-new_mol_start_index:x-old_mol_start_index for x in old_complex.select("resname == 'MOL' ") if x in complex_atom_map.keys()}
        #adjust the atom map to account for the presence of solvent degrees of freedom:
        #By design, all atoms after the ligands are water, and should be mapped.
        n_water_atoms = solvent_only_topology.to_openmm().getNumAtoms()
        for i in range(n_water_atoms):
            new_to_old_atom_map[new_mol_len+i] = old_mol_len + i

        #change the map to accomodate the TP:
        #new_to_old_atom_map = {value : key for key, value in atom_map_adjusted.items()}

        #make a TopologyProposal
        ligand_topology_proposal = TopologyProposal(new_topology=new_solvated_ligand_omm_topology, new_system=new_solvated_system,
                                                    old_topology=old_solvated_topology, old_system=old_solvated_system,
                                                    new_to_old_atom_map=new_to_old_atom_map, old_chemical_state_key='A',
                                                    new_chemical_state_key='B')

        return ligand_topology_proposal, old_solvated_positions

    @property
    def complex_topology_proposal(self):
        return self._complex_topology_proposal
    @property
    def complex_old_positions(self):
        return self._complex_positions_old_solvated
    @property
    def complex_new_positions(self):
        return self._complex_positions_new_solvated
    @property
    def solvent_topology_proposal(self):
        return self._solvent_topology_proposal
    @property
    def solvent_old_positions(self):
        return self._old_solvent_positions
    @property
    def solvent_new_positions(self):
        return self._new_solvent_positions

class NonequilibriumSwitchingFEP(object):
    """
    This class manages Nonequilibrium switching based relative free energy calculations, carried out on a distributed computing framework.
    """

    default_forward_functions = {
        'lambda_sterics' : 'lambda',
        'lambda_electrostatics' : 'lambda',
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

        self._forward_positions_after_switch = []
        self._reverse_positions_after_switch = []

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
            work_values, positions = self._current_nonequilibrium_work_result.join()
            self._forward_work.append(work_values[:self._concurrency])
            self._reverse_work.append(work_values[self._concurrency:])
            self._forward_positions_after_switch.append()

    def collect_ne_work(self):
        """
        Collect the nonequilibrium work, if using async mode.
        """
        if not self._current_nonequilibrium_work_result:
            return

        work_values, positions = self._current_nonequilibrium_work_result.join()
        self._forward_work.append(work_values[:self._concurrency])
        self._reverse_work.append(work_values[self._concurrency:])
        self._current_nonequilibrium_work_result = None

    @property
    def forward_work(self):
        return self._forward_work

    @property
    def reverse_work(self):
        return self._reverse_work

    @property
    def current_free_energy_estimate(self):
        [df, ddf] = pymbar.BAR(self._forward_work, self._reverse_work)
        return [df, ddf]

if __name__=="__main__":
    import os
    gaff_filename = get_data_filename("data/gaff.xml")
    forcefield_files = [gaff_filename, 'tip3p.xml', 'amber99sbildn.xml']
    path_to_schrodinger_inputs = "/Users/grinawap/Downloads"
    protein_file = os.path.join(path_to_schrodinger_inputs, "/Users/grinawap/Downloads/CDK2_fixed_nohet.pdb")
    molecule_file = os.path.join(path_to_schrodinger_inputs, "/Users/grinawap/Downloads/Inputs_for_FEP/CDK2_ligands.mol2")
    fesetup = NonequilibriumFEPSetup(protein_file, molecule_file, 0, 2, forcefield_files)

    ne_fep = NonequilibriumSwitchingFEP(fesetup.solvent_topology_proposal, fesetup.solvent_old_positions, fesetup.solvent_new_positions)
    ne_fep.minimize()
    ne_fep.run_equilibrium()
    ne_fep.run_nonequilibrium_task()
    ne_fep.run_equilibrium()
    ne_fep.collect_ne_work()

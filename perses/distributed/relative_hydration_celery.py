from perses.distributed import feptasks
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
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

    def __init__(self, topology_proposal, pos_old, new_positions, use_dispersion_correction=False, forward_functions=None, concurrency=4, platform_name="OpenCL", temperature=300.0*unit.kelvin):
        self._factory = HybridTopologyFactory(topology_proposal, pos_old, new_positions, use_dispersion_correction=use_dispersion_correction)
        if forward_functions == None:
            self._forward_functions = self.default_forward_functions
        else:
            self._forward_functions = forward_functions
        self._reverse_functions = {param : param_formula.replace("lambda", "(1-lambda)") for param, param_formula in self._forward_functions.items()}

        self._hybrid_system = self._factory.hybrid_system
        self._initial_hybrid_positions = self._factory.hybrid_positions
        self._concurrency = concurrency

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

        equilibrated_result_forward = feptasks.run_equilibrium.delay(current_positions_forward, self._hybrid_system, n_steps, 0.0, self._forward_functions, temperature=self._temperature, platform_name=self._platform_name)
        equilibrated_result_reverse = feptasks.run_equilibrium.delay(current_positions_reverse, self._hybrid_system, n_steps, 1.0, self._reverse_functions, temperature=self._temperature, platform_name=self._platform_name)

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

        minimized_forward_result = feptasks.minimize.delay(current_positions_forward, self._hybrid_system, 0.0, self._forward_functions, max_steps, temperature=self._temperature, platform_name=self._platform_name)
        minimized_reverse_result = feptasks.minimize.delay(current_positions_reverse, self._hybrid_system, 1.0, self._forward_functions, max_steps, temperature=self._temperature, platform_name=self._platform_name)

        self._current_positions_forward_result = minimized_forward_result
        self._current_positions_reverse_result = minimized_reverse_result

    def run_nonequilibrium_task(self, ncmc_nsteps=100, async=True):
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
            tasks.append(feptasks.run_protocol.s(current_forward_positions, self._hybrid_system, ncmc_nsteps, 'forward', self._forward_functions))
        for i in range(self._concurrency):
            tasks.append(feptasks.run_protocol.s(current_reverse_positions, self._hybrid_system, ncmc_nsteps, 'reverse', self._reverse_functions))

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
    pass
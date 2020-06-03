from __future__ import absolute_import

from perses.dispersed import feptasks
from perses.samplers import multistate
from perses.utils.openeye import *
from perses.utils.data import load_smi
from perses.annihilation.relative import HybridTopologyFactory
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol
from perses.rjmc.topology_proposal import TopologyProposal, TwoMoleculeSetProposalEngine, SmallMoleculeSetProposalEngine, PointMutationEngine
from perses.rjmc.geometry import FFAllAngleGeometryEngine

from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState

import pymbar
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
from openmoltools import forcefield_generators
import copy
import mdtraj as md
from io import StringIO
from openmmtools.constants import kB
import logging
import os
import dask.distributed as distributed
import parmed as pm
from collections import namedtuple
from typing import List, Tuple, Union, NamedTuple
from collections import namedtuple
import random
from scipy.special import logsumexp
from pkg_resources import resource_filename
from perses.tests.utils import validate_endstate_energies

ENERGY_THRESHOLD = 1e-2
temperature = 300 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT



class PointMutationExecutor(object):
    """
    Simple, stripped-down class to create a protein-ligand system and allow a mutation of a protein.
    this will allow support for the creation of _two_ relative free energy calculations:
        1. 'wildtype' - 'point mutant' complex hybrid.
        2. 'wildtype' - 'point mutant' receptor hybrid (i.e. with ligand of interest unbound)

    Example (create full point mutation executor and run parallel tempering on both complex and apo phases):
        receptor_path = 'data/perses_jacs_systems/thrombin/Thrombin_protein.pdb'
        ligands_path = 'data/perses_jacs_systems/thrombin/Thrombin_ligands.sdf'
        receptor_filename = resource_filename('openmmforcefields', receptor_path)
        ligand_filename = resource_filename('openmmforcefields', ligands_path)

        pm_delivery = PointMutationExecutor(receptor_filename = receptor_filename,
                                    ligand_filename = ligand_filename,
                                    mutation_chain_id = '2',
                                    mutation_residue_id = '198',
                                     proposed_residue = 'THR',
                                     phase = 'complex',
                                     conduct_endstate_validation = False,
                                     ligand_index = 0,
                                     forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                                     barostat = openmm.MonteCarloBarostat(1.0 * unit.atmosphere, temperature, 50),
                                     forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus}, periodic_forcefield_kwargs = {'nonbondedMethod': app.PME}
                                     small_molecule_forcefields = 'gaff-2.11')

        complex_htf = pm_delivery.get_complex_htf()
        apo_htf = pm_delivery.get_apo_htf()

        #now we can build the hybrid repex samplers
        from perses.annihilation.lambda_protocol import LambdaProtocol
        from openmmtools.multistate import MultiStateReporter
        from perses.samplers.multistate import HybridRepexSampler
        from openmmtools import mcmc

        suffix = 'run'; selection = 'not water'; checkpoint_interval = 10; n_states = 11; n_cycles = 5000

        for htf in [complex_htf, apo_htf]:
            lambda_protocol = LambdaProtocol(functions='default')
            reporter_file = pkl[:-3]+suffix+'.nc'
            reporter = MultiStateReporter(reporter_file, analysis_particle_indices = htf.hybrid_topology.select(selection), checkpoint_interval = checkpoint_interval)
            hss = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep= 4.0 * unit.femtoseconds,
                                                                                  collision_rate=5.0 / unit.picosecond,
                                                                                  n_steps=250,
                                                                                  reassign_velocities=False,
                                                                                  n_restart_attempts=20,
                                                                                  splitting="V R R R O R R R V",
                                                                                  constraint_tolerance=1e-06),
                                                                                  hybrid_factory=htf, online_analysis_interval=10)
            hss.setup(n_states=n_states, temperature=300*unit.kelvin, storage_file = reporter, lambda_protocol = lambda_protocol, endstates=False)
            hss.extend(n_cycles)

    """
    def __init__(self,
                 receptor_filename,
                 ligand_filename,
                 mutation_chain_id,
                 mutation_residue_id,
                 proposed_residue, phase = 'complex',
                 conduct_endstate_validation = False,
                 ligand_index = 0,
                 forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                 barostat = openmm.MonteCarloBarostat(1.0 * unit.atmosphere, temperature, 50),
                 forcefield_kwargs = {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus},
                 periodic_forcefield_kwargs={'nonbondedMethod': app.PME}
                 small_molecule_forcefields = 'gaff-2.11',
                 **kwargs):
        """
        arguments
            receptor_filename : str
                path to receptor; .pdb
            ligand_filename : str
                path to ligand of interest; .sdf or .pdb
            mutation_chain_id : str
                name of the chain to be mutated
            mutation_residue_id : str
                residue id to change
            proposed_residue : str
                three letter code of the residue to mutate to
            phase : str, default complex
                if phase == vacuum, then the complex will not be solvated with water; else, it will be solvated with tip3p
            conduct_endstate_validation : bool, default True
                whether to conduct an endstate validation of the hybrid topology factory
            ligand_index : int, default 0
                which ligand to use
            forcefield_files : list of str, default ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
                forcefield files for proteins and solvent
            barostat : openmm.MonteCarloBarostat, default openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 50)
                barostat to use
            forcefield_kwargs : dict, default {'removeCMMotion': False, 'ewaldErrorTolerance': 1e-4, 'nonbondedMethod': app.NoCutoff, 'constraints' : app.HBonds, 'hydrogenMass' : 4 * unit.amus}
                forcefield kwargs for system parametrization
            periodic_forcefield_kwargs : dict default {'nonbondedMethod':app.PME}
            small_molecule_forcefields : str, default 'gaff-2.11'
                the forcefield string for small molecule parametrization

        TODO : allow argument for separate apo structure if it exists separately
               allow argument for specator ligands besides the 'ligand_filename'
        """
        from openforcefield.topology import Molecule
        from openmmforcefields.generators import SystemGenerator

        # first thing to do is make a complex and apo...
        pdbfile = open(receptor_filename, 'r')
        pdb = app.PDBFile(pdbfile)
        pdbfile.close()
        receptor_positions, receptor_topology, receptor_md_topology = pdb.positions, pdb.topology, md.Topology.from_openmm(pdb.topology)
        receptor_topology = receptor_md_topology.to_openmm()
        receptor_n_atoms = receptor_md_topology.n_atoms

        molecules = []
        ligand_mol = createOEMolFromSDF(ligand_filename, index = ligand_index)
        ligand_mol = generate_unique_atom_names(ligand_mol)
        molecules.append(Molecule.from_openeye(ligand_mol,allow_undefined_stereo = False))
        ligand_positions, ligand_topology = extractPositionsFromOEMol(ligand_mol),  forcefield_generators.generateTopologyFromOEMol(ligand_mol)
        ligand_md_topology = md.Topology.from_openmm(ligand_topology)
        ligand_n_atoms = ligand_md_topology.n_atoms

        #now create a complex
        complex_md_topology = receptor_md_topology.join(ligand_md_topology)
        complex_topology = complex_md_topology.to_openmm()
        complex_positions = unit.Quantity(np.zeros([receptor_n_atoms + ligand_n_atoms, 3]), unit=unit.nanometers)
        complex_positions[:receptor_n_atoms, :] = receptor_positions
        complex_positions[receptor_n_atoms:, :] = ligand_positions

        #now for a system_generator
        self.system_generator = SystemGenerator(forcefields = forcefield_files,
                                                barostat=barostat,
                                                forcefield_kwargs=forcefield_kwargs,
                                                periodic_forcefield_kwargs=periodic_forcefield_kwargs,
                                                small_molecule_forcefield = small_molecule_forcefields,
                                                molecules=molecules,
                                                cache=None)

        #create complex and apo inputs...
        complex_topology, complex_positions, complex_system = self._solvate(complex_topology, complex_positions, 'tip3p', phase = phase)
        apo_topology, apo_positions, apo_system = self._solvate(receptor_topology, receptor_positions, 'tip3p', phase = 'phase')

        geometry_engine = FFAllAngleGeometryEngine(metadata=None,
                                                use_sterics=False,
                                                n_bond_divisions=100,
                                                n_angle_divisions=180,
                                                n_torsion_divisions=360,
                                                verbose=True,
                                                storage=None,
                                                bond_softening_constant=1.0,
                                                angle_softening_constant=1.0,
                                                neglect_angles = False,
                                                use_14_nonbondeds = True)


        #run pipeline...
        htfs = []
        for (top, pos, sys) in zip([complex_topology, apo_topology], [complex_positions, apo_positions], [complex_system, apo_system]):
            point_mutation_engine = PointMutationEngine(wildtype_topology = top,
                                                                 system_generator = self.system_generator,
                                                                 chain_id = mutation_chain_id, #denote the chain id allowed to mutate (it's always a string variable)
                                                                 max_point_mutants = 1,
                                                                 residues_allowed_to_mutate = [mutation_residue_id], #the residue ids allowed to mutate
                                                                 allowed_mutations = [(mutation_residue_id, proposed_residue)], #the residue ids allowed to mutate with the three-letter code allowed to change
                                                                 aggregate = True) #always allow aggregation

            topology_proposal = point_mutation_engine.propose(sys, top)

            new_positions, logp_proposal = geometry_engine.propose(topology_proposal, pos, beta)
            logp_reverse = geometry_engine.logp_reverse(topology_proposal, new_positions, pos, beta)

            forward_htf = HybridTopologyFactory(topology_proposal = topology_proposal,
                                                 current_positions =  pos,
                                                 new_positions = new_positions,
                                                 use_dispersion_correction = False,
                                                 functions=None,
                                                 softcore_alpha = None,
                                                 bond_softening_constant=1.0,
                                                 angle_softening_constant=1.0,
                                                 soften_only_new = False,
                                                 neglected_new_angle_terms = [],
                                                 neglected_old_angle_terms = [],
                                                 softcore_LJ_v2 = True,
                                                 softcore_electrostatics = True,
                                                 softcore_LJ_v2_alpha = 0.85,
                                                 softcore_electrostatics_alpha = 0.3,
                                                 softcore_sigma_Q = 1.0,
                                                 interpolate_old_and_new_14s = False,
                                                 omitted_terms = None)

            if not topology_proposal.unique_new_atoms:
                assert geometry_engine.forward_final_context_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.forward_final_context_reduced_potential})"
                assert geometry_engine.forward_atoms_with_positions_reduced_potential == None, f"There are no unique new atoms but the geometry_engine's forward atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.forward_atoms_with_positions_reduced_potential})"
                vacuum_added_valence_energy = 0.0
            else:
                added_valence_energy = geometry_engine.forward_final_context_reduced_potential - geometry_engine.forward_atoms_with_positions_reduced_potential

            if not topology_proposal.unique_old_atoms:
                assert geometry_engine.reverse_final_context_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's final context reduced potential is not None (i.e. {self._geometry_engine.reverse_final_context_reduced_potential})"
                assert geometry_engine.reverse_atoms_with_positions_reduced_potential == None, f"There are no unique old atoms but the geometry_engine's atoms-with-positions-reduced-potential in not None (i.e. { self._geometry_engine.reverse_atoms_with_positions_reduced_potential})"
                subtracted_valence_energy = 0.0
            else:
                subtracted_valence_energy = geometry_engine.reverse_final_context_reduced_potential - geometry_engine.reverse_atoms_with_positions_reduced_potential


            if conduct_endstate_validation:
                zero_state_error, one_state_error = validate_endstate_energies(forward_htf._topology_proposal, forward_htf, added_valence_energy, subtracted_valence_energy, beta = beta, ENERGY_THRESHOLD = ENERGY_THRESHOLD)
            else:
                pass

            htfs.append(forward_htf)


        self.complex_htf = htfs[0]
        self.apo_htf = htfs[1]

    def get_complex_htf(self):
        return copy.deepcopy(self.complex_htf)
    def get_apo_htf(self):
        return copy.deepcopy(self.apo_htf)


    def _solvate(self,
                        topology,
                        positions,
                        model,
                        phase):
        """
        Generate a solvated topology, positions, and system for a given input topology and positions.
        For generating the system, the forcefield files provided in the constructor will be used.

        Parameters
        ----------
        topology : app.Topology
            Topology of the system to solvate
        positions : [n, 3] ndarray of Quantity nm
            the positions of the unsolvated system
        forcefield : SystemGenerator.forcefield
            forcefield file of solvent to add
        model : str, default 'tip3p'
            solvent model to use for solvation

        Returns
        -------
        solvated_topology : app.Topology
            Topology of the system with added waters
        solvated_positions : [n + 3(n_waters), 3] ndarray of Quantity nm
            Solvated positions
        solvated_system : openmm.System
            The parameterized system, containing a barostat if one was specified.
        """
        from pdbfixer import PDBFixer
        from simtk.openmm.app import PDBFile
        import os
        modeller = app.Modeller(topology, positions)


        #now we have to add missing atoms
        if phase != 'vacuum':
            modeller.addSolvent(self.system_generator.forcefield, model=model, padding = 1.0 * unit.nanometers, ionicStrength=0.15*unit.molar)
        else:
            pass

        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()

        # canonicalize the solvated positions: turn tuples into np.array
        solvated_positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
        solvated_system = self.system_generator.create_system(solvated_topology)

        return solvated_topology, solvated_positions, solvated_system

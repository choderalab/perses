#!/bin/env python

"""
Example illustrating use of expanded ensembles framework to perform a generic expanded ensembles simulation over chemical species.

This could represent sampling of small molecules, protein mutants, or both.

"""


import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import openeye.oechem as oechem
import openeye.oeomega as oeomega
import openmoltools
import logging
import copy
import perses.rjmc.topology_proposal as topology_proposal
import perses.bias.bias_engine as bias_engine
import perses.rjmc.geometry as geometry
import perses.annihilation.ncmc_switching as ncmc_switching


def generate_initial_molecule(mol_smiles):
    """
    Generate an oemol with a geometry
    """
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, mol_smiles)
    oechem.OEAddExplicitHydrogens(mol)
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega(mol)
    return mol

def oemol_to_openmm_system(oemol, molecule_name):
    """
    Create an openmm system out of an oemol

    Returns
    -------
    system : openmm.System object
        the system from the molecule
    positions : [n,3] np.array of floats
    """
    _ , tripos_mol2_filename = openmoltools.openeye.molecule_to_mol2(oemol, tripos_mol2_filename=molecule_name + '.tripos.mol2', conformer=0, residue_name='MOL')
    gaff_mol2, frcmod = openmoltools.openeye.run_antechamber(molecule_name, tripos_mol2_filename)
    prmtop_file, inpcrd_file = openmoltools.utils.run_tleap(molecule_name, gaff_mol2, frcmod)
    prmtop = app.AmberPrmtopFile(prmtop_file)
    system = prmtop.createSystem(implicitSolvent=app.OBC1)
    crd = app.AmberInpcrdFile(inpcrd_file)
    return system, crd.getPositions(asNumpy=True), prmtop.topology


def run():
    # Create initial model system, topology, and positions.
    smiles_list = ["CC", "CCC", "CCCC"]
    initial_molecule = generate_initial_molecule("CC")
    initial_sys, initial_pos, initial_top = oemol_to_openmm_system(initial_molecule, "ligand_old")
    smiles = 'CC'
    stats = {ms : 0 for ms in smiles_list}
    # Run parameters
    temperature = 300.0 * unit.kelvin # temperature
    pressure = 1.0 * unit.atmospheres # pressure
    collision_rate = 5.0 / unit.picoseconds #collision rate for Langevin dynamics


    #Create proposal metadata, such as the list of molecules to sample (SMILES here)
    proposal_metadata = {'smiles_list': smiles_list}

    transformation = topology_proposal.SingleSmallMolecule(proposal_metadata)

    #initialize weight calculation engine, along with its metadata
    bias_calculator = bias_engine.MinimizedPotentialBias(smiles_list)

    #Initialize NCMC engines.
    switching_timestep = 1.0 * unit.femtosecond # timestep for NCMC velocity Verlet integrations
    switching_nsteps = 10 # number of steps to use in NCMC integration
    switching_functions = { # functional schedules to use in terms of `lambda`, which is switched from 0->1 for creation and 1->0 for deletion
        'lambda_sterics' : 'lambda',
        'lambda_electrostatics' : 'lambda',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'sqrt(lambda)',
        'lambda_torsions' : 'lambda'
        }
    ncmc_engine = ncmc_switching.NCMCEngine(temperature=temperature, timestep=switching_timestep, nsteps=switching_nsteps, functions=switching_functions)

    #initialize GeometryEngine
    geometry_metadata = {'data': 0} #currently ignored
    geometry_engine = geometry.FFGeometryEngine(geometry_metadata)

    # Run a anumber of iterations.
    niterations = 50
    system = initial_sys
    topology = initial_top
    positions = initial_pos
    current_log_weight = bias_calculator.g_k(smiles)
    n_accepted = 0
    propagate = True
    for i in range(niterations):
        # Store old (system, topology, positions).

        # Propose a transformation from one chemical species to another.
        state_metadata = {'molecule_smiles' : smiles}
        top_proposal = transformation.propose(system, topology, positions, state_metadata) #get a new molecule

        # QUESTION: What about instead initializing StateWeight once, and then using
        # log_state_weight = state_weight.computeLogStateWeight(new_topology, new_system, new_metadata)?
        log_weight = bias_calculator.g_k(top_proposal.metadata['molecule_smiles'])

        # Perform alchemical transformation.

        # Alchemically eliminate atoms being removed.

        #print(old_alchemical_system)
        [ncmc_old_positions, ncmc_elimination_logp] = ncmc_engine.integrate(top_proposal, positions, direction='delete')
        #print(ncmc_old_positions)
        #print(ncmc_elimination_logp)
        # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
        # QUESTION: Again, maybe we want to have the geometry engine initialized once only?
        geometry_proposal = geometry_engine.propose(top_proposal.new_to_old_atom_map, top_proposal.new_system, system, ncmc_old_positions)

        # Alchemically introduce new atoms.
        [ncmc_new_positions, ncmc_introduction_logp] = ncmc_engine.integrate(top_proposal, geometry_proposal.new_positions, direction='insert')
        #print(ncmc_new_positions)
        #print(ncmc_introduction_logp)

        # Compute total log acceptance probability, including all components.
        logp_accept = top_proposal.logp_proposal + geometry_proposal.logp + ncmc_elimination_logp + ncmc_introduction_logp + log_weight/log_weight.unit - current_log_weight/current_log_weight.unit
        #print(logp_accept)

        # Accept or reject.
        if ((logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept))) and not np.any(np.isnan(ncmc_new_positions)):
            # Accept.
            n_accepted+=1
            (system, topology, positions, current_log_weight, smiles) = (top_proposal.new_system, top_proposal.new_topology, ncmc_new_positions, log_weight, top_proposal.metadata['molecule_smiles'])
        else:
            # Reject.
            logging.debug("reject")
        stats[smiles]+=1
        print(positions)
        if propagate:
            p_system = copy.deepcopy(system)
            integrator = openmm.LangevinIntegrator(temperature, collision_rate, switching_timestep)
            context = openmm.Context(p_system, integrator)
            context.setPositions(positions)
            print(context.getState(getEnergy=True).getPotentialEnergy())
            integrator.step(1000)
            state = context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True)
            del context, integrator, p_system

    print("The total number accepted was %d out of %d iterations" % (n_accepted, niterations))
    print(stats)

#
# MAIN
#

if __name__=="__main__":
    logger = logging.getLogger('antechamber')
    logger.setLevel(logging.INFO)
    run()

import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import openeye.oechem as oechem
import openeye.oeomega as oeomega
import openmoltools
import logging
from pkg_resources import resource_filename
import os
import copy
import perses.rjmc.topology_proposal as topology_proposal
import perses.bias.bias_engine as bias_engine
import perses.rjmc.geometry as geometry
import perses.annihilation.ncmc_switching as ncmc_switching

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

# TURN LOGGER ON
#logging.basicConfig(level=logging.DEBUG)

def extractPositionsFromOEMOL(molecule):
    positions = unit.Quantity(np.zeros([molecule.NumAtoms(), 3], np.float32), unit.angstroms)
    coords = molecule.GetCoords()
    for index in range(molecule.NumAtoms()):
        positions[index,:] = unit.Quantity(coords[index], unit.angstroms)
    return positions

def generate_initial_molecule(mol_smiles):
    """
    Generate an oemol with a geometry
    """
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, mol_smiles)
    mol.SetTitle("MOL")
    oechem.OEAddExplicitHydrogens(mol)
    oechem.OETriposAtomNames(mol)
    oechem.OETriposBondTypeNames(mol)
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
    from openmoltools.amber import run_tleap, run_antechamber
    from openmoltools.openeye import molecule_to_mol2
    _ , tripos_mol2_filename = molecule_to_mol2(oemol, tripos_mol2_filename=molecule_name + '.tripos.mol2', conformer=0, residue_name='MOL')
    gaff_mol2, frcmod = run_antechamber(molecule_name, tripos_mol2_filename)
    prmtop_file, inpcrd_file = run_tleap(molecule_name, gaff_mol2, frcmod)
    prmtop = app.AmberPrmtopFile(prmtop_file)
    system = prmtop.createSystem(implicitSolvent=None, removeCMMotion=False)
    crd = app.AmberInpcrdFile(inpcrd_file)
    return system, crd.getPositions(asNumpy=True), prmtop.topology

def oemol_to_omm_ff(oemol, molecule_name):
    from perses.rjmc import topology_proposal
    from openmoltools import forcefield_generators
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])
    topology = forcefield_generators.generateTopologyFromOEMol(oemol)
    system = system_generator.build_system(topology)
    positions = extractPositionsFromOEMOL(oemol)
    return system, positions, topology

def test_run_example():
    # Run parameters
    temperature = 300.0 * unit.kelvin # temperature
    pressure = 1.0 * unit.atmospheres # pressure
    collision_rate = 5.0 / unit.picoseconds # collision rate for Langevin dynamics
    timestep = 1.0 * unit.femtoseconds # propagation timestep
    nsteps_per_iteration = 50 # number of timesteps for propagation step iteration
    switching_timestep = 1.0 * unit.femtosecond # timestep for NCMC velocity Verlet integrations
    switching_nsteps = 50 # number of steps to use in NCMC integration
    niterations = 50 # number of sampler iterations

    # Compute kT and inverse temperature.
    kT = kB * temperature
    beta = 1.0 / kT

    # Create initial model system, topology, and positions.
    smiles_list = ["CC", "CCC", "CCCC", "CCC(C)C", "CC(C)(C)C", "Cc1ccccc1"]
    #smiles_list = ["CC", "CCC", "CCCC", "CCCCC"]
    stats = { smiles : 0 for smiles in smiles_list } # stats[smiles] is the number of times molecule 'smiles' has been visited

    # Initialize sampler state.
    smiles = 'CC' # current sampler state
    initial_molecule = generate_initial_molecule("CC")
    initial_sys, initial_pos, initial_top = oemol_to_omm_ff(initial_molecule, "ligand_old")
    gaff_xml_filename = get_data_filename('data/gaff.xml')
    system_generator = topology_proposal.SystemGenerator([gaff_xml_filename])

    # Create proposal metadata, such as the list of molecules to sample (SMILES here)
    proposal_metadata = {'smiles_list': smiles_list}
    transformation = topology_proposal.SmallMoleculeSetProposalEngine(smiles_list, app.Topology(), system_generator)

    # Initialize weight calculation engine, along with its metadata
    bias_calculator = bias_engine.MinimizedPotentialBias(smiles_list, implicit_solvent=None)

    # Initialize NCMC engines.
    switching_functions = { # functional schedules to use in terms of `lambda`, which is switched from 0->1 for creation and 1->0 for deletion
        'lambda_sterics' : 'lambda',
        'lambda_electrostatics' : 'lambda',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
        }
    platform = openmm.Platform.getPlatformByName("Reference")
    ncmc_engine = ncmc_switching.NCMCEngine(temperature=temperature, timestep=switching_timestep, nsteps=switching_nsteps, functions=switching_functions, platform=platform)

    #initialize GeometryEngine
    geometry_metadata = {'data': 0} #currently ignored
    geometry_engine = geometry.FFAllAngleGeometryEngine(geometry_metadata)

    # TODO: bias calculator should return unitless quantities
    current_log_weight = bias_calculator.g_k(smiles) / kT

    # Run a number of iterations.
    # TODO: This should be incorporated into an MCMCSampler / SAMSSampler class.
    n_accepted = 0
    system = initial_sys
    topology = initial_top
    positions = initial_pos
    print("")
    for iteration in range(niterations):

        #
        # PROPAGATE POSITIONS
        #

        # Propagate with Langevin dynamics to achieve ergodic sampling
        integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        potential = context.getState(getEnergy=True).getPotentialEnergy()
        if np.isnan(potential/kT):
            raise Exception("Potential energy of full system is NaN")
        print('Iteration %5d: potential = %12.3f kT' % (iteration, potential/kT))
        integrator.step(nsteps_per_iteration)
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        del context, integrator

        #
        # UPDATE CHEMICAL STATE
        #

        # Propose a transformation from one chemical species to another.
        state_metadata = {'molecule_smiles' : smiles}
        top_proposal = transformation.propose(system, topology, positions, beta, state_metadata) # get a new molecule

        # QUESTION: What about instead initializing StateWeight once, and then using
        # log_state_weight = state_weight.computeLogStateWeight(new_topology, new_system, new_metadata)?
        # TODO: bias calculator should return unitless quantities
        log_weight = bias_calculator.g_k(top_proposal.molecule_smiles) / kT

        # Alchemically eliminate atoms being removed.
        [ncmc_old_positions, ncmc_elimination_logp] = ncmc_engine.integrate(top_proposal, positions, direction='delete')
        # Check that positions are not NaN
        if np.any(np.isnan(ncmc_old_positions)):
            raise Exception("Positions are NaN after NCMC delete with %d steps" % switching_nsteps)

        # Generate coordinates for new atoms and compute probability ratio of old and new probabilities.
        # QUESTION: Again, maybe we want to have the geometry engine initialized once only?
        top_proposal.old_positions = ncmc_old_positions
        geometry_new_positions, geometry_logp  = geometry_engine.propose(top_proposal)

        # Alchemically introduce new atoms.
        [ncmc_new_positions, ncmc_introduction_logp] = ncmc_engine.integrate(top_proposal, geometry_new_positions, direction='insert')
        # Check that positions are not NaN
        if np.any(np.isnan(ncmc_new_positions)):
            raise Exception("Positions are NaN after NCMC insert with %d steps" % switching_nsteps)

        # Compute total log acceptance probability, including all components.
        logp_accept = top_proposal.logp_proposal + geometry_logp + ncmc_elimination_logp + ncmc_introduction_logp + log_weight - current_log_weight
        print("Proposal from '%12s' -> '%12s' : logp_accept = %+10.4e [logp_proposal %+10.4e geometry_logp %+10.4e ncmc_elimination_logp %+10.4e ncmc_introduction_logp %+10.4e log_weight %+10.4e current_log_weight %+10.4e]"
            % (smiles, top_proposal.molecule_smiles, logp_accept, top_proposal.logp_proposal, geometry_logp, ncmc_elimination_logp, ncmc_introduction_logp, log_weight, current_log_weight))

        # Accept or reject.
        accept = ((logp_accept>=0.0) or (np.random.uniform() < np.exp(logp_accept)))
        if accept:
            logging.debug("accept")
            n_accepted += 1
            (system, topology, positions, current_log_weight, smiles) = (top_proposal.new_system, top_proposal.new_topology, ncmc_new_positions, log_weight, top_proposal.molecule_smiles)
        else:
            logging.debug("reject")

        # Update statistics.
        stats[smiles] += 1

    print("The total number accepted was %d out of %d iterations" % (n_accepted, niterations))
    print(stats)

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing
    In the source distribution, these files are in ``perses/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the openmoltools folder).
    """

    fn = resource_filename('perses', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn


if __name__=="__main__":
    test_run_example()

__author__ = 'dominic rufa'

"""
Application API for setting up perses relative free energy calculations on Folding@Home.

argv[1]: setup.yaml (argument for perses.app.setup_relative_calculation.getSetupOptions)
"""
import numpy as np
import os
import simtk.unit as unit
from simtk import openmm
import logging
import datetime

# TODO: Move logging filters to utils module
class TimeFilter(logging.Filter):
    """
    Logging filter that shows how much time (in seconds) have elapsed since the last logging statement.
    """
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated
        delta = datetime.datetime.fromtimestamp(record.relativeCreated/1000.0) - datetime.datetime.fromtimestamp(last/1000.0)
        record.relative = '{0:.2f}'.format(delta.seconds + delta.microseconds/1000000.0)
        self.last = record.relativeCreated
        return True

# TODO: Move logging configuration helpers to utils module
fmt = logging.Formatter(fmt="%(asctime)s:(%(relative)ss):%(name)s:%(message)s")
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
[hndl.addFilter(TimeFilter()) for hndl in _logger.handlers]
[hndl.setFormatter(fmt) for hndl in _logger.handlers]

# Default alchemical protocols
# TODO: Move these to alchemical module that encapsulates alchemical factory and best-practices protocols
x = 'lambda'
# TODO change this for perses.annihilation.LambdaProtocol.default_functions
DEFAULT_ALCHEMICAL_FUNCTIONS = {
                             'lambda_sterics_core': x,
                             'lambda_electrostatics_core': x,
                             'lambda_sterics_insert': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                             'lambda_sterics_delete': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
                             'lambda_electrostatics_insert': f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
                             'lambda_electrostatics_delete': f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
                             'lambda_bonds': x,
                             'lambda_angles': x,
                             'lambda_torsions': x}


# TODO: Reconsider this for integration into some kind of class hierarchy that represents free energy methods that can be run on Folding@home?
# For example, we could have
# FoldingAtHomeSimulation
#  + FoldingAtHomeFreeEnergyCalculation
#    + NonequilibriumCycling
#    + NonequilibriumShooting
#    + IndependentAlchemicalSimulations
#    + SAMS
#    + TimesSquareSampling
#
# These classes could all generate their own complete RUN{run_id}/... directory hierarchies for Folding@home, complete with all necessary
# auxiliary files: {system, state, integrator, core}.xml, metadata, etc

def make_neq_integrator(nsteps_eq=250000, nsteps_neq=250000, neq_splitting='V R H O R V', timestep=4.0 * unit.femtosecond, alchemical_functions=DEFAULT_ALCHEMICAL_FUNCTIONS):
    """
    Construct an openmmtools.integrators.PeriodicNonequilibriumIntegrator for collecting nonequilibrium work measurement data on Folding@home

    Parameters
    ----------
    nsteps_eq : int, default=250000
        Number of equilibration steps to dwell within lambda = 0 or 1 when reached
    nsteps_neq : int, default=250000
        Number of nonequilibrium switching steps for 0->1 and 1->0 switches
    neq_splitting : str, optional, default='V R H O R V' (BAOAB nonequilibrum integrator)
        Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.
        "H" increments the global parameter `lambda` by 1/nsteps_neq for each step and accumulates protocol work.
    timestep : int, default=4.0 * unit.femtosecond
        Timestep to use for integrator.
    alchemical_functions : dict, optional, default=DEFAULT_ALCHEMICAL_FUNCTIONS
        Dictionary containing alchemical functions describing how `lambda` should modify all interactions. See DEFAULT_ALCHEMICAL_FUNCTIONS for example.

    Returns
    -------
    integrator : openmmtools.integrators.PeriodicNonequilibriumIntegrator

    """
    from openmmtools.integrators import PeriodicNonequilibriumIntegrator
    integrator = PeriodicNonequilibriumIntegrator(alchemical_functions, nsteps_eq, nsteps_neq, neq_splitting, timestep=timestep)
    return integrator

# TODO: Integrate this into class hierarchy that understands what kind of free energy calculations can be run on Folding@home?
def make_core_file(numSteps,
                   xtcFreq,
                   globalVarFreq,
                   xtcAtoms='solute',
                   precision='mixed',
                   globalVarFilename='globals.csv',
                   directory='.'):
    """ 
    Generate core.xml file for Folding@home OpenMM core22 

    See all options for core.xml described in core22 README:
    https://github.com/foldingathome/openmm-core

    .. note :: 
    
    * OpenMM core22 can mis-order atoms if `xtcAtoms = solute`, so it is 
      recommended that a comma-separated list of atoms to be written be specified 
      until this is resolved.
    
      https://github.com/FoldingAtHome/openmm-core/issues/360

    Parameters
    ----------
    numSteps : int
        Number of steps to perform
    xtcFreq : int
        Frequency to save configuration to disk
    globalVarFreq : int
        Frequency to save variables to globalVarFilename
    xtcAtoms : str, optional, default='solute'
        Which atoms to save
    precision : str, optional, default='mixed'
        Precision of simulation
    globalVarFilename : str, optional, default='globals.csv'
        Filename to store global simulation results
    directory : str, optional, default='.'
        Location on disk to save core.xml file

    """
    core_parameters = {
        'numSteps': numSteps,
        'xtcFreq': xtcFreq, # once per ns
        'xtcAtoms': xtcAtoms,
        'precision': precision,
        'globalVarFilename': globalVarFilename,
        'globalVarFreq': globalVarFreq,
    }
    # Serialize core.xml
    import dicttoxml
    with open(f'{directory}/core.xml', 'wt') as outfile:
        #core_parameters = create_core_parameters(phase)
        xml = dicttoxml.dicttoxml(core_parameters, custom_root='config', attr_type=False)
        from xml.dom.minidom import parseString
        dom = parseString(xml)
        outfile.write(dom.toprettyxml())

def relax_structure(temperature,
                    system,
                    positions,
                    nequil=1000,
                    n_steps_per_iteration=250,
                    platform_name='CUDA',
                    timestep=4.*unit.femtosecond,
                    collision_rate=90./unit.picosecond):
    """
    Minimize and equilibrate the hybrid system to prepare for Folding@home simulation.

    Parameters
    ----------
    temperature : simtk.unit.Quantity with units compatible with kelvin
        temperature of simulation
    system : openmm.System
        system object for simulation
    positions : simtk.unit.Quantity of shape (natoms,3) with units compatible with nanometers
        Positions of the atoms in the system
    nequil : int, optional, default = 1000
        number of equilibration applications
    n_steps_per_iteration : int, optional, default = 250
        numper of steps per nequil
    platform name : str, optional, default='CUDA'
        platform to run openmm on. OpenCL is best as this is what is used on FAH
    timestep : simtk.unit.Quantity, default = 4*unit.femtosecond
       timestep for equilibration NOT for production
    collision_rate : simtk.unit.Quantity, default=90./unit.picosecond
        Collision rate for equilibration
    
    return
        state : openmm.State
            state of simulation (getEnergy=True, getForces=True, getPositions=True, getVelocities=True, getParameters=True)
    """
    _logger.info(f'Starting to relax')

    # Create integrator for equilibration
    from openmmtools.integrators import LangevinIntegrator
    integrator = LangevinIntegrator(temperature=temperature, timestep=timestep, collision_rate=collision_rate)

    # Prepare the plaform
    platform = openmm.Platform.getPlatformByName(platform_name)
    if platform_name in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
    if platform_name in ['CUDA']:
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
    context = openmm.Context(system, integrator, platform)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    context.setPositions(positions)

    # Minimize
    _logger.info(f'Starting to minimise')
    openmm.LocalEnergyMinimizer.minimize(context)

    # Equilibrate
    _logger.info(f'set velocities to temperature')
    context.setVelocitiesToTemperature(temperature)
    _logger.info(f'Starting to equilibrate for {nequil*n_steps_per_iteration*timestep}')
    integrator.step(nequil*n_steps_per_iteration)
    context.setVelocitiesToTemperature(temperature)
    state = context.getState(getEnergy=True, getForces=True, getPositions=True, getVelocities=True, getParameters=True)
    _logger.info(f'Relax done')

    # Clean up
    del context, integrator

    # Return final state for Folding@Home packaging
    return state

# TODO: Replace this with a sensible class hierarchy based API once we refactor perses API
def run_neq_fah_setup(ligand_file,
                      old_ligand_index,
                      new_ligand_index,
                      forcefield_files,
                      trajectory_directory,
                      complex_box_dimensions=(9.8, 9.8, 9.8),
                      solvent_box_dimensions=(3.5, 3.5, 3.5),
                      timestep=4.0, # femtoseconds (implicit)
                      eq_splitting='V R O R V',
                      neq_splitting='V R H O R V',
                      measure_shadow_work=False,
                      pressure=1.0,
                      temperature=300. * unit.kelvin,
                      solvent_padding=9*unit.angstroms,
                      phases=['complex','solvent','vacuum'],
                      phase_project_ids=None,
                      protein_pdb=None,
                      receptor_mol2=None,
                      small_molecule_forcefield='openff-2.0.0',
                      small_molecule_parameters_cache=None,
                      atom_expression=['IntType'],
                      bond_expression=['DefaultBonds'],
                      spectators=None,
                      neglect_angles=False,
                      anneal_14s=False,
                      nonbonded_method='PME',
                      map_strength=None,
                      softcore_v2=False,
                      save_setup_pickle_as=None,
                      render_atom_map=False,
                      alchemical_functions=DEFAULT_ALCHEMICAL_FUNCTIONS,
                      num_equilibration_iterations=1000,
                      num_equilibration_steps_per_iteration=250,
                      nsteps_eq=250000,
                      nsteps_neq=250000,
                      fe_type='fah',
                      collision_rate=1./unit.picoseconds,
                      collision_rate_setup=90./unit.picoseconds,
                      constraint_tolerance=1e-6,
                      n_steps_per_move_application=250,
                      globalVarFreq=250,
                      setup='small_molecule',
                      protein_kwargs=None,
                      ionic_strength=0.15*unit.molar,
                      remove_constraints=False,
                      rmsd_restraint=False,
                      **kwargs):
    """
    Set up perses relative free energy calculations for Folding@home

    main execution function that will:
        - create a directory for each phase according to the `trajectory_directory` argument
        - make a subdirectory named f"RUN_{old_ligand_index}_{new_ligand_index}" given the specified ligand indices of the `ligand_file`
        - create topology proposals for all phases
        - create/serialize hybrid factories or all phases (and validate endstates)
        - create/serialize an openmmtools.integrators.PeriodicNonequilibriumIntegrator for all phases
        - relax generated structures with a minimizer and LangevinIntegrator for all phases
        - create/serialize a state associated with the relaxed structures
        - create/serialize a `core.xml` object for all phases

    Examples
    --------

    >>> run_neq_fah_setup('ligand.sdf', 0, 1,['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml'],'RUN0',protein_pdb='protein.pdb', phases=['complex','solvent','vacuum'],phase_project_ids={'complex':14320,'solvent':14321,'vacuum':'vacuum'})
    
    TODO
    ----
    * Check whether temperature, pressure, and timestep are all with or without units specified
    * Always place a virtual bond between protein subunits and ligand and closest protein subunit
    * Refector this to use a class with configurable default parameters? Or just use new perses class hierarchy and provide a different executor?

    Parameters
    ----------
    ligand_file : str
        .sdf (or any openeye-readable) file containing ligand labeled indices and structures
    old_ligand_index : int
        index of the old ligand within ligand_file (0-indexed)
    new_ligand_index : int
        index of the new ligand within ligand_file (0-indexed)
    forcefield_files : list of str
        list of OpenMM ffxml forcefields to use for complex/solvent parameterization
    trajectory_directory : str
        RUNXXX directory to generate for FAH deployment
    complex_box_dimensions : Vec3, default=(9.8, 9.8, 9.8)
        define box dimensions of complex phase (in nm) to ensure simulations run at near-uniform speed
    solvent_box_dimensions : Vec3, default=(3.5, 3.5, 3.5)
        define box dimensions of solvent phase (in nm) to ensure simulations run at near-uniform speed
    timestep : simtk.unit.Quantity or float, optional, default=4.
        timestep for production integration; float or int will be interpreted as femtoseconds
    eq_splitting : str, optional, default = 'V R O R V'
        splitting string of relaxation dynamics
    neq_splitting : str, optional, default = 'V R H O R V'
        splitting string of nonequilibrium dynamics
    measure_shadow_work : bool, optional, default=False
        True/False to measure shadow work in NonequilibriumLangevinIntegrator
        Measuring shadow work will significantly slow integration
        Shadow work is added to the integrator globals.csv file
    pressure : float or simtk.unit.Quantity, optional, default=1.
        pressure for simulation; float or int will be interepreted as atm
    temperature : float or simtk.unit.Quantity, optional, default=300.*unit.kelvin,
        temperature for simulation; float or int will be interpreted as kelvin
    phases : list, optional, default = ['complex','solvent','vacuum','apo']
        phases to run, where allowed phases are:
        'complex','solvent','vacuum','apo'
    phase_project_ids : dict, optional, default=None
        Each phase in 'phases' must have a corresponding FAH project id specified
        e.g. phase_project_ids = { 'complex' : 13458, 'solvent' : 13459 }
    protein_pdb : str, optional, default=None
        name of protein file
        Protein file can include molecules (such as solvent) for which parameters exist in the specified OpenMM ffxml files
    receptor_mol2 : str, optional, default=None
        If the receptor is not a protein, a receptor mol2 file can be specified; this will be parameterized by the SystemGenerator
        e.g. for macromolecular hosts
    small_molecule_forcefield : str, optional, default='openff-2.0.0'
        Small molecule forcefield filename for use with the openmmforcefields SystemGenerator
        Available options are described at https://github.com/openmm/openmmforcefields
        and include `gaff-*` and `openff-*` force field generations
    small_molecule_parameters_cache : str, optional, default=None
        cache file containing small molecule forcefield files (to avoid repeatedly reparameterizing)
    atom_expression : list, optional, default=['IntType']
        list of string for atom mapping criteria. see oechem.OEExprOpts for options
    bond_expression : list, optional, default=['DefaultBonds']
        list of string for bond mapping criteria. see oechem.OEExprOpts for options
    map_strength : 'str', optional, default=None
        atom and bond expressions will be ignored, and either a 'weak', 'default', or 'strong' map_strength will be used.
    spectators : str, optional, default=None
        file describing chemical species of any non-alchemical molecules (e.g. sdf, mol2) in simulation that must be parameterized separately by SystemGenerator
        These molecular identities will be fed to SystemGenerator
    neglect_angles : bool, optional, default=False
        wether to use angle terms in building of unique-new groups. False is strongly recommended
    anneal_14s : bool, optional, default False
        Whether to anneal 1,4 interactions over the protocol;
    nonbonded_method : str, optional, default='PME'
        nonbonded method to use
    softcore_v2 : bool, optional, default=False
        If True, will use v2 softcore from de Groot and colleagues
    alchemical_functions : dict, optional, default=DEFAULT_ALCHEMICAL_FUNCTIONS
        alchemical functions for transformation
    num_equilibration_iterations : int, optional, default=1000
        number of equilibration steps to do during set up
    num_equilibration_steps_per_iteration : int, optional, default=250,
        number of steps per iteration during EQUILIBRATION. default is 250 steps of 2fs, 1000 times which is 500ps of equilibration for SETUP
    nsteps_eq : int, optional, default=250000
        number of normal MD steps to take for FAH integrator for PRODUCTION
    nsteps_neq : int, optional, default=250000
        number of nonequilibrium steps to take for FAH integrator for PRODUCTION
    fe_type : str, optional, default='fah'
        tells setup_relative_calculation() to use the fah pipeline
    collision_rate : simtk.unit.Quantity, optional, default=1./unit.picosecond
        collision_rate for PRODUCTION
    collision_rate_setup : simtk.unit.Quantity, default=90./unit.picosecond
        collision_rate for EQUILIBRATION
    constraint_tolerance : float, optional, default=1e-6
        tolerance to use for constraints
    n_steps_per_move_application : int, optional, default=250
        number of equilibrium steps to take per move
        TODO: How is this different from num_equilibration_steps_per_iteration?
    globalVarFreq : int, optional, default=250
        Interval at which the globals.csv integrator output with nonequilibrium work measurements is to be written out
    setup : str, optional, default='small_molecule'
        Specify whether we are setting up a 'small_molecule' transformation or 'protein' mutation
    protein_kwargs : dict, optional, default=None
        ?????
    ionic_strength : openmm.unit.Quantity, optional, default=0.15*unit.molar
        Ionic strength to use in simulation setup
    remove_constraints : str, optional, default=False
        If True, constraints will be removed from the specific MDTraj DSL selection
        e.g. "not water"
        This is useful if the alchemical system is to be treated without bond constraints 
    rmsd_restraint : bool, optional, default=False
        If True, will restraint the core atoms and protein CA atoms within 6.5A of the core atoms.
    kwargs
        Other arguments are passed on in setup_options
    """
    from perses.utils import data
    if isinstance(temperature,float) or isinstance(temperature,int):
        temperature = temperature * unit.kelvin

    if isinstance(timestep,float) or isinstance(timestep,int):
        timestep = timestep * unit.femtosecond

    if isinstance(pressure, float) or isinstance(pressure, int):
        pressure = pressure * unit.atmosphere

    # Turn all of the args into a dict for passing to run_setup
    # TODO: This is unsafe; replace this with the new class-based perses API.
    # See API design notes at https://gist.github.com/jchodera/b8d2396a5953e391bb08335f1fc97f4e
    # HBM - this doesn't feel particularly safe
    # Also, this means that the function can't run without being called by run(), as we are requiring things that aren't arguments to this function, like 'solvent_projid'...etc
    setup_options = locals() # WARNING: This is incredibly dangerous! We are just feeding all variables in scope via setup_options to run_setup() or PointMutationExecutor()
    if 'kwargs' in setup_options.keys(): #update the setup options w.r.t. kwargs
        # TODO: This is dangerous if there is no argument validation
        setup_options.update(setup_options['kwargs'])
    if protein_kwargs is not None: #update the setup options w.r.t. the protein kwargs
        setup_options.update(setup_options['protein_kwargs'])
        if 'apo_box_dimensions' not in list(setup_options.keys()):
            setup_options['apo_box_dimensions'] = setup_options['complex_box_dimensions']

    # Sanity check for whether we are setting up a small molecule perturbation or protein mutation
    setups_allowed = ['small_molecule', 'protein']
    assert setup in setups_allowed, f"setup {setup} not in setups_allowed: {setups_allowed}"

    # Ensure each phase has a project id specified
    for phase in phases:
        assert (phase in phase_project_ids), f"Phase {phase} requested, but not in phase_project_ids {phase_project_ids.keys()}"

    # Modify some perses setup options for fah-specific functionality
    # TODO: Replace this with new perses class-based API
    setup_options['trajectory_prefix'] = None
    setup_options['anneal_1,4s'] = False
    from perses.utils.openeye import generate_expression
    setup_options['atom_expr'] = generate_expression(setup_options['atom_expression'])
    setup_options['bond_expr'] = generate_expression(setup_options['bond_expression'])

    # Generate topology proposals and hybrid topology factories via perses run_setup
    _logger.info(f"spectators: {setup_options['spectators']}")
    if setup == 'small_molecule':
        _logger.info(f"Setting up a small molecule transformation")
        from perses.app.setup_relative_calculation import run_setup
        setup_dict = run_setup(setup_options, serialize_systems=False, build_samplers=False)
        topology_proposals = setup_dict['topology_proposals'] # WARNING: This is not just a collection of TopologyProposal objects accessed via topology_proposals[phase]
        htfs = setup_dict['hybrid_topology_factories']
    elif setup == 'protein':
        _logger.info(f"Setting up a protein point mutation")
        from perses.app.relative_point_mutation_setup import PointMutationExecutor
        setup_engine = PointMutationExecutor(**setup_options)
        topology_proposals = {'complex': setup_engine.get_complex_htf()._topology_proposal, 'apo': setup_engine.get_apo_htf()._topology_proposal}
        htfs = {'complex': setup_engine.get_complex_htf(), 'apo': setup_engine.get_apo_htf()}

    # Create solvent and complex phase directories
    for phase in htfs.keys():
        _logger.info(f'Setting up phase {phase} for project ID {phase_project_ids[phase]}')
        phase_dir = f"{phase_project_ids[phase]}/RUNS"
        dir = os.path.join(os.getcwd(), phase_dir, trajectory_directory)
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Store some useful information in human- and computer-readable format
        # TODO: Once we switch to the new perses class API, all metadata will be serialized by appropriate objects
        #       and we will not need this
        metadata = {
            'transformation' : setup,
            'protein_pdb': protein_pdb,      
        }      
        if setup == 'small_molecule':
            metadata['old_ligand_index'] = old_ligand_index,
            metadata['new_ligand_index'] = new_ligand_index,
            metadata['ligand_file'] = ligand_file
            # Get useful information about the molecular transformation
            from openff.toolkit.topology import Molecule
            for endpoint in ['old', 'new']:
                molecule = Molecule.from_openeye(getattr(tp, f'ligand_oemol_{endpoint}'))
                metadata[f'{endpoint}_smiles'] = molecule.to_smiles()
                metadata[f'{endpoint}_name'] = molecule.name
        elif setup == 'protein':
            # TODO: Adapt this for protein mutations
            pass

        import yaml
        metadata_filename = f'{dir}/metadata.yaml'
        with open(metadata_filename, 'wt') as outfile:
            outfile.write(yaml.dump(metadata))

        # Render atom mapping
        tp = topology_proposals
        atom_map_filename = f'{dir}/atom_map.png'
        if setup == 'small_molecule':
            from perses.utils.smallmolecules import render_atom_mapping
            old_ligand_oemol, new_ligand_oemol = tp['ligand_oemol_old'], tp['ligand_oemol_new']
            _map = tp['non_offset_new_to_old_atom_map']
            render_atom_mapping(atom_map_filename, old_ligand_oemol, new_ligand_oemol, _map)
        elif setup == 'protein':
            from perses.utils.smallmolecules import render_protein_residue_atom_mapping # TODO: Why do we need a separate utility for this?
            render_protein_residue_atom_mapping(topology_proposal, atom_map_filename)

        # TODO: Determine atom mappings and slices we need
        # careful to include only protein and small molecule atoms
        hybrid_solute_to_real_solute_map = dict() # hybrid_solute_to_real_solute_map[endpoint][hybrid_atom_index] 
        real_solute_to_hybrid_solute_map = dict() # real_solute_to_hybrid_solute_map[endpoint][real_atom_index] is the 
        hybrid_to_real_map = dict()
        real_to_hybrid_map = dict()
        hybrid_to_hybrid_solute_map = dict()    
        for endpoint in ['old', 'new']:
            htf = htfs[phase]
            openmm_topology = getattr(htf._topology_proposal, f'{endpoint}_topology')
            mdtraj_topology = md.Topology.from_openmm(openmm_topology)
            # Select non-solvent
            # WARNING: This is fragile because it reproduces the behavior of mdtraj's remove_solvent()
            # TODO: Implement 'solvent' into mdtraj DSL following same logic and replace this code with
            # public API code only. Alternatively, use MDAnalysis.
            # https://github.com/mdtraj/mdtraj/blob/master/mdtraj/core/residue_names.py#L1826
            from mdtraj.core.residue_names import _SOLVENT_TYPES
            solvent_types = list(_SOLVENT_TYPES)
            solute_indices = [ atom.index for atom in mdtraj_topology.atoms if
                               atom.residue.name not in solvent_types]
            real_to_hybrid_map = getattr(htf,f'_{endpoint}_to_hybrid_map')
            # real_solute_to_hybrid_solute_map 
            hybrid_solute_to_real_solute_map[endpoint] = { [solute_indices[real_index]] : real_index for real_index in range(len(solute_indices)) }



        # * full hybrid system to full {old | new} system
        # * solute-only hybrid system to full hybrid system
        # * solute-only hybrid system to solute-only {old | new} system

        # Serialize atom mappings
        # These can be accessed with:
        # np.load('/path/to/file.npz', allow_pickle=True)['hybrid_to_old_map'].flat[0]
        # TODO: This is really clunky.
        # TODO: Improve the way we serialize these atom mappings
        htf = htfs[phase]
        np.savez(f'{dir}/hybrid_atom_mappings.npz',
                 # Full system maps
                 hybrid_to_old_map=htf._hybrid_to_old_map,
                 hybrid_to_new_map=htf._hybrid_to_new_map,
                 old_to_hybrid_map=htf._old_to_hybrid_map,
                 new_to_hybrid_map=htf._new_to_hybrid_map,
                 # Solute only maps
                 old_solute_to_hybrid_solute_map=real_solute_to_hybrid_solute_map['old'],
                 new_solute_to_hybrid_solute_map=real_solute_to_hybrid_solute_map['new'],
                 hybrid_solute_to_old_solute_map=hybrid_solute_to_real_solute_map['old'],                 
                 hybrid_solute_to_new_solute_map=hybrid_solute_to_real_solute_map['new'],                 
                 # Full to solute subsets
                 hybrid_solute_atom_indices=hybrid_solute_atom_indices,
        )

        # Create a core.xml for Folding@home OpenMM core22
        # TODO: Specify xtcAtoms as hybrid solute atom indices
        nsteps_per_cycle = 2*nsteps_eq + 2*nsteps_neq
        ncycles = 1
        nsteps_per_ps = 250
        nsteps = ncycles * nsteps_per_cycle
        make_core_file(numSteps=nsteps,
                       xtcFreq=1000*nsteps_per_ps,
                       globalVarFreq=10*nsteps_per_ps,
                       #xtcAtoms=hybrid_solute_atom_indices,
                       directory=dir)

        # Serialize the hybrid topology factory for this phase
        # NOTE: This uses a fragile, slow, and inconvenient numpy savez + pickle scheme
        # TODO: Replace this with a better serialization scheme to enable more rapid access to useful information
        _logger.info(f'Serializing hybrid topology factory...')
        np.savez_compressed(f'{dir}/htf',htfs[phase])

        # Serialize the hybrid_system OpenMM System for execution on Folding@home
        _logger.info(f'Serializing hybrid System...')
        data.serialize(htfs[phase].hybrid_system, f"{dir}/system.xml.bz2")

        # Create and serialize an OpenMM Integrator for execution on Folding@home
        _logger.info(f'Serializing integrator...')
        integrator = make_neq_integrator(**setup_options)
        data.serialize(integrator, f"{dir}/integrator.xml.bz2")

        # Minimize, and equilibrate, then serialize a State to initiate simulations from
        _logger.info(f'Minimizing and equilibrating...')
        state = relax_structure(temperature=temperature,
                                system = htfs[phase].hybrid_system,
                                positions = htfs[phase].hybrid_positions,
                                nequil = num_equilibration_iterations,
                                n_steps_per_iteration=num_equilibration_steps_per_iteration, collision_rate=collision_rate_setup, **kwargs)
        
        _logger.info(f'Serializing equilibrated State...')
        data.serialize(state, f"{dir}/state.xml.bz2")

        # TODO: Save complete hybrid topology
            
        # Write old and new equilibrated snapshots
        import mdtraj as md
        pos = state.getPositions(asNumpy=True)
        pos = np.asarray(pos) # QUESTION: Why is this here? Doesn't it strip units?
        for endpoint in ['old', 'new']:
            openmm_topology = getattr(htfs[phase]._topology_proposal, f'{endpoint}_topology')
            mdtraj_topology = md.Topology.from_openmm(openmm_topology)
            positions = getattr(htfs[phase], f'{endpoint}_positions')(pos)
            traj = md.Trajectory(positions, mdtraj_topology)
            traj.remove_solvent(exclude=['CL', 'NA'], inplace=True)
            traj.save(f'{dir}/{endpoint}_{phase}.pdb')

def run(yaml_filename=None):
    """
    Main application entry point for perses-fah to set up a single perses free energy calculation on Folding@home
    
    Parameters
    ----------
    yaml_filename : str, optional, default=None
        YAML file specifying which alchemical free energy calculation to set up and which options to use
        If no filename is specified, will try sys.argv[1]
    """
    import sys
    if yaml_filename is None:
        try:
            yaml_filename = sys.argv[1]
            _logger.info(f"Detected yaml file: {yaml_filename}")
        except IndexError as e:
            _logger.critical(f"{e}: You must specify the setup yaml file as an  argument to the script.")

    # Read setup options from the YAML file
    import yaml
    yaml_file = open(yaml_filename, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()

    import os
    from shutil import copyfile
    # Create all project directories and copy input YAML file to these directories
    if 'complex_projid' in setup_options:
        if not os.path.exists(f"{setup_options['complex_projid']}"):
            #os.makedirs(f"{setup_options['complex_projid']}/RUNS/")
            dst = f"{setup_options['complex_projid']}/RUNS/{setup_options['trajectory_directory']}"
            os.makedirs(dst)
            copyfile(yaml_filename, dst)
    if 'solvent_projid' in setup_options:
        if not os.path.exists(f"{setup_options['solvent_projid']}"):
            #os.makedirs(f"{setup_options['solvent_projid']}/RUNS/")
            dst = f"{setup_options['solvent_projid']}/RUNS/{setup_options['trajectory_directory']}"
            os.makedirs(dst)
            copyfile(yaml_filename, dst)
    if 'apo_projid' in setup_options:
        if not os.path.exists(f"{setup_options['apo_projid']}"):
            #os.makedirs(f"{setup_options['apo_projid']}/RUNS/")
            dst = f"{setup_options['apo_projid']}/RUNS/{setup_options['trajectory_directory']}"
            os.makedirs(dst)
            copyfile(yaml_filename, dst)
    if 'vacuum_projid' in setup_options:
        if not os.path.exists(f"{setup_options['vacuum_projid']}"):
            #os.makedirs(f"{setup_options['vacuum_projid']}/RUNS/")
            dst = f"{setup_options['vacuum_projid']}/RUNS/{setup_options['trajectory_directory']}"
            os.makedirs(dst)
            copyfile(yaml_filename, dst)

    # Set up alchemical free energy calculation for Folding@home
    run_neq_fah_setup(**setup_options)

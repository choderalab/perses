__author__ = 'dominic rufa'

"""
Folding@Home perses executor

# prereqs
conda config --add channels omnia --add channels conda-forge
conda create -n perses python3.7 perses tqdm dicttoxml
pip uninstall --yes openmmtools
pip install git+https://github.com/choderalab/openmmtools.git

argv[1]: setup.yaml (argument for perses.app.setup_relative_calculation.getSetupOptions)
"""
import numpy as np
import os
import simtk.unit as unit
from simtk import openmm
import logging

import datetime
class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated
        delta = datetime.datetime.fromtimestamp(record.relativeCreated/1000.0) - datetime.datetime.fromtimestamp(last/1000.0)
        record.relative = '{0:.2f}'.format(delta.seconds + delta.microseconds/1000000.0)
        self.last = record.relativeCreated
        return True

fmt = logging.Formatter(fmt="%(asctime)s:(%(relative)ss):%(name)s:%(message)s")
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
[hndl.addFilter(TimeFilter()) for hndl in _logger.handlers]
[hndl.setFormatter(fmt) for hndl in _logger.handlers]

#let's make a default lambda protocol
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


def make_neq_integrator(nsteps_eq=250000, nsteps_neq=250000, neq_splitting='V R H O R V', timestep=4.0 * unit.femtosecond, alchemical_functions=DEFAULT_ALCHEMICAL_FUNCTIONS, **kwargs):
    """
    generate an openmmtools.integrators.PeriodicNonequilibriumIntegrator

    arguments
        nsteps_eq : int, default=250000
            Number of equilibration steps to dwell within lambda = 0 or 1 when reached
        nsteps_neq : int, default=250000
            Number of nonequilibrium switching steps for 0->1 and 1->0 switches
        neq_splitting : str, default='V R H O R V'
            Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.
            "H" increments the global parameter `lambda` by 1/nsteps_neq for each step and accumulates protocol work.
        timestep : int, default=4.0 * unit.femtosecond
            integrator timestep
        alchemical_functions=dict
            dictionary containing alchemical functions of how to perturb each group. See DEFAULT_ALCHEMICAL_FUNCTIONS for example
        **kwargs :
            miscellaneous arguments for openmmtools.integrators.LangevinIntegrator

    returns
        integrator : openmmtools.integrators.PeriodicNonequilibriumIntegrator

    """
    from openmmtools.integrators import PeriodicNonequilibriumIntegrator
    integrator = PeriodicNonequilibriumIntegrator(alchemical_functions, nsteps_eq, nsteps_neq, neq_splitting, timestep=timestep)
    return integrator


def make_core_file(numSteps,
                   xtcFreq,
                   globalVarFreq,
                   xtcAtoms='solute',
                   precision='mixed',
                   globalVarFilename='globals.csv',
                   directory='.'):
    """ Makes core.xml file for simulating on folding at home

    Parameters
    ----------
    numSteps : int
        Number of steps to perform
    xtcFreq : int
        Frequency to save configuration to disk
    globalVarFreq : int
        Frequency to save variables to globalVarFilename
    xtcAtoms : str, default='solute'
        Which atoms to save
    precision : str, default='mixed'
        Precision of simulation
    globalVarFilename : str, default='globals.csv'
        Filename to store global simulation results
    directory : str, default='.'
        Location on disk to save core.xml file

    # TODO - unhardcode 'core.xml' or would it always be this?
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
                    platform_name='OpenCL',
                    timestep=2.*unit.femtosecond,
                    collision_rate=90./unit.picosecond,
                    **kwargs):
    """
    arguments
        temperature : simtk.unit.Quantity with units compatible with kelvin
            temperature of simulation
        system : openmm.System
            system object for simulation
        positions : simtk.unit.Quantity of shape (natoms,3) with units compatible with nanometers
            Positions of the atoms in the system
        nequil : int, default = 1000
            number of equilibration applications
        n_steps_per_iteration : int, default = 250
            numper of steps per nequil
        platform name : str default='OpenCL'
            platform to run openmm on. OpenCL is best as this is what is used on FAH
        timestep : simtk.unit.Quantity, default = 2*unit.femtosecond
            timestep for equilibration NOT for production
        collision_rate : simtk.unit.Quantity, default=90./unit.picosecond

    return
        state : openmm.State
            state of simulation (getEnergy=True, getForces=True, getPositions=True, getVelocities=True, getParameters=True)
    """

    from openmmtools.integrators import LangevinIntegrator
    _logger.info(f'Starting to relax')
    integrator = LangevinIntegrator(temperature=temperature, timestep=timestep, collision_rate=collision_rate)
    platform = openmm.Platform.getPlatformByName(platform_name)

    # prepare the plaform
    if platform_name in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
    if platform_name in ['CUDA']:
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
    context = openmm.Context(system, integrator, platform)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    context.setPositions(positions)

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

    del context, integrator
    return state


def run_neq_fah_setup(ligand_file,
                      old_ligand_index,
                      new_ligand_index,
                      forcefield_files,
                      trajectory_directory,
                      complex_box_dimensions=(9.8, 9.8, 9.8),
                      solvent_box_dimensions=(3.5, 3.5, 3.5),
                      timestep=4.0,
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
                      small_molecule_forcefield='openff-1.2.0',
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
                      remove_constraints='not water',
                      **kwargs):
    """
    main execution function that will:
        - create a directory for each phase according to the `trajectory_directory` argument
        - make a subdirectory named f"RUN_{old_ligand_index}_{new_ligand_index}" given the specified ligand indices of the `ligand_file`
        - create topology proposals for all phases
        - create/serialize hybrid factories or all phases (and validate endstates)
        - create/serialize an openmmtools.integrators.PeriodicNonequilibriumIntegrator for all phases
        - relax generated structures with a minimizer and LangevinIntegrator for all phases
        - create/serialize a state associated with the relaxed structures
        - create/serialize a `core.xml` object for all phases


    >>> run_neq_fah_setup('ligand.sdf', 0, 1,['amber/ff14SB.xml','amber/tip3p_standard.xml','amber/tip3p_HFE_multivalent.xml'],'RUN0',protein_pdb='protein.pdb', phases=['complex','solvent','vacuum'],phase_project_ids={'complex':14320,'solvent':14321,'vacuum':'vacuum'})

    arguments
        ligand_file : str
            .sdf (or any openeye-readable) file containing ligand labeled indices and structures
        old_ligand_index : int
            index of the old ligand
        new_ligand_index : int
            inded of the new ligand
        forcefield_files : list of str
            list of forcefields to use for complex/solvent parameterization
        trajectory_directory : str
            RUNXXX for FAH deployment
        complex_box_dimensions : Vec3, default=(9.8, 9.8, 9.8)
            define box dimensions of complex phase (in nm)
        solvent_box_dimensions : Vec3, default=(3.5, 3.5, 3.5)
            define box dimensions of solvent phase (in nm)
        timestep : float, default=4.
            step size of nonequilibrium integration
        eq_splitting : str, default = 'V R O R V'
            splitting string of relaxation dynamics
        neq_splitting : str, default = 'V R H O R V'
            splitting string of nonequilibrium dynamics
        measure_shadow_work : bool, default=False
            True/False to measure shadow work
        pressure: float, default=1.
            pressure in atms for simulation
        temperature: simtk.unit.Quantity, default=300.*unit.kelvin,
            temperature in K for simulation
        phases: list, default = ['complex','solvent','vacuum','apo']
            phases to run, where allowed phases are:
            'complex','solvent','vacuum','apo'
        protein_pdb : str, default=None
            name of protein file
        receptor_mol2 : str, default=None
            name of receptor file if protein_pdb not provided
        small_molecule_forcefield : str, default='openff-1.0.0'
            small molecule forcefield filename
        small_molecule_parameters_cache : str, default=None
            cache file containing small molecule forcefield files
        atom_expression : list default=['IntType']
            list of string for atom mapping criteria. see oechem.OEExprOpts for options
        bond_expression : list default=['DefaultBonds']
            list of string for bond mapping criteria. see oechem.OEExprOpts for options
        map_strength : 'str', default=None
            atom and bond expressions will be ignored, and either a 'weak', 'default' or 'strong' map_strength will be used.
        spectators : str, default=None
            path to any non-alchemical atoms in simulation
        neglect_angles : bool, default=False
            wether to use angle terms in building of unique-new groups. False is strongly recommended
        anneal_14s : bool, default False
            Whether to anneal 1,4 interactions over the protocol;
        nonbonded_method : str, default='PME'
            nonbonded method to use
        softcore_v2=bool, default=False
            wether to use v2 softcore
        alchemical_functions : dict, default=DEFAULT_ALCHEMICAL_FUNCTIONS
            alchemical functions for transformation
        num_equilibration_iterations: int, default=1000
            number of equilibration steps to do during set up
        num_equilibration_steps_per_iteration: int, default=250,
            number of steps per iteration. default is 250 steps of 2fs, 1000 times which is 500ps of equilibration for SETUP
        nsteps_eq : int, default=250000
            number of normal MD steps to take for FAH integrator for PRODUCTION
        nsteps_neq : int, default=250000
            number of nonequilibrium steps to take for FAH integrator for PRODUCTION
        fe_type : str, default='fah'
            tells setup_relative_calculation() to use the fah pipeline
        collision_rate : simtk.unit.Quantity, default=1./unit.picosecond
            collision_rate for PRODUCTION
        collision_rate_setup : simtk.unit.Quantity, default=90./unit.picosecond
        constraint_tolerance : float, default=1e-6
            tolerance to use for constraints
        n_steps_per_move_application : int default=250
            number of equilibrium steps to take per move
    """
    from perses.utils import data
    if isinstance(temperature,float) or isinstance(temperature,int):
        temperature = temperature * unit.kelvin

    if isinstance(timestep,float) or isinstance(timestep,int):
        timestep = timestep* unit.femtosecond

    if isinstance(pressure, float) or isinstance(pressure, int):
        pressure = pressure  * unit.atmosphere

    #turn all of the args into a dict for passing to run_setup
    # HBM - this doesn't feel particularly safe
    # Also, this means that the function can't run without being called by run(), as we are requiring things that aren't arguments to this function, like 'solvent_projid'...etc
    setup_options = locals()
    if 'kwargs' in setup_options.keys(): #update the setup options w.r.t. kwargs
        setup_options.update(setup_options['kwargs'])
    if protein_kwargs is not None: #update the setup options w.r.t. the protein kwargs
        setup_options.update(setup_options['protein_kwargs'])
        if 'apo_box_dimensions' not in list(setup_options.keys()):
            setup_options['apo_box_dimensions'] = setup_options['complex_box_dimensions']

    #setups_allowed
    setups_allowed = ['small_molecule', 'protein']
    assert setup in setups_allowed, f"setup {setup} not in setups_allowed: {setups_allowed}"

    # check there is a project_id for each phase
    for phase in phases:
        assert (phase in phase_project_ids), f"Phase {phase} requested, but not in phase_project_ids {phase_project_ids.keys()}"

    #some modification for fah-specific functionality:
    setup_options['trajectory_prefix'] = None
    setup_options['anneal_1,4s'] = False
    from perses.utils.openeye import generate_expression
    setup_options['atom_expr'] = generate_expression(setup_options['atom_expression'])
    setup_options['bond_expr'] = generate_expression(setup_options['bond_expression'])

    #run the run_setup to generate topology proposals and htfs
    _logger.info(f"spectators: {setup_options['spectators']}")
    if setup == 'small_molecule':
        from perses.app.setup_relative_calculation import run_setup
        setup_dict = run_setup(setup_options, serialize_systems=False, build_samplers=False)
        topology_proposals = setup_dict['topology_proposals']
        htfs = setup_dict['hybrid_topology_factories']
    elif setup == 'protein':
        from perses.app.relative_point_mutation_setup import PointMutationExecutor
        setup_engine = PointMutationExecutor(**setup_options)
        topology_proposals = {'complex': setup_engine.get_complex_htf()._topology_proposal, 'apo': setup_engine.get_apo_htf()._topology_proposal}
        htfs = {'complex': setup_engine.get_complex_htf(), 'apo': setup_engine.get_apo_htf()}

    #create solvent and complex directories
    for phase in htfs.keys():
        _logger.info(f'Setting up phase {phase}')
        phase_dir = f"{phase_project_ids[phase]}/RUNS"
        dir = os.path.join(os.getcwd(), phase_dir, trajectory_directory)
        if not os.path.exists(dir):
            os.makedirs(dir)

        # TODO - replace this with actually saving the importand part of the HTF
        np.savez_compressed(f'{dir}/htf',htfs[phase])

        #serialize the hybrid_system
        data.serialize(htfs[phase].hybrid_system, f"{dir}/system.xml.bz2")

        #make and serialize an integrator
        integrator = make_neq_integrator(**setup_options)
        data.serialize(integrator, f"{dir}/integrator.xml")

        #create and serialize a state
        try:
            state = relax_structure(temperature=temperature,
                            system = htfs[phase].hybrid_system,
                            positions = htfs[phase].hybrid_positions,
                            nequil = num_equilibration_iterations,
                            n_steps_per_iteration=num_equilibration_steps_per_iteration, collision_rate=collision_rate_setup, **kwargs)

            data.serialize(state, f"{dir}/state.xml.bz2")
        except Exception as e:
            _logger.warning(e)
            passed = False
        else:
            passed = True

        pos = state.getPositions(asNumpy=True)
        pos = np.asarray(pos)

        import mdtraj as md
        top = htfs[phase].hybrid_topology
        np.save(f'{dir}/hybrid_topology', top)
        traj = md.Trajectory(pos, top)
        traj.remove_solvent(exclude=['CL', 'NA'], inplace=True)
        traj.save(f'{dir}/hybrid_{phase}.pdb')

        #lastly, make a core.xml
###
        nsteps_per_cycle = 2*nsteps_eq + 2*nsteps_neq
        ncycles = 1
        nsteps_per_ps = 250
        nsteps = ncycles * nsteps_per_cycle
        make_core_file(numSteps=nsteps,
                       xtcFreq=1000*nsteps_per_ps,
                       globalVarFreq=10*nsteps_per_ps,
                       directory=dir)

     #create a logger for reference
        # TODO - add more details to this
        references = {'start_ligand': old_ligand_index,
                      'end_ligand': new_ligand_index,
                      'protein_pdb': protein_pdb,
                      'passed_strucutre_relax': passed}

        np.save(f'{dir}/references',references)

        tp = topology_proposals
        from perses.utils.smallmolecules import render_atom_mapping
        atom_map_filename = f'{dir}/atom_map.png'
        if setup=='protein':
            from perses.utils.smallmolecules import  render_protein_residue_atom_mapping
            render_protein_residue_atom_mapping(tp['apo'], atom_map_filename)
        else:
            old_ligand_oemol, new_ligand_oemol = tp['ligand_oemol_old'], tp['ligand_oemol_new']
            _map = tp['non_offset_new_to_old_atom_map']
            render_atom_mapping(atom_map_filename, old_ligand_oemol, new_ligand_oemol, _map)


def run(yaml_filename=None):
    import sys
    if yaml_filename is None:
        try:
            yaml_filename = sys.argv[1]
            _logger.info(f"Detected yaml file: {yaml_filename}")
        except IndexError as e:
            _logger.critical(f"{e}: You must specify the setup yaml file as an  argument to the script.")

    # this is imported, but not used --- why?
    # from perses.app.setup_relative_calculation import getSetupOptions
    import yaml
    yaml_file = open(yaml_filename, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()

    import os
    # make master and run directories
    if 'complex_projid' in setup_options:
        if not os.path.exists(f"{setup_options['complex_projid']}"):
            os.makedirs(f"{setup_options['complex_projid']}/RUNS/")
            os.makedirs(f"{setup_options['complex_projid']}/RUNS/{setup_options['trajectory_directory']}")
    if 'solvent_projid' in setup_options:
        if not os.path.exists(f"{setup_options['solvent_projid']}"):
            os.makedirs(f"{setup_options['solvent_projid']}/RUNS/")
            os.makedirs(f"{setup_options['solvent_projid']}/RUNS/{setup_options['trajectory_directory']}")
    if 'apo_projid' in setup_options:
        if not os.path.exists(f"{setup_options['apo_projid']}"):
            os.makedirs(f"{setup_options['apo_projid']}/RUNS/")
            os.makedirs(f"{setup_options['apo_projid']}/RUNS/{setup_options['trajectory_directory']}")
    if 'vacuum_projid' in setup_options:
        if not os.path.exists(f"{setup_options['vacuum_projid']}"):
            os.makedirs(f"{setup_options['vacuum_projid']}/RUNS/")
            os.makedirs(f"{setup_options['vacuum_projid']}/RUNS/{setup_options['trajectory_directory']}")

    run_neq_fah_setup(**setup_options)

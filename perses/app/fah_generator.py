#1/usr/bin/env python

__author__ = 'dominic rufa'

"""
Folding@Home perses executor

# prereqs
conda config --add channels omnia --add channels conda-forge
conda create -n perses python3.7 perses tqdm dicttoxml
pip uninstall --yes openmmtools
pip install git+https://github.com/choderalab/openmmtools.git

argv[1]: setup.yaml (argument for perses.app.setup_relative_calculation.getSetupOptions)
argv[2]: neq_setup.yaml (contains keywords for openmmtools.integrators.PeriodicNonequilibriumIntegrator arguments)
argv[3]: run_number (project run number; defined by f"setup_options['trajectory_directory']_phase/RUN_{run_number}")
"""
import yaml
import numpy as np
import os
import sys
import simtk.unit as unit
from simtk import openmm
import logging
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


#let's make a default lambda protocol
x = 'lambda'
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


def make_neq_integrator(nsteps_eq=250000, nsteps_neq=250000, neq_splitting='V R H O R V', timestep=4.0 * unit.femtosecond, alchemical_functions = DEFAULT_ALCHEMICAL_FUNCTIONS, **kwargs):
    """
    generate an openmmtools.integrators.PeriodicNonequilibriumIntegrator

    arguments
        nsteps_eq : int, default=1000
            Number of equilibration steps to dwell within lambda = 0 or 1 when reached
        nsteps_neq : int, default=1000
            Number of nonequilibrium switching steps for 0->1 and 1->0 switches
        neq_splitting : str, default='V R H O R V'
            Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.
            "H" increments the global parameter `lambda` by 1/nsteps_neq for each step and accumulates protocol work.
        timestep : int, default=4.0 * unit.femtosecond
            integrator timestep
        **kwargs :
            miscellaneous arguments for openmmtools.integrators.LangevinIntegrator

    returns
        integrator : openmmtools.integrators.PeriodicNonequilibriumIntegrator

    """
    from openmmtools.integrators import PeriodicNonequilibriumIntegrator
    #from copy import deepcopy
    #integrator_kwargs = deepcopy(setup_options_dict)
    integrator = PeriodicNonequilibriumIntegrator(alchemical_functions, nsteps_eq, nsteps_neq, neq_splitting,timestep=timestep)
    return integrator

def relax_structure(temperature, system, positions, nequil = 4, n_steps_per_iteration=250,platform_name='CUDA'):
    """
    arguments
        temperature : simtk.unit.Quantity with units compatible with kelvin
            temperature of simulation
        system : openmm.System
            system object for simulation
        positions : simtk.unit.Quantity of shape (natoms,3) with units compatible with nanometers
            Positions of the atoms in the system
        nequil : int
            number of equilibration applications
        n_steps_per_iteration : int
            numper of steps per nequil
        platform name : str default='CUDA'
            platform to run openmm on

    return
        state : openmm.State
            state of simulation (getEnergy=True, getForces=True, getPositions=True, getVelocities=True, getParameters=True)
    """

    from perses.dispersed.feptasks import minimize
    from openmmtools.integrators import LangevinIntegrator
    import tqdm
    _logger.info(f'Starting to relax')
    integrator = LangevinIntegrator(temperature = temperature)
    platform = openmm.Platform.getPlatformByName(platform_name)
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
    _logger.info(f'Starting to equilibrate')
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
                      index=0,
                      box_dimensions=(8.5,8.5,8.5),
                      timestep=4.0 * unit.femtosecond,
                      eq_splitting = 'V R O R V',
                      neq_splitting='V R H O R V',
                      measure_shadow_work=False,
                      pressure=1.0,
                      temperature=300,
                      solvent_padding=9*unit.angstroms,
                      set_solvent_box_dims_to_complex=True,
                      phases=['complex','solvent','vacuum'],
                      protein_pdb=None,
                      receptor_mol2=None,
                      small_molecule_forcefield = 'openff-1.0.0',
                      small_molecule_parameters_cache = None,
                      spectators=None,
                      neglect_angles=True,
                      anneal_14s=False,
                      nonbonded_method='PME',
                      atom_expr=None,
                      bond_expr=None,
                      map_strength=None,
                      softcore_v2=False,
                      save_setup_pickle_as=None,
                      render_atom_map=False,
                      alchemical_functions=DEFAULT_ALCHEMICAL_FUNCTIONS,
                      num_minimize_steps=100,
                      num_equilibration_iterations=4,
                      num_equilibration_steps_per_iteration=250,
                      nsteps_eq=250000,
                      nsteps_neq=250000,
                      fe_type='fah',
                      n_steps_per_move_application=1,
                      n_equilibrium_steps_per_iteration=1,
                      collision_rate=1.0/unit.picoseconds,
                      constraint_tolerance=1e-6,
                      measure_heat=False,
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
            name of project
        timestep : simtk.unit.Quantity with units compatible with picoseconds
            step size of nonequilibrium integration
        eq_splitting : str
            splitting string of relaxation dynamics
        neq_splitting : str, default = 'V R H O R V'
            splitting string of nonequilibrium dynamics

    """
    from perses.app.setup_relative_calculation import run_setup
    from simtk.openmm import XmlSerializer
    from perses.utils import data
    #turn all of the args into a dict for passing to run_setup
    setup_options = locals()
    if 'kwargs' in setup_options.keys():
        setup_options.update(setup_options['kwargs'])

    #some modification for fah-specific functionality:
    setup_options['trajectory_prefix']=None
    setup_options['anneal_1,4s'] = setup_options['anneal_14s']

    #run the run_setup to generate topology proposals and htfs
    _logger.info('SETUP STARTED') 
    setup_dict = run_setup(setup_options, serialize_systems=False, build_samplers=False)
    _logger.info('SETUP DONE') 
    topology_proposals = setup_dict['topology_proposals']
    htfs = setup_dict['hybrid_topology_factories']

    #create solvent and complex directories
    for phase in htfs.keys():
        _logger.info(f'Setting up phase {phase}')
        if phase == 'solvent':
            phase_dir = '13405/RUNS'
        if phase == 'complex':
            phase_dir = '13404/RUNS'
        if phase == 'vacuum':
            phase_dir = 'VACUUM/RUNS'
        dir = os.path.join(os.getcwd(), phase_dir, f'RUN{index}')
        if not os.path.exists(dir):
            os.mkdir(dir)

        np.savez(f'{dir}/htf',htfs[phase])

        #serialize the hybrid_system
        data.serialize(htfs[phase].hybrid_system, f"{dir}/system.xml")

        #make and serialize an integrator
        integrator = make_neq_integrator(**setup_options)
        data.serialize(integrator, f"{dir}/integrator.xml")

        #create and serialize a state
        try:
            state = relax_structure(temperature=temperature,
                            system = htfs[phase].hybrid_system,
                            positions = htfs[phase].hybrid_positions,
                            nequil = num_equilibration_iterations,
                            n_steps_per_iteration=num_equilibration_steps_per_iteration)

            data.serialize(state, f"{dir}/state.xml")
        except Exception as e:
            print(e)
            passed=False
        else:
            passed=True

        pos = state.getPositions(asNumpy=True)
        pos = np.asarray(pos)

        np.save(f'{dir}/positions',pos)
        import mdtraj as md
        top = htfs[phase].hybrid_topology
        np.save(f'{dir}/hybrid_topology',top)
        traj = md.Trajectory(pos, top)
        traj.remove_solvent(exclude=['CL','NA'],inplace=True)
        traj.save(f'{dir}/hybrid_{phase}.pdb')

        #lastly, make a core.xml
        nsteps_per_cycle = 2*nsteps_eq + 2*nsteps_neq
        ncycles = 1    
        nsteps_per_ps = 250
        core_parameters = {
            'numSteps' : ncycles * nsteps_per_cycle,
            'xtcFreq' : 1000*nsteps_per_ps, # once per ns
            'xtcAtoms' : 'solute',
            'precision' : 'mixed',
            'globalVarFilename' : 'globals.csv',
            'globalVarFreq' : nsteps_per_ps,
        }
        # Serialize core.xml
        import dicttoxml
        with open(f'{dir}/core.xml', 'wt') as outfile:
            #core_parameters = create_core_parameters(phase)
            xml = dicttoxml.dicttoxml(core_parameters, custom_root='config', attr_type=False)
            from xml.dom.minidom import parseString
            dom = parseString(xml)
            outfile.write(dom.toprettyxml())

        #create a logger for reference
        references = {'start_ligand': old_ligand_index,
                      'end_ligand': new_ligand_index,
                      'protein_pdb': protein_pdb,
                      'passed_strucutre_relax': passed}

        np.save(f'{dir}/references',references)



def run(yaml_filename=None,index=None):
    import sys
    if yaml_filename is None:
       try:
          yaml_filename = sys.argv[1]
          _logger.info(f"Detected yaml file: {yaml_filename}")
       except IndexError as e:
           _logger.critical(f"You must specify the setup yaml file as an argument to the script.")
    if index is None:
       try:
          index = sys.argv[2]
          _logger.info(f"Detected index: {index}")
       except IndexError as e:
           _logger.critical(f"FAH generator needs an index to know which run this job is for")
    from perses.app.setup_relative_calculation import getSetupOptions
    import yaml
    yaml_file = open(yaml_filename, 'r')
    setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()


    import os
    # make master directories
    if not os.path.exists('13404'):
        os.makedirs('13404/RUNS/')
    if not os.path.exists('13405'):
        os.makedirs('13405/RUNS/')
    if not os.path.exists('VACUUM'):
        os.makedirs('VACUUM/RUNS/')

    # make run directories
    #if not os.path.exists(f'13404/RUN{index}'):
    os.makedirs(f'13404/RUNS/RUN{index}')
    #if not os.path.exists(f'13405/RUN{index}'):
    os.makedirs(f'13405/RUNS/RUN{index}')
    #if not os.path.exists(f'VACUUM/RUN{index}'):
    os.makedirs(f'VACUUM/RUNS/RUN{index}')

    ligand_file = setup_options['ligand_file']
    old_ligand_index = setup_options['old_ligand_index']
    new_ligand_index = setup_options['new_ligand_index']
    forcefield_files = setup_options['forcefield_files']
    protein_pdb = setup_options['protein_pdb']
    trajectory_directory = f'RUN{index}'

    run_neq_fah_setup(ligand_file,old_ligand_index,new_ligand_index,forcefield_files,trajectory_directory,index=index, protein_pdb=protein_pdb)

# if __name__ == "__main__":
#     setup_yaml, neq_setup_yaml, run_number = sys.argv[1], sys.argv[2], sys.argv[3] #define args
#
#     #open the setup yaml to pull trajectory_directory
#     yaml_file = open(setup_yaml, 'r')
#     setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
#     yaml_file.close()
#     traj_dir = setup_options['trajectory_directory']
#
#     gather all
#     phases = setup_options['phases']
#     for phase in phases:
#         now_path = os.path.join(os.getcwd(), f"{traj_dir}_{phase}")
#         if not os.path.exists(now_path):
#             os.mkdir(now_path)
#         else:
#             raise Exception(f"{now_path} already exists.  Aborting.")
#     run_neq_fah_setup(setup_yaml, neq_yaml, int(run_number), **kwargs)

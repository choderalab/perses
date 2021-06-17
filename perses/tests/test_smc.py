###########################################
# IMPORTS
###########################################
from simtk import unit, openmm
import numpy as np
import os
from nose.tools import nottest
from unittest import skipIf
from perses.app.setup_relative_calculation import *
from perses.annihilation.relative import HybridTopologyFactory
from perses.app.relative_setup import RelativeFEPSetup
from perses.dispersed.smc import SequentialMonteCarlo
from openmmtools.constants import kB
from perses.dispersed.utils import *
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol
#######################
running_on_github_actions = os.environ.get('GITHUB_ACTIONS', None) == 'true'

#default arguments
lambda_protocol = 'default'
temperature = 300 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
ENERGY_THRESHOLD = 1e-6
trajectory_directory = 'test_smc'
trajectory_prefix = 'out'
atom_selection = 'not water'
timestep = 1 * unit.femtoseconds
collision_rate = 1 / unit.picoseconds
eq_splitting_string = 'V R O R V'
neq_splitting_string = 'V R O R V'
ncmc_save_interval = None
measure_shadow_work = False
neq_integrator = 'langevin'
external_parallelism = None
internal_parallelism = {'library': ('dask', 'LSF'), 'num_processes': 2}
os.system(f"mkdir {trajectory_directory}")
rng = np.random.RandomState(42)
#######################

@nottest
@skipIf(running_on_github_actions, "Skip helper function on GH Actions")
def sMC_setup():
    """
    function to setup local sMC
    """
    from pkg_resources import resource_filename
    smiles_filename = resource_filename("perses", os.path.join("data", "test.smi"))
    fe_setup = RelativeFEPSetup(ligand_input = smiles_filename,
                                old_ligand_index = 0,
                                new_ligand_index = 1,
                                forcefield_files = [],
                                small_molecule_forcefield = 'gaff-2.11',
                                phases = ['vacuum'])

    hybrid_factory = HybridTopologyFactory(topology_proposal = fe_setup._vacuum_topology_proposal,
                                           current_positions = fe_setup._vacuum_positions_old,
                                           new_positions = fe_setup._vacuum_positions_new,
                                           neglected_new_angle_terms = fe_setup._vacuum_forward_neglected_angles,
                                           neglected_old_angle_terms = fe_setup._vacuum_reverse_neglected_angles,
                                           softcore_LJ_v2 = True,
                                           interpolate_old_and_new_14s = False)

    zero_state_error, one_state_error = validate_endstate_energies(fe_setup._vacuum_topology_proposal,
                                                                   hybrid_factory,
                                                                   added_energy = fe_setup._vacuum_added_valence_energy,
                                                                   subtracted_energy = fe_setup._vacuum_subtracted_valence_energy,
                                                                   beta = beta,
                                                                   platform = openmm.Platform.getPlatformByName('Reference'),
                                                                   ENERGY_THRESHOLD = ENERGY_THRESHOLD)
    ne_fep = SequentialMonteCarlo(factory = hybrid_factory,
                                      lambda_protocol = lambda_protocol,
                                      temperature = temperature,
                                      trajectory_directory = trajectory_directory,
                                      trajectory_prefix = trajectory_prefix,
                                      atom_selection = atom_selection,
                                      timestep = timestep,
                                      eq_splitting_string = eq_splitting_string,
                                      neq_splitting_string = neq_splitting_string,
                                      collision_rate = collision_rate,
                                      ncmc_save_interval = ncmc_save_interval,
                                      external_parallelism = None,
                                      internal_parallelism = None)

    assert ne_fep.external_parallelism is False and ne_fep.internal_parallelism is True, f"all parallelism should be None"
    assert ne_fep.workers == 0, f"local annealing definition only allows 0 workers"
    assert ne_fep.parallelism_parameters == {'library': None, 'num_processes': None}, f"the parallelism_parameters are not supported"

    #get reduced energies
    ne_fep.minimize_sampler_states()
    ne_fep.equilibrate(n_equilibration_iterations = 10,
                           n_steps_per_equilibration = 1,
                           endstates = [0,1],
                           decorrelate = True,
                           timer = True,
                           minimize = False)
    assert all(sum([ne_fep._eq_dict[state][i][-1] for i in range(len(ne_fep._eq_dict[state]))]) == 10 for state in [0,1]), f"there should be 10 snapshots per endstate"
    assert all(len(ne_fep._eq_dict[f"{state}_reduced_potentials"]) == 10 for state in [0,1]), f"there should be 10 reduced potentials per endstate"
    assert all(len(ne_fep._eq_dict[f"{state}_decorrelated"]) <= 10 for state in [0,1]), f"the decorrelated indices must be less than or equal to the total number of snapshots"

    #now to check for decorrelation in ne_fep._eq_files_dict...
    _filenames_0, _filenames_1 = [ne_fep._eq_dict[0][i][0] for i in range(len(ne_fep._eq_dict[0]))], [ne_fep._eq_dict[1][i][0] for i in range(len(ne_fep._eq_dict[1]))]
    decorrelated_0 = [item for sublist in [ne_fep._eq_files_dict[0][filename] for filename in _filenames_0] for item in sublist]
    decorrelated_1 = [item for sublist in [ne_fep._eq_files_dict[1][filename] for filename in _filenames_1] for item in sublist]
    decorrelated_0_files = [subitem for ssublist in [item for sublist in [list(ne_fep._eq_files_dict[0].values())] for item in sublist] for subitem in ssublist]
    decorrelated_1_files = [subitem for ssublist in [item for sublist in [list(ne_fep._eq_files_dict[1].values())] for item in sublist] for subitem in ssublist]
    assert decorrelated_0 == sorted(decorrelated_0_files), f"there is a discrepancy between the decorrelated 0 equilibrium states and the decorrelated equilibria saved to disk"
    assert decorrelated_1 == sorted(decorrelated_1_files), f"there is a discrepancy between the decorrelated 1 equilibrium states and the decorrelated equilibria saved to disk"
    return ne_fep

def test_local_AIS():
    """
    test local annealed importance sampling method in it's entirety
    """
    ne_fep = sMC_setup()



    #separate the tests...
    #1. _activate_annealing_workers(self) and _deactivate_annealing_workers(self)
    ne_fep._activate_annealing_workers()
    #above function should create a sublcass of the local ne_fep called 'annealing_class'
    #which should hold the variables that were initialized with AIS and an extra attribute called 'succeed'
    assert ne_fep.annealing_class.succeed, f"initialization of annealing class failed"
    ne_fep._deactivate_annealing_workers()
    assert not hasattr(ne_fep, 'annealing_class') #the annealing class should be deleted

    #2. call _activate_annealing_workers() again and call_anneal_method()
    ne_fep._activate_annealing_workers()
    #forego the self.parallelism.deploy() and just make sure that
    #call_anneal_method() properly calls self.annealing_class.anneal
    #where annealing_class is LocallyOptimalAnnealing
    incremental_work, sampler_state, timer, _pass, endstates = call_anneal_method(remote_worker = ne_fep,
                                                                       sampler_state = copy.deepcopy(ne_fep.sampler_states[0]),
                                                                       lambdas = np.array([0.0, 1e-6]),
                                                                       noneq_trajectory_filename = None,
                                                                       num_integration_steps = 1,
                                                                       return_timer = True,
                                                                       return_sampler_state = True,
                                                                       rethermalize = False,
                                                                       compute_incremental_work = True)
    if _pass: #the function is nan-safe
        assert  incremental_work is not None and sampler_state is not None and timer is not None, f"no returns can be None if the method passes"
    ne_fep._deactivate_annealing_workers()

    #3. call a dummy compute_sMC_free_energy with artificial values
    cumulative_work_dict = {'forward': np.array([[0., 0.5, 1.]]*3),
                            'reverse': np.array([[0., -0.5, -1.]]*3)}
    ne_fep.compute_sMC_free_energy(cumulative_work_dict)


    #test vanilla AIS
    print('run AIS with protocol')
    ne_fep.AIS(num_particles = 10,
               protocols = {'forward': np.linspace(0,1,9), 'reverse': np.linspace(1,0,9)},
               num_integration_steps = 1,
               return_timer = True,
               rethermalize = False)

    try:
        os.system(f"rm -r {trajectory_directory}")
    except Exception as e:
        print(e)

def test_configure_platform():
    """
    check utils.configure_platform
    """
    configure_platform(platform_name = 'CPU')
    check_platform(openmm.Platform.getPlatformByName("CPU"))

def test_compute_survival_rate():
    """
    test utils.compute_survival_rate()
    """
    def dummy_ancestry_generator_function():
        """
        dummy function to generate particle ancestries
        """
        ancestries = [np.arange(10)]
        for i in range(10):
            new_ancestries = rng.choice(ancestries[-1], 10)
            ancestries.append(new_ancestries)
        return ancestries

    artificial_data = {'forward': dummy_ancestry_generator_function(),
                       'reverse': dummy_ancestry_generator_function()}
    survival_rates = compute_survival_rate(artificial_data)
    for key, value in survival_rates.items():
        for idx in range(1, len(value)):
            assert value[idx] <= value[idx - 1], f"the survival rate is not decreasing"

def test_multinomial_resample():
    """
    test the multinomial resampler
    """
    total_works = rng.rand(10)
    num_resamples = 10
    resampled_works, resampled_indices = multinomial_resample(total_works, num_resamples)
    assert all(_val == np.average(total_works) for _val in resampled_works), f"the returned resampled works are not a uniform average"
    assert set(resampled_indices).issubset(set(np.arange(10))), f"the resampled indices can only be a subset of the resampled works"
    assert len(resampled_works) == num_resamples, f"there have to be the {num_resamples} resampled works"

def test_ESS():
    """
    test the effective sample size computation
    """
    #the ESS already passes with a normalization assertion
    dummy_prev_works, dummy_works_incremental = rng.rand(10), rng.rand(10)
    normalized_ESS = ESS(dummy_prev_works, dummy_works_incremental)

def test_CESS():
    """
    test the conditional effective sample size computation
    """
    #the CESS must be guaranteed to be between 0 and 1
    dummy_prev_works, dummy_works_incremental = rng.rand(10), rng.rand(10)
    _CESS = CESS(dummy_prev_works, dummy_works_incremental)

def test_compute_timeseries():
    """
    test the compute_timeseries function
    """
    reduced_potentials = rng.rand(100)
    data = compute_timeseries(reduced_potentials)
    assert len(data[3]) <= len(reduced_potentials), f"the length of uncorrelated data is at most the length of the raw data"

def test_create_endstates():
    """
    test the creation of unsampled endstates
    """
    from pkg_resources import resource_filename
    smiles_filename = resource_filename("perses", os.path.join("data", "test.smi"))
    fe_setup = RelativeFEPSetup(ligand_input = smiles_filename,
                                old_ligand_index = 0,
                                new_ligand_index = 1,
                                forcefield_files = [],
                                small_molecule_forcefield = 'gaff-2.11',
                                phases = ['vacuum'])

    hybrid_factory = HybridTopologyFactory(topology_proposal = fe_setup._vacuum_topology_proposal,
                                           current_positions = fe_setup._vacuum_positions_old,
                                           new_positions = fe_setup._vacuum_positions_new,
                                           neglected_new_angle_terms = fe_setup._vacuum_forward_neglected_angles,
                                           neglected_old_angle_terms = fe_setup._vacuum_reverse_neglected_angles,
                                           softcore_LJ_v2 = True,
                                           interpolate_old_and_new_14s = False)

    zero_state_error, one_state_error = validate_endstate_energies(fe_setup._vacuum_topology_proposal,
                                                                   hybrid_factory,
                                                                   added_energy = fe_setup._vacuum_added_valence_energy,
                                                                   subtracted_energy = fe_setup._vacuum_subtracted_valence_energy,
                                                                   beta = beta,
                                                                   platform = openmm.Platform.getPlatformByName('Reference'),
                                                                   ENERGY_THRESHOLD = ENERGY_THRESHOLD)

    lambda_alchemical_state = RelativeAlchemicalState.from_system(hybrid_factory.hybrid_system)
    lambda_protocol = LambdaProtocol(functions = 'default')
    lambda_alchemical_state.set_alchemical_parameters(0.0, lambda_protocol)
    thermodynamic_state = CompoundThermodynamicState(ThermodynamicState(hybrid_factory.hybrid_system, temperature = temperature),composable_states = [lambda_alchemical_state])
    zero_endstate = copy.deepcopy(thermodynamic_state)
    one_endstate = copy.deepcopy(thermodynamic_state)
    one_endstate.set_alchemical_parameters(1.0, lambda_protocol)
    new_endstates = create_endstates(zero_endstate, one_endstate)

if __name__ == '__main__':
    test_local_AIS()

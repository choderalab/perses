import time
import yaml
import networkx as nx

# environments
"""
- ligand-protein complex (includes small-molecule ligands and peptides)
- ligand-water
- ligand-vacuum
- ligand-membrane
- ligand-amorphous solid

*for clarity's sake, 'ligand' is the object containing a residue that will be transformed alchemically
"""

from openmmforcefields.generators import SystemGenerator
system_generator = SystemGenerator(forcefields=["protein.ff14SB.xml", "tip3p_standard.xml"],small_molecule_forcefield="gaff-2.11")

# Define the thermodynamic state we will be simulating in
from openmmtools.states import ThermodynamicState

thermodynamic_state = ThermodynamicState(
    temperature=300.0 * unit.kelvin, pressure=1.0 * unit.atmospheres, pH=7.0
)

# Load receptor and ligand using the openforcefield topology objects
from openforcefield.topology import Molecule

# will need to add biopolymer support, but can be done quickly
receptor = Molecule.from_file("receptor.pdb")
# multiple ligands pre-docked to site
ligands = Molecule.from_file("docked_ligands.sdf")


"""
CASE 1 : Relative Binding Free Energies to a protein target
"""
receptor = Molecule.from_file("receptor.pdb")
complex_environment = SmallMoleculeComplexEnvironment(thermodynamic_state, receptor=receptor)
aqueous_environment = AqueousEnvironment(thermodynamic_state)
aqueous_environment.solvent_buffer = 9 * unit.angstroms
octanol_environment = OctanolEnvironment(thermodynamic_state)
#amorphous_solid_environment = AmorphousSolidEnvironment(thermodynamic_state)

#presumably there are specialized arguments for each Environment that is supported

receptor2 =  Molecule.from_file("receptor2.pdb") #maybe we want to add another phase with a similar protein that could have either some fixed mutations or spectators
spectator_ligands = ligands = Molecule.from_file("spectator_ligands.sdf")
complex2_environment = SmallMoleculeComplexEnvironment(thermodynamic_state, receptor = receptor2, spectators = spectator_ligands)

#can pass attributes like above OR pass a yaml into the `Environment` object __init__



from perses.network import FreeEnergyNetwork
#add pre-specified topology_proposal, hybrid_factory, geometry_engine, and sampler arguments from a yaml
yaml_file = open('setup.yaml', 'r')
setup_options = yaml.load(yaml_file, Loader=yaml.FullLoader)
yaml_file.close()

network = FreeEnergyNetwork(system_generator,
                            ligands, #list of ligands (to which more ligands can be appended)
                            transformation_specifications = setup_options, #a yaml dict that has prespecified parameters for proposals and/or samplers that overwrite defaults
                            environments = [complex_environment, aqueous_environment, octanol_environment, complex2_environment] #a list of environments
                            )
#each enviornment instance in the list will make sure that every edge created in the network is labeled by the following tuple: (environment, transform_start, transform_end)
"""
transformation_specifications has a prespecified hierarchical dict of indexed transformations
Example:
    setup_options = {
                        SmallMoleculeComplexEnvironment.__class__.__name__ : {
                                                                                (0,2) : {
                                                                                            topology_proposal : {
                                                                                                                    'map_strength' : 'default',
                                                                                                                    'geometry' : 'strong'
                                                                                                                },
                                                                                            hybrid_factory : {
                                                                                                                'softcore_LJ_v2_alpha' : 0.5

                                                                                                             },
                                                                                            sampler : {
                                                                                                        'type' : 'repex',
                                                                                                        'ncycles' : 1000,
                                                                                                        'n_steps_per_move_application' : 200

                                                                                                      }
                                                                                        }
                                                                                }
    }
options that are not explicitly written are defaulted
"""

#add all to all edges (forward and backward)
for primary_idx in range(len(ligands)):
    for secondary_idx in range(len(ligands)):
        if primary_idx != secondary_idx:
            #add edges to each network; options will overwrite cached setup_options
            network.add_edge(primary_idx, secondary_idx, weight = 1., options = {SmallMoleculeComplexEnvironment.__class__.__name__ : {'sampler': {'type': 'neq', 'steps_neq': 10000}},
                                                                                OctanolEnvironment.__class__.__name__ : {'geometry_proposal': {'angle_softening_constant' : 0.5}}})
            #some options can overwrite the defaults and the yaml
            #some options (especially ones that change mapping criteria) may result in extra energetic contributions that might not cancel out between `Environment` objects
            #   so there has to be an internal consistency check to make sure that there is one consistent atom map broadcast across all phases

            #other arguments, like sampers, can be changed with each phase without issue

FreeEnergyNetwork.validate_network(network) #class method to make sure that every edge in one phase has a matching edge in another

###subsequent modifications to network edges can be done with networkx edge querying API and modifying `data`
edge_to_change = full_network([SmallMoleculeComplexEnvironment.__class__.__name__, 0,3])
edge_to_change.sampler.dt = 2.0 * unit.femtosecond #some attributes can always be changed in place
edge_to_change.topology_proposal.map_strength = 'weak' #some attributes that attempt to change topology_proposal, geometry_proposal, or htf parameters _after_ htf as already been created should raise an error
    # this should raise an error as the topology proposal was conducted when the
"""
CASE 2 : Protein Mutations
    - should handle WT:spectator_ligands, mutant_{mutate to}{resid}:spectator_ligands
    - should handle WT:apo, mutant_{mutate to}{resid}:spectator_ligands #though this would probably require two separate 'receptor' pdbs (one in apo, and one in complex)
"""
receptor_bound = Molecule.from_file("receptor_bound.pdb")
receptor_apo = Molecule.from_file("receptor_apo.pdb")
#presumably, these have the same topology EXCEPT for the existence of spectator ligands, which may

spectators = Molecule.from_file("docked_spectators.sdf")
complex_environment = PeptideMutationEnvironment(thermodynamic_state, receptor=receptor, spectators = spectators)
apo_environment = PeptideMutationEnvironment(thermodynamic_state, receptor=receptor, spectators = spectators) #just to be explicit, there are no spectator ligands

network = FreeEnergyNetwork(system_generator,
                            ligands = None, #the ligands will not be mutated
                            mutatable_residues = complex_environment.topology.residues(), #i don't think this is the right API, but it gets the point across that we allow all residues to mutate
                            transformation_specifications = setup_options, #a yaml dict that has prespecified parameters for proposals and/or samplers that overwrite defaults
                            environments = [complex_environment, apo_environment] #a list of environments
                            )
network.add_edge(23, 'K', weight=1, options = {}) #the second argument here is not a ligand index, but a string corresponding to a residue to mutate to
# does ^ necessitate that if the second argument is a string, then we must assert that each environment in the list is a PeptideMutationEnvironment
# it must also assert that the `FreeEnergyNetwork` `ligand` argument is None since we aren't mutating ligands

"""
CASE 3 : simultaneous transforming residues...
i.e. there is more than one alchemical transformation happening at once...
maybe this means that the `add_edge` method's first argument should actually be a list of tuples specifying all of the transformations happening at once?
"""

"""
I think that the API for the `compute engine` below seems pretty solid
"""

######################
# First, let's consider how we define each environment or system composition

# Environments define different kinds of environments we will
# be computing free energies between.
# They can represent a variety of environments, such as
# * ligand:protein complexes
# * protein:protein complexes
# * ligand in aqueous or organic solvent
# * ligand in an amorphous solvent

# Here are some examples of how we might define environments.


#
# Create a receptor in which we have pre-generated poses for several ligands
#

# Create a SystemGenerator that will construct openmm System objects
# from openforcefield or openmm Topology objects
from openmmforcefields.generators import SystemGenerator

system_generator = SystemGenerator(
    forcefields=["protein.ff14SB.xml", "tip3p_standard.xml"],
    small_molecule_forcefield="gaff-2.11",
)

# Define the thermodynamic state we will be simulating in
from openmmtools.states import ThermodynamicState

thermodynamic_state = ThermodynamicState(
    temperature=300.0 * unit.kelvin, pressure=1.0 * unit.atmospheres, pH=7.0
)

# Load receptor and ligand using the openforcefield topology objects
from openforcefield.topology import Molecule

# will need to add biopolymer support, but can be done quickly
receptor = Molecule.from_file("receptor.pdb")
# reference ligand posed in the receptor
reference_ligand = Molecule.from_file("reference_ligand.sdf")
# multiple ligands pre-docked to site
ligands = Molecule.from_file("docked_ligands.sdf")


# Create the environment containing a receptor and defining reference ligand
complex_environment = SmallMoleculeComplexEnvironment('complex',
    thermodynamic_state, receptor=receptor, reference_ligand=reference_ligand
)
# An Environment encapsulates all the machinery for building systems
# and positions when fed a new type of molecule.
# Configure some options for the complex environment
complex_environment.salt_concentration = {"NaCl": 200 * unit.millimolar}
complex_environment.solvent_buffer = 9 * unit.angstorms
# don't dock the ligand; they're already pre-aligned
complex_environment.dock_ligand = False

# When we feed a new ligand to environment.create_topology(), we will get a new topology with positions
# Later, the openforcefield Topology object may also hold positions, so 'positions' may not be needed
topology = complex_environment.create_topology(reference_ligand)
# We can then parameterize the system with the SystemGenerator
system = system_generator.create_system(topology)

# If we want to dock the ligands in using hybrid docking, we simply set an option
complex_environment.dock_ligand = "hybrid"
complex_topologies = [
    complex_environment.create_topology(ligand=ligand) for ligand in ligands
]  # off topologies with positions?

# JRG: in my head topologies do not ever hold positions :/

#
# Create an environment for the ligands in aqueous solvent
#

from perses.environments import AqueousEnvironment

solvent = AqueousEnvironment(thermodynamic_state)
solvent_environment.salt_concentration = {"NaCl": 200 * unit.millimolar}
complex_environment.solvent_buffer = 9 * unit.angstorms

#
# Create an environment representing the ligand in vacuum
#

from perses.environments import VacuumEnvironment

vacuum = VacuumEnvironment('vacuum', thermodynamic_state)

#
# More complex environments are possible too
#

from perses.environments import OctanolEnvironment

octanol = OctanolEnvironment('octanol', thermodynamic_state)
octanol.water_content = "saturated"  # maybe we have options that control how much water is in the environment

#
# Now create a free energy network that will represent the calculations we want to perform
#

from perses.adaptive import FreeEnergyNetwork

#network = FreeEnergyNetwork(system_generator)
network = FreeEnergyNetwork(system_generator, environments=[complex_environment, solvent_environment, vacuum_environments])
# JRG: Decide whether we use init args or not

#
# Add some complex and solvent edges in a star map
#

i = 0
for j in range(1, nligands):
    # Options are hierarchical dicts and lists, which are easy to specify in a YAML file
    # Look to YANK for inspiration on how we can organize parameters
    # to constructors hierarchically by class name?
    options = {
        "sampler": {
            "type": "ReplicaExchangeSampler",
            "number_of_iterations": 1000
        },
        "alchemical_factory": {
            "type": "RelativeAlchemicalFactory",
            "softcore_LJ_v2": True,
        },
    }

    network.mapper = SmallMoleculeProposalGenerator.mapper

	# When we add an edge, should we add it for all environments?

    edge = network.add_edge(
        f"{i} -> {j}",  # edge name is different from environment name
        ligands[i],
        ligands[j],
        weight=1.0,
        options=options,
    )
    edge.environments['complex']['sampler']['type'] = 'SAMSSampler' # How would we override options for different environments
    # or change for all environments?

    # Add complex edge
    #complex_edge = network.add_edge(
    #    f"complex {i} -> {j}",  # JRG: this label could be autogenerated from the arguments
    #    complex_environment,
    #    ligands[i],
    #    ligands[j],
    #    mapping=mapping,
    #    weight=1.0,
    #    options=options,
    #)
    # Add solvent edge
    #solvent_edge = network.add_edge(
    #    f"solvent {i} -> {j}",  # JRG: this label could be autogenerated from the arguments
    #    solvent_environment,
    #    ligands[i],
    #    ligands[j],
    #    weight=1.0,
    #  	mapping=mapping,
    #    options=options,
    )
    # Change some options
    #solvent_edge.options["sampler"]["type"] = "SAMSSampler"  # is this better
    #solvent_edge.options.sampler.number_of_iterations = 500  # or this, or both?
    # JRG: I prefer the latter one (both can be used)


# JRG: How do you retrieve an edge after network.add_edge? Do we use slicing? Which key?

# Check that graph is not decomposible for each environment
network.validate()

#
# Execute a standard non-adaptive free energy calculation
#

# Connect to a compute engine, which can service multiple calculations running at once
# This could start the calculation running once an engine is available
# Or we could have explicit network.run(n_iterations=10) methods, or network.run(async=True),
# or other start/stop methods
# JRG: I strongly prefer the explicit methods in this case.
network.engine = DaskComputeEngine(
    scheduler="127.0.0.1:8786"
)  # subclass of ComputeEngine to provide a common interface to different backends

# Synchronous execution
network.run(n_iterations=10)

# Check on free energies, error estimates as a networkx graph
g = network.to_networkx()
print(g)

# Check in on the calculation
f_i, df_i = network.get_free_energies() # would automatically use MLE
delta_f_ij, ddelta_f_ij = network.get_free_energy_differences() # would automatically use MLE

# Update the network manually by adjusting a weight
edge = network.get_edge(complex_topologies[i], complex_topologies[j])
edge.weight = 0.5

# Add another edge (while the calculation is running)
# JRG: This will be very cool to have and maybe hard to implement too :D
network.add_edge('new_edge', ligand[1], ligand[2], weight=1.0, options=options
)

# Wait a bit
time.sleep(60)

# Add a weight adaptation strategy that will start to adapt the weights based on the current results
# These strategies could have multiple policies or goals, and would have a common API for adaptively
# changing weights, etc.
network.strategy = DiffNetAdaptiveStrategy(goal="A-optimal")

network.strategy = BudgetAdaptiveStrategy(budget='$3000')
# network will have to have statistics of each edge in terms of production rate, runtime, cost, etc.

# Wait a bit
time.sleep(60)

# Check on weights by rendering free energies, error estimates, and weights to a static networkx graph
g = network.to_networkx()
print(g)

# Stop the calculation
network.engine = None
network.stop_calculation()
network.terminate()
# Or maybe this should be network.stop()?
# JRG: Yes, I'd rather have a .stop() method or a criteria for self-stopping.

################################################################################
# OLD (GOOD) STUFF FROM DOMINIC FOR INSPIRATION
################################################################################

from perses.app.experiments import *

# first, we want to make a client to distribute edge-building across
# the graph and add them as the attributes return asynchronously
network_builder_parallelism = Parallelism()
network_builder_parallelism.activate_client(
    library=("dask", "LSF"),
    num_processes=10,
    processor_type="cpu",  # we only need cpus to build the edges
)

connectivity_data = {
    (0, 1, "solvent", 1): {
        "anneal_14s": True,
        "softcore_LJ_v2": False,
        "simulation_flavor": "repex",
        "n_states": 11,
        "n_cycles": 5000,
    },
    (0, 1, "complex", 1): {
        "softcore_LJ_v2": True,
        "simulation_flavor": "neq",
        "protocol_forward": range(0, 1, 10000),
        "num_particles": 100,
    },
    (1, 2, "solvent", 1): None,
    (1, 2, "complex", 1): None,
}
# we probably need a check to make sure that if the ligand_i, ligand_j is specified,
# that there is an edge for each appropriate phase

# or we can load the connectivity data from a .yaml
connectivity_data = load_yaml(f"example_connectivity.yml")

network_engine = NetworkBuilder(
    parallelism=network_builder_parallelism,
    ligand_input=f"ligands.sdf",  # suppose this just has ligands 0, 1, 2
    receptor_filename=f"thrombin.pdb",
    connectivity_data=connectivity_data,
)

# deactivate the parallelism client and kill it's associated workers
network_builder_parallelism.deactivate_client()


# perhaps we want to add an edge
network_engine.add_edge(
    ligand_i=0,
    ligand_j=2,
    weight=1,
    parameters={"softcore_LJ_v2": True, "simulation_flavor": "neq"},
)

# change weight for good measure
network_edge.network.edges(0, 1)["weight"] = 2

# now we can inspect the network to make sure that it has built and validated all of the edges...
network_engine.print_pretty_network()

# now let's create some input data;
# since we will run complex phase edges on GPUS, we will create a client that handles _just_ gpu jobs
complex_graph_client = Parallelism(
    library=("dask", "LSF"), num_processes=10, processor="gpu"
)  # we only need cpus to build the edges

solvent_graph_client = Parallelism(
    library=("dask", "LSF"), num_processes=10, processor="cpu"
)  # we only need cpus to build the edges

# define a dictionary where we allocate certain edges to certain clients
# based on the computational needs of the edge
# in this example, we delegate all solvent edges to the
# cpu_client and all complex edges to the gpu_client
delegate_edge_clients_dict = {
    set(
        [edge for edge in network_engine.edges() if "solvent" in edge(data=True).keys()]
    ): solvent_graph_client,
    set(
        [edge for edge in network_engine.edges() if "complex" in edge(data=True).keys()]
    ): complex_graph_client,
}
##########################################
##########################################

# once we are satisfied with the initial graph, let's build a sampler
equal_allocation_sampler = EqualAllocationNetworkSampler(
    network=network_engine.network,
    edge_clients=delegeate_edge_clients_dict,
    iteration_per_edge=None,  # this is allocated by `allotted_time_per_edge`
)
equal_allocation_sampler.extend_sampler(
    allotted_time_per_edge={
        edge: 3600 * unit.seconds for edge in network_engine.network.edges()
    }
)
# then we wait...
equal_allocation_sampler.save_results()

##########################################
##########################################

# let's build a generalizable adaptive sampler with a specified policy...
policy = FindBestBinderPolicy(
    error_tolerance=1 * unit.kilojoules_per_mole
)  # this policy will leverage a policy for Bayesian Inference

adaptive_sampler = KArmBanditSampler(
    network=network_engine.network,
    policy=policy,
    edge_clients=delegate_edge_clients_dict,
)

adaptive_sampler.extend_sampler(
    total_wallclock_time=sum(
        complex_graph_client.num_workers + solvent_graph_client.num_workers
    )
    * unit.hours
)
adaptive_sampler.save_results()



#
#
#

import time

# ENVIRONMENTS

# First, let's consider how we define each environment or system composition

# Environments define different kinds of environments we will be computing free energies between.
# They can represent a variety of environments, such as
# * ligand:protein complexes
# * protein:protein complexes
# * ligand in aqueous or organic solvent
# * ligand in an amorphous solvent

# Here are some examples of how we might define environments.


#
# Create a receptor in which we have pre-generated poses for several ligands
#

# Create a SystemGenerator that will construct openmm System objects from openforcefield or openmm Topology objects
from openmmforcefields.generators import SystemGenerator
system_generator = SystemGenerator(
    forcefields=['protein.ff14SB.xml', 'tip3p_standard.xml'],
    small_molecule_forcefield='gaff-2.11')

# Define the thermodynamic state we will be simulating in
from openmmtools.states import ThermodynamicState
thermodynamic_state = ThermodynamicState(
    temperature = 300.0 * unit.kelvin,
    pressure = 1.0 * unit.atmospheres,
    pH = 7.0)

# Load receptor and ligand using the openforcefield topology objects
from openforcefield.topology import Molecule
receptor = Molecule.from_file('receptor.pdb') # will need to add biopolymer support, but can be done quickly
reference_ligand = Molecule.from_file('reference_ligand.sdf') # reference ligand posed in the receptor
ligands = Molecule.from_file('docked_ligands.sdf') # multiple ligands pre-docked to site


# Create the environment containing a receptor and defining reference ligand
complex_environment = SmallMoleculeComplexEnvironment(thermodynamic_state, receptor=receptor, reference_ligand=reference_ligand)
# An Environment encapsulates all the machinery for building systems and positions when fed a new type of molecule.
# Configure some options for the complex environment
complex_environment.salt_concentration = { 'NaCl' : 200 * unit.millimolar }
complex_environment.solvent_buffer = 9 * unit.angstorms
complex_environment.dock_ligand = False # don't dock the ligand; they're already pre-aligned

# When we feed a new ligand to environment.create_topology(), we will get a new topology with positions
# Later, the openforcefield Topology object may also hold positions, so 'positions' may not be needed
topology, positions = complex_environment.create_topology(reference_ligand)
# We can then parameterize the system with the SystemGenerator
system = system_generator.create_system(topology)

# If we want to dock the ligands in using hybrid docking, we simply set an option
complex_environment.dock_ligand = 'hybrid'
complex_topologies = [ complex_environment.create(ligand=ligand) for ligand in ligands ] # off topologies with positions?

#
# Create an environment for the ligands in aqueous solvent
#

from perses.environments import AqueousEnvironment
solvent = AqueousEnvironment(thermodynamic_state)
solvent_environment.salt_concentration = { 'NaCl' : 200 * unit.millimolar }
complex_environment.solvent_buffer = 9 * unit.angstorms

#
# Create an environment representing the ligand in vacuum
#

from perses.environments import VacuumEnvironment
vacuum = VacuumEnvironment(thermodynamic_state)

#
# More complex environments are possible too
#

from perses.environments import OctanolEnvironment
octanol = OctanolEnvironment(thermodynamic_state)
octanol.water_content = 'saturated' # maybe we have options that control how much water is in the environment

#
# Now create a free energy network that will represent the calculations we want to perform
#

from perses.adaptive import FreeEnergyNetwork
network = FreeEnergyNetwork(system_generator)

#
# Add some complex and solvent edges in a star map
#

i = 0
for j in range(1, nligands):
    # Options are hierarchical dicts and lists, which are easy to specify in a YAML file
    # Look to YANK for inspiration on how we can organize parameters to constructors hierarchically by class name?
    options = {
        'sampler' : {
            'type' : 'ReplicaExchangeSampler',
            'number_of_iterations' : 1000
            },
        'alchemical_factory' : {
            'type' : 'RelativeAlchemicalFactory',
            'softcore_LJ_v2' : True,
            }
    }
    # Add complex edge
    complex_edge = network.add_edge(f'complex {i} -> {j}', complex_environment, ligands[i], ligands[j], weight=1.0, options=options)
    # Add solvent edge
    solvent_edge = network.add_edge(f'solvent {i} -> {j}', solvent_environment, ligands[i], ligands[j], weight=1.0, options=options)
    # Change some options
    solvent_edge.options['sampler']['type'] = 'SAMSSampler' # is this better
    solvent_edge.options.sampler.number_of_iterations = 500 # or this, or both?

#
# Execute a standard non-adaptive free energy calculation
#

# Connect to a compute engine, which can service multiple calculations running at once
# This could start the calculation running once an engine is available
# Or we could have explicit network.run(n_iterations=10) methods, or network.run(async=True), or other start/stop methods
network.engine = DaskComputeEngine(scheduler='127.0.0.1:8786') # subclass of ComputeEngine to provide a common interface to different backends

# Wait a bit
time.sleep(60)

# Check in on the calculation
f_i, df_i = network.get_free_energies()
delta_f_ij, ddelta_f_ij = network.get_free_energy_differences()

# Update the network manually by adjusting a weight
edge = network.get_edge(complex_topologies[i], complex_topologies[j])
edge.weight = 0.5

# Add another edge (while the calculation is running)
network.add_edge(complex_topologies[1], complex_topologies[2], weight=1.0, options=options)

# Wait a bit
time.sleep(60)

# Add a weight adaptation strategy that will start to adapt the weights based on the current results
# These strategies could have multiple policies or goals, and would have a common API for adaptively
# changing weights, etc.
network.strategy = DiffNetAdaptiveStrategy(goal='A-optimal')

# Wait a bit
time.sleep(60)

# Check on weights by rendering free energies, error estimates, and weights to a static networkx graph
g = network.to_networkx()
print(g)

# Stop the calculation
network.engine = None
# Or maybe this should be network.stop()?

################################################################################
# OLD (GOOD) STUFF FROM DOMINIC FOR INSPIRATION
################################################################################

from perses.app.experiments import *
#first, we want to make a client to distribute edge-building across the graph and add them as the attributes return asynchronously
network_builder_parallelism = Parallelism()
network_builder_parallelism.activate_client(library = ('dask', 'LSF'),
                                            num_processes = 10,
                                            processor_type = 'cpu', #we only need cpus to build the edges
                                            )

connectivity_data = {(0,1, 'solvent', 1) : {'anneal_14s': True,
                                            'softcore_LJ_v2': False,
                                            'simulation_flavor': 'repex',
                                            'n_states': 11,
                                            'n_cycles': 5000
                                            }
                     (0,1, 'complex', 1) : {'softcore_LJ_v2': True,
                                            'simulation_flavor': 'neq',
                                            'protocol_forward': range(0,1,10000),
                                            'num_particles': 100
                                            },
                    (1,2,'solvent', 1): None,
                    (1,2, 'complex', 1): None,
                    }
#we probably need a check to make sure that if the ligand_i, ligand_j is specified, that there is an edge for each appropriate phase

#or we can load the connectivity data from a .yaml
connectivity_data = load_yaml(f"example_connectivity.yml")

network_engine = NetworkBuilder(parallelism = network_builder_parallelism,
                                ligand_input = f"ligands.sdf", #suppose this just has ligands 0, 1, 2
                                receptor_filename = f"thrombin.pdb",
                                connectivity_data = connectivity_data)

#deactivate the parallelism client and kill it's associated workers
network_builder_parallelism.deactivate_client()


#perhaps we want to add an edge
network_engine.add_edge(ligand_i = 0, ligand_j = 2, weight = 1, parameters = {'softcore_LJ_v2': True, 'simulation_flavor': 'neq'} )

#change weight for good measure
network_edge.network.edges(0,1)['weight'] = 2

#now we can inspect the network to make sure that it has built and validated all of the edges...
network_engine.print_pretty_network()

#now let's create some input data;
#since we will run complex phase edges on GPUS, we will create a client that handles _just_ gpu jobs
complex_graph_client = Parallelism(library = ('dask', 'LSF'),
                                    num_processes = 10,
                                    processor = 'gpu') #we only need cpus to build the edges

solvent_graph_client = Parallelism(library = ('dask', 'LSF'),
                                    num_processes = 10,
                                    processor = 'cpu') #we only need cpus to build the edges

#define a dictionary where we allocate certain edges to certain clients based on the computational needs of the edge
#in this example, we delegate all solvent edges to the cpu_client and all complex edges to the gpu_client
delegate_edge_clients_dict = {set([edge for edge in network_engine.edges() if 'solvent' in edge(data=True).keys()]) : solvent_graph_client,
                              set([edge for edge in network_engine.edges() if 'complex' in edge(data=True).keys()]): complex_graph_client
                                }
##########################################
##########################################

#once we are satisfied with the initial graph, let's build a sampler
equal_allocation_sampler = EqualAllocationNetworkSampler(network = network_engine.network,
                                                         edge_clients = delegeate_edge_clients_dict,
                                                         iteration_per_edge = None #this is allocated by `allotted_time_per_edge`
                                                         )
equal_allocation_sampler.extend_sampler(allotted_time_per_edge = {edge: 3600 * unit.seconds for edge in network_engine.network.edges()})
#then we wait...
equal_allocation_sampler.save_results()

##########################################
##########################################

#let's build a generalizable adaptive sampler with a specified policy...
policy = FindBestBinderPolicy(error_tolerance = 1 * unit.kilojoules_per_mole) #this policy will leverage a policy for Bayesian Inference

adaptive_sampler = KArmBanditSampler(network = network_engine.network,
                                     policy = policy,
                                     edge_clients = delegate_edge_clients_dict)

adaptive_sampler.extend_sampler(total_wallclock_time = sum(complex_graph_client.num_workers + solvent_graph_client.num_workers) * unit.hours)
adaptive_sampler.save_results()

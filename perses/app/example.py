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

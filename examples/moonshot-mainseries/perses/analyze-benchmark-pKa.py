"""
Analyze perses calculations for a benchmark set of ligands annotated with experimental data

"""

# SDF filename containing all ligands annotated with experimental data
# Ligands should be annotated with experimental data using the SD tags
# EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL
# EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR

import argparse
import numpy as np

def get_molecule_titles(filename, add_warts=False):
    """
    Get the list of molecule titles (names) from the specified SDF file.

    Parameters
    ----------
    filename : str
        The filename to read molecules from
    add_warts : bool, optional, default=False
        If True, if multiple molecules with the same title are read, a wart (_0, _1, ...) is appended.

    Returns
    -------
    titles : list of str
        List of molecule titles from the provided SDF file
    """
    # Read titles
    titles = list()
    from openeye import oechem
    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            title = oemol.GetTitle()
            # Trim warts
            title = title.rsplit('_')[0]
            titles.append(title)
            
    if not add_warts:
        return titles
            
    #
    # Add warts to duplicate titles
    #

    # Count titles
    from collections import defaultdict
    title_counts = defaultdict(int)
    for title in titles:
        title_counts[title] += 1

    # Add warts to titles that appear multiple times
    wart_index = defaultdict(int)
    for index, title in enumerate(titles):
        if title_counts[title] > 1:
            wart = f'_{wart_index[title]}'
            wart_index[title] += 1
            titles[index] = title + wart

    return titles

def get_experimental_data(filename):
    """
    Get experimental data for all physical molecules from a CSV file.

    The following SD tags will be examined
    * `Title`
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL`
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR`
    
    Parameters
    ----------
    filename : str
        The SDF or CSV filename of real molecules with annotations to read molecules from

    Returns
    -------
    molecules : dict
        molecules[title] is a dict containing entries:
            `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL`
            `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR`
    """     
    import pandas as pd
    df = pd.read_csv(filename)
    df.set_index('Title', inplace=True)
    molecules = df.to_dict('index')
    return molecules

def get_microstate_info(filename):
    """Get the microstate free energy penalties (in kcal/mol) and parent state name associated with each microstate.

    Parameters
    ----------
    filename : str
        Name of SDF file to read from

    Returns
    -------
    microstates : list
        microstates[index] is a dict containing entries:
            `compound` : title of this microstate
            `parent_molecule` : str, title of parent molecule
            `r_epik_State_Penalty` : float, free energy solution penalty for this microstate

    """
    # Get kcal/mol -> kT conversion factor
    from openmm import unit
    from openmmtools.constants import kB
    kT = kB * 300 * unit.kelvin # thermal energy at 300 K

    # Read microstates
    microstates = list()
    from openeye import oechem
    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()        
        while oechem.OEReadMolecule(ifs, oemol):
            # The microstate name is stored in the `compound` SD tag
            title = oechem.OEGetSDData(oemol, 'compound')

            microstate = dict()
            for dp in oechem.OEGetSDDataPairs(oemol):
                try:
                    microstate[dp.GetTag()] = float(dp.GetValue())
                except ValueError as e:
                    microstate[dp.GetTag()] = dp.GetValue()
            
            microstates.append(microstate)

    return microstates

def get_experimental_data_network(filename):
    """
    Get experimental data for all physical molecules from an SDF or CSV file.

    The following SD tags will be examined
    * `TITLE`
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL`
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR`
    
    Parameters
    ----------
    filename : str
        The SDF or CSV filename of real molecules with annotations to read molecules from

    Returns
    -------
    graph : networkx.DiGraph
        graph.nodes[title] contains the following attributes for the molecule with title 'title':
            exp_g_i : the experimental free energy of binding in kT
            exp_dg_i : standard error in exp_g_i

        300 K is assumed for the experimental measurements in converting from kcal/mol to kT
     
    """
    import networkx as nx
    graph = nx.DiGraph()

    from openmm import unit
    from openmmtools.constants import kB
    kT = kB * 300 * unit.kelvin # thermal energy at 300 K

    # Get titles with warts
    titles = get_molecule_titles(filename, add_warts=False)

    molecule_index = 0
    from openeye import oechem
    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            #title = oemol.GetTitle()
            title = titles[molecule_index] # use title with warts added
            tagname = 'EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL'
            if oechem.OEHasSDData(oemol, tagname):
                node_data = {
                    'exp_g_i' : float(oechem.OEGetSDData(oemol, tagname)) * unit.kilocalories_per_mole / kT,
                    'exp_dg_i' : float(oechem.OEGetSDData(oemol, tagname+'_STDERR')) * unit.kilocalories_per_mole  / kT,
                    }

                graph.add_node(title, **node_data)
            molecule_index += 1

    return graph

def collapse_states(perses_graph, microstates):
    """
    Read epik state penalties from SDF and collapse protonation/tautomeric states

    The following SD tags will be examined
    * `r_epik_State_Penalty` : penalty of solution microstate in kcal/mol

    Parameters
    ----------
    perses_graph : networkx.DiGraph
        The perses graph of microstates where nodes are named with microstate titles
    microstates : list of dict
        microstates[index] is a dict that contains
            'compound' : microstate title
            'parent_compound' : parent macrostate title
            'r_epik_State_Penalty' : solution microstate penalty in kcal/mol
     
    """
    # Create a copy of the graph
    import copy
    perses_graph = copy.deepcopy(perses_graph)

    # Compute conversion from kcal/mol to kT
    from openmm import unit
    from openmmtools.constants import kB
    kT = kB * 300 * unit.kelvin # thermal energy at 300 K

    macrostates = { microstate['compound'] : microstate['parent_molecule'] for microstate in microstates }

    # Annotate graph with macrostate titles
    for microstate in perses_graph.nodes:
        perses_graph.nodes[microstate]['ligand_title'] = macrostates[microstate]

    # Read state penalties into perses network
    microstate_penalties_in_kT = { microstate['compound'] : microstate['r_epik_State_Penalty'] * unit.kilocalories_per_mole / kT for microstate in microstates } # microstate penalties in kT
    for microstate in perses_graph.nodes:
        perses_graph.nodes[microstate]['state_penalty_dg_i'] = microstate_penalties_in_kT[microstate]

    # Create new graph
    print('Creating macrostate graph...')
    import networkx as nx
    import numpy as np
    collapsed_graph = nx.DiGraph() # graph of free energy estimates
    unique_titles = set(macrostates.values())
    for ligand_title in unique_titles:
        # Retrieve all absolute free energy estimates
        state_penalty_i = list()
        g_i = list()
        dg_i = list()
        microstates = list()
        for microstate in perses_graph.nodes:
            if perses_graph.nodes[microstate]['ligand_title'] == ligand_title:
                microstates.append(microstate)
                state_penalty_i.append(perses_graph.nodes[microstate]['state_penalty_dg_i'])
                g_i.append(perses_graph.nodes[microstate]["mle_g_i"])
                dg_i.append(perses_graph.nodes[microstate]["mle_dg_i"])
        state_penalty_i = np.array(state_penalty_i)
        g_i = np.array(g_i)
        dg_i = np.array(dg_i)

        print(f'{ligand_title} {microstates} {state_penalty_i} {g_i} {dg_i} {state_penalty_i + g_i}')
        
        from scipy.special import logsumexp
        state_penalty_i = state_penalty_i + logsumexp(-state_penalty_i) # normalize
        g = -logsumexp(-(state_penalty_i + g_i))
        w_i = np.exp(-(state_penalty_i + g_i) + g)
        dg = np.sqrt(np.sum( (w_i * dg_i)**2 ))

        collapsed_graph.add_node(ligand_title, mle_g_i=g, mle_dg_i=dg)

    # DEBUG : Set lowest free energy to zero
    g0 = min([ node['mle_g_i'] for name, node in collapsed_graph.nodes.items() ])
    for name, node in collapsed_graph.nodes.items():
        node['mle_g_i'] -= g0

    # Compute edges
    print('Computing edges...')
    for i in collapsed_graph.nodes:
        for j in collapsed_graph.nodes:
            if i != j:
                g_ij = collapsed_graph.nodes[j]['mle_g_i'] - collapsed_graph.nodes[i]['mle_g_i']
                dg_ij = np.sqrt(collapsed_graph.nodes[j]['mle_dg_i']**2 + collapsed_graph.nodes[i]['mle_dg_i']**2)
                collapsed_graph.add_edge(i, j, g_ij=g_ij, g_dij=dg_ij)


    return collapsed_graph

def get_perses_realtime_statistics(basepath, trajectory_prefix='out'):
    """
    Retrieve contents of perses realtime analysis YAML files for a given perses output directory.
    
    Parameters
    ----------
    basepath : str
        Filepath pointing to the output of a single perses transformation
    trajectory_prefix : str, optional, default='out'
        trajectory_prefix used for output files in setup YAML file

    Returns
    -------
    statistics : dict
        statistics[phase] is the contents of the analysis YAML 

    """
    
    statistics = dict()

    import yaml
    import os
    # TODO: Auto-detect phase names from filenames that are present
    for phase in ['vacuum', 'complex', 'solvent']:
        filename = f'{basepath}/{trajectory_prefix}-{phase}_real_time_analysis.yaml'
        if os.path.exists(filename):
            with open(filename, 'rt') as infile:
                statistics[phase] = yaml.safe_load(infile)

    return statistics


def get_perses_network_results(basepath, trajectory_prefix='out'):
    """
    Read real-time statistics for all perses transformations in 'basepath' launched via the perses CLI and build a network of estimated free energies.

    .. todo ::
    
    * Enable user to specify one or more experimental measurements that can be passed to DiffNet to improve the resulting estimate

    Parameters
    ----------
    basepath : str
        Filepath glob pointing to the output of transformations in a network
    trajectory_prefix : str, optional, default='out'
        trajectory_prefix used for output files in setup YAML file

    Returns
    -------
    graph : networkx.DiGraph()
        NetworkX graph containing the estimated free energies of all edges and overall MLE free energy estimate solution.
        graph.edges(data=True) will return a list of (i, j, properties) directed i -> j edges with these properties:
           'g_ij' : MBAR free energy estimate for i -> j transformation (in units of kT); negative indicates j should bind more tightly than i
           'g_dij' : standard error uncertainty estimate for g_ij (also in units of kT)
        
    """

    # DEBUG
    #import os
    #filename = 'test.edgelist'
    #if os.path.exists(filename):
    #    import networkx as nx
    #    return nx.read_edgelist(filename)

    # Get list of all YAML files generated by the CLI
    import glob
    yaml_filenames = glob.glob(f'{basepath}/perses-*.yaml')

    # Read each transformation summary and assemble the statistics into a graph
    import networkx as nx
    graph = nx.DiGraph() # graph of free energy estimates
    
    import yaml
    import numpy as np
    from rich.progress import track

    for filename in track(yaml_filenames, description='[blue]Retrieving results of perses calculations...'):
        try:
            with open(filename, 'rt') as infile:
                perses_input = yaml.safe_load(infile)
        except yaml.scanner.ScannerError as e:
            # Some files may become corrupted for unknown reasons
            print(e)
            continue
            
        # Extract initial and final ligand indices
        old_ligand_index = perses_input['old_ligand_index']
        new_ligand_index = perses_input['new_ligand_index']
        path = perses_input['trajectory_directory']
        # Read microstate names and epik state penalites of microstates
        ligand_file = perses_input['ligand_file']        
        microstates = get_microstate_info(ligand_file)
        old_ligand_title = microstates[old_ligand_index]['compound']
        new_ligand_title = microstates[new_ligand_index]['compound']

        # Retrieve realtime statistics for this edge
        statistics = get_perses_realtime_statistics(path, trajectory_prefix=trajectory_prefix)
        # Include this edge if both complex and solvent have useful data
        if ('solvent' in statistics) and ('complex' in statistics):
            # TODO: Extract more statistics about run completion
            # NOTE: We will provide an API for making it easier to gather information about overall binding free energy statistics

            # Package up edge attributes
            edge_attributes = {
                'g_ij' : statistics['complex'][-1]['mbar_analysis']['free_energy_in_kT'] - statistics['solvent'][-1]['mbar_analysis']['free_energy_in_kT'],
                'g_dij' : np.sqrt(statistics['complex'][-1]['mbar_analysis']['standard_error_in_kT']**2 + statistics['solvent'][-1]['mbar_analysis']['standard_error_in_kT']**2),
            }
            
            # Ensure error is not zero
            if edge_attributes['g_dij'] == 0.0:
                print(f'{filename} has g_dij = 0.0; correcting to 1.0 so that analysis will run, but this should not happen')
                edge_attributes['g_dij'] = 1.0

            graph.add_edge(old_ligand_title, new_ligand_title, **edge_attributes)

    # DEBUG
    nx.write_edgelist(graph, "test.edgelist")

    return graph

def solve_mle(graph):

    # Use DiffNet maximum likelihood estimator (MLE) to estimate overall absolute free energies of each ligand 
    # omitting any experimental measurements
    #
    # https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00528
    from openff.arsenic import stats
    g_i, C_ij = stats.mle(graph, factor='g_ij')

    # Populate graph with MLE estimates
    dg_i = np.sqrt(np.diag(C_ij))
    for node, g, dg in zip(graph.nodes, g_i, dg_i):
        graph.nodes[node]["mle_g_i"] = g
        graph.nodes[node]["mle_dg_i"] = dg

    return graph

def generate_arsenic_plots(experimental_data_graph, perses_graph, arsenic_csv_filename='benchmark.csv', target='benchmark',
                           relative_plot_filename='relative.pdf', absolute_plot_filename='absolute.pdf'):
    """
    Generate an arsenic CSV file and arsenic plots

    .. warning:: The CSV file will be deprecated once arsenic object model is improved.

    Parameters
    ----------
    experimental_data_graph : networkx.DiGraph
        graph.nodes[title] contains the following attributes for the molecule with title 'title':
            exp_g_i : the experimental free energy of binding in kT
            exp_dg_i : standard error in exp_g_i
    perses_graph : networkx.DiGraph()
        NetworkX graph containing the estimated free energies of all edges and overall MLE free energy estimate solution.
        graph.edges(data=True) will return a list of (i, j, properties) directed i -> j edges with these properties:
           'g_ij' : MBAR free energy estimate for i -> j transformation (in units of kT); negative indicates j should bind more tightly than i
           'g_dij' : standard error uncertainty estimate for g_ij (also in units of kT)
    arsenic_csv_filename : str, optional, default='arsenic.csv'
        Path to arsenic CSV input file to be generated
    target : str, optional, default='target'
        Target name to use in plots
    relative_plot_filename : str, optional, default='relative.pdf'
        Relative free energy comparison with experiment plot
        This plot compares the direct computed edges (without MLE corrections) with experimental free energy differences
    absolute_plot_filename : str, optional, default='absolute.pdf'
        Absolute free energy comparison with experiment plot
        This plot compares the MLE-derived absolute comptued free energies with experimental free energies
        with the computed free energies shifted to the experimental mean
    """
    from openmm import unit
    from openmmtools.constants import kB
    kT = kB * 300 * unit.kelvin # thermal energy at 300 K

    # Write arsenic CSV file
    with open(arsenic_csv_filename, 'w') as csv_file:
        # Experimental block
        # print header for block
        csv_file.write("# Experimental block\n")
        csv_file.write("# Ligand, expt_DG, expt_dDG\n")
        # Extract ligand name, expt_DG and expt_dDG from ligands dictionary
        for ligand_name, data in experimental_data_graph.nodes(data=True):
            csv_file.write(f"{ligand_name}, {data['exp_g_i'] * kT/unit.kilocalories_per_mole}, {data['exp_dg_i'] * kT/unit.kilocalories_per_mole}\n")            

        # Calculated block
        # print header for block
        csv_file.write("# Calculated block\n")
        csv_file.write("# Ligand1,Ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)\n")
        # Loop through simulation, extract ligand1 and ligand2 indices, convert to names, create string with
        # ligand1, ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)
        # write string in csv file
        for ligand1, ligand2, data in perses_graph.edges(data=True):
            csv_file.write(
                f"{ligand1}, {ligand2}, {data['g_ij'] * kT/unit.kilocalories_per_mole}, {data['g_dij'] * kT/unit.kilocalories_per_mole}, 0.0\n")  # hardcoding additional error as 0.0

    # Generate comparison plots
    from openff.arsenic import plotting, wrangle

    # Generate arsenic plots comparing experimental and calculated free energies
    fe = wrangle.FEMap(arsenic_csv_filename)

    # Generate relative plot
    print(f'Generating {relative_plot_filename}...')
    plotting.plot_DDGs(fe.graph,
                       target_name=f'{target}',
                       title=f'Relative binding energies - {target}',
                       figsize=5,
                       units='kcal/mol',
                       filename=relative_plot_filename,
                   )

    # Generate absolute plot, with experimental data shifted to correct mean
    print(f'Generating {absolute_plot_filename}...')
    #experimental_mean_dg = np.asarray([node[1]["exp_DG"] for node in fe.graph.nodes(data=True)]).mean()
    experimental_mean_dg = np.asarray([data['exp_g_i']*kT/unit.kilocalories_per_mole for node, data in experimental_data_graph.nodes(data=True)]).mean()
    plotting.plot_DGs(fe.graph,
                      target_name=f'{target}',
                      title=f'Absolute binding energies - {target}',
                      figsize=5,
                      units='kcal/mol',
                      filename=absolute_plot_filename,
                      shift=experimental_mean_dg,
                  )

def display_predictions(graph):
    """
    Display the predicted free energies in a table.
    """
    # Get kcal/mol -> kT conversion factor
    from openmm import unit
    from openmmtools.constants import kB
    kT = kB * 300 * unit.kelvin # thermal energy at 300 K
    kT_in_kcal_per_mole = kT/unit.kilocalories_per_mole
    
    from rich.console import Console
    from rich.table import Table

    # Display absolute free energies
    table = Table(title="perses free energy estimates ΔG")

    table.add_column("ligand", justify="left", style="cyan", no_wrap=True)
    table.add_column("perses ΔG (kcal/mol)", justify="centered", style="magenta")

    # Sort ligands
    sorted_ligands = list(graph.nodes)
    sorted_ligands.sort(key = lambda ligand_name : graph.nodes[ligand_name]['mle_g_i'])
    for ligand_name in sorted_ligands:
        data = graph.nodes[ligand_name]
        table.add_row(ligand_name, f"{data['mle_g_i']*kT_in_kcal_per_mole:6.1f} ± {data['mle_dg_i']*kT_in_kcal_per_mole:5.1f}")

    console = Console()
    console.print(table)

    # Display differences
    table = Table(title="perses free energy differences ΔΔG")

    table.add_column("old ligand", justify="left", style="cyan", no_wrap=True)
    table.add_column("new ligand", justify="left", style="cyan", no_wrap=True)
    table.add_column("perses ΔΔG (kcal/mol)", justify="centered", style="magenta")

    # Sort ligands
    sorted_ligands = list(graph.nodes)
    sorted_ligands.sort(key = lambda ligand_name : graph.nodes[ligand_name]['mle_g_i'])
    for edge in graph.edges:
        data = graph.edges[edge]
        table.add_row(edge[0], edge[1], f"{data['g_ij']*kT_in_kcal_per_mole:6.1f} ± {data['g_dij']*kT_in_kcal_per_mole:5.1f}")

    console = Console()
    console.print(table)

def plot_sign_accuracy(perses_graph, experimental_graph, filename):
    """
    Generate a plot showing accuracy in predicting sign of DDG as a function of threshold.

    Parameters
    ----------
    perses_graph : networkx.DiGraph
        Perses analysis graph containing MLE and experimental free energies
    experimental_graph : networkx.DiGraph
        Experimental data graph containing MLE and experimental free energies
    filename : str
        Filename to write to    

    """

    dgmax = 5
    thresholds = np.linspace(0, dgmax, 100)
    accuracies = 0 * thresholds
    for index, threshold in enumerate(thresholds):
        correct_predictions = [ np.sign(data['g_ij']) == np.sign(experimental_graph.nodes[j]['exp_g_i']-experimental_graph.nodes[i]['exp_g_i']) for i, j, data in perses_graph.edges(data=True) if (abs(data['g_ij']) > threshold) ]
        accuracies[index] =  np.mean(correct_predictions)

    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(5,5))
    plt.plot(thresholds, accuracies*100, 'k-')
    plt.xlabel('threshold (kcal/mol)')
    plt.ylabel('accuracy (%)')
    plt.title('accuracy in predicting sign of transformation')
    plt.axis([0, dgmax, 0, 100])
    plt.savefig(filename)

if __name__ == '__main__':
    # Set up argument parser
    arg_parser = argparse.ArgumentParser(prog='analyze-benchmark.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--basepath", type=str, help="Base path for globbing results for free energy network", default='step1-His41(0)-Cys145(0)-His163(0)-*')
    arg_parser.add_argument("--docked", type=str, help="Path to docked ligands SDF file", default='../docked/step1-x2646_0A-dimer-His41(0)-Cys145(0)-His163(0).sdf')
    arg_parser.add_argument("--expdata", type=str, help="Path to experimental data CSV file", default='../molecules/step1.csv')
    args = arg_parser.parse_args()

    # Read molecule experimental data
    experimental_data_graph = get_experimental_data(args.expdata)

    # Read protonation state penalties and parent compound IDs for each microstate
    microstate_info = get_microstate_info(args.docked)

    # Read perses free energy network results
    perses_graph = get_perses_network_results(args.basepath)

    # Solve for MLE estimate (without including experimental data)
    perses_graph = solve_mle(perses_graph)

    # Check that we have sufficient data to analyze the graph
    if len(perses_graph.nodes) == 0:
        raise Exception('No edges have generated sufficient data to compare with experiment yet. Both solvent and complex phases must have provided data to analyze.')

    # Filter perses graph to include only those with experimental data
    #print('Only including nodes with experimental data...')
    #for node in list(perses_graph.nodes):
    #    if node not in experimental_data_graph.nodes:
    #        perses_graph.remove_node(node)

    collapsed_graph = collapse_states(perses_graph, microstate_info)
    
    #plot_sign_accuracy(collapsed_graph, experimental_data_graph, 'accuracy.pdf')

    # Show the predictions
    display_predictions(collapsed_graph)

    # Generate arsenic plots comparing experimental and calculated
    #arsenic_csv_filename = 'arsenic.csv' # CSV file to generate containing experimental absolute free energies and raw edge computed free energies from perses    
    #target_name = 'benchmark'
    #generate_arsenic_plots(experimental_data_graph, collapsed_graph, arsenic_csv_filename=arsenic_csv_filename, target=target_name)

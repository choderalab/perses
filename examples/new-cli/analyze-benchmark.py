"""
Analyze perses calculations for a benchmark set of ligands annotated with experimental data

"""

# SDF filename containing all ligands annotated with experimental data
# Ligands should be annotated with experimental data using the SD tags
# EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL
# EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR
ligands_sdf_filename = 'ligands-old.sdf'




import argparse
import glob
import itertools
import re
import warnings

import numpy as np
import urllib.request
import yaml

from openmmtools.constants import kB

from simtk import unit


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

def get_molecule_titles(filename):
    """
    Get the list of molecule titles (names) from the specified SDF file

    Parameters
    ----------
    filename : str
        The filename to read molecules from

    Returns
    -------
    titles : list of str
        List of molecule titles from the provided SDF file
    """
    titles = list()
    from openeye import oechem
    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            titles.append( oemol.GetTitle() )

    return titles

def get_molecule_experimental_data(filename):
    """
    Get experimental data for molecules using an SDF file

    The following SD tags will be examined
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL`
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR`
    

    Parameters
    ----------
    filename : str
        The filename to read molecules from

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

    experimental_free_energies_in_kT = dict()
    from openeye import oechem
    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            title = oemol.GetTitle()
            tagname = 'EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL'
            if oechem.OEHasSDData(oemol, tagname):
                node_data = {
                    'exp_g_i' : float(oechem.OEGetSDData(oemol, tagname)) * unit.kilocalories_per_mole / kT,
                    'exp_dg_i' : float(oechem.OEGetSDData(oemol, tagname+'_STDERR')) * unit.kilocalories_per_mole  / kT
                    }

                graph.add_node(title, **node_data)

    return graph

def get_perses_network_results(basepath, trajectory_prefix='out'):
    """
    Read real-time statistics for all perses transformations in 'basepath' launched via the perses CLI and build a network of estimated free energies.

    .. todo ::
    
    * Enable user to specify one or more experimental measurements that can be passed to DiffNet to improve the resulting estimate

    Parameters
    ----------
    basepath : str
        Filepath pointing to the output of a single perses transformation
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
    # Get list of all YAML files generated by the CLI
    import glob, os
    yaml_filenames = glob.glob(f'{basepath}/parsed-*.yaml')

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
        # Extract names of molecules from the input SDF file used to launch simulations
        # NOTE: This requires the ligand SDF file to be present and have the same path name
        # TODO: We don't need to do this over and over again if the ligand file is the same
        ligand_titles = get_molecule_titles(perses_input['ligand_file'])
        old_ligand_title = ligand_titles[old_ligand_index]
        new_ligand_title = ligand_titles[new_ligand_index]

        # DEBUG: Exclude charge change and major ring changes (aromatic to non-aromatic)
        exclude_list =  ['0001_Nuv0252', '487_Nuv0273', '490_Nuv0671'] 
        if (old_ligand_title in exclude_list) or (new_ligand_title in exclude_list):
            continue
        
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

            graph.add_edge(old_ligand_title, new_ligand_title, **edge_attributes)

    print(f'Read {len(graph.edges)} perses transformations')

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
    # Write arsenic CSV file
    with open(arsenic_csv_filename, 'w') as csv_file:
        # Experimental block
        # print header for block
        csv_file.write("# Experimental block\n")
        csv_file.write("# Ligand, expt_DG, expt_dDG\n")
        # Extract ligand name, expt_DG and expt_dDG from ligands dictionary
        for ligand_name, data in experimental_data_graph.nodes(data=True):
            csv_file.write(f"{ligand_name}, {data['exp_g_i']}, {data['exp_dg_i']}\n")            

        # Calculated block
        # print header for block
        csv_file.write("# Calculated block\n")
        csv_file.write("# Ligand1,Ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)\n")
        # Loop through simulation, extract ligand1 and ligand2 indices, convert to names, create string with
        # ligand1, ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)
        # write string in csv file
        for ligand1, ligand2, data in perses_graph.edges(data=True):
            csv_file.write(
                f"{ligand1}, {ligand2}, {data['g_ij']}, {data['g_dij']}, 0.0\n")  # hardcoding additional error as 0.0

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
                       units='kT',
                       filename=relative_plot_filename,
                   )

    # Generate absolute plot, with experimental data shifted to correct mean
    print(f'Generating {absolute_plot_filename}...')
    #experimental_mean_dg = np.asarray([node[1]["exp_DG"] for node in fe.graph.nodes(data=True)]).mean()
    experimental_mean_dg = np.asarray([data['exp_g_i'] for node, data in experimental_data_graph.nodes(data=True)]).mean()
    plotting.plot_DGs(fe.graph,
                      target_name=f'{target}',
                      title=f'Absolute binding energies - {target}',
                      figsize=5,
                      units='kT',
                      filename=absolute_plot_filename,
                      shift=experimental_mean_dg,
                  )

def display_predictions(graph):
    """
    Display the predicted free energies in a table.
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title="perses free energy estimates (up to additive constant)")

    table.add_column("ligand", justify="left", style="cyan", no_wrap=True)
    table.add_column("perses ΔG / kT", justify="centered", style="magenta")

    # Sort ligands
    sorted_ligands = list(graph.nodes)
    sorted_ligands.sort(key = lambda ligand_name : graph.nodes[ligand_name]['mle_g_i'])
    for ligand_name in sorted_ligands:
        data = graph.nodes[ligand_name]
        table.add_row(ligand_name, f"{data['mle_g_i']:6.1f} ± {data['mle_dg_i']:5.1f}")

    console = Console()
    console.print(table)

if __name__ == '__main__':

    # Get molecule experimental data
    experimental_data_graph = get_molecule_experimental_data('ligands.sdf')

    # Get perses free energy estimates and MLE estimates
    perses_results_basepath = '/data/chodera/chodera/perses/perses/examples/new-cli'
    perses_graph = get_perses_network_results(perses_results_basepath)

    # Check that we have sufficient data to analyze the graph
    if len(perses_graph.nodes) == 0:
        raise Exception('No edges have generated sufficient data to compare with experiment yet. Both solvent and complex phases must have provided data to analyze.')

    # Show the predictions
    display_predictions(perses_graph)

    # Generate arsenic plots comparing experimental and calculated
    arsenic_csv_filename = 'arsenic.csv' # CSV file to generate containing experimental absolute free energies and raw edge computed free energies from perses    
    target_name = 'benchmark'
    generate_arsenic_plots(experimental_data_graph, perses_graph, arsenic_csv_filename=arsenic_csv_filename, target=target_name)

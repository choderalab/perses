"""
Integrated retrospective and prospective analysis for perses free energy calculations.

Issue: https://github.com/choderalab/perses/issues/1136




# INPUTS
- ligands.sdf : SDF file from which the calculations are run
    - Assume it's a single file specified in the perses yaml ligand_file
    - Assume all ligands represent single protonation/tautomer/stereochemical macrostate
    - If there are multiple protonation/tautomer states that represent one real chemical compound:
        - SDTag called PARENT_LIGAND_ID that corresponds to the LIGAND_ID of the real compound, for which this microstate represents a protonation/tautomer/stereochemical state.
        - If this is a protonation/tautomer microstate it will have the following SDTags:
            - `r_epik_State_Penalty` : state penalty of this microstate in kcal/mol.
                - if no state penalty is specified, assume equal populations. This can be useful for stereoisomers.
- experimental-data.sdf/csv : Tagged experimental data for one or more ligands
    - could be the same ligands.sdf if tags are present with the following fields:
            * EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL
            * EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR
            - if csv it should have this format: SMILES, LIGAND_ID, SDData...
            - if sdf the title should match the molecule LIGAND_ID and SDDATA is sdtags
-


# OUTPUTS
- optimal-predictions.yaml : Predicted optimal binding FEs of all compounds, using ALL available experimental data.
    - Format:
        metadata:
            energy_units : kilocalories_per_mole
            temperature : <temperature in kelvin>
        ligand_data :
            ligand_id :
                absolute_binding_free_energy :
                    estimate :
                    standard_error :
                microstates : [if any ligand has multiple microstates]
                    microstate_id :
                        estimate :
                        standard_error :
                        microstate_penalty :
                        microstate_population_in_complex :
                    ...
                experimental_data:
                    measurement :
                    standard_error :
            ...
    Question: Do we also want to produce all ligand pairs with correct uncertainties?
- retrospective-optimal-reliability.yaml : Reliability indicator for optimal predictions using experimental data for everything. For each compound with exp data we use all other experimental data to compute an optimal MLE estimate.
    ligand_id :
        absolute_binding_free_energy :
            estimate :
            standard_error :
        microstates : [if any ligand has multiple microstates]
            microstate_id :
                estimate :
                standard_error :
                microstate_penalty :
                microstate_population_in_complex :
            ...
        experimental_data:
            measurement :
            standard_error :
        error :
            deviation :
            standard_error :
    ...
- retrospective-edges.yaml : Relative FE results sorted by absolute errors (magnitude) for those with experimental data. Only for edges between ligands with single microstates.
    - initial_ligand_id :
      final_ligand_id :
      absolute_error :
      signed_error :
      error_standard_error :
      calculated :
      calculated_standard_error :
      experimental :
      experimental_standard_error :
      statistics : [Optional]
    ...

- retrospective-absolutes.yaml : (For compounds with experimental data) MLE estimates of absoute binding free energies computed without experimental data, listed in order of absolute FE deviations from experiment
    ligand_id :
      absolute_error :
      signed_error :
      error_standard_error :
      calculated :
      calculated_standard_error :
      experimental :
      experimental_standard_error :
      statistics : [Optional]
      microstates : [Optional]
        microstate_id :
            estimate :
            standard_error :
            microstate_penalty :
            microstate_population_in_complex :







This script produces three major outputs:
* Optimal predictions: The predicted absolute ΔGs with uncertainties 
  (and ΔΔGs between compounds with uncertainties) integrating all available experimental data in the network
* Reliability indicator for optimal predictions: If there is more than one compound with experimental data, 
  we also provide a leave-one-out assessment of how accurately we predicted each compound with experimental data 
  by leaving its experimental free energy out of the DiffNet analysis and reporting the deviation statistics
* Retrospective debugging: For compounds with experimental data, a comparison of directly computed 
  edge ΔΔGs(without using any experimental data) with the experimental ΔΔGs, as well as how the 
  absolute ΔGs (without experimental data corrections) correlate with experimental absolute ΔGs 
  (shifting to the experimental mean in both cases)

Experimental data format:
* CSV: CSV file containing the following fields



TODO:
* Refactor this stand-alone script into an integrated analysis class and CLI integrated into perses.
* Support experimental data in YAML format as well

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

def read_sddata(filename):
    """Read SD tag data from the specified SDF or CSV file.

    Parameters
    ----------
    filename : str
        The SDF or CSV file to read from

    Returns
    -------
    molecules : list of dict
        molecules[index] is a dict containing the SD tag : data pairs from the file
        Numeric values are converted to float when possible

    """

    molecules = list()
    from openeye import oechem
    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()        
        while oechem.OEReadMolecule(ifs, oemol):
            # The parent compound name name is stored in the `Name` SD tag
            name = oechem.OEGetSDData(oemol, 'Name')

            sddata = dict()
            for dp in oechem.OEGetSDDataPairs(oemol):
                try:
                    sddata[dp.GetTag()] = float(dp.GetValue())
                except ValueError as e:
                    sddata[dp.GetTag()] = dp.GetValue()            
            molecules.append(sddata)

    return molecules


def get_experimental_data_network(filename,
                                  exp_tag_name="EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL",
                                  uncertainty_tag_name="EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR"):
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
    exp_tag_name : str
        Name for the SDF tag name to look for the experimental data in kcal/mol.
    uncertainty_tag_name : str
        Name for the SDF tag name to look for the uncertainty of the experimental data in kcal/mol.

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

    from scipy.special import logsumexp

    # Compute conversion from kcal/mol to kT
    from openmm import unit
    from openmmtools.constants import kB
    kT = kB * 300 * unit.kelvin # thermal energy at 300 K

    macrostates = { microstate['compound'] : microstate['parent_compound'] for microstate in microstates }

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

        # Skip compounds where there are no microstates completed yet
        if len(microstates) == 0:
            continue

        print(f'{ligand_title} {microstates} {state_penalty_i} {g_i} {dg_i} {state_penalty_i + g_i}')
        
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

def get_perses_realtime_statistics(basepath):
    """
    Retrieve contents of perses realtime analysis YAML files for a given perses output directory.
    
    Parameters
    ----------
    basepath : str
        Filepath pointing to the output of a single perses transformation

    Returns
    -------
    statistics : dict
        statistics[phase] is the contents of the analysis YAML 

    """
    
    statistics = dict()

    import yaml
    import os

    # Get list of all YAML files generated by the CLI
    import glob
    filenames = glob.glob(f'{basepath}/*_real_time_analysis.yaml')
    for filename in filenames:
        for phase in ['vacuum', 'complex', 'solvent']:
            if phase in filename:
                with open(filename, 'rt') as infile:
                    statistics[phase] = yaml.safe_load(infile)

    return statistics


def get_perses_network_results(basepath, mirostates):
    """
    Read real-time statistics for all perses transformations in 'basepath' launched via the perses CLI and build a network of estimated free energies.

    .. todo ::
    
    * Enable user to specify one or more experimental measurements that can be passed to DiffNet to improve the resulting estimate

    Parameters
    ----------
    basepath : str
        Filepath glob pointing to the output of transformations in a network
    microstates : list of dict
        List of microstates
        Each microstate contains the following SD tags:
        * 'Microstate_Name' denotes microstate name
        * 'Name' name of parent compound
        * TODO: microstate state penalties

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
        old_ligand_title = microstates[old_ligand_index]['compound']
        new_ligand_title = microstates[new_ligand_index]['compound']

        # Retrieve realtime statistics for this edge
        edge_path = os.path.join(basepath, path)
        print(edge_path)
        statistics = get_perses_realtime_statistics(edge_path)
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

def read_edge(filename):
    """Read informatiion for a single transformation.
    The perses YAML filename is specified, and the transformation information in this directory is read.

    Parameters
    ----------
    filename : str
        Perses YAML filename

    """
    import yaml
    import os
    try:
        with open(filename, 'rt') as infile:
            perses_input = yaml.safe_load(infile)
    except yaml.scanner.ScannerError as e:
        # Some files may become corrupted for unknown reasons
        print(e)
        return None

    # Extract initial and final ligand names
    # TODO: We will have to make sure this uses microstate names
    old_ligand_title = perses_input['old_ligand_name']
    new_ligand_title = perses_input['new_ligand_name']
    path = perses_input['trajectory_directory']

    # Retrieve realtime statistics for this edge
    edge_path = os.path.dirname(filename)
    statistics = get_perses_realtime_statistics(edge_path)
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

        return (old_ligand_title, new_ligand_title, edge_attributes)

    return None

def get_perses_network_results_multiprocessing(basepath):
    """
    Read real-time statistics for all perses transformations in 'basepath' launched via the perses CLI and build a network of estimated free energies.

    .. todo ::
    
    * Enable user to specify one or more experimental measurements that can be passed to DiffNet to improve the resulting estimate

    Parameters
    ----------
    basepath : str
        Filepath glob pointing to the output of transformations in a network

    Returns
    -------
    graph : networkx.DiGraph()
        NetworkX graph containing the estimated free energies of all edges and overall MLE free energy estimate solution.
        graph.edges(data=True) will return a list of (i, j, properties) directed i -> j edges with these properties:
           'g_ij' : MBAR free energy estimate for i -> j transformation (in units of kT); negative indicates j should bind more tightly than i
           'g_dij' : standard error uncertainty estimate for g_ij (also in units of kT)
        
    """

    # DEBUG: Read edgelist directly form a file
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
    from multiprocessing import Pool
    from tqdm import tqdm
    import os
    processes = None
    if 'LSB_DJOB_NUMPROC' in os.environ:
        processes = int(os.environ['LSB_DJOB_NUMPROC'])
    pool = Pool(processes=processes)
    for edge in track(pool.imap_unordered(read_edge, yaml_filenames), total=len(yaml_filenames), description='[blue]Retrieving results of perses calculations...'):
        if edge is not None:
            (old_ligand_title, new_ligand_title, edge_attributes) = edge
            graph.add_edge(old_ligand_title, new_ligand_title, **edge_attributes)

    # DEBUG
    nx.write_edgelist(graph, "test.edgelist")

    return graph

def solve_mle(graph):

    # Use DiffNet maximum likelihood estimator (MLE) to estimate overall absolute free energies of each ligand 
    # omitting any experimental measurements
    #
    # https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00528
    from cinnabar import stats
    g_i, C_ij = stats.mle(graph, factor='g_ij')

    # Populate graph with MLE estimates
    dg_i = np.sqrt(np.diag(C_ij))
    for node, g, dg in zip(graph.nodes, g_i, dg_i):
        graph.nodes[node]["mle_g_i"] = g
        graph.nodes[node]["mle_dg_i"] = dg

    return graph

def generate_arsenic_plots(experimental_data, perses_graph, arsenic_csv_filename='benchmark.csv', target='benchmark',
                           relative_plot_filename='relative.pdf', absolute_plot_filename='absolute.pdf', pIC50_sdtag="pChEMBL Value"):
    """
    Generate an arsenic CSV file and arsenic plots

    .. warning:: The CSV file will be deprecated once arsenic object model is improved.

    Parameters
    ----------
    experimental_data : dict
        experimental_data[title] contains the following attributes for the molecule with title 'title':
            `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL`
            `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR`
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
    import numpy as np
    kT = kB * 300 * unit.kelvin # thermal energy at 300 K

    # Write arsenic CSV file
    with open(arsenic_csv_filename, 'w') as csv_file:
        # Experimental block
        # print header for block
        csv_file.write("# Experimental block\n")
        csv_file.write("# Ligand, expt_DG, expt_dDG\n")
        # Extract ligand name, expt_DG and expt_dDG from ligands dictionary
        for sddata in experimental_data:
            # Compute experimental affinity in kcal/mol
            ligand_name = sddata['Name']
            dg_exp = - kT * np.log(10) * sddata['pChEMBL Value']
            ddg_exp = 0.3 # estimate
            # Write CSV
            csv_file.write(f"{ligand_name}, {dg_exp}, {ddg_exp}\n")            

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
    from cinnabar import plotting, wrangle

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
    experimental_mean_pChEMBL = np.asarray([sddata['pChEMBL Value'] for sddata in experimental_data]).mean()
    experimental_mean_dg = - 0.592 * np.log(10) * experimental_mean_pChEMBL
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
    #table = Table(title="perses free energy differences ΔΔG")
    #table.add_column("old ligand", justify="left", style="cyan", no_wrap=True)
    #table.add_column("new ligand", justify="left", style="cyan", no_wrap=True)
    #table.add_column("perses ΔΔG (kcal/mol)", justify="centered", style="magenta")

    # Sort ligands
    # sorted_ligands = list(graph.nodes)
    # sorted_ligands.sort(key = lambda ligand_name : graph.nodes[ligand_name]['mle_g_i'])
    # for edge in graph.edges:
    #     data = graph.edges[edge]
    #     table.add_row(edge[0], edge[1], f"{data['g_ij']*kT_in_kcal_per_mole:6.1f} ± {data['g_dij']*kT_in_kcal_per_mole:5.1f}")

    # console = Console()
    # console.print(table)

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

#
# MAIN
#

if __name__ == '__main__':
    # Set up argument parser
    arg_parser = argparse.ArgumentParser(prog='analyze.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--basepath", type=str, help="Base path for globbing results for free energy network", default='20230410/jak2_5ns_*')
    arg_parser.add_argument("--macrostates", type=str, help="Path to macrostate data (which can contain experimental data) SDF or CSV file; must contain Name SD tag", default='20230410/ligands_from_FEP_cations_SUBSET_IC50_uM.sdf')
    arg_parser.add_argument("--microstates", type=str, help="Path to microstate data (e.g. docked ligands) SDF or CSV file; Titles must be unique, and must contain Name SD tag", default='20230410/ligands_from_FEP_cations_SUBSET_IC50_uM.sdf')
    arg_parser.add_argument("--sdtag", type=str, help="SDF tag to look for experimental pIC50 value.", default="pChEMBL Value")
    args = arg_parser.parse_args()

    # Notes:
    # We expect the experimental data file to contain a 'Name' tag for each molecule
    # If experimental data is present, we need this to be expressed as pChEMBL in the experimental data file
    # The microstates must include a unique Title for each molecule, and contain the parent compound 'Name' in the molecule

    # Get experimental pIC50 sdtag name
    pIC50_sdtag = args.sdtag

    # Read molecule experimental data
    print(f"Reading macrostate data from SD tags in '{args.macrostates}'...")
    macrostates = read_sddata(args.macrostates)

    # Read all microstates, including microstate state penalties (if present)
    print(f"Reading microstate data from SD tags in '{args.microstates}'...")
    microstates = read_sddata(args.microstates)

    # Read perses free energy network results
    print(f"Reading free energy network results from '{args.basepath}'...")
    perses_graph = get_perses_network_results_multiprocessing(args.basepath)

    # Solve for MLE estimate (without including experimental data)
    print(f"Solving MLE estimate...")
    perses_graph = solve_mle(perses_graph)

    # Check that we have sufficient data to analyze the graph
    if len(perses_graph.nodes) == 0:
        raise Exception('No edges have generated sufficient data to compare with experiment yet. Both solvent and complex phases must have provided data to analyze.')

    # Filter perses graph to include only those with experimental data
    #print('Only including nodes with experimental data...')
    #for node in list(perses_graph.nodes):
    #    if node not in experimental_data_graph.nodes:
    #        perses_graph.remove_node(node)

    # TODO: Collapse microstates
    #collapsed_graph = collapse_states(perses_graph, microstate_info)
    
    #plot_sign_accuracy(collapsed_graph, experimental_data_graph, 'accuracy.pdf')

    # Show the predictions
    display_predictions(perses_graph)

    # Generate arsenic plots comparing experimental and calculated
    arsenic_csv_filename = 'arsenic.csv' # CSV file to generate containing experimental absolute free energies and raw edge computed free energies from perses    
    target_name = 'benchmark'
    generate_arsenic_plots(macrostates, perses_graph, arsenic_csv_filename=arsenic_csv_filename, target=target_name, pIC50_sdtag=pIC50_sdtag)

"""
Functions to plot free energy maps for sets of ligands

"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import linregress

import logging

logger = logging.getLogger(__name__)

def pathway(nodes):
    """
    Converts a list of joined nodes into the pairwise steps

    Parameters
    ----------
    nodes : list
        list of integers of nodes to convert

    Returns
    -------
    List of tuples of edges involved in pathway

    """
    steps = []
    for i in range(len(nodes) - 1):
        steps.append((nodes[i], nodes[i + 1]))
    steps.append((nodes[-1], nodes[0]))
    return steps


def get_attr(G, nodeA, nodeB, attr='dg'):
    """
    Reads data dict assigned to a given edge and returns the item

    Parameters
    ----------
    G : networkx.Graph
    nodeA : int
        index of first node of edge
    nodeB : int
        index of second node of edge
    attr : str
        key of data of edge to get

    Returns
    -------
    Value corresponding to attr for edge (nodeA, nodeB) of graph G
    """
    return G.get_edge_data(nodeA, nodeB)[0][attr]


def combine_errors(errors):
    return (np.sum([x ** 2 for x in errors])) ** 0.5

def cycle_closure(G, steps, verbose=False):
    """
    Finds all cycles of n steps in free energy map G and returns those which the cycle closure is larger than the combined error

    Parameters
    ----------
    G : networkx.Graph
    steps : int
        number of steps for cycles to consider
    verbose : bool, default False
        option to plot information for all cycles

    Returns
    -------
    list of cycles that do not close to within the combined error of the constituent edges

    """
    bad = []

    for i in nx.simple_cycles(G):
        if len(i) == steps:
            path = pathway(i)
            total = 0
            errors = []
            for step in path:
                total += get_attr(G,step[0], step[1])
                errors.append(get_attr(G,step[0], step[1], 'ddg'))
            total_error = combine_errors(errors)

            if total > total_error:
                total = np.round(total, 1)
                total_error = np.round(total_error, 1)
                print(f"Cycle {i} does not close")
                print(f"Closure: {total}")
                print(f"Cycle error: {total_error}")
                bad.append(i)
            elif verbose:
                # print anyway
                print(f"Cycle {i}")
                print(f"Closure: {total}")
                print(f"Cycle error: {total_error}")
    return bad


def plot_comparison(x,y,X,Y, title='',shaded=True,color='blue'):
    """
    Function to compare different attributes and/or graphs. Both the attributes can be the same (i.e. comparing 'dg'
    from two different maps) or two attributes of the same graph ('exp' vs 'dg').

    Parameters
    ----------
    x : str
        data attribute of graph X to plot on x-axis
    y : str
        data attribute of graph Y to plot on y-axis
    X : networkx.Graph
        graph to plot on x-axis
    Y : networkx.Graph
        graph t plot on y-axis. This can be the same as X
    title : str
        title of figure
    shaded : bool
        add grey shaded region to illustrate 1kT error from unity
    color : str
        color of datapoints

    Returns
    -------

    R2 : float
        R^2 of datapoints plotted
    std_err : float
        standard error of datapoints plotted

    """
    errors = {'dg': 'ddg', 'exp': 'experr', 'calc': 'calcerr'}

    plt.figure(figsize=(10, 10))


    xs = []
    ys = []
    for edge in X.edges():
        if Y.has_edge(edge[0], edge[1]):
            xval = get_attr(X,edge[0], edge[1], x)
            yval = get_attr(Y,edge[0], edge[1], y)
            xerr = get_attr(X,edge[0], edge[1], errors[x])
            yerr = get_attr(Y,edge[0], edge[1], errors[y])
            if xval < 0:
                xval = -xval
                yval = -yval
            plt.scatter(xval, yval, color=color)
            plt.errorbar(xval, yval, xerr=xerr, yerr=yerr, color=color)
            xs.append(xval)
            ys.append(yval)

    slope, intercept, r_value, _value, std_err = linregress(xs, ys)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    limits = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    plt.xlim(limits)
    plt.ylim(limits)
    limits = list(limits)
    plt.plot(limits, limits, '--', color='grey', linewidth=1)
    if shaded:
        plt.fill_between(limits, [x - 1 for x in limits], [x + 1 for x in limits], color='grey', alpha=0.2)

    plt.xlabel(x + ' / kT')
    plt.ylabel(y + ' / kT')
    plt.text(limits[0]+1, limits[1]-1, f"$R^2$ = {r_value**2:.2f}")
    plt.title(title)
    plt.show()


    return r_value**2, std_err
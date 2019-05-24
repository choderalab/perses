"""
Functions to plot free energy maps for sets of ligands

"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def pathway(nodes):
    steps = []
    for i in range(len(nodes) - 1):
        steps.append((nodes[i], nodes[i + 1]))
    steps.append((nodes[-1], nodes[0]))
    return steps


def get_attr(G, nodeA, nodeB, attr='dg'):
    return G.get_edge_data(nodeA, nodeB)[0][attr]


def combine_errors(errors):
    return (np.sum([x ** 2 for x in errors])) ** 0.5

def cycles(G, steps):
    cycles = []

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
    return c

def cycle_closure(G, steps, verbose=False):
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


def plot_comparison(x,y,X,Y, title='', shaded=True,color='blue'):
    errors = {'dg': 'ddg', 'exp': 'experr', 'calc': 'calcerr'}

    plt.figure(figsize=(10, 10))

    for edge in X.edges():
        xval = get_attr(X,edge[0], edge[1], x)
        yval = get_attr(Y,edge[0], edge[1], y)
        xerr = get_attr(X,edge[0], edge[1], errors[x])
        yerr = get_attr(Y,edge[0], edge[1], errors[y])
        plt.scatter(xval, yval, color=color)
        plt.errorbar(xval, yval, xerr=xerr, yerr=yerr, color=color)

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
    plt.title(title)
    plt.show()
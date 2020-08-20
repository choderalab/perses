import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns
import numpy as np
from simtk.openmm import unit
from openmmtools.constants import kB

temperature_kelvin = 300.
temperature = temperature_kelvin * unit.kelvin
kT = kB * temperature


def _produce_plot(name, show_plots=False, plot_file_format="pdf"):
    if show_plots:
        plt.show()
    else:
        fname = os.extsep.join([name, plot_file_format])
        logging.info(f"Writing {fname}")
        plt.savefig(fname)


def plot_single_work_distribution(f, r, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(f, shade=True, color="cornflowerblue", ax=ax)
    sns.rugplot(
        f,
        ax=ax,
        color="cornflowerblue",
        alpha=0.5,
        label=f"forward : N={len(f)}",
    )

    sns.kdeplot(
        [-x for x in r], shade=True, color="hotpink", ax=ax
    )
    sns.rugplot(
        [-x for x in r],
        ax=ax,
        color="hotpink",
        alpha=0.5,
        label=f"reverse : N={len(r)}",
    )
    if title is not None:
        ax.set_title(title)


def plot_two_work_distribution(f1, r1, f2, r2, phases=(None, None), title=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 3.25))
    plot_single_work_distribution(f1, r1, ax1, phases[0])
    plot_single_work_distribution(f2, r2, ax2, phases[1])

    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
        )
    fig.subplots_adjust(top=0.9, wspace=0.15)
    ax1.legend()
    ax2.legend()

    _produce_plot('title')


def plot_relative_distribution(relative_fes, bins=100, title="rel_fe_hist"):
    """ Plots the distribution of relative free energies

    Parameters
    ----------
    relative_fes : list
        Relative free energies in kcal/mol
    bins : int, default=100
        Number of bins for histogramming


    """
    sns.kdeplot(
        relative_fes, shade=True, color="hotpink")
    sns.rugplot(
        relative_fes,
        color="hotpink",
        alpha=0.5,
        label=f"N={len(relative_fes)}",
    )
    plt.xlabel("Relative free energy to ligand 0 / kcal/mol")
    _produce_plot(title)


def plot_convergence(results, n_gens=3, title=None):
    if "complex_fes" in results and "solvent_fes" in results:
        max_gen = 0
        for i in range(n_gens):
            try:
                DDG = ((results[f"solvent_fes_GEN{i}"][0]- results[f"complex_fes_GEN{i}"][0])* kT).value_in_unit(unit.kilocalories_per_mole)
                low = ((results[f"solvent_fes_GEN{i}"][1]- results[f"complex_fes_GEN{i}"][2])* kT).value_in_unit(unit.kilocalories_per_mole)
                high = ((results[f"solvent_fes_GEN{i}"][2]- results[f"complex_fes_GEN{i}"][1])* kT).value_in_unit(unit.kilocalories_per_mole)
                plt.scatter(i, DDG, color='green')
                plt.vlines(i,low,high,color='green')
                if i > max_gen:
                    max_gen = i
            except KeyError:
                continue

        colors = {'solvent': 'blue', 'complex': 'red'}
        for phase in ['solvent', 'complex']:
            y = []
            low = []
            high = []
            for i in range(n_gens):
                try:
                    y.append((results[f"{phase}_fes_GEN{i}"][0] * kT)
                             .value_in_unit(unit.kilocalories_per_mole))
                    low.append((results[f"{phase}_fes_GEN{i}"][1] * kT)
                             .value_in_unit(unit.kilocalories_per_mole))
                    high.append((results[f"{phase}_fes_GEN{i}"][2] * kT)
                             .value_in_unit(unit.kilocalories_per_mole))
                    if i > max_gen:
                        max_gen = i
                except KeyError:
                    continue
            shift = np.mean(y)
            y = y - shift
            low = low - shift
            high = high - shift
            plt.scatter([i for i in range(0, max_gen+1)],  y,
                        color=colors[phase],label=phase)
            for i,_ in enumerate(y):
                plt.vlines(i,low[i],high[i],color=colors[phase])

        plt.xlabel('GEN')
        plt.ylabel('Relative free energy /'+r' kcal mol${^-1}$')
        plt.plot([0, max_gen],
                 [(results['binding_fe'][0]* kT).value_in_unit(unit.kilocalories_per_mole), (results['binding_fe'][0]* kT).value_in_unit(unit.kilocalories_per_mole)],
                 color='green', linestyle=":", label='free energy (all GENS)')
        plt.fill_between([0, max_gen],
                         (results['binding_fe'][1]* kT).value_in_unit(unit.kilocalories_per_mole),
                         (results['binding_fe'][2]* kT).value_in_unit(unit.kilocalories_per_mole),
                         alpha=0.2, color='green')
        plt.xticks([i for i in range(0, max_gen+1)])
        plt.legend()
        _produce_plot(f"fe_convergence_{title}")


def plot_cumulative_distributions(results, minimum=None, maximum=5, cmap='PiYG', n_bins=100,
                                  markers=[-2, -1, 0, 1, 2],
                                  title='Cumulative distribution'):
    """Plots cumulative distribution of ligand affinities

    Parameters
    ----------
    results : list(float)
        List of affinities to plot
    maximum : int, default=5
        Maximum affinity to plot, saves plotting boring plateaus
    cmap : str, default='PiYG'
        string name of colormap to use
    n_bins : int, default=100
        Number of bins to use
    markers : list(float), default=range(-2,3)
        Affinity values at which to label
    title : str, default='Cumulative distribution'
        Title to label plot

    """
    if minimum is None:
        results = [x for x in results if x < maximum]
    else:
        results = [x for x in results if minimum < x < maximum]

    # the colormap could be a kwarg
    cm = plt.cm.get_cmap(cmap)

    # Get the histogramp
    Y, X = np.histogram(list(results), n_bins)
    Y = np.cumsum(Y)
    x_span = X.max()-X.min()
    C = [cm(((X.max()-x)/x_span)) for x in X]

    plt.bar(X[:-1], Y, color=C, width=X[1]-X[0], edgecolor='k')

    for v in markers:
        plt.vlines(-v, 0, Y.max(), 'grey', linestyles='dashed')
        plt.text(v-0.5, 0.8*Y.max(),
                 f"$N$ = {len([x for x in results if x < v])}",
                 rotation=90,
                 verticalalignment='center',
                 color='green')
    plt.xlabel('Affinity relative to ligand 0 / '+r'kcal mol$^{-1}$')
    plt.ylabel('Cumulative $N$ ligands')
    plt.title(title)
    _produce_plot(f"cumulative_{title}")

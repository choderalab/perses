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


def _produce_plot(name, show_plots=False, plot_file_format = "pdf"):
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


def _plot_relative_distribution(relative_fes, bins=100):
    """ Plots the distribution of relative free energies

    Parameters
    ----------
    relative_fes : list
        Relative free energies in kcal/mol
    bins : int, default=100
        Number of bins for histogramming


    """
    plt.hist(relative_fes, bins=bins)
    plt.xlabel("Relative free energy to ligand 0 / kcal/mol")
    _produce_plot("rel_fe_lig0_hist")


### this will be useful for looking at looking at shift in relative FEs over GENS
def _plot_convergence(results, n_gens=3, title=None):
    # TODO add plotting of complex and solvent legs independently (shifted to zero) as there might be more repetitions of one than the other
    if "complex_fes" in results and "solvent_fes" in results:
        for i in range(n_gens):
            try:
                DDG = (
                    (
                        np.mean(results[f"complex_fes_GEN{i}"])
                        - np.mean(results[f"solvent_fes_GEN{i}"])
                    )
                    * kT
                ).value_in_unit(unit.kilocalories_per_mole)
                dDDG = (
                    (
                        np.mean(results[f"complex_dfes_GEN{i}"]) ** 2
                        + np.mean(results[f"solvent_dfes_GEN{i}"]) ** 2
                    )
                    ** 0.5
                    * kT
                ).value_in_unit(unit.kilocalories_per_mole)
                plt.errorbar(i, DDG, yerr=dDDG)
                plt.scatter(i, DDG)
            except KeyError:
                continue
        # TODO add axis labels
        _produce_plot(f"fe_delta_{title}")

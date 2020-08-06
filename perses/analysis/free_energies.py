import numpy as np
import seaborn as sns
from pymbar import BAR
import matplotlib.pyplot as plt
import seaborn
import pickle
from simtk.openmm import unit
import bz2
import json
import tqdm
from openmmtools.constants import kB
import random
import joblib
from typing import Optional


def _strip_outliers(w, max_value=1e4, n_devs=100):
    w = w[w.abs() < max_value]
    return w[(w - w.mean()).abs() < n_devs * w.std()]


def _get_works(df, run, project, GEN=None):
    works = df[df["RUN"] == run]

    if GEN:
        works = works[works["GEN"] == GEN]

    f = works[works["PROJ"] == project].forward_work
    r = works[works["PROJ"] == project].reverse_work
    return f, r


def free_energies(
    details_file_path: str,
    work_file_path: str,
    complex_project: str,
    solvent_project: str,
    temperature_kelvin: float = 300.0,
    n_bootstrap: int = 100,
    show_plots: bool = False,
    plot_file_format: str = "png",
    cache_dir: Optional[str] = None,
):
    r"""Compute free energies from a set of runs.

    Parameters
    ----------
    details_file_path : str
        Path to json file containing run metadata.
    work_file_path : str
        Path to bz2-compressed pickle file containing work values from simulation
    complex_project : str
        Project identifier (of the form "PROJXXXXX") of the FAH project for complex simulation
    solvent_project : str
        Project identifier (of the form "PROJXXXXX") of the FAH project for solvent simulation
    temperature_kelvin : float
    n_bootstrap : int, optional
        Number of bootstrap steps used for BAR free energy estimation
    show_plots : bool, optional
        If true, block to display plots interactively (using `plt.show()`).
        If false, save plots as images in current directory (for batch usage).
    plot_file_format : str, optional
        Image file format for saving plots. Accepts any file extension supported by `plt.savefig()`
    cache_dir : str, optional
        If given, local directory for caching BAR calculations. Results are cached by run, phase, and generation.
        If None, no caching is performed.
    """

    with bz2.BZ2File(work_file_path, "r") as infile:
        work = pickle.load(infile)

    with open(details_file_path, "r") as f:
        details = json.load(f)

    work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna()

    temperature = temperature_kelvin * unit.kelvin
    kT = kB * temperature

    projects = {"complex": complex_project, "solvent": solvent_project}

    def _produce_plot(name):
        if show_plots:
            plt.show()
        else:
            plt.savefig('.'.join([name, plot_file_format]))

    def _bootstrap_BAR(run, phase, gen_id):
        f_works, r_works = _get_works(work, RUN, projects[phase], GEN=f"GEN{gen_id}")
        f_works = _strip_outliers(f_works)
        r_works = _strip_outliers(r_works)
        fes = []
        errs = []

        if len(f_works) > 10 and len(r_works) > 10:
            for _ in range(n_bootstrap):
                f = random.choices(f_works, k=len(f_works))
                r = random.choices(r_works, k=len(r_works))
                fe, err = BAR(np.asarray(f), np.asarray(r))
                fes.append(fe)
                errs.append(err)

        return fes, errs, f_works, r_works

    if cache_dir is not None:
        # cache results in local directory
        memory = joblib.Memory(backend="local", location=cache_dir, verbose=0)
        bootstrap_BAR = memory.cache(_bootstrap_BAR)
    else:
        bootstrap_BAR = _bootstrap_BAR

    for d in tqdm.tqdm(details.values()):
        RUN = d["directory"]
        if show_plots:
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        for i, phase in enumerate(projects.keys()):
            if show_plots:
                axes[i].set_title(phase)

            all_forward = []
            all_reverse = []
            # There will be 6 gens for this project I think
            for gen_id in range(0, 7):
                fes, errs, f_works, r_works = bootstrap_BAR(RUN, phase, gen_id)
                d[f"{phase}_fes_GEN{gen_id}"] = fes
                d[f"{phase}_dfes_GEN{gen_id}"] = errs
                all_forward.extend(f_works)
                all_reverse.extend(r_works)
            #         print(all_forward)
            if len(all_forward) < 10:
                print(f"Cant calculate {RUN} {phase}")
                continue
            if len(all_reverse) < 10:
                print(f"Cant calculate {RUN} {phase}")
                continue
            if show_plots:
                sns.kdeplot(all_forward, shade=True, color="cornflowerblue", ax=axes[i])
                sns.rugplot(
                    all_forward,
                    ax=axes[i],
                    color="cornflowerblue",
                    alpha=0.5,
                    label=f"forward : N={len(f_works)}",
                )
                sns.rugplot(
                    all_forward,
                    ax=axes[i],
                    color="darkblue",
                    label=f"forward (gen0) : N={len(f_works)}",
                )
                sns.rugplot(
                    [-x for x in all_reverse],
                    ax=axes[i],
                    color="mediumvioletred",
                    label=f"reverse (gen0) : N={len(r_works)}",
                )
                sns.kdeplot(
                    [-x for x in all_reverse], shade=True, color="hotpink", ax=axes[i]
                )
                sns.rugplot(
                    [-x for x in all_reverse],
                    ax=axes[i],
                    color="hotpink",
                    alpha=0.5,
                    label=f"reverse : N={len(r_works)}",
                )

            if any([True for x in [all_reverse, all_forward] if len(x) < 20]):
                print(f"Cant calculate {RUN} {phase}")
            else:
                # TODO add bootstrapping here
                d[f"{phase}_fes"] = BAR(
                    np.asarray(all_forward), np.asarray(all_reverse)
                )
        #             d[f'n_{phase}'] = len(all_forward) + len(all_reverse)

        if show_plots:
            fig.suptitle(
                f"{RUN}: {d['protein'].split('_')[0]} {d['start']}-{d['end']}",
                fontsize=16,
            )
            fig.subplots_adjust(top=0.9, wspace=0.15)
            axes[0].legend()
            axes[1].legend()
            _produce_plot(f"{RUN}")

    ligand_result = {0: 0.0}
    ligand_result_uncertainty = {0: 0.0}

    for d in details.values():
        if "complex_fes" in d and "solvent_fes" in d:
            DDG = ((d["complex_fes"][0] - d["solvent_fes"][0]) * kT).value_in_unit(
                unit.kilocalories_per_mole
            )
            dDDG = (
                (d["solvent_fes"][1] ** 0.5 + d["complex_fes"][1] ** 0.5) ** 2 * kT
            ).value_in_unit(unit.kilocalories_per_mole)
            ligand_result[d["end"]] = DDG
            ligand_result_uncertainty[d["end"]] = DDG

    plt.hist(ligand_result.values(), bins=100)
    plt.xlabel("Relative free energy to ligand 0 / kcal/mol")
    _produce_plot("rel_fe_lig0_hist")

    ### this will be useful for looking at looking at shift in relative FEs over GENS

    for d in details.values():
        RUN = d["directory"]
        if "complex_fes" in d and "solvent_fes" in d:
            for i in range(0, 7):
                try:
                    DDG = (
                        (
                            np.mean(d[f"complex_fes_GEN{i}"])
                            - np.mean(d[f"solvent_fes_GEN{i}"])
                        )
                        * kT
                    ).value_in_unit(unit.kilocalories_per_mole)
                    dDDG = (
                        (
                            np.mean(d[f"complex_dfes_GEN{i}"]) ** 0.5
                            + np.mean(d[f"solvent_dfes_GEN{i}"]) ** 0.5
                        )
                        ** 2
                        * kT
                    ).value_in_unit(unit.kilocalories_per_mole)
                    plt.errorbar(i, DDG, yerr=dDDG)
                    plt.scatter(i, DDG)
                except:
                    continue
            _produce_plot(f"fe_delta_{RUN}")


if __name__ == "__main__":
    import fire

    fire.Fire(free_energies)


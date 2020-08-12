import numpy as np
import pandas as pd
import seaborn as sns
from pymbar import BAR
import matplotlib.pyplot as plt
import seaborn
import pickle
from simtk.openmm import unit
import bz2
import json
from tqdm.auto import tqdm
from openmmtools.constants import kB
import random
import joblib
import logging
from fah_analysis import *
import os
from typing import Optional


def _strip_outliers(w, max_value=1e4, n_devs=5):
    """Removes work values that are more than 5 (n_devs) standard deviations from the mean or larger than 1E4 (max_value).

    Parameters
    ----------
    w : list(float)
        List of works
    max_value : int, default=1E4
        Work values larger than this will be discarded
    n_devs : int, default=5
        Number of standard deviations from mean to be returned

    Returns
    -------
    list(float)
        Tidied list of work values

    """
    w = w[w.abs() < max_value]
    return w[(w - w.mean()).abs() < n_devs * w.std()]


def _get_works(df, run, project, GEN=None):
    """ Get set of work values from a dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Information generated from folding at home
    run : str
        run to collect data for (i.e. 'RUN0')
    project : str
        project to collect data for (i.e. 'PROJ13420')
    GEN : str, optional default=None
        if provided, will only return work values from a given run, (i.e. 'GEN0'), otherwise all generations are returned

    Returns
    -------
    list, list
        Returns lists for forwards and backwards works

    """
    works = df[df["RUN"] == run]

    if GEN:
        works = works[works["GEN"] == GEN]

    f = works[works["PROJ"] == project].forward_work
    r = works[works["PROJ"] == project].reverse_work
    return f, r


def free_energies(
    details_file_path: list,
    work_file_path: str,
    complex_project: str,
    solvent_project: str,
    temperature_kelvin: float = 300.0,
    n_bootstrap: int = 100,
    show_plots: bool = False,
    plot_file_format: str = "pdf",
    cache_dir: Optional[str] = None,
    min_num_work_values: int = 10,
):
    r"""Compute free energies from a set of runs.

    Parameters
    ----------
    details_file_path : list
        Path or list of paths to json file containing run metadata.
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
    min_num_work_values : int, optional
        Minimum number of forward and reverse work values for valid calculation
    """

    # load pandas dataframe from FAH
    work = pd.read_pickle(work_file_path)

    # load the json that contains the information as to what has been computed in each run
    if isinstance(details_file_path, str):
        details_file_path = [details_file_path]

    details = {}
    for path in details_file_path:
        with open(path, 'r') as f:
            new = json.load(f)
            details = {**details, **new}

    # remove any unuseable values
    with pd.option_context("mode.use_inf_as_na", True):
        work = work.dropna()

    temperature = temperature_kelvin * unit.kelvin
    kT = kB * temperature

    projects = {"complex": complex_project, "solvent": solvent_project}


    def _bootstrap_BAR(run, phase, gen_id, n_bootstrap):
        f_works, r_works = _get_works(work, RUN, projects[phase], GEN=f"GEN{gen_id}")
        f_works = _strip_outliers(f_works)
        r_works = _strip_outliers(r_works)
        fes = []
        errs = []

        if len(f_works) > 10 and len(r_works) > 10:
            for _ in range(n_bootstrap):
                f = random.choices(f_works.values, k=len(f_works))
                r = random.choices(r_works.values, k=len(r_works))
                fe, err = BAR(np.asarray(f), np.asarray(r))
                fes.append(fe)
                errs.append(err)

        # TODO - I think we can just return the mean of the fes and the stderrs, rather than passing n_bootstrap values around
        return fes, errs, f_works, r_works

    if cache_dir is not None:
        # cache results in local directory
        memory = joblib.Memory(backend="local", location=cache_dir, verbose=0)
        bootstrap_BAR = memory.cache(_bootstrap_BAR)
    else:
        bootstrap_BAR = _bootstrap_BAR

    def _max_gen(RUN):
        df = work[work["RUN"] == RUN]
        if df.empty:
            raise ValueError(f"No work values found for {RUN}")
        return df["GEN"].str[3:].astype(int).max()

    def _process_run(RUN):
        def _process_phase(i, phase):
            all_forward = []
            all_reverse = []
            for gen_id in range(_max_gen(RUN)):
                fes, errs, f_works, r_works = bootstrap_BAR(RUN, phase, gen_id)
                d[f"{phase}_fes_GEN{gen_id}"] = fes
                d[f"{phase}_dfes_GEN{gen_id}"] = errs
                all_forward.extend(f_works)
                all_reverse.extend(r_works)

            if len(all_forward) < min_num_work_values:
                raise ValueError(f"less than {min_num_work_values} forward work values")
            if len(all_reverse) < min_num_work_values:
                raise ValueError(f"less than {min_num_work_values} reverse work values")

            # TODO add bootstrapping here
            d[f"{phase}_fes"] = BAR(np.asarray(all_forward), np.asarray(all_reverse))

        for i, phase in enumerate(projects.keys()):
            try:
                _process_phase(i, phase)
            except ValueError as e:
                logging.warn(f"Can't calculate {RUN} {phase}: {e}")
                continue

        title = f"{RUN}: {d['protein'].split('_')[0]} {d['start']}-{d['end']}"
        plot_two_work_distribution(title=title)

    for d in tqdm(details.values()):
        RUN = d["directory"]
        try:
            _process_run(RUN)
        except ValueError as e:
            logging.warn(f"Can't calculate {RUN}: {e}")
            continue

    ligand_result = {0: 0.0}
    ligand_result_uncertainty = {0: 0.0}

    # TODO -- this assumes that everything is star-shaped, linked to ligand 0. If it's not, the values in ligand_result and ligand_result_uncertainty won't be correct.
    for d in details.values():
        if "complex_fes" in d and "solvent_fes" in d:
            DDG = ((d["complex_fes"][0] - d["solvent_fes"][0]) * kT).value_in_unit(
                unit.kilocalories_per_mole
            )
            dDDG = (
                (d["solvent_fes"][1] ** 2 + d["complex_fes"][1] ** 2) ** 0.5 * kT
            ).value_in_unit(unit.kilocalories_per_mole)
            ligand_result[d["end"]] = DDG
            ligand_result_uncertainty[d["end"]] = DDG

    _plot_relative_distribution(ligand_result.values())


if __name__ == "__main__":
    import fire

    fire.Fire(free_energies)

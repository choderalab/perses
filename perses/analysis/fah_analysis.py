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
from perses.analysis.fah_plotting import *
import os
from typing import Optional
from perses.analysis.resample import bootstrap_uncorrelated


_logger = logging.getLogger()
_logger.setLevel(logging.INFO)


def _strip_outliers(w, max_value=1e4, n_devs=5):
    """Removes work values that are more than 5 (n_devs) standard deviations from the mean or larger than 1E4 (max_value).

    Parameters
    ----------
    w : np.array
        array of works
    max_value : int, default=1E4
        Work values larger than this will be discarded
    n_devs : int, default=5
        Number of standard deviations from mean to be returned

    Returns
    -------
    list(float)
        Tidied list of work values

    """
    w = w[np.abs(w) < max_value]
    return np.asarray(w[np.abs((w - np.mean(w))) < n_devs * np.std(w)])


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


def _ci(values, ci=0.95):
    low_frac = (1.0 - ci) / 2.0
    high_frac = 1.0 - low_frac
    low, high = np.percentile(values, [low_frac, high_frac])
    return low, high


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
    _logger.info(f'Loading FAH output from {work_file_path}')
    work = pd.read_pickle(work_file_path)
    _logger.info(f'{work.size} switches loaded')

    # convert columns to numeric
    for c in [
        "forward_work",
        "reverse_work",
        "forward_final_potential",
        "reverse_final_potential",
    ]:
        work[c] = pd.to_numeric(work[c])

    # load the json that contains the information as to what has been computed in each run
    if isinstance(details_file_path, str):
        details_file_path = [details_file_path]

    _logger.info('Loading simulation parameters from...')
    details = {}
    for path in details_file_path:
        with open(path, 'r') as f:
            _logger.info(f'\t {path}')
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

        if len(f_works) < min_num_work_values:
            raise ValueError(
                f"less than {min_num_work_values} forward work values (got {len(f_works)})"
            )
        if len(r_works) < min_num_work_values:
            raise ValueError(
                f"less than {min_num_work_values} reverse work values (got {len(r_works)})"
            )

        fes = islice(samples_uncorrelated(f_works.values, r_works.values), n_bootstrap)

        return fes, f_works, r_works

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
                fes, f_works, r_works = bootstrap_BAR(f_works, r_works, n_bootstrap)
                low, high = _ci(fes)
                d[f"{phase}_fes_GEN{gen_id}"] = (np.mean(fes), low, high)
                d[f"{phase}_dfes_GEN{gen_id}"] = np.std(fes)
                all_forward.extend(f_works)
                all_reverse.extend(r_works)

            if len(all_forward) < min_num_work_values:
                raise ValueError(f"fewer than {min_num_work_values} forward work values")
            if len(all_reverse) < min_num_work_values:
                raise ValueError(f"fewer than {min_num_work_values} reverse work values")

            # TODO add bootstrapping here
            d[f"{phase}_fes"], _, _ = bootstrap_BAR(all_forward, all_reverse, n_bootstrap)

        for i, phase in enumerate(projects.keys()):
            try:
                _process_phase(i, phase)
            except ValueError as e:
                logging.warning(f"Can't calculate {RUN} {phase}: {e}")
                continue

            try:
                binding = np.asarray(d[f"solvent_fes"]) - np.asarray(d[f"complex_fes"])
                low_binding, high_binding = _ci(binding)
                d['binding_fe'] = (np.mean(binding), low_binding, high_binding)
                d['binding_dfe'] = np.std(binding)
                low_sol, high_sol = _ci(d[f"solvent_fes"])
                d[f"solvent_fes"] = (np.mean(d[f"solvent_fes"]), low_sol, high_sol)
                d[f"solvent_dfes"] = np.std(d[f"solvent_fes"])
                low_com, high_com = _ci(d[f"complex_fes"])
                d[f"complex_fes"] = (np.mean(d[f"complex_fes"]), low_com, high_com)
                d[f"complex_dfes"] = np.std(d[f"complex_fes"])

            except KeyError:
                continue

        # title = f"{RUN}: {d['protein'].split('_')[0]} {d['start']}-{d['end']}"
        # plot_two_work_distribution(title=title)

    for d in tqdm(details.values()):
        RUN = d["directory"]
        try:
            _process_run(RUN)
        except ValueError as e:
            logging.warning(f"Can't calculate {RUN}: {e}")
            continue

    #
    # # TODO I think this belongs somewhere else, but not sure where
    # ligand_result = {0: 0.0}
    # ligand_result_uncertainty = {0: 0.0}
    #
    # # TODO -- this assumes that everything is star-shaped, linked to ligand 0. If it's not, the values in ligand_result and ligand_result_uncertainty won't be correct.
    # for d in details.values():
    #     if "complex_fes" in d and "solvent_fes" in d:
    #         DDG = ((d["complex_fes"][0] - d["solvent_fes"][0]) * kT).value_in_unit(
    #             unit.kilocalories_per_mole
    #         )
    #         dDDG = (
    #             (d["solvent_fes"][1] ** 2 + d["complex_fes"][1] ** 2) ** 0.5 * kT
    #         ).value_in_unit(unit.kilocalories_per_mole)
    #         ligand_result[d["end"]] = DDG
    #         ligand_result_uncertainty[d["end"]] = DDG

    #plot_relative_distribution(ligand_result.values())

    return details


def store_json(contents, filename='analysed.json'):
    """ Allows jsons with numpy arrays to be stored

    Parameters
    ----------
    contents : dict
        Thing to save to json
    filename : string, default='analysed.json'
        filename or path for storage


    """
    import json

    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if 'json' not in filename:
        filename += '.json'
    with open(filename, 'w') as f:
        json.dump(contents, f, cls=NumpyArrayEncoder)
        _logger.info(f'json file saved at {filename}')


if __name__ == "__main__":
    import fire

    fire.Fire(free_energies)

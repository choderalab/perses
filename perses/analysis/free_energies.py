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


def _strip_outliers(w, n_devs=100):
    w = [x for x in w if np.abs(x) < 10 ** 4]
    mean = np.mean(w)
    std = np.std(w)
    good_w = [x for x in w if np.abs(x - mean) < n_devs * std]
    return np.asarray(good_w)


def _get_works(df, run, project, GEN=None):
    works = df[(df["RUN"] == run)]

    if GEN:
        works = works[works["GEN"] == GEN]

    f = works[works["PROJ"] == project].forward_work
    r = works[works["PROJ"] == project].reverse_work
    return f, r


def run(
    details_file_path,
    work_file_path,
    complex_project,
    solvent_project,
    temperature_kelvin=300.0,
    n_bootstrap=100,
    plotting=True,
):

    with bz2.BZ2File(work_file_path, "r") as infile:
        work = pickle.load(infile)

    with open(details_file_path, "r") as f:
        details = json.load(f)

    work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna()

    temperature = temperature_kelvin * unit.kelvin
    kT = kB * temperature

    projects = {"complex": complex_project, "solvent": solvent_project}

    for d in tqdm.tqdm(details.values()):
        RUN = d["directory"]
        if plotting:
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        for i, phase in enumerate(projects.keys()):
            if plotting:
                axes[i].set_title(phase)

            all_forward = []
            all_reverse = []
            # There will be 6 gens for this project I think
            for gen_id in range(0, 7):
                f_works, r_works = _get_works(
                    work, RUN, projects[phase], GEN=f"GEN{gen_id}"
                )
                f_works = _strip_outliers(f_works)
                r_works = _strip_outliers(r_works)
                d[f"{phase}_fes_GEN{gen_id}"] = []
                d[f"{phase}_dfes_GEN{gen_id}"] = []

                if len(f_works) > 10 and len(r_works) > 10:
                    for _ in range(n_bootstrap):
                        f = random.choices(f_works, k=len(f_works))
                        r = random.choices(r_works, k=len(r_works))
                        fe, err = BAR(np.asarray(f), np.asarray(r))
                        d[f"{phase}_fes_GEN{gen_id}"].append(fe)
                        d[f"{phase}_dfes_GEN{gen_id}"].append(err)

                all_forward.extend(f_works)
                all_reverse.extend(r_works)
            #         print(all_forward)
            if len(all_forward) < 10:
                print(f"Cant calculate {RUN} {phase}")
                continue
            if len(all_reverse) < 10:
                print(f"Cant calculate {RUN} {phase}")
                continue
            if plotting:
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

        if plotting:
            fig.suptitle(
                f"{RUN}: {d['protein'].split('_')[0]} {d['start']}-{d['end']}",
                fontsize=16,
            )
            fig.subplots_adjust(top=0.9, wspace=0.15)
            axes[0].legend()
            axes[1].legend()
            fig.savefig(f"{RUN}.png")

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
    plt.show()

    ### this will be useful for looking at looking at shift in relative FEs over GENS

    for d in details.values():
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
            plt.show()


if __name__ == "__main__":
    import fire

    fire.Fire(run)


def test_run():
    run(
        details_file_path="./2020-07-24.json",
        work_file_path="../data/work-13420.pkl.bz2",
        complex_project="PROJ13420",
        solvent_project="PROJ13421",
    )

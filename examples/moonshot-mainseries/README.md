# Relative free energy calculations of main medicinal chemistry milestone compounds for COVID Mooonshot

## Prerequisites

In addition to perses (and OpenEye Toolkit dependencies), to expand protonation states with the `expand_protonation_states: true` argument in `setup.yaml`, we require the Schrodinger Suite to be installed and `$SCHRODINGER` to be configured so that `$SCHRODINGER/epik` can be found.

## Prepare the receptor structures

First, download and unpack Mpro source structures from Fragalysis using [permalink](https://fragalysis.diamond.ac.uk/viewer/react/download/tag/91448cc6-45c8-4707-94c3-3c59fc45c6da) into `structures/` subdirectory.

Generate a Spruce loop database from the retrieved structures:
```bash
$OPENEYE_APPLICATIONS/bin/loopdb_builder -in structures/aligned/ -source_name fragalysis -prefix mainseries
```
Note that this requires having [downloaded and installed the OpenEye Applications](https://www.eyesopen.com/customer-software-download) into the `$OPENEYE_APPLICATIONS` directory.

Prep receptors to generate fully protonated protein PDB files containing X-ray waters along with ligand SDF files:
```bash
python 00-prep-receptor.py
```
This will prepare the receptors listed in the `setup.yaml` file.

If desired, check poses of reference ligands are stable with short dynamics runs.
For example:
```bash
export FRAGALYSIS_ID="P0744_0A"
export PROTONATION_STATE="His41(0)-Cys145(0)-His163(0)"
simulate.py --receptor="Mpro-${FRAGALYSIS_ID}_bound-${PROTONATION_STATE}-protein.pdb" --ligand="Mpro-${FRAGALYSIS_ID}_bound-${PROTONATION_STATE}-ligand.pdb" --nsteps=250000 --selection="not water" --initial=initial.pdb --final=final.pdb --xtctraj=trajectory.xtc 
```

## Generate docked poses

Generate docked poses for the molecule sets in `molecules/`:
```bash
python 02-generate-poses.py
```

## Run the free energy calculations

To run the calculations in the `perses/` subdirectory, we provide a script for the LSF batch queue manager:
```bash
cd perses/
bsub < submit-all.sh
```
The LSF script uses `run-perses.py` to launch the appropriate free energy calculation as directed by the `setup.yaml` in the base directory.

To figure out how many calculations will be run (e.g. to identify how many jobs in the job array should be run), you can use 
```bash
python run-perses.py --count
```

## Analyze the free energy calculations

Run the analysis script to analyze the neutral charge state:
```bash
python analyze-benchmark-pKa.py --basepath "step1-His41(0)-Cys145(0)-His163(0)-*" --docked "../docked/step1-x2646_0A-dimer-His41(0)-Cys145(0)-His163(0).sdf" --expdata "../molecules/step1.csv"
python analyze-benchmark-pKa.py --basepath "step2-His41(0)-Cys145(0)-His163(0)-*" --docked "../docked/step2-x10959_0A-dimer-His41(0)-Cys145(0)-His163(0).sdf" --expdata "../molecules/step2.csv"
python analyze-benchmark-pKa.py --basepath "step3-His41(0)-Cys145(0)-His163(0)-*" --docked "../docked/step3-x11612_0A-dimer-His41(0)-Cys145(0)-His163(0).sdf" --expdata "../molecules/step3.csv"
python analyze-benchmark-pKa.py --basepath "step4-His41(0)-Cys145(0)-His163(0)-*" --docked "../docked/step4-P0744_0A-dimer-His41(0)-Cys145(0)-His163(0).sdf" --expdata "../molecules/step4.csv"
```

# Things to fix/improve

- [ ] The docking output should generate microstates with unique microstate names, rather than microstaes with parent compound names; this would simplify our workflow.
- [ ] Standardize naming convention (`parent_molecule`, `compound`)
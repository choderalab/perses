# Run neq switching for a NTRK1:entrectinib mutation (G613V in NTRK1)
This script shows how to perform a non-equilibrium switching simulation for a protein-ligand apo and complex phases,
by mutating the GLY613 residue in NTRK1 to a valine (VAL), computing and storing the resulting work for the forward
and reverse trajectory simulations for both phases.

Simulation parameters are hardcoded in the script file, modify as needed. 

Run the example by using:
```bash
python run_neq_distrib_flattened.py
```
By default, running the script stores the output in numpy format in the `output/` directory.
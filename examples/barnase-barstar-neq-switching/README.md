# Run neq switching for a barnase:barstar mutation (T42A in barstar)
This script shows how to perform a non-equilibrium switching simulation for the barnase-barstar protein-protein complex,
by mutating the second residue in barstar to an alanine (ALA), computing and storing the resulting work for the forward
and reverse trajectory simulations.

Simulation parameters are hardcoded in the script file, modify as needed. 

Run the example by using:
```bash
python run_neq_distrib_flattened.py
```
By default, running the script stores the output in numpy format in the `output/` directory.
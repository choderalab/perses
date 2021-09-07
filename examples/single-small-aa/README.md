# Run neq switching for a protein mutation (ALA->ASP dipeptide)
This script shows how to perform a non-equilibrium switching simulation for the mutation of a capped alanine (ALA) to a
capped aspartic acid (ASP), computing and storing the resulting work for the forward and reverse simulations.

Simulation parameters are hardcoded in the script file, modify as needed. 

Run the example by using:
```bash
python run_neq_distrib_flattened.py
```
Where the output is stored in numpy format in the `output/` directory, by default.

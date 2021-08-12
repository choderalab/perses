# Run neq switching for a protein mutation (ALA->ASP dipeptide)
Run the `run_neq_distrib_flattened.py` script, according to the following parameters:
```bash
usage: run_neq_distrib_flattened.py [-h] [--outdir OUTDIR] [--platform PLATFORM] [--eq_save_period EQ_SAVE_PERIOD] [--neq_save_period NEQ_SAVE_PERIOD] time_step eq_steps neq_steps

run perses protein mutation on capped amino acid

positional arguments:
  time_step             time step in femtoseconds
  eq_steps              Number of steps for equilibrium simulation
  neq_steps             Number of steps for non-equilibrium simulation

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR       path to output directory. Defaults to output/
  --platform PLATFORM   compute platform: Reference, CPU, CUDA or OpenCL. Defaults to OpenCL.
  --eq_save_period EQ_SAVE_PERIOD
                        Save period for equlibrium simulation, in steps. Defaults to 1000.
  --neq_save_period NEQ_SAVE_PERIOD
                        Save period for non-equlibrium simulation, in steps. Defaults to 1000.
```
For example, if you want to run the simulation with a time step of `4 fs`, `10` steps for equilibration, and `100` steps
for non-equilibrium simulation, all while saving every step in the equilibration part, and every `10` steps in the 
non-equilibrium part, you would run something like:
```bash
python run_neq_distrib_flattened.py 4.0 10 100 --platform CPU --eq_save_period 1 --neq_save_period 10 --outdir output/
```
Where the output is saved in numpy format in the `output/` directory.

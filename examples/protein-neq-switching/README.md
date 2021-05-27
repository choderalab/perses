# Run neq switching for a protein mutation (ALA->ASP dipeptide)
1. First generate a pickled htf. 
An example htf is present in `2/`. The code to generate one:
```python
import pickle
import os
from perses.app.relative_point_mutation_setup import PointMutationExecutor
from simtk import unit

solvent_delivery = PointMutationExecutor("ala_vacuum.pdb",
                        '1',
                        '2',
                        'ASP',
                        ionic_strength=0.15*unit.molar,
                        flatten_torsions=True,
                        flatten_exceptions=True, 
                        conduct_endstate_validation=False
                       )
apo_htf = solvent_delivery.get_apo_htf()
with open(os.path.join(outdir, f"2_solvent.pickle"), "wb") as f:
    pickle.dump(apo_htf, f)

```
2. Run neq switching: (copied from a bash script)
```bash
old_aa="ala"
new_aa="asp"
resid="2"
outdir="2/"
phase='solvent'
eq_length=1 # in ns to run eq
neq_length=1 # in ns to run neq

python run_neq_distrib_flattened.py $outdir $phase "$((${LSB_JOBINDEX}-1))" $eq_length $neq_length
```

Note: `run_neq_distrib_flattened.py` will search `$outdir` for a pickled htf with the name `{outdir number}_{phase}.pickle` (here, it's `2_solvent.pickle`)

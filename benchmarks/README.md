# Perses benchmarks

This subdirectory exposes a CLI tool for running automated benchmarks from
[OpenFF's protein ligand benchmark dataset](https://github.com/openforcefield/protein-ligand-benchmark) using perses.

## Running benchmarks

Assuming you have a clone of the perses code repository, and you are standing in the `benchmarks` subdirectory
(where this file lives). Then the benchmarks can be run using the following command syntax:
```bash
python run_benchmarks.py --target [protein-name] --edge [edge-index]
```

For example, for running the seventh edge (zero-based, according to [plbenchmark data](https://github.com/openforcefield/protein-ligand-benchmark) )
for `tyk2` protein, you would run:
```bash
# Set up and run edge 6
python run_benchmarks.py --target tyk2 --edge 6
```
Should the calculation for an edge fail, you can simply re-run the same command-line and the calculation will resume:
```bash
# Resume failed edge 6
python run_benchmarks.py --target tyk2 --edge 6
```

### Running benchmark with local files
The script accepts the `--local` flag to run benchmarks using the files already in the same directory as the 
`run_benchmarks.py` script. It requires the following files and names in the same directory: `template.yaml`, 
`ligands.sdf` and `target.pdb`. And you can run using local files with:
```bash
python run_benchmarks.py --target tyk2 --edge 6 --local
```

### Specifying a git revision/tag
Since the upstream repository is prone to changes. We can specify a revision or a tag using the `--revision` argument.
For example if we want to use the 0.2.0 tag release, we can run with:
```bash
python run_benchmarks.py --target tyk2 --edge 6 --revision=0.2.0
```
By default, it uses `0.2.1` as the revision to date. You can also specify branches of the repository such as `main`.

For more information on how to use the tool, you can run `python run_benchmarks.py -h`.

## Analyzing benchmarks

To analyze the simulations a script called `benchmark_analysis.py` is used as follows:
```bash
python benchmark_analysis.py --target [protein-name]
```

For example, for tyk2 results:
```bash
python benchmark_analysis.py --target tyk2
```
This will generate an output CSV file for [`arsenic`](https://github.com/openforcefield/arsenic) and corresponding 
absolute and relative free energy plots as PNG files produced according to best practices.)

For more information on how to use the cli analysis tool use `python benchmark_analysis.py -h`.

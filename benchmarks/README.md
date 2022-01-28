# Perses benchmarks

This subdirectory exposes a CLI tool for running automated benchmarks from
[OpenFF's protein ligand benchmark dataset](OpenFF's protein ligand benchmark dataset) using perses.

## Running benchmarks

Assuming you have a clone of the perses code repository and you are standing in the `benchmarks` subdirectory
(where this file lives). Then the benchmarks can be run using the following command syntax:

```bash
python run_benchmarks.py --target [protein-name] --edge [edge-index]
```

For example, for running the seventh edge (zero-based, according to [plbenchmark data](https://github.com/openforcefield/protein-ligand-benchmark) )
for `tyk2` protein, you would run:

```bash
python run_benchmarks.py --target tyk2 --edge 6
```

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

For more information on how to use the cli analysis tool use `python benchmark_analysis.py -h`.
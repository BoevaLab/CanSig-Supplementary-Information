# Benchmark


## Preparing the environment

Remember to run

```
$ git submodule update --init --recursive --remote
```
in the main directory to get a copy of CanSig checked out at the right version.

Then, create a new environment and install it:
```
$ cd benchmark
$ pip install scib
$ pip install -e CanSig
```

## Getting the data

Download the data from

TODO: Make the ZIP version of the dataset available.

Now unzip the dataset:
```
$ cd data
$ unzip data.zip
```
A new file `data.h5ad` should appear.

## Running the benchmark

We run individual methods using a Python script

```
$ python scripts/run.py data/data.h5ad --method combat
```
which produces a JSON with result for a particular method (and the choice of hyperparameters).

To see all hyperparameter combinations, see the `scripts/experiment.sh`. It can be run (inside the environment) using

```
source scripts/experiment.sh
```

## Disclaimer

This code was **not** carefully designed and tested. Hence, there may be bugs inside – we advise treating these results with caution.


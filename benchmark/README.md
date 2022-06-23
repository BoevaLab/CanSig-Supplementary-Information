# Benchmark

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

Now unzip the dataset:
```
$ cd data
$ unzip data.zip
```
A new file `data.h5ad` should appear.

### Caution

This code was **not** carefully designed and tested. Hence, it is likely that there are bugs inside – we advise treating the results with caution.


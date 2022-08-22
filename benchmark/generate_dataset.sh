#!/bin/bash

#SBATCH --job-name=R
#SBATCH --output=generate_dataset_py.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

pip show anndata
python ~/scRNA_shared_signatures/benchmark/generate_datasets.py
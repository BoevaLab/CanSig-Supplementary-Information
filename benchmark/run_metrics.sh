#!/bin/sh

LAUNCHER="slurm_cpu"
DATA_PATH="/cluster/work/boeva/scRNAdata/benchmark/real_data"
RESULTS_PATH="/cluster/work/boeva/scRNAdata/benchmark/real_data_results"

python run_metrics.py hydra/launcher=$LAUNCHER +method=harmony,scanorama,mnn,combat,scvi,dhaka \
++data.data_path=$DATA_PATH \
++results_path=$RESULTS_PATH \
hydra/launcher=submitit_local \
--multirun
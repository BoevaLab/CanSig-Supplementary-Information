#!/bin/sh

LAUNCHER="slurm"
DATA_PATH="/cluster/work/boeva/scRNAdata/benchmark/real_data"
RESULTS_PATH="/cluster/work/boeva/scRNAdata/benchmark/real_data_results"

python run_integration.py hydra/launcher=$LAUNCHER \
+model=harmony,scanorama,mnn,combat,liger \
+data.data_path=$DATA_PATH \
++results_path=$RESULTS_PATH --multirun
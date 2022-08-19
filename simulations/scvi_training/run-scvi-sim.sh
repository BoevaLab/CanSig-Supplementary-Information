#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=scvi.out.txt
#

source ~/Documents/home_dir/gpu_env/bin/activate
python run_scVI.py
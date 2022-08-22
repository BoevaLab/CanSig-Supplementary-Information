
source ~/scRNA/bin/activate

python main.py hydra/launcher=submitit_slurm +model=harmony model.theta=2.0,2.25,2.5,2.75,3.0 model.lamb=1.0,1.25,1.5,1.75,2.0 --multirun &
python main.py hydra/launcher=submitit_slurm +model=scanorama model.sigma=12.5,15.,17.5 model.alpha=0.05,0.1,0.15  --multirun &


source ~/gpu_env/bin/activate
python main.py hydra/launcher=submitit_slurm +model=cansig model.n_latent=2,4,6,8 model.n_latent_cnv=2,5,10 model.n_latent_batch_effect=2,5,10 --multirun &
python main.py hydra/launcher=submitit_slurm +model=scvi model.n_latent=2,4,6,8 model.n_hidden=128,256 model.n_layers=1,2 --multirun &
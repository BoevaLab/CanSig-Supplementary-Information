

python main.py hydra/launcher=submitit_slurm +model=harmony model.theta=2.0,2.25,2.5,2.75,3.0 model.lamb=1.0,1.25,1.5,1.75,2.0 --multirun &
python main.py hydra/launcher=submitit_slurm +model=scanorama model.sigma=12.5,15.,17.5 model.alpha=0.05,0.1,0.15  --multirun &
python main.py hydra/launcher=submitit_slurm +model=bbknn model.neighbors_within_batch=2,3,4,5 model.n_top_genes=1000,2000,3000 --multirun &
python main.py hydra/launcher=submitit_slurm +model=mnn model.k=10,15,20,25,30 model.sigma=0.75,1.,1.25  --multirun &
python main.py hydra/launcher=submitit_slurm +model=combat model.n_top_genes=1000,2000,3000 model.cell_cycle=True,False model.log_counts=True,False  --multirun &
python main.py hydra/launcher=submitit_slurm +model=desc model.res=0.2,0.5,0.8,1. model.learning_rate=150,500,750 model.tol=0.001,0.005,0.01  --multirun &


source ~/gpu_env/bin/activate
python main.py hydra/launcher=submitit_slurm +model=cansig model.n_latent=2,4,6,8 model.n_latent_cnv=4,5,6 model.n_latent_batch_effect=4,5,6 model.n_layers=1,2 --multirun &
python main.py hydra/launcher=submitit_slurm +model=scvi model.n_latent=2,4,6,8 model.n_hidden=128,256 model.n_layers=1,2 --multirun &
"""This script is specific to the dataset we used from Pelka et al. 
The original adata we use is the version preprocessed with CanSig's preprocessing module"""
import scanpy as sc
import pathlib
import scvi

######### INSERT PATH TO ORIGINAL DATA ############
DATAPATH = pathlib.Path("/insert/path/here")

adata = sc.read_h5ad(DATAPATH / "non_malignant.h5ad")

adata = adata[
    adata.obs.celltype.isin(["TCD4", "TCD8", "Tgd", "Macro", "Plasma"])
].copy()

scvi.model.SCVI.setup_anndata(adata, batch_key="sample_id")

model = scvi.model.SCVI(adata, n_hidden=64, n_layers=2, n_latent=12)
model.train(early_stopping=True, max_epochs=300)
model.save("../scvi-model/")

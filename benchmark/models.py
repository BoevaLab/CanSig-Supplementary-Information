import warnings
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple, List, Optional, Union

import bbknn
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import scvi
from anndata import AnnData
from cansig.integration.model import CanSig
from omegaconf import MISSING

from benchmark.utils import split_batches


@dataclass
class ModelConfig:
    name: str = MISSING
    gpu: bool = False
    malignant_only: bool = True
    batch_key: str = "sample_id"
    latent_key: str = "latent"


def run_model(adata: AnnData, cfg) -> Tuple[AnnData, float]:
    start = timer()
    if cfg.name == "bbknn":
        adata = run_bbknn(adata, config=cfg)
    elif cfg.name == "scvi":
        adata = run_scvi(adata, config=cfg)
    elif cfg.name == "scanorama":
        adata = run_scanorama(adata, config=cfg)
    elif cfg.name == "harmony":
        adata = run_harmony(adata, config=cfg)
    elif cfg.name == "cansig":
        adata = run_cansig(adata, config=cfg)
    elif cfg.name == "nmm":
        adata = run_mnn(adata, config=cfg)
    elif cfg.name == "combat":
        adata = run_combat(adata, config=cfg)
    elif cfg.name == "desc":
        adata = run_desc(adata, config=cfg)
    else:
        raise NotImplementedError(f"{cfg.name} is not implemented.")
    run_time = timer() - start
    return adata, run_time


@dataclass
class BBKNNConfig(ModelConfig):
    name: str = "bbknn"
    neighbors_within_batch: int = 3


def run_bbknn(adata: AnnData, config: BBKNNConfig) -> AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    bbknn.bbknn(
        adata,
        batch_key=config.batch_key,
        neighbors_within_batch=config.neighbors_within_batch,
    )

    return adata


@dataclass
class SCVIConfig(ModelConfig):
    name: str = "scvi"
    gpu: bool = True
    n_latent: int = 4
    n_hidden: int = 128
    n_layers: int = 1
    n_top_genes: int = 2000
    max_epochs: int = 400


def run_scvi(adata: AnnData, config: SCVIConfig) -> AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=config.n_top_genes)
    bdata = adata[:, adata.var["highly_variable"]].copy()

    scvi.model.SCVI.setup_anndata(bdata, layer="counts", batch_key=config.batch_key)
    model = scvi.model.SCVI(
        bdata,
        n_latent=config.n_latent,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
    )
    model.train(
        max_epochs=config.max_epochs,
        # TODO: add this to cansig!
        train_size=1.0,
        plan_kwargs={"n_epochs_kl_warmup": config.max_epochs},
    )
    adata.obsm[config.latent_key] = model.get_latent_representation()
    return adata


@dataclass
class ScanoramaConfig(ModelConfig):
    name: str = "scanorama"
    knn: int = 20
    n_top_genes: int = 2000
    sigma: float = 15.0
    approx: bool = True
    alpha: float = 0.1


def run_scanorama(adata: AnnData, config: ScanoramaConfig) -> AnnData:
    # scanorama requires that cells from the same batch must
    # be contiguously stored in adata
    idx = np.argsort(adata.obs[config.batch_key])
    adata = adata[idx, :].copy()
    sc.pp.recipe_zheng17(adata, n_top_genes=config.n_top_genes)
    sc.tl.pca(adata)
    sce.pp.scanorama_integrate(
        adata,
        config.batch_key,
        adjusted_basis=config.latent_key,
        knn=config.knn,
        sigma=config.sigma,
        approx=config.approx,
        alpha=config.alpha,
    )
    return adata


@dataclass
class HarmonyConfig(ModelConfig):
    name: str = "harmony"
    n_top_genes: int = 2000
    max_iter_harmony: int = 100
    max_iter_kmeans: int = 100
    theta: float = 2.0
    lamb: float = 1.0
    epsilon_cluster: float = 1e-5
    epsilon_harmony: float = 1e-4
    random_state: int = 0


def run_harmony(adata: AnnData, config: HarmonyConfig) -> AnnData:
    sc.pp.recipe_zheng17(adata, n_top_genes=config.n_top_genes)
    sc.tl.pca(adata)
    sce.pp.harmony_integrate(
        adata,
        config.batch_key,
        theta=config.theta,
        lamb=config.lamb,
        adjusted_basis=config.latent_key,
        max_iter_harmony=config.max_iter_harmony,
        max_iter_kmeans=config.max_iter_kmeans,
        epsilon_cluster=config.epsilon_cluster,
        epsilon_harmony=config.epsilon_harmony,
        random_state=config.random_state,
    )

    return adata


@dataclass
class CanSigConfig(ModelConfig):
    name: str = "cansig"
    gpu: bool = True
    malignant_only: bool = False
    n_latent: int = 4
    n_layers: int = 1
    n_hidden: int = 128
    n_latent_batch_effect: int = 5
    n_latent_cnv: int = 10
    n_top_genes: int = 2000
    max_epochs: int = 400
    cnv_max_epochs: int = 400
    batch_effect_max_epochs: int = 400
    beta: float = 1.0
    batch_effect_beta: float = 1.0
    covariates: Optional[List] = field(
        default_factory=lambda: ["log_counts", "pct_counts_mt", "S_score", "G2M_score"]
    )
    annealing: str = "linear"
    malignant_key: str = "malignant_key"
    malignant_cat: str = "malignant"
    non_malignant_cat: str = "non-malignant"
    subclonal_key: str = "subclonal"
    celltype_key: str = "program"


def run_cansig(adata: AnnData, config: CanSigConfig) -> AnnData:
    bdata = CanSig.preprocessing(
        adata.copy(),
        n_highly_variable_genes=config.n_top_genes,
        malignant_key=config.malignant_key,
        malignant_cat=config.malignant_cat,
    )
    CanSig.setup_anndata(
        bdata,
        celltype_key=config.celltype_key,
        malignant_key=config.malignant_key,
        malignant_cat=config.malignant_cat,
        non_malignant_cat=config.non_malignant_cat,
        continuous_covariate_keys=config.covariates,
        layer="counts",
    )
    model = CanSig(
        bdata,
        n_latent=config.n_latent,
        n_layers=config.n_layers,
        n_hidden=config.n_hidden,
        n_latent_cnv=config.n_latent_cnv,
        n_latent_batch_effect=config.n_latent_batch_effect,
        sample_id_key=config.batch_key,
        subclonal_key=config.subclonal_key,
    )

    model.train(
        max_epochs=config.max_epochs,
        cnv_max_epochs=config.cnv_max_epochs,
        batch_effect_max_epochs=config.batch_effect_max_epochs,
        train_size=1.0,
        plan_kwargs={
            "n_epochs_kl_warmup": config.max_epochs,
            "beta": config.beta,
            "annealing": config.annealing,
        },
        batch_effect_plan_kwargs={"beta": config.batch_effect_beta},
    )

    save_model_history(model)

    save_latent_spaces(model, adata)

    idx = model.get_index(malignant_cells=True)
    adata = adata[idx, :].copy()
    adata.obsm[config.latent_key] = model.get_latent_representation()

    return adata


@dataclass
class MNNConfig(ModelConfig):
    name: str = "nmm"
    k: int = 20
    sigma: float = 1.
    n_top_genes: int = 2000


def run_mnn(adata: AnnData, config: MNNConfig) -> AnnData:
    split = split_batches(adata, config.batch_key)

    bdata = adata.copy()
    sc.pp.normalize_total(bdata, target_sum=1e4)
    sc.pp.log1p(bdata)
    sc.pp.highly_variable_genes(bdata, n_top_genes=config.n_top_genes)
    hvg = bdata.var.index[bdata.var["highly_variable"]].tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrected, _, _ = sce.pp.mnn_correct(*split, var_subset=hvg)
    corrected = corrected[0].concatenate(corrected[1:])

    corrected.obsm[config.latent_key] = corrected.X

    return corrected


@dataclass
class CombatConfig(ModelConfig):
    name: str = "combat"
    cell_cycle: bool = False
    n_top_genes: int = 2000
    log_counts: bool = False

def run_combat(adata: AnnData, config: CombatConfig) -> AnnData:
    covariates = []
    if config.cell_cycle:
        covariates += ["G2M_score", "S_score"]

    if config.log_counts:
        covariates += ["log_counts"]

    covariates = covariates or None

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=config.n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    X = sc.pp.combat(adata, config.batch_key, covariates=covariates,
                     inplace=False)
    adata.obsm[config.latent_key] = X
    return adata


@dataclass
class DescConfig(ModelConfig):
    name: str = "desc"
    gpu: bool = False  # TODO: add GPU acceleration
    res: float = 0.8
    n_top_genes: int = 2000
    n_neighbors: int = 10
    batch_size: int = 256
    tol: float = 0.005
    learning_rate: float = 500
    save_dir: Union[str, Path] = "."


def run_desc(adata: AnnData, config: DescConfig) -> AnnData:
    import desc
    # Preprocessing and parameters taken from https://github.com/eleozzr/desc/issues/28.
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=config.n_top_genes, inplace=True)
    sc.pp.scale(adata, zero_center=True, max_value=6)
    adata = desc.scale_bygroup(adata, groupby=config.batch_key, max_value=6)
    adata_out = desc.train(adata,
                           dims=[adata.shape[1], 128, 32],  # or set 256
                           tol=config.tol,
                           # suggest 0.005 when the dataset less than 5000
                           n_neighbors=config.n_neighbors,
                           batch_size=config.batch_size,
                           louvain_resolution=config.res,
                           save_dir=config.save_dir,
                           do_tsne=False,
                           use_GPU=config.gpu,
                           num_Cores=8,
                           save_encoder_weights=False,
                           save_encoder_step=2,
                           use_ae_weights=False,
                           do_umap=False,
                           num_Cores_tsne=4,
                           learning_rate=config.learning_rate)

    adata_out.obsm[config.latent_key] = adata_out.obsm["X_Embeded_z" + str(config.res)]

    return adata_out


def save_model_history(model: CanSig, name: str = ""):
    modules = {
        "combined": model.module,
        "batch_effect": model.module_batch_effect,
        "cnv": model.module_cnv,
    }

    for key, module in modules.items():
        df = pd.concat([df for df in module.history.values()], axis=1)
        df.to_csv(f"{key}_{name}.csv")


def save_latent_spaces(model: CanSig, adata: AnnData, name: str = ""):
    latent = model.get_batch_effect_latent_representation()
    idx = model.get_index(malignant_cells=False)
    df = pd.DataFrame(latent, index=adata.obs_names[idx])
    df.to_csv(f"{name}_batch_effect_latent.csv")

    latent = model.get_cnv_latent_representation()
    idx = model.get_index(malignant_cells=True)
    df = pd.DataFrame(latent, index=adata.obs_names[idx])
    df.to_csv(f"{name}_cnv_latent.csv")

from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Tuple, List, Optional

import bbknn
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import scvi
from anndata import AnnData
from cansig.integration.model import CanSig
from omegaconf import MISSING


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
        adata = run_scvi(adata, cfg)
    elif cfg.name == "scanorama":
        adata = run_scanorama(adata, config=cfg)
    elif cfg.name == "harmony":
        adata = run_harmony(adata, config=cfg)
    elif cfg.name == "cansig":
        adata = run_cansig(adata, config=cfg)
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
    sc.tl.pca(adata)
    sc.pp.scale(adata)
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


def run_cansig(adata: AnnData, config: CanSigConfig):
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
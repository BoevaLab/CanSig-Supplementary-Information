import logging
import pathlib as pl
from dataclasses import dataclass, field, MISSING
from typing import Dict, Any, Optional

import anndata
import cansig.cluster.api as cluster  # pytype: disable=import-error
import cansig.filesys as fs  # pytype: disable=import-error
import cansig.gsea as gsea  # pytype: disable=import-error
import hydra
import numpy as np
import scanpy as sc
from cansig.filesys import get_directory_name
from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import OmegaConf

_LOGGER = logging.getLogger(__name__)


def hydra_run_sweep():
    RUN_SWEEP_DEFAULTS = {
        "run": {"dir": "${results_path}/${data.cancer}/${run_dir:}"},
        "sweep": {
            "dir": "${results_path}",
            "subdir": "${data.cancer}/${run_dir:}",
        }
    }
    return RUN_SWEEP_DEFAULTS


def get_partition(gpu: bool) -> str:
    if gpu:
        return "gpu"
    return "compute"


def get_gres(gpu: bool) -> Optional[str]:
    if gpu:
        return "gpu:rtx2080ti:1"
    return None


@dataclass
class ModelConfig:
    name: str = MISSING
    gpu: bool = False
    malignant_only: bool = True
    batch_key: str = "sample_id"
    latent_key: str = "latent"
    n_top_genes: int = 2000


@dataclass
class SCVIConfig(ModelConfig):
    name: str = "scvi"
    gpu: bool = True
    n_latent: int = 10
    n_hidden: int = 128
    n_layers: int = 1
    max_epochs: int = 400
    cell_cycle: bool = False
    log_counts: bool = False
    pct_counts_mt: bool = False


@dataclass
class DataConfig:
    cancer: str = "npc"  # TODO: make this optinal.
    base_dir: str = "/cluster/work/boeva/scRNAdata/preprocessed"
    data_path: str = field(
        default_factory=lambda: "${data_path:${data.base_dir}, ${data.cancer}}")
    malignant_key: str = "malignant_key"
    malignant_cat: str = "malignant"


@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: SCVIConfig = SCVIConfig()
    results_path: str = "/cluster/work/boeva/scRNAdata/scvi_results"
    hydra: Dict[str, Any] = field(default_factory=hydra_run_sweep)


@dataclass
class Slurm(SlurmQueueConf):
    mem_gb: int = 16
    timeout_min: int = 720
    partition: str = field(default_factory=lambda: "${get_partition:${model.gpu}}")
    gres: Optional[str] = field(default_factory=lambda: "${get_gres:${model.gpu}}")


def run_scvi(adata: anndata.AnnData, config: SCVIConfig) -> anndata.AnnData:
    import scvi

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=config.n_top_genes)
    bdata = adata[:, adata.var["highly_variable"]].copy()

    covariates = []
    if config.cell_cycle:
        covariates += ["S_score", "G2M_score"]

    if config.log_counts:
        covariates += ["log_counts"]

    if config.pct_counts_mt:
        covariates += ["pct_counts_mt"]

    covariates = covariates or None

    scvi.model.SCVI.setup_anndata(bdata, layer="counts", batch_key=config.batch_key,
                                  continuous_covariate_keys=covariates)
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


def data_path(base_dir: str, cancer: str) -> str:
    data_path = pl.Path(base_dir).joinpath(cancer).joinpath("_LAST")
    return str(data_path)


def read_anndata(data_config: DataConfig) -> anndata.AnnData:
    data_path = data_config.data_path
    _LOGGER.info(f"Loading adata from {data_path}.")
    adata = anndata.read_h5ad(data_path)
    adata.X = adata.X.astype(np.float32)
    adata.layers["counts"] = adata.X.copy()
    return adata


OmegaConf.register_new_resolver("run_dir", get_directory_name)
OmegaConf.register_new_resolver("get_gres", get_gres)
OmegaConf.register_new_resolver("get_partition", get_partition)
OmegaConf.register_new_resolver("data_path", data_path)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="hydra/launcher", name="slurm", node=Slurm, provider="submitit_launcher")


@hydra.main(config_name="config", config_path=None, version_base="1.1")
def main(cfg: Config) -> None:
    try:
        adata = read_anndata(data_config=cfg.data)
        n_clusters = sum(adata.obs.columns.str.endswith("_GT"))
        _LOGGER.info(f"Found {n_clusters} ground truth signatures.")
        adata = run_scvi(adata, cfg.model)
        _LOGGER.info("Running clustering.")
        cluster_config = cluster.LeidenNClusterConfig(clusters=n_clusters)
        clustering_algorithm = cluster.LeidenNCluster(cluster_config)
        labels = clustering_algorithm.fit_predict(adata.obsm[cfg.model.latent_key])

        # Read the anndata and add the cluster labels
        # TODO(Pawel, Florian, Josephine): Apply preprocessing, e.g., selecting HVGs?
        #  Or maybe this should be in the GEX object?
        adata = anndata.read_h5ad(cfg.data.data_path)
        cluster_col = "new-cluster-column"
        adata.obs[cluster_col] = labels

        # Find the signatures
        _LOGGER.info("Running DE-analysis.")
        gex_object = gsea.GeneExpressionAnalysis(cluster_name=cluster_col)
        gene_ranks = gex_object.diff_gex(adata)

        # *** Signature saving and scoring ***
        # by default, all de novo found signatures are saved as the result of the differential gene expression
        # and the signatures are scored an all cells using n_genes_sig top positively diff expressed genes
        _LOGGER.info("Saving signatures.")
        output_dir = fs.PostprocessingDir(path=".")
        output_dir.make_sig_dir()
        gsea.save_signatures(diff_genes=gene_ranks, res_dir=output_dir.signature_output)
    except Exception as e:
        _LOGGER.exception(e)


if __name__ == "__main__":
    main()

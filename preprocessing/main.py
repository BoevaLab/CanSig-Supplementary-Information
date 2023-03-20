import os

import anndata
import hydra
import numpy as np
import scanpy as sc
from cansig import run_preprocessing
from omegaconf import DictConfig
import logging

_LOGGER = logging.getLogger(__name__)


from utils import (
    get_samples,
    load_adata,
    get_scoring_dict,
    get_reference_groups,
    mkdirs,
    symlink_force
)


def quality_control_10x(adata: anndata.AnnData) -> anndata.AnnData:
    _LOGGER.info(f"Starting qc with {adata.n_obs} cells and {adata.n_vars} genes.")
    sc.pp.filter_cells(adata, min_counts=1_500)
    sc.pp.filter_cells(adata, max_counts=50_000)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False,
                               inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20.0].copy()
    sc.pp.filter_genes(adata, min_cells=int(0.01 * adata.n_obs))
    _LOGGER.info(f"Finished qc with {adata.n_obs} cells and {adata.n_vars} genes.")
    return adata


def quality_control_smart_seq(adata:anndata.AnnData) -> anndata.AnnData:
    _LOGGER.info(f"Starting qc with {adata.n_obs} cells and {adata.n_vars} genes.")
    sc.pp.filter_cells(adata, min_genes=3_000)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False,
                               inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20.0].copy()
    sc.pp.filter_genes(adata, min_cells=int(0.01 * adata.n_obs))
    _LOGGER.info(f"Finished qc with {adata.n_obs} cells and {adata.n_vars} genes.")
    return adata


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    mkdirs(cfg)
    adatas = []
    samples = get_samples(cfg.cancer.data)
    for sample in samples:
        if sample in cfg.data.excluded_samples:
            continue
        adata = load_adata(sample_info=sample)
        adatas.append(adata)

    adata = anndata.concat(adatas, join="outer")

    if cfg.cancer.type not in ["glioblastoma", "tirosh_mel"]:
        adata = quality_control_10x(adata)
        min_malignant_cells=50
    else:
        adata = quality_control_smart_seq(adata)
        min_malignant_cells=30

    scoring_dict = get_scoring_dict(cfg.cancer.scoring)
    reference_groups = get_reference_groups(cfg.cancer.reference)

    assert "celltype" in adata.obs.columns, "Missing celltype from adata.obs"
    assert "sample_id" in adata.obs.columns, "Missing sample_id from adata.obs"

    adata = run_preprocessing(
        adata,
        malignant_celltypes=list(cfg.cancer.unhealthy),
        undetermined_celltypes=list(cfg.cancer.undecided),
        reference_groups=reference_groups,
        celltype_key="celltype",
        batch_key="sample_id",
        gene_order=cfg.annotations.gene_order,
        scoring_dict=scoring_dict,
        window_size=cfg.cancer.infercnv.window_size,
        min_reference_cells=cfg.cancer.infercnv.min_reference_cells,
        min_malignant_cells=min_malignant_cells,
        step=cfg.cancer.infercnv.step,
        figure_dir=cfg.dirs.figures,
    )
    adata.obs["n_counts"] = adata.X.sum(1)
    adata.obs["log_counts"] = np.log(adata.obs["n_counts"])
    adata.write_h5ad(cfg.files.data)
    bdata = adata[adata.obs["malignant_key"] == "malignant", :].copy()
    bdata.write(cfg.files.malignant)
    bdata = adata[adata.obs["malignant_key"] == "non-malignant", :].copy()
    bdata.write(cfg.files.non_malignant)
    symlink_force(os.path.abspath(cfg.files.malignant), cfg.dirs.last_path)


if __name__ == "__main__":
    main()



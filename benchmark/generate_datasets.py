import os
from pathlib import Path
from typing import List, Optional, Tuple

import anndata
import infercnvpy as cnv
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from numpy.random import uniform
from scipy.io import mmread


def get_adata(path):
    files = os.listdir(path)
    if "sim_counts.csv" in files:
        X = pd.read_csv(f"{path}/sim_counts.csv", index_col=0).transpose()
    elif "sim_counts.mtx" in files:
        X = mmread(f"{path}/sim_counts.mtx").tocsr().transpose()
    else:
        raise NotImplementedError()
    gene_info = pd.read_csv(f"{path}/sim_genes.csv", index_col=0)
    cell_info = pd.read_csv(f"{path}/sim_cells.csv", index_col=0)
    adata = anndata.AnnData(X=X, var=gene_info, obs=cell_info)
    return adata


def generate_cnvs(
    adata,
    batch_key: str = "Batch",
    malignant_key: str = "Group",
    non_malignant_cat: str = "Group5",
    n_cnvs=15,
    min_length=25,
    max_length=150,
    n_chromosomes=20,
    min_cnv_per_patient: int = 8,
    max_cnv_per_patient: int = 12,
    drop_percentage: float = 0.5,
    min_cells_per_subclone: Optional[int] = None,
    subclonal_key: str = "subclonal",
    cnv_loss=(0.3, 1.1),
    cnv_gain=(0.9, 1.7),
):
    cancer_cnvs = generate_params(n_cnvs, n_chromosomes)
    adata.obs[subclonal_key] = adata.obs[batch_key].values
    for batch in adata.obs[batch_key].unique():
        batch_idx = (adata.obs[batch_key] == batch) & (
            adata.obs[malignant_key] != non_malignant_cat
        )
        n_cnvs_per_patient = np.random.choice(
            np.arange(min_cnv_per_patient, max_cnv_per_patient)
        )
        patient_cnvs = np.random.choice(
            list(cancer_cnvs.keys()), replace=False, size=n_cnvs_per_patient
        )
        for chromosome in patient_cnvs:
            len_chr = sum(adata.var["chromosome"] == chromosome)
            length = np.random.choice(np.arange(min_length, max_length))
            gain = cancer_cnvs[chromosome]

            start_idx = np.random.choice(np.arange(len_chr - (length + 1)))
            start_idx += adata.var["chromosome"].str.find(chromosome).argmax()
            cnv_effect = get_cnv_effect(
                sum(batch_idx),
                length,
                gain,
                drop_percentage,
                cnv_gain=cnv_gain,
                cnv_loss=cnv_loss,
            )
            adata[batch_idx, start_idx : (start_idx + length)].X = np.rint(
                adata[batch_idx, start_idx : (start_idx + length)].X * cnv_effect
            )

            if min_cells_per_subclone:
                n_cells = sum(
                    adata[batch_idx, :].obs[malignant_key] != non_malignant_cat
                )
                max_subclones = (n_cells // min_cells_per_subclone) - 1
                print(max_subclones)


def infercnvs(
    adata: AnnData,
    window_size=200,
    step=10,
    batch_key="Batch",
    group_key="Group",
    non_malignant_groups: Optional[List[str]] = None,
):
    if non_malignant_groups is None:
        non_malignant_groups = ["Group5"]

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    adata.obsm["X_cnv"] = np.zeros((adata.n_obs, adata.n_vars // 10))
    for sample in adata.obs[batch_key].unique():
        cnv_dict, X_cnv = cnv.tl.infercnv(
            adata[adata.obs[batch_key] == sample].copy(),
            reference_key=group_key,
            reference_cat=non_malignant_groups,
            window_size=window_size,
            step=step,
            inplace=False,
        )
        adata.obsm["X_cnv"][adata.obs[batch_key] == sample, :] = X_cnv.todense()
    adata.uns["cnv"] = {"chr_pos": cnv_dict}
    adata.obs["malignant_key"] = adata.obs["Group"].apply(
        lambda x: "malignant" if x != "Group5" else "non-malignant"
    )
    adata.X = adata.layers["counts"].copy()
    del adata.layers["counts"]
    if "log1p" in adata.uns_keys():
        del adata.uns["log1p"]


def get_cnv_effect(
    n_cells: int,
    length: int,
    gain: bool,
    drop_percentage: float,
    cnv_gain: Tuple[float],
    cnv_loss: Tuple[float],
):
    size = (n_cells, length)
    cnv_effect = (
        uniform(*cnv_gain, size=size) if gain else uniform(*cnv_loss, size=size)
    )
    drop_effect = np.random.uniform(size=size) <= drop_percentage
    cnv_effect[drop_effect] = 1.0
    return cnv_effect


def generate_params(n_cnvs: int, len_chromosome: int = 20):
    cnv_chromosomes = np.random.choice(
        np.arange(1, len_chromosome + 1), size=n_cnvs, replace=False
    )
    cnv_params = {}
    for i in cnv_chromosomes:
        cnv_params[f"chr{i}"] = True if np.random.uniform() < 0.5 else False
    return cnv_params


def annotate(adata: anndata.AnnData, n_chromosomes: int = 20):
    length_chromosome = adata.n_vars // n_chromosomes
    chromosome = [
        f"chr{i}" for i in range(1, n_chromosomes + 1) for _ in range(length_chromosome)
    ]
    adata.var["chromosome"] = chromosome
    adata.var["start"] = np.arange(1, adata.shape[1] + 1)
    adata.var["end"] = np.arange(1, adata.shape[1] + 1) + 1


def drop_rare(
    adata: AnnData, group: str, batch_key: str, p_drop: float, group_key: str
) -> AnnData:
    idx = np.ones((adata.n_obs))
    idx.fill(False)
    for batch in adata.obs[batch_key].unique():
        if np.random.uniform() <= p_drop:
            idx = idx | (
                (adata.obs[batch_key] == batch) & (adata.obs[group_key] == group)
            )

    return adata[~idx, :].copy()


def drop_groups(
    adata: AnnData, batch_key: str, group_key: str, groups: List[str]
) -> AnnData:
    idx = np.ones((adata.n_obs))
    idx.fill(False)
    chocies = groups
    for batch in adata.obs[batch_key].unique():
        group = np.random.choice(chocies)
        if group:
            idx = idx | (
                (adata.obs[batch_key] == batch) & (adata.obs[group_key] == group)
            )

    return adata[~idx, :].copy()


def main():
    configs = [
        {
            "min_cnv_per_patient": 10,
            "max_cnv_per_patient": 12,
            "drop_rare": True,
            "drop_random": False,
            "name": "rare",
        },
        {
            "min_cnv_per_patient": 10,
            "max_cnv_per_patient": 12,
            "drop_rare": False,
            "drop_random": True,
            "name": "random",
        },
        {
            "min_cnv_per_patient": 6,
            "max_cnv_per_patient": 8,
            "drop_rare": False,
            "drop_random": False,
            "name": "few_cnv",
        },
        {
            "min_cnv_per_patient": 10,
            "max_cnv_per_patient": 12,
            "drop_rare": False,
            "drop_random": False,
            "name": "many_cnv",
        },
    ]

    path = "/cluster/work/boeva/scRNAdata/benchmark"
    # path = "/home/barkmann/BoevaLab/scRNA_shared_signatures/benchmark"
    for dataset_path in Path(f"{path}/raw_datasets").iterdir():
        print(f"Processing {dataset_path.stem}")

        for n_config, config in enumerate(configs):
            print(f"\t Processing config {n_config}")
            adata = get_adata(dataset_path)
            adata.X = adata.X.todense()
            annotate(adata)

            if config["drop_rare"]:
                adata = drop_rare(
                    adata, "Group4", batch_key="Batch", group_key="Group", p_drop=0.5
                )
            if config["drop_random"]:
                adata = drop_groups(
                    adata,
                    batch_key="Batch",
                    group_key="Group",
                    groups=["Group1", "Group2", "Group3", "Group4"],
                )

            generate_cnvs(
                adata,
                min_cnv_per_patient=config["min_cnv_per_patient"],
                max_cnv_per_patient=config["max_cnv_per_patient"],
                min_length=50,
                max_length=150,
                n_cnvs=18,
                non_malignant_cat="Group5",
                drop_percentage=0.0,
            )

            infercnvs(adata)
            # For some input adata.X is a matrix which is not saved correctly by
            # anndata. I have no idea why it becomes a matrix...
            if type(adata.X) == np.matrix:
                adata.X = np.squeeze(np.asarray(adata.X))
            adata.write_h5ad(
                f"{path}/datasets/{dataset_path.stem}_{config['name']}.h5ad"
            )


if __name__ == "__main__":
    main()

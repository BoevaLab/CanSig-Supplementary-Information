from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


def save_latent(adata: AnnData, latent_key: str, dataset_name: str) -> None:
    if latent_key in adata.obsm_keys():
        latent = pd.DataFrame(adata.obsm[latent_key], index=adata.obs_names)
        latent.to_csv(f"{dataset_name}_latent.csv")

def plot_integration(
        adata: AnnData, dataset_name: str, batch_key: str, group_key: str
) -> None:
    sc.settings.figdir = "."
    sc.tl.umap(adata)
    sc.pl.umap(
        adata, color=[batch_key, group_key], save=f"_{dataset_name}.png", show=False
    )


def get_partition(gpu: bool) -> str:
    if gpu:
        return "gpu"
    return "compute"


def get_gres(gpu: bool) -> Optional[str]:
    if gpu:
        return "gpu:rtx2080ti:1"
    return None


def diffusion_nn(adata, k, max_iterations=26):
    """
    Diffusion neighbourhood score
    This function generates a nearest neighbour list from a connectivities matrix
    as supplied by BBKNN or Conos. This allows us to select a consistent number
    of nearest neighbours across all methods.

    Return:
       `k_indices` a numpy.ndarray of the indices of the k-nearest neighbors.
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "`neighbors` not in adata object. " "Please compute a neighbourhood graph!"
        )

    if "connectivities" not in adata.obsp:
        raise ValueError(
            "`connectivities` not in `adata.obsp`. "
            "Please pass an object with connectivities computed!"
        )

    T = adata.obsp["connectivities"]

    # Row-normalize T
    T = sparse.diags(1 / T.sum(1).A.ravel()) * T

    T_agg = T ** 3
    M = T + T ** 2 + T_agg
    i = 4

    while ((M > 0).sum(1).min() < (k + 1)) and (i < max_iterations):
        # note: k+1 is used as diag is non-zero (self-loops)
        print(f"Adding diffusion to step {i}")
        T_agg *= T
        M += T_agg
        i += 1

    if (M > 0).sum(1).min() < (k + 1):
        raise ValueError(
            f"could not find {k} nearest neighbors in {max_iterations}"
            "diffusion steps.\n Please increase max_iterations or reduce"
            " k.\n"
        )

    M.setdiag(0)
    k_indices = np.argpartition(M.A, -k, axis=1)[:, -k:]

    return k_indices


def diffusion_conn(adata, min_k=50, copy=True, max_iterations=26):
    """
    Diffusion for connectivites matrix extension
    This function performs graph diffusion on the connectivities matrix until a
    minimum number `min_k` of entries per row are non-zero.

    Note:
    Due to self-loops min_k-1 non-zero connectivies entries is actually the stopping
    criterion. This is equivalent to `sc.pp.neighbors`.

    Returns:
       The diffusion-enhanced connectivities matrix of a copy of the AnnData object
       with the diffusion-enhanced connectivities matrix is in
       `adata.uns["neighbors"]["conectivities"]`
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "`neighbors` not in adata object. " "Please compute a neighbourhood graph!"
        )

    if "connectivities" not in adata.obsp:
        raise ValueError(
            "`connectivities` not in `adata.obsp`. "
            "Please pass an object with connectivities computed!"
        )

    T = adata.obsp["connectivities"]

    # Normalize T with max row sum
    # Note: This keeps the matrix symmetric and ensures |M| doesn't keep growing
    T = sparse.diags(1 / np.array([T.sum(1).max()] * T.shape[0])) * T

    M = T

    # Check for disconnected component
    n_comp, labs = sparse.csgraph.connected_components(
        adata.obsp["connectivities"], connection="strong"
    )

    if n_comp > 1:
        tab = pd.value_counts(labs)
        small_comps = tab.index[tab < min_k]
        large_comp_mask = np.array(~pd.Series(labs).isin(small_comps))
    else:
        large_comp_mask = np.array([True] * M.shape[0])

    T_agg = T
    i = 2
    while ((M[large_comp_mask, :][:, large_comp_mask] > 0).sum(1).min() < min_k) and (
            i < max_iterations
    ):
        print(f"Adding diffusion to step {i}")
        T_agg *= T
        M += T_agg
        i += 1

    if (M[large_comp_mask, :][:, large_comp_mask] > 0).sum(1).min() < min_k:
        raise ValueError(
            "could not create diffusion connectivities matrix"
            f"with at least {min_k} non-zero entries in"
            f"{max_iterations} iterations.\n Please increase the"
            "value of max_iterations or reduce k_min.\n"
        )

    M.setdiag(0)

    if copy:
        adata_tmp = adata.copy()
        adata_tmp.uns["neighbors"].update({"diffusion_connectivities": M})
        return adata_tmp

    else:
        return M


def split_batches(adata: AnnData, batch_key: str) -> List[AnnData]:
    splits = []
    for batch in adata.obs[batch_key].cat.categories:
        splits.append(adata[adata.obs[batch_key] == batch].copy())
    return splits

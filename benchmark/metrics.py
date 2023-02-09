import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from anndata import AnnData
import scib_metrics as sm
from scib_metrics.benchmark import Benchmarker
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from _cluster import LeidenNClusterConfig, LeidenNCluster
from models import ModelConfig


_LOGGER = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    n_neighbors: int = 50
    group_key: str = "program"
    cluster_key: str = "leiden"
    n_random_seeds: int = 10
    clustering_range: Tuple[int] = tuple(range(2, 6))


def run_metrics(adata: AnnData, config: ModelConfig, metric_config: MetricsConfig):
    metrics = {}
    _LOGGER.info("Computing neighbors.")
    compute_neighbors(
        adata,
        latent_key=config.latent_key,
        n_neighbors=metric_config.n_neighbors,
    )

    # Biological conservation metrics
    _LOGGER.info("Computing ASW.")
    metrics.update(compute_asw(adata, metric_config.group_key, config.latent_key))
    _LOGGER.info("Computing dv-score.")
    metrics.update(
        compute_davies_bouldin(adata, metric_config.group_key, config.latent_key)
    )
    metrics.update(
        compute_calinski_harabasz(adata, metric_config.group_key, config.latent_key)
    )
    _LOGGER.info("Computing ARI and NMI.")
    metrics.update(compute_ari_nmi(adata, metric_config))

    # Batch effect metrics
    _LOGGER.info("Computing KBET.")
    metrics.update(
        kbet(
            adata,
            latent_key=config.latent_key,
            label_key=metric_config.group_key,
            batch_key=config.batch_key,
        )
    )

    return metrics


def kbet(
        adata: AnnData, latent_key: str, label_key: str, batch_key: str
) -> Dict[str, float]:
    """This implementation of kBet is taken from scib and combined with the
    kbet_single implementation from scETM."""
    adata.strings_to_categoricals()
    if latent_key in adata.obsm_keys():
        adata_tmp = sc.pp.neighbors(adata, n_neighbors=50, use_rep=latent_key,
                                    copy=True)
    else:
        adata_tmp = adata.copy()
    # check if pre-computed neighbours are stored in input file
    connectivities = diffusion_conn(adata_tmp, min_k=50, copy=False)
    adata_tmp.obsp["connectivities"] = connectivities

    # set upper bound for k0
    size_max = 2 ** 31 - 1

    # prepare call of kBET per cluster
    kBET_scores = {"cluster": [], "kBET": []}
    for clus in adata_tmp.obs[label_key].unique():

        # subset by label
        adata_sub = adata_tmp[adata_tmp.obs[label_key] == clus, :].copy()

        # check if neighborhood size too small or only one batch in subset
        if np.logical_or(
                adata_sub.n_obs < 10, len(adata_sub.obs[batch_key].cat.categories) == 1
        ):
            print(f"{clus} consists of a single batch or is too small. Skip.")
            score = np.nan
        else:
            quarter_mean = np.floor(
                np.mean(adata_sub.obs[batch_key].value_counts()) / 4
            ).astype("int")
            k0 = np.min([70, np.max([10, quarter_mean])])
            # check k0 for reasonability
            if k0 * adata_sub.n_obs >= size_max:
                k0 = np.floor(size_max / adata_sub.n_obs).astype("int")

            n_comp, labs = scipy.sparse.csgraph.connected_components(
                adata_sub.obsp["connectivities"], connection="strong"
            )

            if n_comp == 1:  # a single component to compute kBET on
                adata_sub.obsm["knn_indices"] = diffusion_nn(adata_sub, k=k0)
                adata_sub.uns["neighbors"]["params"]["n_neighbors"] = k0

                score = calculate_kbet(
                    adata_sub,
                    use_rep="",
                    batch_col=batch_key,
                    calc_knn=False,
                    n_neighbors=adata_sub.uns["neighbors"]["params"]["n_neighbors"],
                )[2]

            else:
                # check the number of components where kBET can be computed upon
                comp_size = pd.value_counts(labs)
                # check which components are small
                comp_size_thresh = 3 * k0
                idx_nonan = np.flatnonzero(
                    np.in1d(labs, comp_size[comp_size >= comp_size_thresh].index)
                )

                # check if 75% of all cells can be used for kBET run
                if len(idx_nonan) / len(labs) >= 0.75:
                    # create another subset of components, assume they are not visited
                    # in a diffusion process
                    adata_sub_sub = adata_sub[idx_nonan, :].copy()
                    adata_sub_sub.obsm["knn_indices"] = diffusion_nn(
                        adata_sub_sub, k=k0
                    )
                    adata_sub_sub.uns["neighbors"]["params"]["n_neighbors"] = k0

                    score = calculate_kbet(
                        adata_sub_sub,
                        use_rep="",
                        batch_col=batch_key,
                        calc_knn=False,
                        n_neighbors=adata_sub_sub.uns["neighbors"]["params"][
                            "n_neighbors"
                        ],
                    )[2]

                else:  # if there are too many too small connected components,
                    score = 0  # i.e. 100% rejection

        kBET_scores["cluster"].append(clus)
        kBET_scores["kBET"].append(score)

    kBET_scores = pd.DataFrame.from_dict(kBET_scores)
    kBET_scores = kBET_scores.reset_index(drop=True)

    final_score = np.nanmean(kBET_scores["kBET"]).item()

    return {"k_bet_acceptance_rate": final_score}


def compute_ari(adata: AnnData, group_key: str, cluster_key: str) -> float:
    return adjusted_rand_score(adata.obs[group_key], adata.obs[cluster_key])


def compute_nmi(adata: AnnData, group_key: str, cluster_key: str) -> float:
    return normalized_mutual_info_score(adata.obs[group_key], adata.obs[cluster_key])

def compute_asw(
        adata: AnnData, group_key: str, latent_key: str
) -> Dict[str, Optional[float]]:
    if latent_key not in adata.obsm_keys():
        return {"average_silhouette_width": np.nan}
    asw = silhouette_score(X=adata.obsm[latent_key], labels=adata.obs[group_key])
    asw = (asw + 1) / 2

    return {"average_silhouette_width": asw}


def compute_calinski_harabasz(
        adata: AnnData, group_key: str, latent_key: str
) -> Dict[str, Optional[float]]:
    if latent_key not in adata.obsm_keys():
        return {"calinski_harabasz_score": np.nan}
    score = calinski_harabasz_score(adata.obsm[latent_key], adata.obs[group_key])
    return {"calinski_harabasz_score": score}


def compute_davies_bouldin(
        adata: AnnData, group_key: str, latent_key: str
) -> Dict[str, Optional[float]]:
    if latent_key not in adata.obsm_keys():
        return {"davies_bouldin": np.nan}
    score = davies_bouldin_score(adata.obsm[latent_key], adata.obs[group_key])
    return {"davies_bouldin": score}


def compute_ari_nmi(
        adata: AnnData, metric_config: MetricsConfig
) -> Dict[str, Optional[float]]:
    metrics = {}
    for k in metric_config.clustering_range:

        for random_seed in range(metric_config.n_random_seeds):
            _LOGGER.info(f"Running {k} clusters and {random_seed} random seed.")
            try:
                leiden_config = LeidenNClusterConfig(
                    random_state=random_seed, clusters=k
                )
                cluster_algo = LeidenNCluster(leiden_config)
                cluster_algo.fit_predict(adata, key_added=metric_config.cluster_key)
            except ValueError as e:
                print(e)
                ari = np.nan
                nmi = np.nan
            else:
                ari = compute_ari(adata, metric_config.group_key,
                                  metric_config.cluster_key)
                nmi = compute_nmi(adata, metric_config.group_key,
                                  metric_config.cluster_key)

            metrics[f"ari_{k}_{random_seed}"] = ari
            metrics[f"nmi_{k}_{random_seed}"] = nmi

    return metrics


def compute_neighbors(adata: AnnData, latent_key: str, n_neighbors: int):
    if latent_key in adata.obsm.keys():
        knn_indices = _get_knn_indices(
            adata,
            use_rep=latent_key,
            n_neighbors=n_neighbors,
            calc_knn=True,
        )
        adata.obsm["knn_indices"] = knn_indices

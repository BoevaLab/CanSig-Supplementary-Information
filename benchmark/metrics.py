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


@dataclass
class MetricsConfig:
    n_neighbors: int = 50
    group_key: str = "program"
    cluster_key: str = "leiden"
    n_random_seeds: int = 10
    clustering_range: Tuple[int] = tuple(range(2, 6))

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

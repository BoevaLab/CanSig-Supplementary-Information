from typing import List, Optional

import anndata
import numpy as np
import pytest
from scib.metrics import kBET

from benchmark.metrics import kbet, run_metrics, MetricsConfig
from benchmark.models import ModelConfig


def get_adata(proportion: Optional[List[int]] = None):
    if proportion is None:
        proportion = [50, 200, 50, 200]

    N_CELLS = 500
    adata = anndata.AnnData(X=np.zeros((N_CELLS, 1)))

    means = [(0.,) * 5 + (5.,) * 5, (5.,) * 5 + (0.,) * 5]
    adata.obsm["latent"] = np.vstack(
        [np.random.normal(loc=means[0], size=(N_CELLS // 2, 10)),
         np.random.normal(loc=means[1], size=(N_CELLS // 2, 10))])

    adata.obs["batch"] = ["batch_1"] * proportion[0] + ["batch_2"] * proportion[1] + [
        "batch_1"] * proportion[2] + ["batch_2"] * proportion[3]
    adata.obs["program"] = ["program_1"] * (N_CELLS // 2) + ["program_2"] * (
            N_CELLS // 2)
    adata.strings_to_categoricals()
    return adata


@pytest.mark.parametrize("proportion", [[50, 200, 50, 200],
                                        [125, 125, 125, 125],
                                        [50, 200, 125, 125]])
def test_kbet(proportion):
    adata = get_adata(proportion)
    res_1 = kbet(adata, latent_key="latent", batch_key="batch", label_key="program")
    res_2 = kBET(adata, embed="latent", batch_key="batch", label_key="program")
    assert pytest.approx(res_1["k_bet_acceptance_rate"], abs=0.02) == res_2


def test_run_metrics():
    adata = get_adata()
    model_config = ModelConfig(batch_key="batch")
    metric_config = MetricsConfig(n_clusters=2)

    results = run_metrics(adata, config=model_config, metric_config=metric_config)

    for key in ["average_silhouette_width", "davies_bouldin",
                "calinski_harabasz_score"]:
        assert key in results

    for metric in ["ari", "nmi"]:
        for key in [f"{metric}_{i}" for i in range(10)]:
            assert pytest.approx(results[key]) == 1.

    assert results["k_bet_acceptance_rate"] >= 0.95


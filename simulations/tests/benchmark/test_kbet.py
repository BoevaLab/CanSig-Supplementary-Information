from typing import List

import anndata
import numpy as np
import pytest
from scib.metrics import kBET

from benchmark.metrics import kbet


def get_adata(proportion: List[int]):
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


@pytest.mark.parametrize("proportion", [[50, 200, 50, 200], [125, 125, 125, 125]])
def test_kbet(proportion):
    adata=get_adata(proportion)
    res_1 = kbet(adata, latent_key="latent", batch_key="batch", label_key="program")
    res_2 = kBET(adata, embed="latent", batch_key="batch", label_key="program")
    assert pytest.approx(res_1["k_bet_acceptance_rate"], abs=0.01) == res_2

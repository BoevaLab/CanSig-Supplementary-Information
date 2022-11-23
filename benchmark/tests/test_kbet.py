"""Test the kBET implementation. This test requires the kBET R library, therefore it
is in an extra file."""
import pytest

from benchmark.metrics import kbet
from test_metrics import get_adata


@pytest.mark.parametrize("proportion", [None])
def test_kbet(proportion):
    from scib.metrics import kBET

    adata = get_adata(proportion)
    res_1 = kbet(adata, latent_key="latent", batch_key="batch", label_key="program")
    res_2 = kBET(adata, embed="latent", batch_key="batch", label_key="program")
    assert pytest.approx(res_1["k_bet_acceptance_rate"], abs=0.02) == res_2


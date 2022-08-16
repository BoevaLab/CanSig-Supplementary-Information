import pytest

import dhaka.run as drun
from dhaka.data import example_data


@pytest.mark.parametrize("n_latent", [3, 5])
def test_run(n_latent: int, n_cells: int = 25) -> None:
    # Generate some example data
    adata = example_data(n_cells=n_cells, n_genes=40)
    # Generate run config
    config = drun.DhakaConfig(n_latent=n_latent, epochs=2)

    # Learn the representations
    adata = drun.run_dhaka(adata, config=config, key_added="X_dhaka")

    # Check that they indeed exist
    assert adata.obsm["X_dhaka"].shape == (n_cells, n_latent)

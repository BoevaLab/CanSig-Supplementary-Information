import torch
import pytest

import dhaka.vae as vae


@pytest.mark.parametrize("n_latent", [3, 5])
@pytest.mark.parametrize("n_input", [10, 20])
def test_encoder(n_latent: int, n_input: int, n_cells: int = 5) -> None:
    x = torch.rand(n_cells, n_input)
    encoder = vae.Encoder(input_dim=n_input, latent_dim=n_latent, hidden1=5, hidden2=3, hidden3=3)
    mu, logvar = encoder(x)

    assert mu.size() == (n_cells, n_latent)
    assert logvar.size() == (n_cells, n_latent)

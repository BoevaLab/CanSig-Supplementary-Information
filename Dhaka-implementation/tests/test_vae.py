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


@pytest.mark.parametrize("n_latent", [4, 10])
def test_sample_shape(n_latent: int, batch_size: int = 5) -> None:
    mu = torch.randn(batch_size, n_latent)
    logvar = torch.randn(batch_size, n_latent)

    sample = vae.sample(mu, logvar)
    assert sample.size() == mu.size()


@pytest.mark.parametrize("n_latent", [4, 10])
def test_sample_right_mean_and_variance(n_latent: int, batch_size: int = 4, n_samples: int = 10000) -> None:
    torch.manual_seed(10)

    mu = torch.randn(batch_size, n_latent)
    # Use very small variance
    logvar = 0.01 * torch.rand(batch_size, n_latent)

    samples = torch.stack(
        [vae.sample(mu, logvar) for _ in range(n_samples)], dim=0
    )

    assert samples.size() == (n_samples, batch_size, n_latent)

    empirical_mu = torch.mean(samples, dim=0)
    empirical_std = torch.std(samples, dim=0)

    assert empirical_mu.size() == mu.size()
    assert empirical_std.size() == mu.size()

    assert empirical_mu == pytest.approx(mu, rel=0.03, abs=0.03)
    assert empirical_std == pytest.approx(torch.exp(0.5 * logvar), rel=0.05, abs=0.005)

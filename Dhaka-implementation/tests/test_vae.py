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


def test_kl_divergence(batch_size: int = 5, n_latent: int = 3) -> None:
    mu = torch.zeros(batch_size, n_latent)
    logvar = torch.zeros(batch_size, n_latent)
    assert vae.kl_divergence(mu, logvar) == 0


@pytest.mark.parametrize("random_seed", [0, 1, 2])
@pytest.mark.parametrize("dim", [2, 3])
def test_kl_divergence(random_seed: int, dim: int) -> None:
    torch.manual_seed(random_seed)

    mu = torch.randn(1, dim)
    logvar = torch.randn(1, dim)

    assert vae.kl_divergence(mu, logvar) > 0


@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_decoder(activation: str, batch_size: int = 5, n_latent: int = 4, output_dim: int = 10) -> None:
    decoder = vae.Decoder(output_dim=output_dim, latent_dim=n_latent, hidden1=3, hidden2=2, hidden3=3)

    some_mus = torch.randn(batch_size, n_latent)
    some_logvars = torch.randn_like(some_mus)

    x_decoded = decoder(some_mus, some_logvars)

    assert x_decoded.size() == (batch_size, output_dim)

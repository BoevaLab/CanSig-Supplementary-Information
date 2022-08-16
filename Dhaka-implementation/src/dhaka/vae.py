from typing import Literal, Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Probabilistic encoder.

    For each datapoint, predicts the mean and the log(variance)
    of the corresponding Gaussian posterior.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden1: int = 1024,
        hidden2: int = 512,
        hidden3: int = 256,
    ) -> None:
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
        )

        self.mu_layers = nn.Linear(hidden3, latent_dim)
        self.logvar_layers = nn.Linear(hidden3, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_layers(x)
        mu = self.mu_layers(shared)
        logvar = self.logvar_layers(shared)

        return mu, logvar


def calculate_std(logvar: torch.Tensor) -> torch.Tensor:
    """Passes from logarithm of the variance to the standard deviation.

    Args:
        logvar: logarithm of the variance, shape (batch, n_latent)

    Returns:
        standard deviations, shape (batch, n_latent)
    """
    return torch.exp(logvar / 2)


def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Returns a one-point sample from each distribution.

    Args:
        mu: mean vectors of the distributions, shape (batch, n_latent)
        logvar: logarithm of the variance for each distribution, shape (batch, n_latent)

    Returns:
        a batch of samples (one from each distribution), shape (batch, n_latent)
    """
    eps = torch.randn_like(mu)
    return mu + eps * calculate_std(logvar)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    """For a batch of multinormal distributions calculates the sum of
    KL divergences between each of them and Normal(0, Identity)

    Args:
        mu: shape (batch, n_latent)
        logvar: shape (batch, n_latent)

    Returns:
        KL divergence summed along all points

    Note:
        There is a _minus_ sign:
            Loss = Reconstruction - KL
    """
    # As in the original VAE paper, the equation is
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        hidden1: int = 256,
        hidden2: int = 512,
        hidden3: int = 1024,
        final_activation: Literal["sigmoid", "relu"] = "sigmoid",
    ) -> None:
        super().__init__()

        assert final_activation in ["sigmoid", "relu"]
        assert min(hidden1, hidden2, hidden3, output_dim, latent_dim) > 0

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim),
        )
        self.final_activation = nn.Sigmoid() if final_activation == "sigmoid" else nn.ReLU()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        z = sample(mu=mu, logvar=logvar)
        x_raw = self.layers(z)
        return self.final_activation(x_raw)


class Dhaka(pl.LightningModule):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 3,
        learning_rate: float = 1e-4,
        hidden_layers: Tuple[int] = (256, 512, 1024),
    ) -> None:
        """

        Args:
            n_genes: number of genes, i.e., batch should be of size (n_cells, n_genes)
            latent_dim: dimensionality of the latent space
            learning_rate: learning rate
            hidden_layers: hidden layers of the DECODER (passing from latent_dim to n_genes)
                We use three layers for the decoder. The encoder architecture is a symmetric one.
        """
        super().__init__()

        assert len(hidden_layers) == 3
        self.encoder = Encoder(
            input_dim=n_genes,
            latent_dim=latent_dim,
            # Note that the encoder has reversed order of layers
            hidden1=hidden_layers[2],
            hidden2=hidden_layers[1],
            hidden3=hidden_layers[0],
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=n_genes,
            hidden1=hidden_layers[0],
            hidden2=hidden_layers[1],
            hidden3=hidden_layers[2],
        )

        self._learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self._learning_rate)
        return optimizer

    def training_step(self, train_batch):
        # Infer the distributions for each data point
        mu, logvar = self.encoder(train_batch)

        # Sample one point from each distribution
        reconstructed = self.decoder(mu, logvar)

        # Calculate the reconstruction loss, averaged per datapoint
        reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed, train_batch, reduction="mean")

        # Calculate the KL term averaged per datapoint (as the loss above is averaged, rather than summed).
        # Note also the minus sign --- we will add different losses in the end.
        kl_loss = -kl_divergence(mu, logvar) / len(train_batch)

        return reconstruction_loss + kl_loss

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(batch)

    def representations(self, batch) -> torch.Tensor:
        mu, logvar = self.forward(batch)
        return mu

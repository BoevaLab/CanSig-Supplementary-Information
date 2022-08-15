from typing import Tuple

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


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        hidden1: int = 256,
        hidden2: int = 512,
        hidden3: int = 1024,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        raise NotImplementedError


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
        optimizer = torch.optim.RMSprop(self.parameters, lr=self._learning_rate)
        return optimizer

    def training_step(self, train_batch):
        raise NotImplementedError


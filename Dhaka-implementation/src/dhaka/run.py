import dataclasses

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data

import dhaka.vae as vae
import dhaka.data as data


_DEFAULT_KEY_ADDED: str = "X_dhaka"


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@dataclasses.dataclass
class DhakaConfig:
    """Controls the Dhaka run.

    Compare with the hyperparameters described in Supplementary Information, Section 1.1.

    *** Latent representations ***
      n_latent: dimensionality of latent representations

    *** Data preprocessing ***
      n_genes: number of genes to be selected (genes are selected by highest average)
      pseudocounts: pseudocounts to be added to the counts data, to mitigate problems with zeros.
        Note: in the original paper pseudocounts are not mentioned. Possibly setting them to 0 may work as well,
          although log2(0) = -inf, so I'd personally discourage it
      total_expression: used to normalize the data (by default we have TPM)

    *** VAE training ***
      epochs: number of training epochs
      batch_size: batch size
      learning_rate: learning rate
      clip_norm: the gradient norm is clipped to prevent exploding/vanishing gradients (see
    """
    # Latent representations
    n_latent: int = 3
    # Data preprocessing
    n_genes: int = 5000
    total_expression: float = 1e6
    pseudocounts: int = 1
    # Training
    epochs: int = 5
    batch_size: int = 50
    learning_rate: float = 1e-4
    clip_norm: float = 2.0


def _validate_config(config: DhakaConfig) -> None:
    assert config.n_latent > 0
    assert config.n_genes > 0
    assert config.total_expression >= 1
    assert config.pseudocounts >= 0
    assert config.epochs >= 1
    assert config.batch_size >= 1
    assert config.learning_rate > 0
    assert config.clip_norm > 0


def run_dhaka(adata: ad.AnnData, config: DhakaConfig, key_added: str = _DEFAULT_KEY_ADDED) -> ad.AnnData:
    """Runs the Dhaka algorithm on the specified dataset, learning the representations.

    Args:
        adata: raw gene expression data, will be modified in-place
        config: config
        key_added: field name with representations (to be added to `adata`)

    Returns:
        `adata` with representations added to `obsm[key_added]`
    """
    _validate_config(config)

    # Construct the dataset and a training dataloader
    dataset = data.NumpyArrayDataset(
        data.normalize(
            adata,
            pseudocounts=config.pseudocounts,
            n_top_genes=config.n_genes,
            total_expression=config.total_expression,
        )
    )
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=config.batch_size)

    # Define a model and the training procedure (with gradient clipping)
    model = vae.Dhaka(n_genes=dataset.n_features, latent_dim=config.n_latent)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        gradient_clip_val=config.clip_norm,
        gradient_clip_algorithm="norm"
    )
    trainer.fit(model, train_dataloaders=train_dataloader)

    # Get the predictions
    prediction_dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=config.batch_size)
    model.eval()

    representations_all = np.concatenate([
        tensor_to_numpy(model.representations(batch)) for batch in prediction_dataloader
    ])
    assert representations_all.shape == (adata.shape[0], config.n_latent)

    # Add the representations to the AnnData object and return it
    adata.obsm[key_added] = representations_all
    return adata

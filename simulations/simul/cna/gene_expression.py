"""How CNA changes affect gene expression."""
from typing import Tuple
import numpy as np
import anndata as ad

from .types import GeneVector
from ..rand import Seed


# Type for storing numbers how CNA affects gene expression of a particular gene.
# Shape (n_genes,)
CNAExpressionChangeVector = GeneVector


def get_mask_high(adata: ad.AnnData, quantile: float = 0.9) -> np.ndarray:
    gex_mean = np.squeeze(np.asarray(adata.X.mean(axis=0)))
    qt = np.quantile(gex_mean, quantile)
    mask_high = gex_mean > qt
    return mask_high


def _sample_gain_vector_high(n_genes: int, rng: Seed = 123) -> np.ndarray:
    """Samples gain changes from a Cauchy distribution for highly expressed genes"""
    generator = np.random.default_rng(rng)
    x = generator.standard_cauchy(size=n_genes)
    # cauchy is part of the location-scale family, so if I draw from standard cauchy just need to shift
    # by location and multiply by scale
    # parameters are fitted on real data
    return 1.3538 + 0.1879 * x


def _sample_loss_vector_high(n_genes: int, rng: Seed = 123) -> np.ndarray:
    """Samples loss changes from a Cauchy distribution for highly expressed genes"""
    generator = np.random.default_rng(rng)
    x = generator.standard_cauchy(size=n_genes)
    # cauchy is part of the location-scale family, so if I draw from standard cauchy just need to shift
    # by location and multiply by scale
    # parameters are fitted on real data
    return 0.5681 + 0.0725 * x


def _sample_gain_vector_low(n_genes: int, rng: Seed = 123) -> np.ndarray:
    """Samples gain changes from a GMM for lowly expressed genes"""
    generator = np.random.default_rng(rng)
    pi = [0.2671, 0.7329]
    mu = [3.0553, 0.9422]
    sigma = [2.2546, 0.6179]

    # to sample from a mixture, you first sample the mixture from a categorical distribution
    mixture = generator.choice([0, 1], size=n_genes, p=pi)
    # then you sample from the normal of the mixture that was chosen
    x = []
    for i in range(len(mixture)):
        x.append(generator.normal(loc=mu[mixture[i]], scale=sigma[mixture[i]]))
    return np.array(x)


def _sample_loss_vector_low(n_genes: int, rng: Seed = 123) -> np.ndarray:
    """Samples loss changes from a GMM for lowly expressed genes"""
    generator = np.random.default_rng(rng)
    pi = [0.1728, 0.8272]
    mu = [2.1843, 0.5713]
    sigma = [2.0966, 0.4377]

    # to sample from a mixture, you first sample the mixture from a categorical distribution
    mixture = generator.choice([0, 1], size=n_genes, p=pi)
    # then you sample from the normal of the mixture that was chosen
    x = []
    for i in range(len(mixture)):
        x.append(generator.normal(loc=mu[mixture[i]], scale=sigma[mixture[i]]))
    return np.array(x)


def sample_gain_vector(mask_high: np.ndarray) -> CNAExpressionChangeVector:
    """Generates a vector controlling by what factor expression should change if a gene copy is gained.

    For each gene `g`:

    `NEW_EXPRESSION[g] = OLD_EXPRESSION[g] * GAIN_VECTOR[g]`

    Args:
        n_genes: for how many genes this vector should be generated
        rng: seed

    Returns:
        a vector controlling the expression change, shape (n_genes,)
    """
    changes = np.zeros((len(mask_high),))
    n_high = len(np.where(mask_high)[0])
    n_low = len(mask_high) - n_high
    #### WARNING: put some random seeds in here so it varies across patients
    gain_high = _sample_gain_vector_high(n_genes=n_high, rng=np.random.randint(100))
    gain_low = _sample_gain_vector_low(n_genes=n_low, rng=np.random.randint(100))

    changes[mask_high] = gain_high
    changes[~mask_high] = gain_low

    return changes


def sample_loss_vector(mask_high: np.ndarray) -> CNAExpressionChangeVector:
    """Generates a vector controlling by what factor expression should change if a gene copy is lost.

    For each gene `g`:

    `NEW_EXPRESSION[g] = OLD_EXPRESSION[g] * GAIN_VECTOR[g]`

    Args:
        n_genes: for how many genes this vector should be generated
        rng: seed

    Returns:
        a vector controlling the expression change, shape (n_genes,)
    """
    changes = np.zeros((len(mask_high),))
    n_high = len(np.where(mask_high)[0])
    n_low = len(mask_high) - n_high
    #### WARNING: put some random seeds in here so it varies across patients
    loss_high = _sample_loss_vector_high(n_genes=n_high, rng=np.random.randint(100))
    loss_low = _sample_loss_vector_low(n_genes=n_low, rng=np.random.randint(100))

    changes[mask_high] = loss_high
    changes[~mask_high] = loss_low

    return changes


def perturb(
    original: CNAExpressionChangeVector, sigma: float, rng: Seed = 542
) -> CNAExpressionChangeVector:
    """Takes an expression changes vector and perturbs it by adding Gaussian noise.

    Args:
        original: expression changes vector, shape (n_genes,)
        sigma: controls the standard deviation of the noise
        rng: seed

    Returns:
        new expression changes vector
    """
    generator = np.random.default_rng(rng)
    noise = generator.normal(loc=0, scale=sigma, size=original.size)

    return np.maximum(original + noise, 0.0)


def _create_changes_vector(
    mask: GeneVector, change: GeneVector, fill: float = 1.0
) -> GeneVector:
    """Creates a change vector using the mask, the change value (to be used if the mask is true) and the fill
    value (to be used in places where the mask is false).

    For each gene `g`:
        OUTPUT[g] = change[g] if mask[g] is True else fill
    """
    return change * mask + fill * (~mask)


def _generate_masks(changes: GeneVector) -> Tuple[GeneVector, GeneVector]:
    """Generates boolean masks for the CNV changes.

    Args:
        changes: integer-valued vector, positive entries correspond to copy number gains,
            and negative to losses. Zeros correspond to no CNVs. Shape (n_genes,)

    Returns:
        boolean array, gain mask, shape (n_genes,)
        boolean array, loss mask, shape (n_genes,)
    """
    gain_mask = changes > 0
    loss_mask = changes < 0
    return gain_mask, loss_mask


def change_expression(
    expression: GeneVector,
    changes: GeneVector,
    gain_change: GeneVector,
    loss_change: GeneVector,
) -> GeneVector:
    """Changes the expression.

    Args:
        expression: base rate of expression
        changes: a vector with positive entries representing CNV gains, negative losses, zeros for no changes
        gain_change: expression change vector, used at places where gains were observed
        loss_change: expression change vector, used at places where losses were observed

    Note:
        For `gain_change` and `loss_change` you may wish to use the `perturb`ed (independently for each cell)
        version of the original vectors (see `gain_vector` and `loss_vector`).
    """
    gain_mask, loss_mask = _generate_masks(changes)

    gains_effect = _create_changes_vector(mask=gain_mask, change=gain_change)
    losses_effect = _create_changes_vector(mask=loss_mask, change=loss_change)

    return expression * gains_effect * losses_effect

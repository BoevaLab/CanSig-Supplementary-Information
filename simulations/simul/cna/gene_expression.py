"""How CNA changes affect gene expression."""

from typing import Tuple, List
import numpy as np
import anndata as ad
import scanpy as sc
from scipy.stats import truncnorm

from .types import GeneVector
from ..rand import Seed


# Type for storing numbers how CNA affects gene expression of a particular gene.
# Shape (n_genes,)
CNAExpressionChangeVector = GeneVector

### Getting truncated cauchy because standard cauchy gives weird results
# for FC as most gains become losses
def truncated_cauchy_rvs(loc=0, scale=1, a=-1, b=1, size=None, rng=123):
    """
    Generate random samples from a truncated Cauchy distribution.

    `loc` and `scale` are the location and scale parameters of the distribution.
    `a` and `b` define the interval [a, b] to which the distribution is to be
    limited.

    With the default values of the parameters, the samples are generated
    from the standard Cauchy distribution limited to the interval [-1, 1].
    """
    generator = np.random.default_rng(rng)
    ua = np.arctan((a - loc) / scale) / np.pi + 0.5
    ub = np.arctan((b - loc) / scale) / np.pi + 0.5
    U = generator.uniform(ua, ub, size=size)
    rvs = loc + scale * np.tan(np.pi * (U - 0.5))
    return rvs


def truncated_normal_rvs(loc=0, scale=1, a=-1, b=1, size=(1,), rng=123):
    """
    Generate random samples from a truncated normal distribution
    `loc` and `scale` are the location and scale parameters of the distribution.
    `a` and `b` define the interval [a, b] to which the distribution is to be
    limited.
    """
    ua = (a - loc) / scale
    ub = (b - loc) / scale
    rvs = truncnorm.rvs(ua, ub, loc=loc, scale=scale, size=size, random_state=rng)
    return rvs


def get_mask_high(adata: ad.AnnData, quantile: float = 0.9) -> np.ndarray:
    gex_mean = np.squeeze(np.asarray(adata.X.mean(axis=0)))
    qt = np.quantile(gex_mean, quantile)
    mask_high = gex_mean > qt
    return mask_high


def _sample_gain_vector_high(n_genes: int, rng: Seed = 123) -> np.ndarray:
    """Samples gain changes from a Cauchy distribution for highly expressed genes"""
    return truncated_cauchy_rvs(loc=1.5, scale=0.1, a=0, b=10, size=(n_genes,), rng=rng)


def _sample_loss_vector_high(n_genes: int, rng: Seed = 123) -> np.ndarray:
    """Samples gain changes from a Cauchy distribution for highly expressed genes"""
    return truncated_cauchy_rvs(loc=0.5, scale=0.1, a=0, b=10, size=(n_genes,), rng=rng)


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
        x.append(
            truncated_normal_rvs(
                a=0,
                b=10,
                loc=mu[mixture[i]],
                scale=sigma[mixture[i]],
                size=(1,),
                rng=np.random.randint(200),
            )
        )
    return np.array(x).ravel()


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
        x.append(
            truncated_normal_rvs(
                a=0,
                b=10,
                loc=mu[mixture[i]],
                scale=sigma[mixture[i]],
                size=(1,),
                rng=np.random.randint(200),
            )
        )
    return np.array(x).ravel()


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

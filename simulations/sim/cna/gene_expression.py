"""How CNA changes affect gene expression."""
from typing import Tuple
import numpy as np

from .types import GeneVector
from ..rand import Seed


# Type for storing numbers how CNA affects gene expression of a particular gene.
# Shape (n_genes,)
CNAExpressionChangeVector = GeneVector


def gain_vector(n_genes: int, rng: Seed = 123) -> CNAExpressionChangeVector:
    """Generates a vector controlling by what factor expression should change if a gene copy is gained.

    For each gene `g`:

    `NEW_EXPRESSION[g] = OLD_EXPRESSION[g] * GAIN_VECTOR[g]`

    Args:
        n_genes: for how many genes this vector should be generated
        rng: seed

    Returns:
        a vector controlling the expression change, shape (n_genes,)
    """
    generator = np.random.default_rng(rng)
    return generator.lognormal(mean=np.log(1.5), sigma=0.3, size=n_genes)


def loss_vector(n_genes: int, rng: Seed = 421) -> CNAExpressionChangeVector:
    """Generates a vector controlling by what factor expression should change if a gene copy is lost.

    For each gene `g`:

    `NEW_EXPRESSION[g] = OLD_EXPRESSION[g] / LOSS_VECTOR[g]`.

    Args:
        n_genes: for how many genes this vector should be generated
        rng: seed

    Returns:
        a vector controlling the expression change, shape (n_genes,)

    Todo:
        Check if this distribution is the right one -- we divide by these values, so there may be a risk
        that we should take 1/this. (I however doubt it).
    """
    generator = np.random.default_rng(rng)
    return generator.lognormal(mean=np.log(2), sigma=0.45, size=n_genes)


def perturb(original: CNAExpressionChangeVector, sigma: float, rng: Seed = 542) -> CNAExpressionChangeVector:
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


def _create_changes_vector(mask: GeneVector, change: GeneVector, fill: float = 1.0) -> GeneVector:
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
    expression: GeneVector, changes: GeneVector, gain_change: GeneVector, loss_change: GeneVector
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

    multiply = _create_changes_vector(mask=gain_mask, change=gain_change)
    divide = _create_changes_vector(mask=loss_mask, change=loss_change)

    return expression * multiply / divide

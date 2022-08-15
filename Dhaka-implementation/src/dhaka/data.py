import anndata as ad
import numpy as np


def example_data(
    n_cells: int = 1000,
    n_genes: int = 15000,
    dropout: float = 0.7,
    poisson_lambda: int = 4000,
    seed=532,
) -> ad.AnnData:
    """Generates example raw counts data."""
    assert n_genes > 0
    assert n_cells > 0
    assert 0 <= dropout < 1

    rng = np.random.default_rng(seed)

    counts = rng.poisson(poisson_lambda, size=(n_cells, n_genes))
    mask = rng.binomial(1, 1 - dropout, size=(n_cells, n_genes))

    return ad.AnnData(X=counts * mask)


def normalize(data: ad.AnnData) -> ad.AnnData:
    """
    """


    raise NotImplementedError

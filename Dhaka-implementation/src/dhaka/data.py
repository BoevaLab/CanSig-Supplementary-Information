import anndata as ad
import numpy as np

import torch
import torch.utils.data


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


def _to_array(array_like):
    if hasattr(array_like, "toarray"):
        return array_like.toarray()
    else:
        return np.asarray(array_like)


def _normalize(X: np.ndarray, total_expression: float) -> np.ndarray:
    """Normalizes the counts matrix such that the total expression per cell is fixed.

    Args:
        X: raw gene expression data, shape (n_cells, n_genes)

    Returns:
        array of the same shape as X, with entries rescaled so that each cell has exactly
        `total_expression` counts
    """
    summed = X.sum(axis=1)[:, None]
    assert summed.shape == (len(X), 1)
    return total_expression * X / summed


def _find_most_expressed_genes(X: np.ndarray, n_top_genes: int) -> np.ndarray:
    """Finds the genes with the highest average expression.

    Args:
        X: shape (n_cells, n_genes)
        n_top_genes: how many genes to select

    Returns:
        array of indices with genes to be selected. Length min(n_genes, n_top_genes).
            The indices are ordered simply by their value, rather than the associated mean gene expression.

    Example:
        Consider a dataset with one cell:
        X = np.asarray([[5, 2, 9]])

        Then
        _find_most_expressed_genes(X, 1) == np.array([2])
        _find_most_expressed_genes(X, 2) == np.array([0, 2])

        and for every K >= 3:
        _find_most_expressed_genes(X, K) == np.array([0, 1, 2])
    """
    mean_expression = np.mean(X, axis=0)
    assert mean_expression.shape == (X.shape[1],)

    sorted_expression = sorted(range(len(mean_expression)), key=lambda i: mean_expression[i], reverse=True)
    genes_to_take = sorted_expression[:n_top_genes]

    return np.asarray(sorted(genes_to_take))


def _select_most_expressed_genes(X: np.ndarray, n_top_genes: int) -> np.ndarray:
    """Selects the genes with the highest average expression.

    Args:
        X: shape (n_cells, n_genes)
        n_top_genes: how many genes to select

    Returns:
        array of shape (n_cells, n_new_genes), where n_new_genes = min(n_top_genes, n_genes)
    """
    genes = _find_most_expressed_genes(X, n_top_genes=n_top_genes)
    return X[:, genes]


def normalize(data: ad.AnnData, pseudocounts: float = 1.0, n_top_genes: int = 5000, total_expression: float = 1e6) -> np.ndarray:
    """Data normalization procedure.

    We tried to mimick the procedure described in Supplement 1.1. We decided to add pseudocounts, as there may be
    zero entries and log2 may return infinite values.
    """
    X = _to_array(data.X)  # Shape (n_cells, n_genes)

    # Normalize the cells so that each has `total_expression` counts
    X = _normalize(X, total_expression=total_expression)

    # Add pseudocounts, to deal with zero entries at logarithm stage
    X = X + pseudocounts

    # Select the most expressed genes
    X = _select_most_expressed_genes(X, n_top_genes=n_top_genes)

    # Return the logarithmied data
    return np.log2(X)


class NumpyArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray) -> None:
        assert X.shape == (X.shape[0], X.shape[1]), f"X should be two-dimensional array. Its shape is {X.shape}."

        self._X = X

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self._X[index])

    @property
    def n_features(self) -> int:
        return self._X.shape[1]

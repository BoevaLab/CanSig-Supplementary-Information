import numpy as np

import dhaka.data as dd


def _test_find_most_expressed_genes() -> None:
    gene_expression = np.asarray([0, 5, 1, 2, 4, 6])

    X = np.stack([
            gene_expression,
            gene_expression + 2,
            gene_expression ** 2,
        ])
    assert X.shape == (2, len(gene_expression))

    assert dd._find_most_expressed_genes(X, 1) == np.array([5])
    assert dd._find_most_expressed_genes(X, 2) == np.array([1, 5])
    assert dd._find_most_expressed_genes(X, 3) == np.array([1, 4, 5])


def _test_find_most_expressed_genes_2() -> None:
    X = np.asarray([[5, 2, 9]])

    assert dd._find_most_expressed_genes(X, 1) == np.array([2])
    assert dd._find_most_expressed_genes(X, 2) == np.array([0, 2])

    for k in [3, 5, 10]:
        assert dd._find_most_expressed_genes(X, k) == np.array([0, 1, 2])

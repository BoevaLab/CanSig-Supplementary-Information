"""Shared types between different CNA submodules."""
from typing import TypeVar

import numpy as np

# Type for storing values per gene, to vectorize operations. Shape (n_genes,)
GeneVector = np.ndarray

# Gene anchors
Anchors = TypeVar("Anchors")

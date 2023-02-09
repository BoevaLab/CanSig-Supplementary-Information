import logging
from typing import Literal  # pytype: disable=not-supported-yet
from typing import Optional

import anndata as an  # pytype: disable=import-error
import numpy as np
import pydantic  # pytype: disable=import-error
import scanpy as sc  # pytype: disable=import-error
from anndata import AnnData

_LOGGER = logging.getLogger(__name__)

_SupportedMetric = Literal[
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


class NeighborsGraphConfig(pydantic.BaseModel):
    """Settings for neighborhood graph computation.
    For description, see
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html
    """

    n_neighbors: int = pydantic.Field(default=15)
    n_pcs: Optional[int] = pydantic.Field(default=None)
    knn: bool = pydantic.Field(default=True)
    # TODO(Pawel): Check whether we can support other methods as well.
    method: Literal["umap"] = pydantic.Field(default="umap")
    metric: _SupportedMetric = pydantic.Field(default="euclidean")


class _LeidenBaseConfig(pydantic.BaseModel):
    nngraph: NeighborsGraphConfig = pydantic.Field(default_factory=NeighborsGraphConfig)
    random_state: int = pydantic.Field(default=0)
    directed: bool = pydantic.Field(default=True)
    use_weights: bool = pydantic.Field(default=True)
    n_iterations: int = pydantic.Field(default=-1)


class BinSearchSettings(pydantic.BaseModel):
    start: pydantic.PositiveFloat = pydantic.Field(
        default=1e-3, description="The minimal resolution."
    )
    end: pydantic.PositiveFloat = pydantic.Field(
        default=5.0, description="The maximal resolution."
    )
    epsilon: pydantic.PositiveFloat = pydantic.Field(
        default=1e-3,
        description="Controls the maximal number of iterations before throwing lookup "
        "error.",
    )

    @pydantic.validator("end")
    def validate_end_greater_than_start(cls, v, values, **kwargs) -> float:
        if v <= values["start"]:
            raise ValueError("In binary search end must be greater than start.")
        return v


class LeidenNClusterConfig(_LeidenBaseConfig):
    clusters: int = pydantic.Field(
        default=5, description="The number of clusters to be returned."
    )
    binsearch: BinSearchSettings = pydantic.Field(default_factory=BinSearchSettings)


class LeidenNCluster:
    def __init__(self, settings: LeidenNClusterConfig) -> None:
        self._settings = settings

    def fit_predict(self, adata: AnnData, key_added: str) -> np.ndarray:
        for offset in [0, 20_000, 30_000, 40_000]:
            points = _binary_search_leiden_resolution(
                adata,
                k=self._settings.clusters,
                key_added=key_added,
                random_state=self._settings.random_state + offset,
                directed=self._settings.directed,
                use_weights=self._settings.use_weights,
                start=self._settings.binsearch.start,
                end=self._settings.binsearch.end,
                _epsilon=self._settings.binsearch.epsilon,
            )
            if points is not None:
                break
        # In case that for multiple random seeds we didn't find a resolution that
        # matches the number of clusters, we raise a ValueError.
        else:
            raise ValueError(
                f"No resolution for the number of clusters {self._settings.clusters}"
                f" found."
            )

        return points.obs[key_added].astype(int).values


def _binary_search_leiden_resolution(
    adata: an.AnnData,
    k: int,
    start: float,
    end: float,
    key_added: str,
    random_state: int,
    directed: bool,
    use_weights: bool,
    _epsilon: float,
) -> Optional[an.AnnData]:
    """Binary search to get the resolution corresponding
    to the right k."""
    # We try the resolution which is in the middle of the interval
    res = 0.5 * (start + end)

    # Run Leiden clustering
    sc.tl.leiden(
        adata,
        resolution=res,
        key_added=key_added,
        random_state=random_state,
        directed=directed,
        use_weights=use_weights,
    )

    # Get the number of clusters found
    selected_k = adata.obs[key_added].nunique()
    if selected_k == k:
        _LOGGER.info(f"Binary seach finished with {res} resolution.")
        return adata

    # If the start and the end are too close (and there is no point in doing another
    # iteration), we raise an error that one can't find the required number of clusters
    if abs(end - start) < _epsilon * res:
        return None

    if selected_k > k:
        return _binary_search_leiden_resolution(
            adata,
            k=k,
            start=start,
            end=res,
            key_added=key_added,
            random_state=random_state,
            directed=directed,
            _epsilon=_epsilon,
            use_weights=use_weights,
        )
    else:
        return _binary_search_leiden_resolution(
            adata,
            k=k,
            start=res,
            end=end,
            key_added=key_added,
            random_state=random_state,
            directed=directed,
            _epsilon=_epsilon,
            use_weights=use_weights,
        )

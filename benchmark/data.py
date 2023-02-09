import logging
import pathlib as pl
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

import anndata
import numpy as np
from omegaconf import MISSING

_LOGGER = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_path: str = MISSING
    malignant_key: str = "malignant_key"
    malignant_cat: str = "malignant"


def read_anndata(
        data_path: Union[str, Path],
        malignant_only: bool = True,
        malignant_key: str = "malignant_key",
        malignant_cat: str = "malignant",
):
    _LOGGER.info(f"Loading adata from {data_path}.")
    adata = anndata.read_h5ad(data_path)
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X.astype(np.float32)
    if not malignant_only:
        return adata
    else:
        return adata[adata.obs[malignant_key] == malignant_cat, :].copy()
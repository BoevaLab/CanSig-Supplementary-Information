from pathlib import Path
from typing import Union

import anndata


def read_anndata(
    data_path: Union[str, Path],
    malignant_only: bool = True,
    malignant_key: str = "malignant_key",
    malignant_cat: str = "malignant",
):
    adata = anndata.read_h5ad(data_path)
    adata.layers["counts"] = adata.X.copy()
    if not malignant_only:
        return adata
    else:
        return adata[adata.obs[malignant_key] == malignant_cat, :].copy()

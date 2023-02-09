import logging
import pathlib as pl
from typing import Optional, List, Union

import pandas as pd
from anndata import AnnData

_LOGGER = logging.getLogger(__name__)


def save_latent(adata: AnnData, latent_key: str, dataset_name: str) -> None:
    latent_path = pl.Path("latent_codes")
    latent_path.mkdir(exist_ok=True)
    if latent_key in adata.obsm_keys():
        latent = pd.DataFrame(adata.obsm[latent_key], index=adata.obs_names)
        latent.to_csv(latent_path / f"{dataset_name}.csv")


def load_latent(path: Union[str, pl.Path]):
    _LOGGER.info(f"Loading latent codes from {path}.")
    return pd.read_csv(path, index_col=0)


def hydra_run_sweep():
    RUN_SWEEP_DEFAULTS = {
        "run": {"dir": "${results_path}/${model.name}/${run_dir:}"},
        "sweep": {
            "dir": "${results_path}",
            "subdir": "${model.name}/${run_dir:}",
        }
    }
    return RUN_SWEEP_DEFAULTS


def get_partition(gpu: bool) -> str:
    if gpu:
        return "gpu"
    return "compute"


def get_gres(gpu: bool) -> Optional[str]:
    if gpu:
        return "gpu:rtx2080ti:1"
    return None


def split_batches(adata: AnnData, batch_key: str) -> List[AnnData]:
    splits = []
    for batch in adata.obs[batch_key].cat.categories:
        splits.append(adata[adata.obs[batch_key] == batch].copy())
    return splits

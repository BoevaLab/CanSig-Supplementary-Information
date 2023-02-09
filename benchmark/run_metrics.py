import logging
import pathlib as pl
from dataclasses import dataclass, field
from typing import Dict, Any

import hydra
import pandas as pd
import scib_metrics.benchmark
from cansig.filesys import get_directory_name
from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import OmegaConf

from data import read_anndata, DataConfig
from utils import load_latent

_LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    method: str
    data: DataConfig = DataConfig()
    results_path: str = "/cluster/work/boeva/scRNAdata/benchmark/results"
    overwrite: bool = True
    hydra: Dict[str, Any] = field(default_factory=lambda: {"job": {"chdir": False}})


@dataclass
class Slurm(SlurmQueueConf):
    mem_gb: int = 16
    timeout_min: int = 720


OmegaConf.register_new_resolver("run_dir", get_directory_name)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="hydra/launcher", name="slurm", node=Slurm, provider="submitit_launcher")


@hydra.main(config_name="config", config_path=None, version_base="1.1")
def main(cfg: Config):
    results_path = pl.Path(cfg.results_path)
    method_path = results_path.joinpath(cfg.method)
    data_path = pl.Path(cfg.data.data_path)
    for run in method_path.iterdir():
        _LOGGER.info(f"Start computing metrics for {run}")
        dfs = []
        # Skip .submitit and logs.
        if run.name.startswith(".") or run.is_file():
            continue

        if not cfg.overwrite and run.joinpath("results.csv").is_file():
            _LOGGER.info(f"Metrics were already computed for {run} and overwrite was set"
                         f"to false. Therefore {run} is skipped.")
            continue

        latent_codes_dir = run.joinpath("latent_codes")

        for latent_path in latent_codes_dir.iterdir():
            latent = load_latent(latent_path)
            dataset = latent_path.stem
            adata = read_anndata(data_path.joinpath(f"{dataset}.h5ad"),
                                 malignant_only=True,
                                 malignant_cat=cfg.data.malignant_cat,
                                 malignant_key=cfg.data.malignant_key)
            if set(latent.index) != set(adata.obs_names):
                raise ValueError(
                    "The index of the latent space is different than the data's index")
            adata = adata[latent.index, :].copy()
            adata.obsm[dataset] = latent.values
            benchmarker = scib_metrics.benchmark.Benchmarker(adata,
                                                             batch_key="sample_id",
                                                             label_key="program",
                                                             embedding_obsm_keys=[
                                                                 dataset])
            _LOGGER.info("Running metrics.")
            benchmarker.benchmark()
            dfs.append(benchmarker.get_results(min_max_scale=False))
        results = pd.concat(dfs)
        results.drop_duplicates(inplace=True, keep="last")
        _LOGGER.info("Saving results.")
        results.to_csv(run.joinpath("results.csv"))

if __name__ == "__main__":
    main()

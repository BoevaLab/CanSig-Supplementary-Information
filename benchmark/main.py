import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import hydra
import pandas as pd
from cansig.filesys import get_directory_name
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from data import read_anndata
from metrics import run_metrics, MetricsConfig
from models import (
    ModelConfig,
    BBKNNConfig,
    SCVIConfig,
    ScanoramaConfig,
    CanSigConfig,
    HarmonyConfig,
    MNNConfig,
    DhakaConfig,
    ScanVIConfig,
    TrVAEpConfig,
    ScGENConfig,
    run_model, CombatConfig, DescConfig,
)
from utils import save_latent, plot_integration, get_gres, get_partition

DEFAULTS = {"hydra/launcher": "submitit_slurm"}


@dataclass
class Config:
    model: ModelConfig
    metric_config: MetricsConfig = MetricsConfig()
    malignant_key: str = "malignant_key"
    malignant_cat: str = "malignant"
    data_path: str = "/cluster/work/boeva/scRNAdata/benchmark/datasets"
    results_path: str = "/cluster/work/boeva/scRNAdata/benchmark/results"
    hydra: Dict[str, Any] = field(
        default_factory=lambda: {
            "run": {"dir": "${results_path}/${model.name}/${run_dir:}"},
            "sweep": {
                "dir": "${results_path}/${model.name}",
                "subdir": "${run_dir:}",
            },
            "launcher": {
                "mem_gb": 32,
                "timeout_min": 720,
                "cpus-per-task": 8,
                "partition": "${get_partition:${model.gpu}}",
                "gres": "${get_gres:${model.gpu}}",
            },
        }
    )


OmegaConf.register_new_resolver("run_dir", get_directory_name)
OmegaConf.register_new_resolver("get_gres", get_gres)
OmegaConf.register_new_resolver("get_partition", get_partition)

cs = ConfigStore.instance()

cs.store(name="config", node=Config)
cs.store(group="model", name="bbknn", node=BBKNNConfig)
cs.store(group="model", name="scvi", node=SCVIConfig)
cs.store(group="model", name="scanorama", node=ScanoramaConfig)
cs.store(group="model", name="cansig", node=CanSigConfig)
cs.store(group="model", name="harmony", node=HarmonyConfig)
cs.store(group="model", name="mnn", node=MNNConfig)
cs.store(group="model", name="combat", node=CombatConfig)
cs.store(group="model", name="desc", node=DescConfig)
cs.store(group="model", name="dhaka", node=DhakaConfig)
cs.store(group="model", name="scanvi", node=ScanVIConfig)
cs.store(group="model", name="trvaep", node=TrVAEpConfig)
cs.store(group="model", name="scgen", node=ScGENConfig)

_LOGGER = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path=None)
def main(cfg: Config):
    dfs = []
    dataset_path = Path(cfg.data_path)
    for dataset in sorted(list(dataset_path.iterdir())):
        _LOGGER.info(f"Processing {dataset.stem}")
        results = {}
        adata = read_anndata(
            dataset,
            cfg.model.malignant_only,
            malignant_key=cfg.malignant_key,
            malignant_cat=cfg.malignant_cat,
        )
        _LOGGER.info(f"Running integration model {cfg.model.name}.")
        adata, run_time = run_model(adata, cfg.model)

        results["run_time"] = run_time
        _LOGGER.info("Start running metrics.")
        results.update(
            run_metrics(adata, config=cfg.model, metric_config=cfg.metric_config)
        )
        dfs.append(pd.DataFrame(results, index=[dataset.stem]))
        _LOGGER.info("Plotting integration results.")
        plot_integration(
            adata,
            dataset_name=dataset.stem,
            batch_key=cfg.model.batch_key,
            group_key=cfg.metric_config.group_key,
        )
        _LOGGER.info("Saving latent spaces.")
        save_latent(adata, latent_key=cfg.model.latent_key, dataset_name=dataset.stem)

    pd.concat(dfs).to_csv("results.csv")


if __name__ == "__main__":
    main()

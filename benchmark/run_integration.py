import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import hydra
import pandas as pd
from cansig.filesys import get_directory_name
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from slurm_utils import SlurmCPU, SlurmGPU
from data import read_anndata, DataConfig
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
from utils import (save_latent, get_gres, get_partition,
                   hydra_run_sweep)

_LOGGER = logging.getLogger(__name__)

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig = DataConfig()
    results_path: str = "/cluster/work/boeva/scRNAdata/benchmark/results"
    hydra: Dict[str, Any] = field(default_factory=hydra_run_sweep)



OmegaConf.register_new_resolver("run_dir", get_directory_name)
OmegaConf.register_new_resolver("get_gres", get_gres)
OmegaConf.register_new_resolver("get_partition", get_partition)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="hydra/launcher", name="slurm_cpu", node=SlurmCPU, provider="submitit_launcher")
cs.store(group="hydra/launcher", name="slurm_gpu", node=SlurmGPU, provider="submitit_launcher")
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

@hydra.main(config_name="config", config_path=None, version_base="1.1")
def main(cfg: Config):
    run_times = {}
    dataset_path = Path(cfg.data.data_path)
    for dataset in sorted(list(dataset_path.iterdir())):
        _LOGGER.info(f"Processing {dataset.stem}")
        adata = read_anndata(
            dataset,
            cfg.model.malignant_only,
            malignant_key=cfg.data.malignant_key,
            malignant_cat=cfg.data.malignant_cat,
        )
        adata, run_time = run_model(adata, cfg.model)

        run_times[dataset.stem] = run_time
        save_latent(adata, latent_key=cfg.model.latent_key, dataset_name=dataset.stem)

    pd.DataFrame.from_dict(run_times, orient='index', columns=["run_time"]).to_csv("run_times.csv")


if __name__ == "__main__":
    main()

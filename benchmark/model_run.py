from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import hydra
import pandas as pd
from cansig.filesys import get_directory_name
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from data import read_anndata
from metrics import MetricsConfig, kbet, compute_neighbors
from models import (
    ModelConfig,
    BBKNNConfig,
    SCVIConfig,
    ScanoramaConfig,
    CanSigConfig,
    HarmonyConfig,
    run_model,
)
from utils import save_latent


@dataclass
class Config:
    model: ModelConfig
    metric_config: MetricsConfig = MetricsConfig()
    data_path: str = (
        "/cluster/work/boeva/scRNAdata/preprocessed/melanoma/"
        "2022-07-10_18-26-30/data/data.h5ad"
    )
    results_path: str = "/cluster/work/boeva/scRNAdata/benchmark/experiments"
    hydra: Dict[str, Any] = field(
        default_factory=lambda: {
            "run": {"dir": "${results_path}/${model.name}/${run_dir:}"}
        }
    )


OmegaConf.register_new_resolver("run_dir", get_directory_name)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="bbknn", node=BBKNNConfig)
cs.store(group="model", name="scvi", node=SCVIConfig)
cs.store(group="model", name="scanorama", node=ScanoramaConfig)
cs.store(group="model", name="cansig", node=CanSigConfig)
cs.store(group="model", name="harmony", node=HarmonyConfig)


@hydra.main(config_name="config", config_path=None)
def main(cfg: Config):
    dataset_path = Path(cfg.data_path)
    print(f"Processing {dataset_path.stem}", flush=True)
    print("Load data", flush=True)
    adata = read_anndata(dataset_path, cfg.model.malignant_only)
    print("Run model", flush=True)
    adata, run_time = run_model(adata, cfg.model)
    save_latent(adata, latent_key=cfg.model.latent_key, dataset_name="latent")
    print("Saved latent space", flush=True)
    compute_neighbors(
        adata,
        n_neighbors=cfg.metric_config.n_neighbors,
        latent_key=cfg.model.latent_key,
        name=cfg.model.name,
    )
    kbet_results = kbet(
        adata, config=cfg.metric_config, n_neighbors=cfg.metric_config.n_neighbors
    )
    print(f"kbet results: {kbet_results}", flush=True)
    pd.DataFrame.from_dict(kbet_results).to_csv("bket_result.csv")
    print("Save latent", flush=True)


if __name__ == "__main__":
    main()

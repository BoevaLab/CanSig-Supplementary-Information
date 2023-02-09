from dataclasses import dataclass
from typing import Optional

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf


@dataclass
class SlurmCPU(SlurmQueueConf):
    mem_gb: int = 16
    timeout_min: int = 720
    partition: str = "compute"
    gres: Optional[str] = None


@dataclass
class SlurmGPU(SlurmQueueConf):
    mem_gb: int = 16
    timeout_min: int = 720
    partition: str = "gpu"
    gres: Optional[str] = "gpu:rtx2080ti:1"
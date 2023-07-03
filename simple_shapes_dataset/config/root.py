from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING
from shimmer import ShimmerInfoConfig

from simple_shapes_dataset.config.dataset import Dataset
from simple_shapes_dataset.config.domains import DomainModules
from simple_shapes_dataset.config.training import Training
from simple_shapes_dataset.config.wandb import WanDB


@dataclass()
class Config:
    default_root_dir: Path = MISSING
    dataset: Dataset = field(default_factory=Dataset)
    training: Training = field(default_factory=Training)
    wandb: WanDB = field(default_factory=WanDB)
    domain_modules: DomainModules = field(default_factory=DomainModules)
    __shimmer__: ShimmerInfoConfig | None = None

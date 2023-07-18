from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING
from shimmer import ShimmerInfoConfig

from simple_shapes_dataset.config.dataset import Dataset
from simple_shapes_dataset.config.domains import DomainModules
from simple_shapes_dataset.config.global_workspace import GlobalWorkspace
from simple_shapes_dataset.config.logging import Logging
from simple_shapes_dataset.config.training import Training
from simple_shapes_dataset.config.visualization import Visualization
from simple_shapes_dataset.config.wandb import WanDB


@dataclass()
class Config:
    seed: int = 0
    default_root_dir: Path = MISSING
    dataset: Dataset = field(default_factory=Dataset)
    training: Training = field(default_factory=Training)
    wandb: WanDB = field(default_factory=WanDB)
    logging: Logging = field(default_factory=Logging)
    domain_modules: DomainModules = field(default_factory=DomainModules)
    global_workspace: GlobalWorkspace = field(default_factory=GlobalWorkspace)
    visualization: Visualization = field(default_factory=Visualization)
    __shimmer__: ShimmerInfoConfig | None = None

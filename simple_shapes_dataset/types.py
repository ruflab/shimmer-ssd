from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from shimmer import __version__ as shimmer_version


@dataclass
class RelativePathToRoot:
    pass


class DomainType(Enum):
    v = ("v", "default")
    attr = ("attr", "default")
    attr_legacy = ("attr", "legacy")
    attr_unpaired = ("attr", "unpaired")
    v_latents = ("v_latents", "default")
    v_latents_unpaired = ("v_latents", "unpaired")

    def __init__(self, kind: str, variant: str) -> None:
        self.kind = kind
        self.variant = variant


class Logging(BaseModel):
    log_train_medias_every_n_epochs: int | None = None
    log_val_medias_every_n_epochs: int | None = None


class Slurm(BaseModel):
    script: str
    run_workdir: str
    python_env: str
    command: str

    pre_modules: list[str]
    run_modules: list[str]
    args: dict[str, Any]

    grid_search: list[str] | None = None
    grid_search_exclude: list[dict[str, Any]] | None = None


class Optim(BaseModel):
    lr: float = 1e-4
    max_lr: float = 5e-3
    weight_decay: float = 0.0


class Training(BaseModel):
    batch_size: int = 1024
    num_workers: int = 0
    devices: int = 1
    accelerator: str = "gpu"

    fast_dev_run: bool = False
    max_steps: int = 100_000
    enable_progress_bar: bool = True

    precision: Any = 32
    float32_matmul_precision: str = "highest"

    optim: Optim


class ExploreVAE(BaseModel):
    checkpoint: str
    num_samples: int = 9
    range_start: int = -3
    range_end: int = 3
    wandb_name: str | None = None


class ExploreGW(BaseModel):
    domain: str
    checkpoint: str
    num_samples: int = 9
    range_start: int = -3
    range_end: int = 3
    wandb_name: str | None = None


class Visualization(BaseModel):
    explore_vae: ExploreVAE
    explore_gw: ExploreGW


class WanDB(BaseModel):
    enabled: bool = False
    save_dir: str
    project: str
    entity: str
    reinit: bool


class Exploration(BaseModel):
    gw_checkpoint: str


class Dataset(BaseModel):
    path: Path


class VisualModule(BaseModel):
    num_channels: int = 3
    ae_dim: int = 256
    latent_dim: int = 8
    beta: float = 0.1

    color_blind: bool = False


class AttributeModule(BaseModel):
    latent_dim: int = 10
    hidden_dim: int = 64
    beta: float = 0.05
    coef_categories: float = 1
    coef_attributes: float = 1

    nullify_rotation: bool = False


class TextModule(BaseModel):
    latent_filename: str = "latent"

    latent_dim: int = 24
    hidden_dim: int = 256
    beta: float = 0.1


class DomainModules(BaseModel):
    visual: VisualModule
    attribute: AttributeModule
    text: TextModule


class EncodersConfig(BaseModel):
    hidden_dim: int = 32
    n_layers: int = 3


class LoadedDomainConfig(BaseModel):
    checkpoint_path: str
    domain_type: DomainType
    args: dict[str, Any]


class DomainProportion(BaseModel):
    domains: list[str]
    proportion: float


class LossCoeffients(BaseModel):
    cycles: float = 1.0
    demi_cycles: float = 1.0
    translations: float = 1.0
    contrastives: float = 1.0
    kl: float = 0.0


class GlobalWorkspace(BaseModel):
    latent_dim: int = 12
    is_variational: bool = False
    var_contrastive_loss: bool = False
    domains: list[LoadedDomainConfig]
    encoders: EncodersConfig
    decoders: EncodersConfig
    sync_prop: float = 1.0
    domain_proportions: list[DomainProportion]
    loss_coefficients: LossCoeffients
    domain_args: dict[str, Any]


class ShimmerConfigInfo(BaseModel):
    version: str = shimmer_version
    debug: bool = False
    cli: Any = {}


class Config(BaseModel):
    seed: int = 0
    ood_seed: int | None = None
    default_root_dir: Path
    domain_checkpoint: LoadedDomainConfig | None = None
    presaved_latents_path: dict[str, str] = {}
    dataset: Dataset
    training: Training
    wandb: WanDB
    logging: Logging
    domain_modules: DomainModules
    global_workspace: GlobalWorkspace
    visualization: Visualization
    exploration: Exploration
    slurm: Slurm
    __shimmer__: ShimmerConfigInfo = ShimmerConfigInfo()

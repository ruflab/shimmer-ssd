from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from cfg_tools import ParsedModel
from pydantic import BaseModel, field_validator
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

    pre_modules: Sequence[str] = []
    run_modules: Sequence[str] = []
    args: Mapping[str, Any] = {}

    grid_search: Mapping[str, Sequence[Any]] | None = None
    grid_search_exclude: Sequence[Mapping[str, Any]] | None = None


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
    args: Mapping[str, Any] = {}

    @field_validator("domain_type", mode="before")
    @classmethod
    def validate_domain_type(cls, v: str) -> DomainType:
        """
        Use names instead of values to select enums
        """
        if not hasattr(DomainType, v):
            raise ValueError(f"{v} is not part of enum DomainType")
        return getattr(DomainType, v)


class DomainProportion(BaseModel):
    domains: Sequence[str]
    proportion: float


class LossCoeffients(BaseModel):
    cycles: float = 1.0
    demi_cycles: float = 1.0
    translations: float = 1.0
    contrastives: float = 0.01
    fused: float = 1.0


class GlobalWorkspace2Domains(BaseModel):
    latent_dim: int = 12
    bayesian_gw: bool = False
    use_fusion_model: bool = False
    selection_temperature: float = 0.2
    learn_logit_scale: bool = False
    vsepp_contrastive_loss: bool = False
    vsepp_margin: float = 0.2
    vsepp_measure: Literal["cosine", "order"] = "cosine"
    vsepp_max_violation: bool = True
    linear_domains: bool = False
    linear_domains_use_bias: bool = False
    domains: Sequence[LoadedDomainConfig]
    encoders: EncodersConfig = EncodersConfig()
    decoders: EncodersConfig = EncodersConfig()
    sync_prop: float = 1.0
    domain_proportions: Sequence[DomainProportion]
    loss_coefficients: LossCoeffients = LossCoeffients()
    domain_args: Mapping[str, Any]
    checkpoint: Path | None = None


class ShimmerConfigInfo(BaseModel):
    version: str = shimmer_version
    debug: bool = False
    cli: Any = {}


class Config(ParsedModel):
    seed: int = 0
    ood_seed: int | None = None
    default_root_dir: Path
    domain_checkpoint: LoadedDomainConfig | None = None
    presaved_latents_path: Mapping[str, str] = {}
    dataset: Dataset
    training: Training
    wandb: WanDB
    logging: Logging
    domain_modules: DomainModules
    global_workspace: GlobalWorkspace2Domains
    visualization: Visualization | None = None
    exploration: Exploration | None = None
    slurm: Slurm | None = None
    __shimmer__: ShimmerConfigInfo = ShimmerConfigInfo()

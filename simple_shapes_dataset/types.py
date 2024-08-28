from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from cfg_tools import ParsedModel
from pydantic import BaseModel, Field, field_validator
from shimmer.data.types import DomainDesc
from shimmer.version import __version__ as shimmer_version


@dataclass
class RelativePathToRoot:
    pass


class DomainType(Enum):
    v = DomainDesc("v", "v")
    v_latents = DomainDesc("v", "v_latents")
    attr = DomainDesc("attr", "attr")
    t = DomainDesc("t", "t")
    raw_text = DomainDesc("t", "raw_text")


class DomainModelVariantType(Enum):
    v = (DomainType.v, "default")
    attr = (DomainType.attr, "default")
    attr_legacy = (DomainType.attr, "legacy")
    attr_unpaired = (DomainType.attr, "unpaired")
    v_latents = (DomainType.v_latents, "default")
    v_latents_unpaired = (DomainType.v_latents, "unpaired")
    t = (DomainType.t, "default")

    def __init__(self, kind: DomainType, model_variant: str) -> None:
        self.kind = kind
        self.model_variant = model_variant


class Logging(BaseModel):
    filter_images: Sequence[str] | None = None
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
    max_train_size: int = -1


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
    vocab_path: str
    merges_path: str
    seq_length: int = 64
    vocab_size: int = 822

    latent_dim: int = 64
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
    domain_type: DomainModelVariantType
    args: Mapping[str, Any] = {}

    @field_validator("domain_type", mode="before")
    @classmethod
    def validate_domain_type(cls, v: str) -> DomainModelVariantType:
        """
        Use names instead of values to select enums
        """
        return DomainModelVariantType[v]


class DomainProportion(BaseModel):
    domains: Sequence[str]
    proportion: float


class LossCoeffients(BaseModel):
    cycles: float = 1.0
    demi_cycles: float = 1.0
    translations: float = 1.0
    contrastives: float = 0.01
    fused: float = 1.0


class GlobalWorkspace(BaseModel):
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
    title: str | None = Field(None, alias="-t")
    desc: str | None = Field(None, alias="-d")
    domain_modules: DomainModules
    global_workspace: GlobalWorkspace
    visualization: Visualization | None = None
    exploration: Exploration | None = None
    slurm: Slurm | None = None
    __shimmer__: ShimmerConfigInfo = ShimmerConfigInfo()

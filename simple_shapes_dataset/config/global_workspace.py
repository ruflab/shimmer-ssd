from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import MISSING


class DomainType(Enum):
    v = ("v", "default")
    attr = ("attr", "default")
    attr_unpaired = ("attr", "unpaired")
    v_latents = ("v_latents", "default")
    v_latents_unpaired = ("v_latents", "unpaired")

    def __init__(self, kind: str, variant: str) -> None:
        self.kind = kind
        self.variant = variant


@dataclass
class EncodersConfig:
    hidden_dim: int = 32
    n_layers: int = 3


@dataclass
class LoadedDomainConfig:
    checkpoint_path: Path = MISSING
    domain_type: DomainType = MISSING


@dataclass
class DomainProportion:
    domains: list[str] = MISSING
    proportion: float = MISSING


@dataclass
class LossCoeffients:
    cycles: float = 1.0
    demi_cycles: float = 1.0
    translations: float = 1.0
    contrastives: float = 1.0
    kl: float = 0.0


@dataclass
class GlobalWorkspace:
    latent_dim: int = 12
    is_variational: bool = False
    var_contrastive_loss: bool = False
    domains: list[LoadedDomainConfig] = field(default_factory=list)
    encoders: EncodersConfig = field(default_factory=EncodersConfig)
    decoders: EncodersConfig = field(default_factory=EncodersConfig)
    sync_prop: float = 1.0
    domain_proportions: list[DomainProportion] = field(default_factory=list)
    loss_coefficients: LossCoeffients = field(default_factory=LossCoeffients)
    domain_args: dict[str, Any] = field(default_factory=dict)

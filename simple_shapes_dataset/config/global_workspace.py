from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from omegaconf import MISSING


class DomainType(StrEnum):
    attr = "attr"
    v = "v"
    v_latents = "v_latents"


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
    cycles: int = 1
    demi_cycles: int = 1
    translations: int = 1
    contrastives: int = 1
    kl: int = 1


@dataclass
class GlobalWorkspace:
    latent_dim: int = 12
    domains: list[LoadedDomainConfig] = field(default_factory=list)
    encoders: EncodersConfig = field(default_factory=EncodersConfig)
    decoders: EncodersConfig = field(default_factory=EncodersConfig)
    domain_proportions: list[DomainProportion] = field(default_factory=list)
    loss_coefficients: LossCoeffients = field(default_factory=LossCoeffients)
    domain_args: dict[str, Any] = field(default_factory=dict)

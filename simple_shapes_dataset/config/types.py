from dataclasses import dataclass, field
from enum import StrEnum


class VAEType(StrEnum):
    sigma = "sigma"
    beta = "beta"


@dataclass
class VisualModule:
    num_channels: int = 3
    ae_dim: int = 64
    latent_dim: int = 12
    beta: float = 1
    vae_type: VAEType = VAEType.beta


@dataclass
class AttributeModule:
    latent_dim: int = 12
    hidden_dim: int = 32
    n_layers: int = 3
    beta: float = 1
    vae_type: VAEType = VAEType.beta


@dataclass
class DomainModules:
    visual: VisualModule = field(default_factory=VisualModule)
    attribute: AttributeModule = field(default_factory=AttributeModule)


@dataclass
class Config:
    domain_modules: DomainModules = field(default_factory=DomainModules)

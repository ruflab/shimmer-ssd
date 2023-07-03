from dataclasses import dataclass, field

from shimmer.modules.vae import VAEType


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

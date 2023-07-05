from dataclasses import dataclass, field


@dataclass
class VisualModule:
    num_channels: int = 3
    ae_dim: int = 64
    latent_dim: int = 12
    beta: float = 1


@dataclass
class AttributeModule:
    latent_dim: int = 12
    hidden_dim: int = 64
    beta: float = 0.01


@dataclass
class DomainModules:
    visual: VisualModule = field(default_factory=VisualModule)
    attribute: AttributeModule = field(default_factory=AttributeModule)

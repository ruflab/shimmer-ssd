from dataclasses import dataclass, field


@dataclass
class VisualModule:
    num_channels: int = 3
    ae_dim: int = 256
    latent_dim: int = 8
    beta: float = 0.1

    color_blind: bool = False


@dataclass
class AttributeModule:
    latent_dim: int = 10
    hidden_dim: int = 64
    beta: float = 0.05
    coef_categories: float = 1
    coef_attributes: float = 1

    nullify_rotation: bool = False


@dataclass
class DomainModules:
    visual: VisualModule = field(default_factory=VisualModule)
    attribute: AttributeModule = field(default_factory=AttributeModule)

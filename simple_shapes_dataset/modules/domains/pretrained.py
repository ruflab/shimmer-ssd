from collections.abc import Sequence
from pathlib import Path

import torch
from shimmer import DomainModule, GWInterface, GWInterfaceBase, VariationalGWInterface
from torch.nn import Linear

from simple_shapes_dataset.ckpt_migrations import (
    attribute_mod_migrations,
    migrate_model,
    visual_mod_migrations,
)
from simple_shapes_dataset.errors import ConfigurationError
from simple_shapes_dataset.modules.domains.attribute import (
    AttributeDomainModule,
    AttributeLegacyDomainModule,
    AttributeWithUnpairedDomainModule,
)
from simple_shapes_dataset.modules.domains.visual import (
    VisualDomainModule,
    VisualLatentDomainModule,
    VisualLatentDomainWithUnpairedModule,
)
from simple_shapes_dataset.types import DomainType, LoadedDomainConfig


class GWLinearInterface(GWInterfaceBase):
    def __init__(
        self, domain_module: DomainModule, workspace_dim: int, bias: bool = False
    ) -> None:
        super().__init__(domain_module, workspace_dim)

        self.encoder = Linear(domain_module.latent_dim, workspace_dim, bias=bias)
        self.decoder = Linear(workspace_dim, domain_module.latent_dim, bias=bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def load_pretrained_module(
    root_path: Path,
    domain: LoadedDomainConfig,
) -> DomainModule:
    domain_checkpoint = root_path / domain.checkpoint_path
    match domain.domain_type:
        case DomainType.v:
            migrate_model(domain_checkpoint, visual_mod_migrations)
            module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )

        case DomainType.v_latents:
            migrate_model(domain_checkpoint, visual_mod_migrations)
            v_module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )
            module = VisualLatentDomainModule(v_module)

        case DomainType.v_latents_unpaired:
            migrate_model(domain_checkpoint, visual_mod_migrations)
            v_module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )
            module = VisualLatentDomainWithUnpairedModule(v_module)

        case DomainType.attr:
            migrate_model(domain_checkpoint, attribute_mod_migrations)
            module = AttributeDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )

        case DomainType.attr_unpaired:
            migrate_model(domain_checkpoint, attribute_mod_migrations)
            module = AttributeWithUnpairedDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )

        case DomainType.attr_legacy:
            module = AttributeLegacyDomainModule()

        case _:
            raise ConfigurationError(f"Unknown domain type {domain.domain_type.name}")
    return module


def load_pretrained_domain(
    default_root_dir: Path,
    domain: LoadedDomainConfig,
    workspace_dim: int,
    encoder_hidden_dim: int,
    encoder_n_layers: int,
    decoder_hidden_dim: int,
    decoder_n_layers: int,
    is_variational: bool = False,
    is_linear: bool = False,
    bias: bool = False,
) -> tuple[DomainModule, GWInterfaceBase]:
    module = load_pretrained_module(default_root_dir, domain)

    interface: GWInterfaceBase
    if is_linear:
        interface = GWLinearInterface(module, workspace_dim=workspace_dim, bias=bias)
    elif is_variational:
        interface = VariationalGWInterface(
            module,
            workspace_dim=workspace_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_n_layers=encoder_n_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_n_layers=decoder_n_layers,
        )
    else:
        interface = GWInterface(
            module,
            workspace_dim=workspace_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_n_layers=encoder_n_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_n_layers=decoder_n_layers,
        )

    return module, interface


def load_pretrained_domains(
    default_root_dir: Path,
    domains: Sequence[LoadedDomainConfig],
    workspace_dim: int,
    encoders_hidden_dim: int,
    encoders_n_layers: int,
    decoders_hidden_dim: int,
    decoders_n_layers: int,
    is_variational: bool = False,
    is_linear: bool = False,
    bias: bool = False,
) -> tuple[dict[str, DomainModule], dict[str, GWInterfaceBase]]:
    modules: dict[str, DomainModule] = {}
    interfaces: dict[str, GWInterfaceBase] = {}
    for domain in domains:
        if domain.domain_type.kind in modules:
            raise ConfigurationError("Cannot load multiple domains of the same kind.")
        model, interface = load_pretrained_domain(
            default_root_dir,
            domain,
            workspace_dim,
            encoders_hidden_dim,
            encoders_n_layers,
            decoders_hidden_dim,
            decoders_n_layers,
            is_variational,
            is_linear,
            bias,
        )
        modules[domain.domain_type.kind] = model
        interfaces[domain.domain_type.kind] = interface
    return modules, interfaces

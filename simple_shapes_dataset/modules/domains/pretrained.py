from collections.abc import Sequence
from pathlib import Path

from shimmer import DomainModule, GWInterface, GWInterfaceBase, VariationalGWInterface

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
            print(domain_checkpoint, visual_mod_migrations)
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
) -> tuple[DomainModule, GWInterfaceBase]:
    module = load_pretrained_module(default_root_dir, domain)

    interface_cls = VariationalGWInterface if is_variational else GWInterface

    return module, interface_cls(
        module,
        workspace_dim=workspace_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_n_layers=encoder_n_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_n_layers=decoder_n_layers,
    )


def load_pretrained_domains(
    default_root_dir: Path,
    domains: Sequence[LoadedDomainConfig],
    workspace_dim: int,
    encoders_hidden_dim: int,
    encoders_n_layers: int,
    decoders_hidden_dim: int,
    decoders_n_layers: int,
    is_variational: bool = False,
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
        )
        modules[domain.domain_type.kind] = model
        interfaces[domain.domain_type.kind] = interface
    return modules, interfaces

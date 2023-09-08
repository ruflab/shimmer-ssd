from typing import cast

from shimmer.modules.domain import DomainDescription, DomainModule

from simple_shapes_dataset.config.global_workspace import (
    DomainType,
    LoadedDomainConfig,
)
from simple_shapes_dataset.errors import ConfigurationError
from simple_shapes_dataset.modules.domains.attribute import (
    AttributeDomainModule,
    AttributeWithUnpairedDomainModule,
)
from simple_shapes_dataset.modules.domains.visual import (
    VisualDomainModule,
    VisualLatentDomainModule,
    VisualLatentDomainWithUnpairedModule,
)
from simple_shapes_dataset.modules.vae import RAEEncoder


def load_pretrained_module(
    domain: LoadedDomainConfig,
) -> tuple[DomainModule, int]:
    match domain.domain_type:
        case DomainType.v:
            module = cast(
                VisualDomainModule,
                VisualDomainModule.load_from_checkpoint(
                    domain.checkpoint_path
                ),
            )
            latent_dim = cast(RAEEncoder, module.vae.encoder).z_dim

        case DomainType.v_latents:
            v_module = cast(
                VisualDomainModule,
                VisualDomainModule.load_from_checkpoint(
                    domain.checkpoint_path
                ),
            )
            module = VisualLatentDomainModule(v_module)

            latent_dim = cast(RAEEncoder, v_module.vae.encoder).z_dim

        case DomainType.v_latents_unpaired:
            v_module = cast(
                VisualDomainModule,
                VisualDomainModule.load_from_checkpoint(
                    domain.checkpoint_path
                ),
            )
            module = VisualLatentDomainWithUnpairedModule(v_module)

            latent_dim = cast(RAEEncoder, v_module.vae.encoder).z_dim + 1

        case DomainType.attr:
            module = cast(
                AttributeDomainModule,
                AttributeDomainModule.load_from_checkpoint(
                    domain.checkpoint_path
                ),
            )
            latent_dim = module.latent_dim

        case DomainType.attr_unpaired:
            module = cast(
                AttributeWithUnpairedDomainModule,
                AttributeWithUnpairedDomainModule.load_from_checkpoint(
                    domain.checkpoint_path
                ),
            )
            latent_dim = module.latent_dim
        case _:
            raise ConfigurationError(
                f"Unknown domain type {domain.domain_type.name}"
            )
    return module, latent_dim


def load_pretrained_domain(
    domain: LoadedDomainConfig,
    encoder_hidden_dim: int,
    encoder_n_layers: int,
    decoder_hidden_dim: int,
    decoder_n_layers: int,
) -> DomainDescription:
    module, latent_dim = load_pretrained_module(domain)

    return DomainDescription(
        module=module,
        latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_n_layers=encoder_n_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_n_layers=decoder_n_layers,
    )


def load_pretrained_domains(
    domains: list[LoadedDomainConfig],
    encoders_hidden_dim: int,
    encoders_n_layers: int,
    decoders_hidden_dim: int,
    decoders_n_layers: int,
) -> dict[str, DomainDescription]:
    modules: dict[str, DomainDescription] = {}
    for domain in domains:
        if domain.domain_type.kind in modules:
            raise ConfigurationError(
                "Cannot load multiple domains of the same kind."
            )
        modules[domain.domain_type.kind] = load_pretrained_domain(
            domain,
            encoders_hidden_dim,
            encoders_n_layers,
            decoders_hidden_dim,
            decoders_n_layers,
        )
    return modules

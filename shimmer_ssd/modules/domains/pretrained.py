from collections.abc import Mapping, Sequence
from pathlib import Path

from shimmer import DomainModule, GWDecoder, GWEncoder
from torch.nn import Linear, Module

from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.ckpt_migrations import (
    migrate_model,
)
from shimmer_ssd.errors import ConfigurationError
from shimmer_ssd.modules.domains.attribute import (
    AttributeDomainModule,
    AttributeLegacyDomainModule,
    AttributeWithUnpairedDomainModule,
)
from shimmer_ssd.modules.domains.text import GRUTextDomainModule, Text2Attr
from shimmer_ssd.modules.domains.visual import (
    VisualDomainModule,
    VisualLatentDomainModule,
    VisualLatentDomainWithUnpairedModule,
)
from shimmer_ssd.types import DomainModelVariantType, LoadedDomainConfig


def load_pretrained_module(
    root_path: Path,
    domain: LoadedDomainConfig,
) -> DomainModule:
    domain_checkpoint = root_path / domain.checkpoint_path
    module: DomainModule
    match domain.domain_type:
        case DomainModelVariantType.v:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_ssd" / "migrations" / "visual_mod",
            )
            module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )

        case DomainModelVariantType.v_latents:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_ssd" / "migrations" / "visual_mod",
            )
            v_module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )
            module = VisualLatentDomainModule(v_module)

        case DomainModelVariantType.v_latents_unpaired:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_ssd" / "migrations" / "visual_mod",
            )
            v_module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )
            module = VisualLatentDomainWithUnpairedModule(v_module)

        case DomainModelVariantType.attr:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_ssd" / "migrations" / "attr_mod",
            )
            module = AttributeDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )

        case DomainModelVariantType.attr_unpaired:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_ssd" / "migrations" / "attr_mod",
            )
            module = AttributeWithUnpairedDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )

        case DomainModelVariantType.attr_legacy:
            module = AttributeLegacyDomainModule()

        case DomainModelVariantType.t:
            module = GRUTextDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args, strict=False
            )
            # Freezes the projector
            # module.embeddings.requires_grad_(False)
            # module.projector.requires_grad_(False)

        case DomainModelVariantType.t_attr:
            assert (
                "text_model_path" in domain.args
            ), 'add "text_model_path" to the domain\'s args.'
            text_model = GRUTextDomainModule.load_from_checkpoint(
                root_path / domain.args["text_model_path"],
                **domain.args.get("t_args", {}),
            )
            module = Text2Attr.load_from_checkpoint(
                domain_checkpoint,
                text_model=text_model,
                **domain.args.get("model_args", {}),
            )

        case _:
            raise ConfigurationError(f"Unknown domain type {domain.domain_type.name}")
    return module


def load_pretrained_domain(
    default_root_dir: Path,
    domain: LoadedDomainConfig,
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[DomainModelVariantType, int],
    encoders_n_layers: int | Mapping[DomainModelVariantType, int],
    decoders_hidden_dim: int | Mapping[DomainModelVariantType, int],
    decoders_n_layers: int | Mapping[DomainModelVariantType, int],
    is_linear: bool = False,
    bias: bool = False,
) -> tuple[DomainModule, Module, Module]:
    module = load_pretrained_module(default_root_dir, domain)
    encoder_hidden_dim = (
        encoders_hidden_dim
        if isinstance(encoders_hidden_dim, int)
        else encoders_hidden_dim[domain.domain_type]
    )
    decoder_hidden_dim = (
        decoders_hidden_dim
        if isinstance(decoders_hidden_dim, int)
        else decoders_hidden_dim[domain.domain_type]
    )
    encoder_n_layers = (
        encoders_n_layers
        if isinstance(encoders_n_layers, int)
        else encoders_n_layers[domain.domain_type]
    )
    decoder_n_layers = (
        decoders_n_layers
        if isinstance(decoders_n_layers, int)
        else decoders_n_layers[domain.domain_type]
    )

    gw_encoder: Module
    gw_decoder: Module
    if is_linear:
        gw_encoder = Linear(module.latent_dim, workspace_dim, bias=bias)
        gw_decoder = Linear(workspace_dim, module.latent_dim, bias=bias)
    else:
        gw_encoder = GWEncoder(
            module.latent_dim, encoder_hidden_dim, workspace_dim, encoder_n_layers
        )
        gw_decoder = GWDecoder(
            workspace_dim, decoder_hidden_dim, module.latent_dim, decoder_n_layers
        )

    return module, gw_encoder, gw_decoder


def load_pretrained_domains(
    default_root_dir: Path,
    domains: Sequence[LoadedDomainConfig],
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[DomainModelVariantType, int],
    encoders_n_layers: int | Mapping[DomainModelVariantType, int],
    decoders_hidden_dim: int | Mapping[DomainModelVariantType, int],
    decoders_n_layers: int | Mapping[DomainModelVariantType, int],
    is_linear: bool = False,
    bias: bool = False,
) -> tuple[dict[str, DomainModule], dict[str, Module], dict[str, Module]]:
    modules: dict[str, DomainModule] = {}
    gw_encoders: dict[str, Module] = {}
    gw_decoders: dict[str, Module] = {}
    for domain in domains:
        if domain.domain_type.kind.value.kind in modules:
            raise ConfigurationError("Cannot load multiple domains of the same kind.")
        model, encoder, decoder = load_pretrained_domain(
            default_root_dir,
            domain,
            workspace_dim,
            encoders_hidden_dim,
            encoders_n_layers,
            decoders_hidden_dim,
            decoders_n_layers,
            is_linear,
            bias,
        )
        modules[domain.domain_type.kind.value.kind] = model
        gw_encoders[domain.domain_type.kind.value.kind] = encoder
        gw_decoders[domain.domain_type.kind.value.kind] = decoder
    return modules, gw_encoders, gw_decoders

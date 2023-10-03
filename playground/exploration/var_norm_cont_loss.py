import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from shimmer import load_structured_config
from shimmer.modules.global_workspace import VariationalGlobalWorkspace
from shimmer.modules.gw_module import VariationalGWModule

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.pre_process import (color_blind_visual_domain,
                                                       nullify_attribute_rotation)
from simple_shapes_dataset.modules.domains.pretrained import load_pretrained_domains


def put_on_device(
    samples: Mapping[frozenset[str], Mapping[str, Any]], device: torch.device
) -> dict[frozenset[str], dict[str, Any]]:
    new_samples: dict[frozenset[str], dict[str, Any]] = {}
    for domain_names, domains in samples.items():
        new_domains: dict[str, Any] = {}
        for domain_name, domain in domains.items():
            if isinstance(domain, torch.Tensor):
                new_domains[domain_name] = domain.to(device)
            elif isinstance(domain, list):
                new_domains[domain_name] = [d.to(device) for d in domain]
            else:
                raise ValueError(f"Unknown type {type(domain)}")
        new_samples[domain_names] = new_domains
    return new_samples


def main():
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["exp_var_cont"],
        debug_mode=DEBUG_MODE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_proportion = {
        frozenset(item.domains): item.proportion
        for item in config.global_workspace.domain_proportions
    }

    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.attribute.nullify_rotation:
        logging.info("Nullifying rotation in the attr domain.")
        additional_transforms["attr"] = [nullify_attribute_rotation]
    if config.domain_modules.visual.color_blind:
        logging.info("v domain will be color blind.")
        additional_transforms["v"] = [color_blind_visual_domain]

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_proportion,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.global_workspace.domain_args,
        additional_transforms=additional_transforms,
    )

    domain_description = load_pretrained_domains(
        config.global_workspace.domains,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )

    domain_module = cast(
        VariationalGlobalWorkspace,
        VariationalGlobalWorkspace.load_from_checkpoint(
            config.exploration.gw_checkpoint,
            domain_descriptions=domain_description,
        ),
    )
    domain_module.eval().freeze()
    domain_module.to(device)
    gw_mod = cast(VariationalGWModule, domain_module.gw_mod)

    val_samples = put_on_device(data_module.get_samples("val", 1), device)
    encoded_samples = domain_module.encode_domains(val_samples)[
        frozenset(["v_latents", "attr"])
    ]
    v_paired = encoded_samples["v_latents"][0]
    attr_paired = encoded_samples["attr"][0]
    v_test = torch.randn(64, 13).to(device)
    attr_test = torch.randn(64, 13).to(device)
    v_test[:, :12] = v_paired[None, :12]
    attr_test[:, :12] = attr_paired[None, :12]
    gw_states_means, gw_stats_std = gw_mod.encoded_distribution(
        {"v_latents": v_test, "attr": attr_test}
    )
    v_gw_var = (0.5 * gw_stats_std["v_latents"]).exp()
    attr_gw_var = (0.5 * gw_stats_std["attr"]).exp()
    print("ok")


if __name__ == "__main__":
    main()

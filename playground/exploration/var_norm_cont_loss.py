import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch
from shimmer.modules.global_workspace import GlobalWorkspaceBayesian

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.ckpt_migrations import (
    migrate_model,
)
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.domain import get_default_domains
from simple_shapes_dataset.dataset.pre_process import (
    color_blind_visual_domain,
    nullify_attribute_rotation,
)
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
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["exp_var_cont.yaml"],
        debug_mode=DEBUG_MODE,
    )

    if config.exploration is None:
        raise ValueError("Exploration config should be set for this script")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_proportion = {
        frozenset(item.domains): item.proportion
        for item in config.global_workspace.domain_proportions
    }

    domain_classes = get_default_domains(
        {domain for domains in domain_proportion for domain in domains}
    )

    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.attribute.nullify_rotation:
        logging.info("Nullifying rotation in the attr domain.")
        additional_transforms["attr"] = [nullify_attribute_rotation]
    if config.domain_modules.visual.color_blind:
        logging.info("v domain will be color blind.")
        additional_transforms["v"] = [color_blind_visual_domain]

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_classes,
        domain_proportion,
        batch_size=config.training.batch_size,
        max_size=config.dataset.max_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.global_workspace.domain_args,
        additional_transforms=additional_transforms,
    )

    domain_description, gw_encoders, gw_decoders = load_pretrained_domains(
        config.default_root_dir,
        config.global_workspace.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )

    ckpt_path = config.default_root_dir / config.exploration.gw_checkpoint
    migrate_model(ckpt_path, PROJECT_DIR / "migrations" / "gw")
    domain_module = GlobalWorkspaceBayesian.load_from_checkpoint(
        ckpt_path,
        domain_mods=domain_description,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )
    domain_module.eval().freeze()
    domain_module.to(device)
    gw_mod = domain_module.gw_mod

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
    gw_states_means = gw_mod.encode({"v_latents": v_test, "attr": attr_test})
    gw_states_std = gw_mod.precisions
    v_gw_var = (0.5 * gw_states_std["v_latents"]).exp()  # noqa: F841
    attr_gw_var = (0.5 * gw_states_std["attr"]).exp()  # noqa: F841
    print(gw_states_means)


if __name__ == "__main__":
    main()

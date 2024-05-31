import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from shimmer import (
    GlobalWorkspaceBayesian,
    GWLossesBayesian,
)

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

    batch_size = 128
    n_rep = 128
    n_unpaired = 8

    val_samples = put_on_device(data_module.get_samples("val", batch_size), device)
    encoded_samples = domain_module.encode_domains(val_samples)[
        frozenset(["v_latents", "attr"])
    ]
    v1 = (
        encoded_samples["v_latents"]
        .unsqueeze(1)
        .expand((batch_size, n_rep, -1))
        .clone()
    )
    attr1 = encoded_samples["attr"].unsqueeze(1).expand((batch_size, n_rep, -1)).clone()
    v_unpaired = torch.randn(batch_size, n_rep, n_unpaired).to(device)
    attr_unpaired = torch.randn(batch_size, n_rep, n_unpaired).to(device)
    v2 = v1[:]
    attr2 = attr1[:]
    v2[:, :, -n_unpaired:] = v_unpaired
    attr2[:, :, -n_unpaired:] = attr_unpaired
    gw_states_means = gw_mod.encode(
        {
            "v_latents": v2.reshape(batch_size * n_rep, -1),
            "attr": attr2.reshape(batch_size * n_rep, -1),
        }
    )
    gw_states_std = gw_mod.precisions

    actual_std_attr = (
        gw_states_means["attr"].reshape(batch_size, n_rep, -1).std(dim=1).mean(dim=0)
    )
    actual_std_v = (
        gw_states_means["v_latents"]
        .reshape(batch_size, n_rep, -1)
        .std(dim=1)
        .mean(dim=0)
    )
    print(f"Actual std attr: {actual_std_attr}")
    print(f"Actual std v: {actual_std_v}")

    predicted_std_attr = gw_states_std["attr"].exp().mean(dim=0)
    predicted_std_v = gw_states_std["v_latents"].exp().mean(dim=0)

    print(f"Predicted std attr: {predicted_std_attr}")
    print(f"Predicted std v: {predicted_std_v}")

    contrastive_fn = cast(GWLossesBayesian, domain_module.loss_mod).contrastive_fn
    assert contrastive_fn is not None

    norm1 = 1.0 + gw_states_std["attr"].exp() + gw_states_std["v_latents"].exp()
    cont_loss1 = contrastive_fn(
        gw_states_means["attr"] / norm1, gw_states_means["v_latents"] / norm1
    )

    norm2 = 1.0 + actual_std_attr.exp() + actual_std_v.exp()
    cont_loss2 = contrastive_fn(
        gw_states_means["attr"] / norm2, gw_states_means["v_latents"] / norm2
    )

    print(f"Contrastive loss 1: {cont_loss1}")
    print(f"Contrastive loss 2: {cont_loss2}")

    print("ok")


if __name__ == "__main__":
    main()

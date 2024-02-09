import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from shimmer import load_structured_config
from shimmer.modules.global_workspace import VariationalGlobalWorkspace
from shimmer.modules.gw_module import VariationalGWModule
from shimmer.modules.losses import (
    VariationalGWLosses,
    contrastive_loss_with_uncertainty,
)

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.pre_process import (
    color_blind_visual_domain,
    nullify_attribute_rotation,
)
from simple_shapes_dataset.modules.domains.pretrained import load_pretrained_domains
from simple_shapes_dataset.types import Config


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
    gw_states_means, gw_states_std = gw_mod.encoded_distribution(
        {
            "v_latents": v2.reshape(batch_size * n_rep, -1),
            "attr": attr2.reshape(batch_size * n_rep, -1),
        }
    )

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

    logit_scale = cast(VariationalGWLosses, domain_module.loss_mod).logit_scale

    cont_loss1 = contrastive_loss_with_uncertainty(
        gw_states_means["attr"],
        gw_states_std["attr"],
        gw_states_means["v_latents"],
        gw_states_std["v_latents"],
        logit_scale,
    )

    cont_loss2 = contrastive_loss_with_uncertainty(
        gw_states_means["attr"],
        actual_std_attr,
        gw_states_means["v_latents"],
        actual_std_v,
        logit_scale,
    )

    print(f"Contrastive loss 1: {cont_loss1}")
    print(f"Contrastive loss 2: {cont_loss2}")

    print("ok")


if __name__ == "__main__":
    main()

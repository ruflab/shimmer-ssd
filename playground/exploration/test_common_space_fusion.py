import logging
from collections.abc import Callable
from typing import Any, cast

from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    RichProgressBar,
)
from shimmer.modules.global_workspace import (
    GlobalWorkspaceBayesian,
    GWPredictionsBase,
)
from torch import set_float32_matmul_precision

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.dataset.domain import get_default_domains
from simple_shapes_dataset.dataset.pre_process import (
    color_blind_visual_domain,
    nullify_attribute_rotation,
)
from simple_shapes_dataset.modules.domains import load_pretrained_domains


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["exp_test_common_space_fusion.yaml"],
        debug_mode=DEBUG_MODE,
    )

    if config.exploration is None:
        raise ValueError("Config value 'exploration' must be set")

    seed_everything(config.seed, workers=True)

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
        ood_seed=config.ood_seed,
        domain_args=config.global_workspace.domain_args,
        additional_transforms=additional_transforms,
    )

    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.default_root_dir,
        config.global_workspace.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
        is_linear=config.global_workspace.linear_domains,
        bias=config.global_workspace.linear_domains_use_bias,
    )

    module = GlobalWorkspaceBayesian.load_from_checkpoint(
        config.exploration.gw_checkpoint,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )

    callbacks: list[Callback] = [RichProgressBar()]
    set_float32_matmul_precision(config.training.float32_matmul_precision)

    trainer = Trainer(
        fast_dev_run=config.training.fast_dev_run,
        max_steps=config.training.max_steps,
        enable_progress_bar=config.training.enable_progress_bar,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
    )

    predictions = cast(
        list[GWPredictionsBase],
        trainer.predict(module, data_module, return_predictions=True),
    )

    for pred in predictions:
        for k in range(pred["states"]["attr"].size(0)):
            for domain in pred["states"]:
                print(domain, pred["states"][domain][k])
            print("--")


if __name__ == "__main__":
    main()

import logging
from collections.abc import Callable
from typing import Any

from lightning.pytorch import Trainer
from shimmer.modules.global_workspace import GlobalWorkspaceBayesian

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.ckpt_migrations import (
    migrate_model,
)
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.pre_process import (
    color_blind_visual_domain,
    nullify_attribute_rotation,
)
from simple_shapes_dataset.modules.domains.pretrained import load_pretrained_domains


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["exp_var_cont.yaml"],
        debug_mode=DEBUG_MODE,
    )

    if config.exploration is None:
        raise ValueError("Exploration config should be set for this script")

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

    ckpt_path = config.exploration.gw_checkpoint
    migrate_model(ckpt_path, PROJECT_DIR / "migrations" / "gw")
    gw = GlobalWorkspaceBayesian.load_from_checkpoint(
        ckpt_path,
        domain_mods=domain_description,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )

    trainer = Trainer(
        fast_dev_run=config.training.fast_dev_run,
        max_steps=config.training.max_steps,
        enable_progress_bar=config.training.enable_progress_bar,
        default_root_dir=config.default_root_dir,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
    )

    predictions = trainer.predict(gw, data_module)
    print(predictions)


if __name__ == "__main__":
    main()

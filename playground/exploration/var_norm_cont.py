import logging
from collections.abc import Callable
from typing import Any, cast

from lightning.pytorch import Trainer
from shimmer import load_structured_config
from shimmer.modules.global_workspace import GlobalWorkspace

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.pre_process import (color_blind_visual_domain,
                                                       nullify_attribute_rotation)
from simple_shapes_dataset.modules.domains.pretrained import load_pretrained_domains


def main():
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["exp_var_cont"],
        debug_mode=DEBUG_MODE,
    )

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

    gw = cast(
        GlobalWorkspace,
        GlobalWorkspace.load_from_checkpoint(
            config.exploration.gw_checkpoint,
            domain_descriptions=domain_description,
        ),
    )
    # gw_mod = cast(VariationalGWModule, gw.gw_mod)

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

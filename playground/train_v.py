from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from migrate_ckpt.migrate import get_folder_migrations

from simple_shapes_dataset import DEBUG_MODE, LOGGER, PROJECT_DIR
from simple_shapes_dataset.ckpt_migrations import SaveMigrations
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.domain import get_default_domains
from simple_shapes_dataset.dataset.pre_process import color_blind_visual_domain
from simple_shapes_dataset.logging import LogVisualCallback
from simple_shapes_dataset.modules.domains.visual import VisualDomainModule


def main():
    LOGGER.debug(f"DEBUG_MODE: {DEBUG_MODE}")

    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_v.yaml"],
        debug_mode=DEBUG_MODE,
    )

    pl.seed_everything(config.seed, workers=True)

    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.visual.color_blind:
        LOGGER.info("v domain will be color blind.")
        additional_transforms["v"] = [color_blind_visual_domain]

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        get_default_domains(["v"]),
        {frozenset(["v"]): 1.0},
        batch_size=config.training.batch_size,
        max_train_size=config.dataset.max_train_size,
        num_workers=config.training.num_workers,
        additional_transforms=additional_transforms,
    )

    v_domain_module = VisualDomainModule(
        num_channels=3,
        ae_dim=config.domain_modules.visual.ae_dim,
        latent_dim=config.domain_modules.visual.latent_dim,
        beta=config.domain_modules.visual.beta,
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
        scheduler_args={
            "max_lr": config.training.optim.max_lr,
            "total_steps": config.training.max_steps,
        },
    )

    val_samples = data_module.get_samples("val", 32)[frozenset(["v"])]["v"]
    train_samples = data_module.get_samples("train", 32)[frozenset(["v"])]["v"]

    LOGGER.debug(
        f"log_val_medias_every_n_epochs {config.logging.log_val_medias_every_n_epochs}"
    )
    callbacks: list[pl.Callback] = [
        LearningRateMonitor(logging_interval="step"),
        LogVisualCallback(
            val_samples,
            log_key="images/val_attr",
            mode="val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
            ncols=8,
        ),
        LogVisualCallback(
            train_samples,
            log_key="images/train_attr",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
            ncols=8,
        ),
    ]

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())

    wandb_logger = None
    if config.wandb.enabled:
        if config.title is not None:
            run_name = config.title
        else:
            run_name = f"v_vae_z={config.domain_modules.visual.latent_dim}"
        wandb_kwargs: dict[str, Any] = {}
        if config.desc is not None:
            wandb_kwargs["notes"] = config.desc
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=["train_gw"],
            name=run_name,
            **wandb_kwargs,
        )
        wandb_logger.experiment.config.update(config.model_dump())

        checkpoint_dir = (
            config.default_root_dir / f"{wandb_logger.name}-{wandb_logger.version}"
        )
        callbacks.extend(
            [
                SaveMigrations(
                    get_folder_migrations(PROJECT_DIR / "migrations" / "visual_mod")
                ),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="{epoch}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                ),
            ]
        )
    LOGGER.debug(f"wandb logger: {wandb_logger}")

    torch.set_float32_matmul_precision(config.training.float32_matmul_precision)

    trainer = pl.Trainer(
        logger=wandb_logger,
        fast_dev_run=config.training.fast_dev_run,
        max_steps=config.training.max_steps,
        enable_progress_bar=config.training.enable_progress_bar,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
    )

    trainer.fit(v_domain_module, data_module)
    trainer.validate(v_domain_module, data_module, "best")


if __name__ == "__main__":
    main()

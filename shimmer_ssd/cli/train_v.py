from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from migrate_ckpt.migrate import get_folder_migrations
from simple_shapes_dataset import (
    SimpleShapesDataModule,
    color_blind_visual_domain,
    get_default_domains,
)

from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR
from shimmer_ssd.ckpt_migrations import SaveMigrations
from shimmer_ssd.config import load_config
from shimmer_ssd.logging import LogVisualCallback
from shimmer_ssd.modules.domains.visual import VisualDomainModule


def train_visual_domain(
    config_path: Path,
    debug_mode: bool | None = None,
    log_config: bool = False,
    extra_config_files: list[str] | None = None,
    argv: list[str] | None = None,
):
    if debug_mode is None:
        debug_mode = DEBUG_MODE
    if extra_config_files is None:
        extra_config_files = ["train_v.yaml"]
    if argv is None:
        argv = []

    LOGGER.debug(f"Debug mode: {debug_mode}")

    config = load_config(
        config_path,
        load_files=extra_config_files,
        debug_mode=debug_mode,
        log_config=log_config,
        argv=argv,
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
                    get_folder_migrations(
                        PROJECT_DIR / "shimmer_ssd" / "migrations" / "visual_mod"
                    )
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


@click.command(
    "v",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help="Train the visual domain",
)
@click.option(
    "--config_path",
    "-c",
    default="./config",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),  # type: ignore
)
@click.option("--debug", "-d", is_flag=True, default=None)
@click.option("--log_config", is_flag=True, default=False)
@click.option(
    "--extra_config_files",
    "-e",
    multiple=True,
    type=str,
    help=(
        "Additional files to `local.yaml` to load in the config path. "
        "By default `train_v.yaml`"
    ),
)
@click.pass_context
def train_v_command(
    ctx: click.Context,
    config_path: Path,
    debug: bool | None,
    log_config: bool,
    extra_config_files: list[str],
):
    return train_visual_domain(
        config_path,
        debug,
        log_config,
        extra_config_files if len(extra_config_files) else None,
        ctx.args,
    )

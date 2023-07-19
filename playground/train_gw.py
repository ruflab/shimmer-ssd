import os
from collections.abc import Mapping

from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf
from shimmer import load_structured_config
from shimmer.modules.domain import DomainDescription
from shimmer.modules.global_workspace import GlobalWorkspace
from torch import set_float32_matmul_precision

from simple_shapes_dataset import PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.logging import LogGWImagesCallback
from simple_shapes_dataset.modules.domains import load_pretrained_domains
from simple_shapes_dataset.modules.global_workspace import (
    GlobalWorkspaceLightningModule,
)


def global_workspace_from_domains(
    domains: Mapping[str, DomainDescription],
    latent_dim: int,
    encoder_hiddent_dim: int,
    encoder_n_layers: int,
    decoder_hidden_dim: int,
    decoder_n_layers: int,
) -> GlobalWorkspace:
    domain_names = set(domains.keys())
    input_dims = {name: domain.latent_dim for name, domain in domains.items()}
    return GlobalWorkspace(
        domain_names,
        latent_dim,
        input_dims,
        {name: encoder_hiddent_dim for name in domains.keys()},
        {name: encoder_n_layers for name in domains.keys()},
        {name: decoder_hidden_dim for name in domains.keys()},
        {name: decoder_n_layers for name in domains.keys()},
    )


def main():
    debug_mode = bool(int(os.getenv("DEBUG", "0")))
    config = load_structured_config(
        PROJECT_DIR / "config",
        Config,
        load_dirs=["train_gw"],
        debug_mode=debug_mode,
    )

    seed_everything(config.seed, workers=True)

    domain_proportion = {
        frozenset(item.domains): item.proportion
        for item in config.global_workspace.domain_proportions
    }

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_proportion,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
    )

    domains = load_pretrained_domains(config.global_workspace.domains)

    global_workspace = global_workspace_from_domains(
        domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )
    module = GlobalWorkspaceLightningModule(
        global_workspace,
        {name: domain.module for name, domain in domains.items()},
        config.global_workspace.loss_coefficients.demi_cycles,
        config.global_workspace.loss_coefficients.cycles,
        config.global_workspace.loss_coefficients.translations,
        config.global_workspace.loss_coefficients.contrastives,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler_args={
            "max_lr": config.training.optim.max_lr,
            "total_steps": config.training.max_steps,
        },
    )

    train_samples = data_module.get_samples("train", 32)
    val_samples = data_module.get_samples("val", 32)
    for domains in val_samples.keys():
        for domain in domains:
            val_samples[frozenset([domain])] = {
                domain: val_samples[domains][domain]
            }
        break

    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        LogGWImagesCallback(
            val_samples,
            log_key="images/val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
        ),
    ]

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())

    wandb_logger = None
    if config.wandb.enabled:
        run_name = f"gw_z={config.global_workspace.latent_dim}"
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=["train_gw"],
            name=run_name,
        )
        wandb_logger.experiment.config.update(
            OmegaConf.to_container(config, resolve=True)
        )

        checkpoint_dir = (
            config.default_root_dir
            / f"{wandb_logger.name}-{wandb_logger.version}"
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            )
        )

    set_float32_matmul_precision(config.training.float32_matmul_precision)

    trainer = Trainer(
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

    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()

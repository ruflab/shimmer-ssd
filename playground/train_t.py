import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.logging import LogTextCallback
from simple_shapes_dataset.modules.domains.text import TextDomainModule


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_t.yaml"],
        debug_mode=DEBUG_MODE,
    )

    pl.seed_everything(config.seed, workers=True)

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        {frozenset(["t"]): 1.0},
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        domain_args={
            "t": {"latent_filename": config.domain_modules.text.latent_filename}
        },
    )

    text_domain_module = TextDomainModule(
        latent_dim=config.domain_modules.text.latent_dim,
        hidden_dim=config.domain_modules.text.hidden_dim,
        beta=config.domain_modules.text.beta,
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
        scheduler_args={
            "max_lr": config.training.optim.max_lr,
            "total_steps": config.training.max_steps,
        },
    )

    val_samples = data_module.get_samples("val", 32)[frozenset(["t"])]["t"]
    train_samples = data_module.get_samples("train", 32)[frozenset(["t"])]["t"]

    callbacks: list[pl.Callback] = [
        LearningRateMonitor(logging_interval="step"),
        LogTextCallback(
            val_samples,
            log_key="images/val_t",
            mode="val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
            image_size=32,
            ncols=8,
        ),
        LogTextCallback(
            train_samples,
            log_key="images/train_t",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
            image_size=32,
            ncols=8,
        ),
    ]

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())

    wandb_logger = None
    if config.wandb.enabled:
        run_name = f"t_vae_z={config.domain_modules.text.latent_dim}"
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=["train_t"],
            name=run_name,
        )
        wandb_logger.experiment.config.update(config.model_dump())

        checkpoint_dir = (
            config.default_root_dir / f"{wandb_logger.name}-{wandb_logger.version}"
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

    trainer.fit(text_domain_module, data_module)


if __name__ == "__main__":
    main()

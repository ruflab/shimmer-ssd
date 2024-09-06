from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer import DomainModule, SaveMigrations
from torch import set_float32_matmul_precision

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.dataset.domain import get_default_domains
from simple_shapes_dataset.dataset.pre_process import (
    TokenizeCaptions,
)
from simple_shapes_dataset.errors import ConfigurationError
from simple_shapes_dataset.logging import LogText2AttrCallback
from simple_shapes_dataset.modules.domains.pretrained import load_pretrained_domain
from simple_shapes_dataset.modules.domains.text import GRUTextDomainModule, Text2Attr
from simple_shapes_dataset.types import DomainModelVariantType


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_text2attr.yaml"],
        debug_mode=DEBUG_MODE,
    )

    seed_everything(config.seed, workers=True)

    domain_proportion = {frozenset(["t", "attr"]): 1.0}
    domain_classes = get_default_domains(["t", "attr"])

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_classes,
        domain_proportion,
        batch_size=config.training.batch_size,
        max_train_size=config.dataset.max_train_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        ood_seed=config.ood_seed,
        domain_args=config.global_workspace.domain_args,
        additional_transforms={
            "t": [
                TokenizeCaptions(
                    config.domain_modules.text.vocab_path,
                    config.domain_modules.text.merges_path,
                    config.domain_modules.text.seq_length,
                )
            ]
        },
    )

    text_model: DomainModule | None = None
    for domain in config.global_workspace.domains:
        if domain.domain_type != DomainModelVariantType.t:
            continue
        text_model, _, _ = load_pretrained_domain(
            config.default_root_dir,
            domain,
            config.global_workspace.latent_dim,
            config.global_workspace.encoders.hidden_dim,
            config.global_workspace.encoders.n_layers,
            config.global_workspace.decoders.hidden_dim,
            config.global_workspace.decoders.n_layers,
            config.global_workspace.linear_domains,
            config.global_workspace.linear_domains_use_bias,
        )

    if not isinstance(text_model, GRUTextDomainModule):
        raise ConfigurationError("text model should be loaded")

    text2attr = Text2Attr(
        text_model.latent_dim,
        128,
        text_model,
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
        scheduler_args={
            "max_lr": config.training.optim.max_lr,
            "total_steps": config.training.max_steps,
        },
    )

    train_samples = data_module.get_samples("train", 32)[frozenset(["attr", "t"])]
    val_samples = data_module.get_samples("val", 32)[frozenset(["attr", "t"])]
    test_samples = data_module.get_samples("test", 32)[frozenset(["attr", "t"])]

    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        LogText2AttrCallback(
            val_samples,
            log_key="images/val",
            mode="val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
            vocab=config.domain_modules.text.vocab_path,
            merges=config.domain_modules.text.merges_path,
        ),
        LogText2AttrCallback(
            test_samples,
            log_key="images/test",
            mode="test",
            every_n_epochs=None,
            vocab=config.domain_modules.text.vocab_path,
            merges=config.domain_modules.text.merges_path,
        ),
        LogText2AttrCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
            vocab=config.domain_modules.text.vocab_path,
            merges=config.domain_modules.text.merges_path,
        ),
    ]
    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())

    wandb_logger = None
    if config.wandb.enabled:
        run_name = f"text2attr_z={config.global_workspace.latent_dim}"
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=["train_gw"],
            name=run_name,
        )
        wandb_logger.experiment.config.update(config.model_dump())

        checkpoint_dir = (
            config.default_root_dir / f"{wandb_logger.name}-{wandb_logger.version}"
        )
        callbacks.extend(
            [
                SaveMigrations(),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="{epoch}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                ),
            ]
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

    trainer.fit(text2attr, data_module)
    trainer.validate(text2attr, data_module, "best")


if __name__ == "__main__":
    main()

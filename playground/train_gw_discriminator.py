import logging
from collections.abc import Callable
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer import (
    ContrastiveLossType,
    SaveMigrations,
)
from shimmer.modules.global_workspace import (
    SchedulerArgs,
)
from torch import set_float32_matmul_precision

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.dataset.domain import get_default_domains
from simple_shapes_dataset.dataset.pre_process import (
    TokenizeCaptions,
    color_blind_visual_domain,
    nullify_attribute_rotation,
)
from simple_shapes_dataset.logging import LogGWImagesCallback
from simple_shapes_dataset.modules.contrastive_loss import VSEPPContrastiveLoss
from simple_shapes_dataset.modules.discriminator import (
    CoefWithDiscriminator,
    GlobalWorkspaceWithDiscriminator,
)
from simple_shapes_dataset.modules.domains import load_pretrained_domains


def main():
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
    )

    seed_everything(config.seed, workers=True)

    domain_proportion = {
        frozenset(item.domains): item.proportion
        for item in config.global_workspace.domain_proportions
    }
    domain_classes = get_default_domains(
        {domain.domain_type.kind.value for domain in config.global_workspace.domains}
    )

    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.attribute.nullify_rotation:
        logging.info("Nullifying rotation in the attr domain.")
        additional_transforms["attr"] = [nullify_attribute_rotation]
    if config.domain_modules.visual.color_blind:
        logging.info("v domain will be color blind.")
        additional_transforms["v"] = [color_blind_visual_domain]
    additional_transforms["t"] = [
        TokenizeCaptions(
            config.domain_modules.text.vocab_path,
            config.domain_modules.text.merges_path,
            config.domain_modules.text.seq_length,
        )
    ]

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

    contrastive_fn: ContrastiveLossType | None = None
    if config.global_workspace.vsepp_contrastive_loss:
        contrastive_fn = VSEPPContrastiveLoss(
            config.global_workspace.vsepp_margin,
            config.global_workspace.vsepp_measure,
            config.global_workspace.vsepp_max_violation,
            torch.tensor([1 / 0.07]).log(),
        )

    gw_type = "gw_fusion"
    loss_coefs_fusion: CoefWithDiscriminator = {
        "contrastives": config.global_workspace.loss_coefficients.contrastives,
        "fused": config.global_workspace.loss_coefficients.fused,
        "translations": config.global_workspace.loss_coefficients.translations,
        "demi_cycles": config.global_workspace.loss_coefficients.demi_cycles,
        "cycles": config.global_workspace.loss_coefficients.cycles,
        "generator": config.global_workspace.discriminator.coef_generator,
        "discriminator": config.global_workspace.discriminator.coef_discriminator,
    }
    module = GlobalWorkspaceWithDiscriminator(
        domain_modules,
        gw_encoders,
        gw_decoders,
        config.global_workspace.latent_dim,
        loss_coefs_fusion,
        config.global_workspace.discriminator.domain_name,
        config.global_workspace.discriminator.hidden_dim,
        config.global_workspace.discriminator.generator_loss_every,
        config.global_workspace.selection_temperature,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=config.training.optim.max_lr,
            total_steps=config.training.max_steps,
        ),
        scheduler=None,
        learn_logit_scale=config.global_workspace.learn_logit_scale,
        contrastive_loss=contrastive_fn,
    )

    train_samples = data_module.get_samples("train", 32)
    val_samples = data_module.get_samples("val", 32)
    test_samples = data_module.get_samples("test", 32)

    for domains in val_samples:
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
            test_samples[frozenset([domain])] = {domain: test_samples[domains][domain]}
        break

    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        LogGWImagesCallback(
            val_samples,
            log_key="images/val",
            mode="val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
            filter=config.logging.filter_images,
            vocab=config.domain_modules.text.vocab_path,
            merges=config.domain_modules.text.merges_path,
        ),
        LogGWImagesCallback(
            val_samples,
            log_key="images/test",
            mode="test",
            every_n_epochs=None,
            filter=config.logging.filter_images,
            vocab=config.domain_modules.text.vocab_path,
            merges=config.domain_modules.text.merges_path,
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
            filter=config.logging.filter_images,
            vocab=config.domain_modules.text.vocab_path,
            merges=config.domain_modules.text.merges_path,
        ),
    ]

    if config.ood_seed is not None:
        train_samples_ood = data_module.get_samples("train", 32, ood=True)
        val_samples_ood = data_module.get_samples("val", 32, ood=True)
        test_samples_ood = data_module.get_samples("test", 32, ood=True)

        for domains in val_samples_ood:
            for domain in domains:
                val_samples_ood[frozenset([domain])] = {
                    domain: val_samples_ood[domains][domain]
                }
                test_samples_ood[frozenset([domain])] = {
                    domain: test_samples_ood[domains][domain]
                }
            break

        callbacks.extend(
            [
                LogGWImagesCallback(
                    val_samples_ood,
                    log_key="images/val/ood",
                    mode="val",
                    every_n_epochs=config.logging.log_val_medias_every_n_epochs,
                    filter=config.logging.filter_images,
                ),
                LogGWImagesCallback(
                    val_samples_ood,
                    log_key="images/test/ood",
                    mode="test",
                    every_n_epochs=None,
                    filter=config.logging.filter_images,
                ),
                LogGWImagesCallback(
                    train_samples_ood,
                    log_key="images/train/ood",
                    mode="train",
                    every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                    filter=config.logging.filter_images,
                ),
            ]
        )

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())

    wandb_logger = None
    if config.wandb.enabled:
        run_name = f"{gw_type}_discriminator_z={config.global_workspace.latent_dim}"
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

    trainer.fit(module, data_module)
    trainer.validate(module, data_module, "best")
    trainer.test(module, data_module, "best")


if __name__ == "__main__":
    main()

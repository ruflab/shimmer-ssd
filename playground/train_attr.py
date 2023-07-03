import os

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer.config import load_config
from utils import PROJECT_DIR

from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.logging import LogAttributesCallback
from simple_shapes_dataset.modules.domains.attribute import (
    AttributeDomainModule,
)


def main():
    print(PROJECT_DIR)
    debug_mode = bool(int(os.getenv("DEBUG", "0")))
    config = load_config(
        PROJECT_DIR / "config",
        load_dirs=["train_attr"],
        structure=Config,
        debug_mode=debug_mode,
    )

    data_module = SimpleShapesDataModule(
        config.dataset.path,
        {frozenset(["attr"]): 1.0},
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    attr_domain_module = AttributeDomainModule(
        latent_dim=config.domain_modules.attribute.latent_dim,
        hidden_dim=config.domain_modules.attribute.hidden_dim,
        n_layers=config.domain_modules.attribute.n_layers,
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
    )

    wandb_logger = None
    if config.wandb.enabled:
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )

    val_samples = data_module.get_samples("val", 2)[frozenset(["attr"])][
        "attr"
    ]
    train_samples = data_module.get_samples("train", 2)[frozenset(["attr"])][
        "attr"
    ]
    callbacks = [
        LogAttributesCallback(
            val_samples,
            log_key="images/val_attr",
            image_size=32,
            ncols=2,
        ),
        LogAttributesCallback(
            train_samples,
            log_key="images/train_attr",
            image_size=32,
            ncols=2,
        ),
    ]

    trainer = pl.Trainer(
        logger=wandb_logger,
        fast_dev_run=config.training.fast_dev_run,
        max_epochs=config.training.max_epochs,
        enable_progress_bar=config.training.enable_progress_bar,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,  # type: ignore
    )

    trainer.fit(attr_domain_module, data_module)


if __name__ == "__main__":
    main()

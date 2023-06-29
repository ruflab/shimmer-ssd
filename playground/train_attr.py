import os

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer.config import load_config
from utils import PROJECT_DIR

from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
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
        latent_dim=12,
        hidden_dim=32,
        n_layers=2,
        optim_lr=1e-3,
    )

    wandb_logger = None
    if config.wandb.enabled:
        wandb_logger = WandbLogger(
            save_dir=config.wandb.save_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )

    trainer = pl.Trainer(
        logger=wandb_logger,
        fast_dev_run=False,
        max_epochs=100,
        enable_progress_bar=True,
        default_root_dir=PROJECT_DIR / "checkpoints",
    )

    trainer.fit(attr_domain_module, data_module)


if __name__ == "__main__":
    main()

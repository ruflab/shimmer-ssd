from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from migrate_ckpt import (
    Migration,
    ckpt_migration_key,
    migrate_from_folder,
)
from shimmer import migrate_model as migrate_shimmer_model

from simple_shapes_dataset import LOGGER


def migrate_model(ckpt_path: str | PathLike, migration_path: str | PathLike, **kwargs):
    default_torch_kwargs: dict[str, Any] = {"weights_only": True}
    default_torch_kwargs.update(kwargs)

    if Path(migration_path).name == "gw":
        migrate_shimmer_model(ckpt_path, **default_torch_kwargs)

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, **default_torch_kwargs)
    new_ckpt, done_migrations = migrate_from_folder(ckpt, migration_path)
    done_migration_log = ", ".join(map(lambda x: x.name, done_migrations))
    LOGGER.debug(f"Migrating: {done_migration_log}")
    if len(done_migrations) or ckpt_migration_key not in ckpt:
        version = 0
        if ckpt_migration_key in ckpt:
            version = len(ckpt[ckpt_migration_key])
        torch.save(ckpt, ckpt_path.with_stem(f"{ckpt_path.stem}-{version}"))
        torch.save(new_ckpt, ckpt_path)


class SaveMigrations(Callback):
    def __init__(self, migrations: Sequence[Migration]):
        self.migrations = migrations

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ):
        checkpoint[ckpt_migration_key] = [mig.name for mig in self.migrations]

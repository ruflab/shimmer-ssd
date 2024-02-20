from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import torch
from migrate_ckpt import Migration, ckpt_migration_key, migrate_ckpt

from simple_shapes_dataset import LOGGER

gw_migrations: list[Migration] = []
var_gw_migrations: list[Migration] = []
visual_mod_migrations: list[Migration] = []
attribute_mod_migrations: list[Migration] = []
text_mod_migrations: list[Migration] = []


def migrate_model(ckpt_path: str | PathLike, migrations: Sequence[Migration], **kwargs):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, **kwargs)
    new_ckpt, done_migrations = migrate_ckpt(ckpt, migrations)
    LOGGER.debug(f"Migrating: {done_migrations}")
    version = 0
    if ckpt_migration_key in ckpt:
        version = len(ckpt[ckpt_migration_key])
    torch.save(ckpt, ckpt_path.with_stem(f"{ckpt_path.stem}-{version}"))
    torch.save(new_ckpt, ckpt_path)

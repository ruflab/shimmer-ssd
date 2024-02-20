from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import torch
from migrate_ckpt import CkptType, Migration, ckpt_migration_key, migrate_ckpt

from simple_shapes_dataset import LOGGER


def add_gw_interfaces(ckpt: CkptType) -> CkptType:
    new_state_dict = {}
    for name, val in ckpt["state_dict"].items():
        new_name = name.replace(
            "gw_mod.encoders.resnet", "gw_mod.gw_interfaces.resnet.encoder"
        )
        new_name = new_name.replace(
            "gw_mod.encoders.bge", "gw_mod.gw_interfaces.bge.encoder"
        )
        new_name = new_name.replace(
            "gw_mod.decoders.resnet", "gw_mod.gw_interfaces.resnet.decoder"
        )
        new_name = new_name.replace(
            "gw_mod.decoders.bge", "gw_mod.gw_interfaces.bge.decoder"
        )
        new_state_dict[new_name] = val
    ckpt["state_dict"] = new_state_dict
    return ckpt


add_gw_interfaces_migration = Migration("add-gw-interfaces", add_gw_interfaces)

gw_migrations: list[Migration] = [add_gw_interfaces_migration]
var_gw_migrations: list[Migration] = [add_gw_interfaces_migration]
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

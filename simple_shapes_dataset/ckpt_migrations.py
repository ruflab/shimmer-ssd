from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
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


def remove_gw_interfaces_hparams(ckpt: CkptType) -> CkptType:
    if "hyper_parameters" in ckpt.keys():
        if "gw_interfaces" in ckpt["hyper_parameters"].keys():
            del ckpt["hyper_parameters"]["gw_interfaces"]
    return ckpt


def replace_gw_interfaces_gw_encoders_decoders(ckpt: CkptType) -> CkptType:
    new_state_dict = {}
    for name, val in ckpt["state_dict"].items():
        if "gw_mod.gw_interfaces" in name and "domain_module" in name:
            continue
        elif "gw_mod.gw_interfaces" in name and "encoder" in name:
            new_name = name.replace(".gw_interfaces", ".gw_encoders")
            new_name = new_name.replace(".encoder", "")
            new_state_dict[new_name] = val
        elif "gw_mod.gw_interfaces" in name and "decoder" in name:
            new_name = name.replace(".gw_interfaces", ".gw_decoders")
            new_name = new_name.replace(".decoder", "")
            new_state_dict[new_name] = val
        elif "gw_interfaces" in name:
            print(name)
        else:
            new_state_dict[name] = val
    ckpt["state_dict"] = new_state_dict
    return ckpt


add_gw_interfaces_migration = Migration("add-gw-interfaces", add_gw_interfaces)
remove_gw_interfaces_hparams_migration = Migration(
    "del-gw-interfaces-hparam", remove_gw_interfaces_hparams
)
replace_gw_interfaces_gw_encoders_decoders_migration = Migration(
    "del-gw-interfaces", replace_gw_interfaces_gw_encoders_decoders
)

gw_migrations: list[Migration] = [
    add_gw_interfaces_migration,
    remove_gw_interfaces_hparams_migration,
    replace_gw_interfaces_gw_encoders_decoders_migration,
]
var_gw_migrations: list[Migration] = [
    add_gw_interfaces_migration,
    remove_gw_interfaces_hparams_migration,
    replace_gw_interfaces_gw_encoders_decoders_migration,
]
visual_mod_migrations: list[Migration] = []
attribute_mod_migrations: list[Migration] = []
text_mod_migrations: list[Migration] = []


def migrate_model(ckpt_path: str | PathLike, migrations: Sequence[Migration], **kwargs):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, **kwargs)
    new_ckpt, done_migrations = migrate_ckpt(ckpt, migrations)
    done_migration_log = ", ".join(map(lambda x: x.name, done_migrations))
    LOGGER.debug(f"Migrating: {done_migration_log}")
    if len(done_migrations) or ckpt_migration_key not in ckpt:
        version = 0
        if ckpt_migration_key in ckpt:
            version = len(ckpt[ckpt_migration_key])
        print(new_ckpt.keys())
        # torch.save(ckpt, ckpt_path.with_stem(f"{ckpt_path.stem}-{version}"))
        # torch.save(new_ckpt, ckpt_path)


class SaveMigrations(Callback):
    def __init__(self, migrations: Sequence[Migration]):
        self.migrations = migrations

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ):
        checkpoint[ckpt_migration_key] = [mig.name for mig in self.migrations]

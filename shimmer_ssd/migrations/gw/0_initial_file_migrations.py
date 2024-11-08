from migrate_ckpt import CkptType, ckpt_migration_key


def handle(ckpt: CkptType) -> CkptType:
    """
    This migration renames migration fields in the checkpoint to be
    consistent with the names of the files.
    Before, it was handled with a list of Migration directly in code.
    """
    if ckpt_migration_key not in ckpt:
        return ckpt

    new_migration_keys = []
    for k, migration in enumerate(ckpt[ckpt_migration_key], 1):
        new_migration_keys.append(f"{k}_{migration.replace('-', '_')}")
    ckpt[ckpt_migration_key] = new_migration_keys

    return ckpt

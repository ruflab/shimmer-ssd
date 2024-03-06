from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.ckpt_migrations import gw_migrations, migrate_model
from simple_shapes_dataset.config import load_config


def main():
    config = load_config(
        PROJECT_DIR / "config",
        debug_mode=DEBUG_MODE,
    )

    if config.global_workspace.checkpoint is not None:
        migrate_model(config.global_workspace.checkpoint, gw_migrations)
        print("Model was migrated!")

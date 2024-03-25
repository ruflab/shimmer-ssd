from shimmer import migrate_model as migrate_shimmer_model

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.ckpt_migrations import migrate_model
from simple_shapes_dataset.config import load_config


def main():
    config = load_config(
        PROJECT_DIR / "config",
        debug_mode=DEBUG_MODE,
    )

    if config.global_workspace.checkpoint is not None:
        migrate_shimmer_model(config.global_workspace.checkpoint)
        migrate_model(
            config.global_workspace.checkpoint, PROJECT_DIR / "migrations" / "gw"
        )
        print("Model was migrated!")


if __name__ == "__main__":
    main()

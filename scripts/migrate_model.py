from shimmer import migrate_model as migrate_shimmer_model

from shimmer_ssd import DEBUG_MODE, PROJECT_DIR
from shimmer_ssd.ckpt_migrations import migrate_model
from shimmer_ssd.config import load_config


def main():
    config = load_config(
        PROJECT_DIR / "config",
        debug_mode=DEBUG_MODE,
    )

    if config.global_workspace.checkpoint is not None:
        migrate_shimmer_model(config.global_workspace.checkpoint)
        migrate_model(
            config.global_workspace.checkpoint,
            PROJECT_DIR / "shimmer_ssd" / "migrations" / "gw",
        )
        print("Model was migrated!")


if __name__ == "__main__":
    main()

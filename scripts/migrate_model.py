from shimmer_ssd import DEBUG_MODE, PROJECT_DIR
from shimmer_ssd.cli.migrate import migrate_domains
from shimmer_ssd.config import load_config

if __name__ == "__main__":
    config = load_config(
        PROJECT_DIR / "config",
        debug_mode=DEBUG_MODE,
    )

    if config.global_workspace.checkpoint is not None:
        migrate_domains(config.global_workspace.checkpoint)

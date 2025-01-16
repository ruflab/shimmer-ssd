from pathlib import Path

import click
from shimmer import migrate_model as migrate_shimmer_model

from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.ckpt_migrations import migrate_model


def migrate_domains(checkpoint_path: Path):
    migrate_shimmer_model(checkpoint_path)
    migrate_model(
        checkpoint_path,
        PROJECT_DIR / "shimmer_ssd" / "migrations" / "gw",
    )
    print("Model was migrated!")


@click.command(
    "migrate",
    help="Migrate checkpoint",
)
@click.argument(
    "checkpoint_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),  # type: ignore
)
def migrate_domains_command(checkpoint_path: Path):
    return migrate_domains(checkpoint_path)

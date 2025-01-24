from pathlib import Path
from typing import Literal

import click
from shimmer import migrate_model as migrate_shimmer_model

from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.ckpt_migrations import migrate_model


def migrate_domains(
    checkpoint_path: Path,
    migration_path: Path | None = None,
    migration_type: Literal["gw", "attr_mod", "text_mod", "visual_mod"] = "gw",
):
    default_path = PROJECT_DIR / "shimmer_ssd" / "migrations"
    migrate_shimmer_model(checkpoint_path, weights_only=False)
    migrate_model(
        checkpoint_path,
        (migration_path or default_path) / migration_type,
        weights_only=False,
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
@click.option(
    "--migration_path",
    default=None,
    type=click.Path(exists=True, dir_okay=True, path_type=Path),  # type: ignore
)
@click.option(
    "--type",
    "-t",
    "migration_type",
    default="gw",
    type=click.Choice(["gw", "attr_mod", "text_mod", "visual_mod"]),
)
def migrate_domains_command(
    checkpoint_path: Path,
    migration_path: Path | None,
    migration_type: Literal["gw", "attr_mod", "text_mod", "visual_mod"],
):
    return migrate_domains(checkpoint_path, migration_path, migration_type)

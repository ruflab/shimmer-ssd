from pathlib import Path
from shutil import copytree

import click

from shimmer_ssd import PROJECT_DIR


@click.group("config")
def config_group():
    pass


@config_group.command("create")
@click.option(
    "--path",
    "-p",
    default="./config",
    type=click.Path(path_type=Path),  # type: ignore
    help="Where to save the config folder. Defaults to `./config`.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    type=bool,
    help=(
        "If the directory already exists, this will force the "
        "reset of the config files."
    ),
)
def create_config_command(path: Path, force: bool):
    if path.exists() and not force:
        click.echo("Config folder already exists. Skipping.")
        return

    config_template_path = PROJECT_DIR / "shimmer_ssd" / "config_template"
    copytree(config_template_path, path, dirs_exist_ok=force)

import tarfile
from pathlib import Path

import click
from simple_shapes_dataset.cli.download import downlad_file

CHECKPOINTS_URL = (
    "https://zenodo.org/records/14747474/files/simple_shapes_checkpoints.tar.gz"
)

TOKENIZER_URL = (
    "https://raw.githubusercontent.com/ruflab/shimmer-ssd/refs/heads/main/tokenizer"
)


@click.group("download")
def download_group():
    pass


@download_group.command("checkpoints", help="Download pretrained checkpoints")
@click.option(
    "--path",
    "-p",
    type=click.Path(path_type=Path),  # type: ignore
    default="./checkpoints",
    help="Where to download the checkpoints. Defaults to `./checkpoints`",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    type=bool,
    help="If the file already exist, his will override with a new file.",
)
def download_dataset(path: Path, force: bool = False):
    click.echo(f"Downloading in {str(path)}.")
    if path.exists() and not force:
        click.echo("Checkpoint path already exists. Skipping.")
        return
    elif path.exists():
        click.echo("Checkpoint path already exists. Overriding.")
    path.mkdir(exist_ok=True)
    archive_path = path / "simple_shapes_checkpoints.tar.gz"
    downlad_file(CHECKPOINTS_URL, archive_path)
    click.echo("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path)
    archive_path.unlink()


@download_group.command("tokenizer", help="Download pretrained tokenizer")
@click.option(
    "--path",
    "-p",
    type=click.Path(path_type=Path),  # type: ignore
    default="./tokenizer",
    help="Where to download the tokenizer files",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    type=bool,
    help="If the file already exist, his will override with a new file.",
)
def download_tokenizer(path: Path, force: bool = False):
    click.echo(f"Downloading in {str(path)}.")
    click.echo(f"Downloading in {str(path)}.")
    if path.exists() and not force:
        click.echo("Tokenizer path already exists. Skipping.")
        return
    elif path.exists():
        click.echo("Tokenizer path already exists. Overriding.")
    path.mkdir(exist_ok=True)
    downlad_file(TOKENIZER_URL + "/merges.txt", path / "merges.txt")
    downlad_file(TOKENIZER_URL + "/vocab.json", path / "vocab.json")

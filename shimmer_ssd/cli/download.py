import tarfile
from pathlib import Path

import click
from simple_shapes_dataset.cli.download import downlad_file

CHECKPOINTS_URL = (
    "https://zenodo.org/records/14289631/files/simple_shapes_checkpoints.tar.gz"
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
def download_dataset(path: Path):
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
def download_tokenizer(path: Path):
    path.mkdir(exist_ok=True)
    downlad_file(TOKENIZER_URL + "/merges.txt", path / "merges.txt")
    downlad_file(TOKENIZER_URL + "/vocab.json", path / "vocab.json")

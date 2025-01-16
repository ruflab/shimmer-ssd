import shutil
import tarfile
from pathlib import Path

import click
from simple_shapes_dataset.cli.download import downlad_file

DATASET_URL = (
    "https://zenodo.org/records/14289631/files/simple_shapes_checkpoints.tar.gz"
)


@click.group("download")
def download_group():
    pass


@click.command("checkpoints", help="Download pretrained checkpoints")
@click.option(
    "--path",
    "-p",
    type=click.Path(path_type=Path),  # type: ignore
    default=".",
    help="Where to download the dataset",
)
def download_dataset(path: Path):
    archive_path = path.parent / "simple_shapes_checkpoints.tar.gz"
    shutil.rmtree(path)
    downlad_file(DATASET_URL, archive_path)
    click.echo("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path)
    archive_path.unlink()

import click

import simple_shapes_dataset.cli.utils as utils

from .create_dataset import create_dataset

__all__ = ["cli", "create_dataset", "utils"]


@click.group()
def cli():
    pass


cli.add_command(create_dataset)

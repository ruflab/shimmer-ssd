import click

import simple_shapes_dataset.cli.utils as utils

from .create_dataset import add_split, create_dataset

__all__ = ["add_split", "cli", "create_dataset", "utils"]


@click.group()
def cli():
    pass


cli.add_command(create_dataset)
cli.add_command(add_split)

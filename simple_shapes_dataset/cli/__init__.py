import click

import simple_shapes_dataset.cli.utils as utils

from .create_dataset import (
    add_alignment_split,
    create_dataset,
    create_unpaired_attributes,
)

__all__ = [
    "add_alignment_split",
    "cli",
    "create_dataset",
    "utils",
    "create_unpaired_attributes",
]


@click.group()
def cli():
    pass


cli.add_command(create_dataset)
cli.add_command(add_alignment_split)
cli.add_command(create_unpaired_attributes)

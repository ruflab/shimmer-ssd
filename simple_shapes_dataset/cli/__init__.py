import click

from .create_dataset import (
    add_alignment_split,
    create_dataset,
    unpaired_attributes_command,
)

__all__ = [
    "add_alignment_split",
    "cli",
    "create_dataset",
    "unpaired_attributes_command",
]


@click.group()
def cli():
    pass


cli.add_command(create_dataset)
cli.add_command(add_alignment_split)
cli.add_command(unpaired_attributes_command)

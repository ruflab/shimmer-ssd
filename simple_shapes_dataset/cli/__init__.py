import click

from simple_shapes_dataset.cli.alignments import alignment_group
from simple_shapes_dataset.cli.odd_one_out import create_odd_one_out_dataset

from .create_dataset import (
    create_dataset,
    create_ood_split,
    unpaired_attributes_command,
)

__all__ = [
    "cli",
    "create_dataset",
    "alignment_group",
    "unpaired_attributes_command",
    "create_ood_split",
]


@click.group()
def cli():
    pass


cli.add_command(create_dataset)
cli.add_command(alignment_group)
cli.add_command(unpaired_attributes_command)
cli.add_command(create_ood_split)
cli.add_command(create_odd_one_out_dataset)

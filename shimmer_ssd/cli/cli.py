import click

from shimmer_ssd.cli.download import download_group
from shimmer_ssd.cli.extract import save_v_latents_command
from shimmer_ssd.cli.migrate import migrate_domains_command
from shimmer_ssd.cli.train_attr import train_attr_command
from shimmer_ssd.cli.train_t import train_t_command
from shimmer_ssd.cli.train_v import train_v_command


@click.group()
def cli():
    pass


cli.add_command(migrate_domains_command)
cli.add_command(download_group)


@cli.group("train")
def train_group():
    pass


train_group.add_command(train_v_command)
train_group.add_command(train_attr_command)
train_group.add_command(train_t_command)


@cli.group("extract")
def extract_group():
    pass


extract_group.add_command(save_v_latents_command)

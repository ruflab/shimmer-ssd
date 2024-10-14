import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import click
import numpy as np

from simple_shapes_dataset.cli.utils import get_deterministic_name, get_domain_alignment


@click.group("alignment")
def alignment_group():
    pass


def create_domain_split(
    seed: int,
    dataset_path: Path,
    domain_alignment: list[tuple[str, float]],
    max_train_size: int | None = None,
):
    if not len(domain_alignment):
        return
    np.random.seed(seed)
    split_path = dataset_path / "domain_splits_v2"
    split_path.mkdir(exist_ok=True)

    domain_sets = {}
    for domain, prop in domain_alignment:
        domain_set = frozenset(domain.split(","))
        if domain_set in domain_sets:
            logging.warning(
                f"Domain set {domain_set} is defined multiple times. "
                f"The value will be overwritten by {prop}."
            )
        domain_sets[domain_set] = prop

    for split in ["train", "val", "test"]:
        labels = np.load(str(dataset_path / f"{split}_labels.npy"))
        allowed_indices = np.arange(labels.shape[0])
        if split == "train":
            allowed_indices = allowed_indices[:max_train_size]
        split_name = get_deterministic_name(domain_sets, seed, allowed_indices.shape[0])
        domain_split = get_domain_alignment(
            seed,
            allowed_indices,
            domain_sets,
        )

        np.save(
            split_path / f"{split}_{split_name}_domain_split.npy",
            domain_split,  # type: ignore
        )


@alignment_group.command("add", help="Create a dataset alignment split")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option(
    "--dataset_path",
    "-p",
    default="./",
    type=str,
    help="Path to the dataset",
)
@click.option(
    "--max_train_size",
    "--ms",
    default=None,
    type=int,
    help="Max index to use for the train set.",
)
@click.option(
    "--domain_alignment",
    "--da",
    "-a",
    multiple=True,
    type=click.Tuple([str, float]),
    help=(
        "Domain alignment proportions. " "Format: 'domain1,domain2,...,domainN prop'."
    ),
)
def add_alignment_split(
    seed: int,
    dataset_path: str,
    max_train_size: int | None,
    domain_alignment: list[tuple[str, float]],
) -> None:
    dataset_location = Path(dataset_path)
    assert dataset_location.exists()

    create_domain_split(seed, dataset_location, domain_alignment, max_train_size)


def print_table(data: Mapping[str, Sequence[str]], space: int = 2) -> str:
    max_per_col = {}
    length = 0
    for colname, col_data in data.items():
        length = len(col_data)
        if length:
            max_size = max(map(len, col_data))
            max_per_col[colname] = max(max_size, len(colname))
        else:
            max_per_col[colname] = len(colname)
    text = ""
    for colname in data:
        text += colname.ljust(max_per_col[colname] + space)
    text += "\n"
    if not length:
        return text
    for k in range(length):
        for colname, coldata in data.items():
            text += coldata[k].ljust(max_per_col[colname] + space)
        text += "\n"
    return text


@alignment_group.command("list", help="List available domain splits")
@click.option(
    "--dataset_path",
    "-p",
    default="./",
    type=str,
    help="Path to the dataset",
)
@click.option(
    "--split",
    default="all",
    type=str,
    help="Which split to show (all, train, val, test)",
)
@click.option(
    "--seed",
    default="all",
    type=str,
    help="Which seed to show (all, a seed)",
)
def list_domain_alignment(dataset_path: str, split: str, seed: str) -> None:
    dataset_location = Path(dataset_path)
    assert dataset_location.exists()
    split_path = dataset_location / "domain_splits_v2"
    table: dict[str, list[str]] = {"split": [], "max_size": [], "seed": [], "props": []}
    for file in split_path.iterdir():
        if file.name.endswith("_domain_split.npy"):
            parts = file.name.removesuffix("_domain_split.npy").split("_")
            selected_split = parts.pop(0)
            if split != "all" and selected_split != split:
                continue
            selected_max_size = parts.pop().removeprefix("ms:")
            selected_seed = parts.pop().removeprefix("seed:")
            if seed != "all" and selected_seed != seed:
                continue
            table["split"].append(selected_split)
            table["max_size"].append(selected_max_size)
            table["seed"].append(selected_seed)
            table["props"].append("_".join(parts))
            # props = {}
            # for part in parts:
            #     domains, prop = part.split(":")
            #     props[frozenset(domains.split(","))] = float(prop)
    click.echo(print_table(table))


@alignment_group.command("update-format", help="Update alignment split format")
@click.option(
    "--dataset_path",
    "-p",
    default="./",
    type=str,
    help="Path to the dataset",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Whether to do a dry-run pass",
)
def update_alignment_split_format(dataset_path: str, dry_run: bool) -> None:
    dataset_location = Path(dataset_path)
    assert dataset_location.exists()
    split_path = dataset_location / "domain_splits_v2"
    if not split_path.exists():
        shutil.copytree(dataset_location / "domain_splits", split_path)
    for file in split_path.iterdir():
        if file.name.endswith("_domain_split.npy"):
            if ":1.0_" not in file.name:
                click.echo(
                    f"Cannot automatically update this file: {file.name}. SKIPPING"
                )
                continue
            parts = file.name.removesuffix("_domain_split.npy").split("_")
            split = parts.pop(0)
            seed = parts.pop()
            domains_str = "_".join(parts)
            domain_split = np.load(file, allow_pickle=True).item()
            max_size = max(map(len, domain_split.values()))
            new_file = f"{split}_{domains_str}_{seed}_ms:{max_size}_domain_split.npy"
            if dry_run:
                print(f"RENAME {file.name} -> {new_file}")
            else:
                file.rename(file.with_name(new_file))

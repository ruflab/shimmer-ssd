import csv
import logging
from pathlib import Path

import click
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from simple_shapes_dataset.cli.ood_splits import filter_dataset, ood_split
from simple_shapes_dataset.text import composer

from .utils import (
    ShapeDependentColorSampler,
    generate_dataset,
    get_deterministic_name,
    get_domain_alignment,
    load_labels,
    save_bert_latents,
    save_dataset,
    save_labels,
)

matplotlib.use("Agg")


def create_unpaired_attributes(
    seed: int,
    dataset_location: Path,
) -> None:
    assert dataset_location.exists()

    num_train_ex = np.load(dataset_location / "train_labels.npy").shape[0]
    num_val_ex = np.load(dataset_location / "val_labels.npy").shape[0]
    num_test_ex = np.load(dataset_location / "test_labels.npy").shape[0]

    np.random.seed(seed)
    train_unpaired = np.random.normal(size=(num_train_ex, 32))
    val_unpaired = np.random.normal(size=(num_val_ex, 32))
    test_unpaired = np.random.normal(size=(num_test_ex, 32))

    np.save(dataset_location / "train_unpaired.npy", train_unpaired)
    np.save(dataset_location / "val_unpaired.npy", val_unpaired)
    np.save(dataset_location / "test_unpaired.npy", test_unpaired)


@click.command("create", help="Create a Simple Shapes dataset")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option("--img_size", default=32, type=int, help="Size of the images")
@click.option(
    "--output_path",
    "-o",
    default="./",
    type=str,
    help="Where to save the dataset",
)
@click.option(
    "--num_train_examples",
    "--ntrain",
    default=500_000,
    type=int,
    help="Number of training examples",
)
@click.option(
    "--num_val_examples",
    "--nval",
    default=1_000,
    type=int,
    help="Number of validation examples",
)
@click.option(
    "--num_test_examples",
    "--ntest",
    default=1_000,
    type=int,
    help="Number of test examples",
)
@click.option(
    "--min_scale",
    default=7,
    type=int,
    help="Minimum size of the shapes (in pixels)",
)
@click.option(
    "--max_scale",
    default=14,
    type=int,
    help="Maximum size of the shapes (in pixels)",
)
@click.option(
    "--min_lightness",
    default=46,
    type=int,
    help="Minimum lightness for the shapes' HSL color. Higher values are lighter.",
)
@click.option(
    "--max_lightness",
    default=256,
    type=int,
    help="Maximum lightness for the shapes' HSL color. Higher values are lighter.",
)
@click.option(
    "--bert_path",
    "-b",
    default="bert-base-uncased",
    type=str,
    help="Pretrained BERT model to use",
)
@click.option(
    "--alignment_train_max_idx",
    default=-1,
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
def create_dataset(
    seed: int,
    img_size: int,
    output_path: str,
    num_train_examples: int,
    num_val_examples: int,
    num_test_examples: int,
    min_scale: int,
    max_scale: int,
    min_lightness: int,
    max_lightness: int,
    bert_path: str,
    alignment_train_max_idx: int,
    domain_alignment: list[tuple[str, float]],
) -> None:
    dataset_location = Path(output_path)
    dataset_location.mkdir(exist_ok=True)

    np.random.seed(seed)

    color_ranges = {
        0: (-30, 30),
        1: (30, 90),
        2: (90, 150),
    }
    color_sampler = ShapeDependentColorSampler(color_ranges)
    train_labels = generate_dataset(
        num_train_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
        color_sampler=color_sampler.set_in_dist(),
    )
    val_labels_in_dist = generate_dataset(
        num_val_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
        color_sampler=color_sampler.set_in_dist(),
    )
    val_labels_ood = generate_dataset(
        num_val_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
        color_sampler=color_sampler.set_ood(),
    )
    test_labels_in_dist = generate_dataset(
        num_test_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
        color_sampler=color_sampler.set_in_dist(),
    )
    test_labels_ood = generate_dataset(
        num_test_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
        color_sampler=color_sampler.set_ood(),
    )

    print("Save labels...")
    save_labels(dataset_location / "train_labels.npy", train_labels)
    save_labels(dataset_location / "val_labels.npy", val_labels_in_dist)
    save_labels(dataset_location / "val_ood_labels.npy", val_labels_ood)
    save_labels(dataset_location / "test_labels.npy", test_labels_in_dist)
    save_labels(dataset_location / "test_ood_labels.npy", test_labels_ood)

    create_domain_split(
        seed, dataset_location, domain_alignment, alignment_train_max_idx
    )

    print("Saving training set...")
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, img_size)
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels_in_dist, img_size)
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels_in_dist, img_size)
    print("Saving ood splits")
    (dataset_location / "ood").mkdir(exist_ok=True)
    print("Saving validation set...")
    (dataset_location / "ood" / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "ood" / "val", val_labels_ood, img_size)
    print("Saving test set...")
    (dataset_location / "ood" / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "ood" / "test", test_labels_ood, img_size)

    print("Saving captions...")
    for split in ["train", "val", "test", "val_ood", "test_ood"]:
        labels = np.load(str(dataset_location / f"{split}_labels.npy"))
        captions = []
        choices = []
        for k in tqdm(range(labels.shape[0]), total=labels.shape[0]):
            caption, choice = composer(
                {
                    "shape": int(labels[k][0]),
                    "rotation": labels[k][4],
                    "color": (labels[k][5], labels[k][6], labels[k][7]),
                    "size": labels[k][3],
                    "location": (labels[k][1], labels[k][2]),
                }
            )
            captions.append(caption)
            choices.append(choice)
        np.save(str(dataset_location / f"{split}_captions.npy"), captions)
        np.save(str(dataset_location / f"{split}_caption_choices.npy"), choices)

        save_bert_latents(
            captions,
            bert_path,
            dataset_location,
            split,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            compute_statistics=(split == "train"),
        )

    create_unpaired_attributes(seed, dataset_location)


def create_domain_split(
    seed: int,
    dataset_path: Path,
    domain_alignment: list[tuple[str, float]],
    train_max_index: int = -1,
):
    if not len(domain_alignment):
        return
    np.random.seed(seed)
    split_path = dataset_path / "domain_splits"
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

    split_name = get_deterministic_name(domain_sets, seed)

    for split in ["train", "val", "test"]:
        labels = np.load(str(dataset_path / f"{split}_labels.npy"))
        allowed_indices = np.arange(labels.shape[0])
        if split == "train":
            allowed_indices = allowed_indices[:train_max_index]
        domain_split = get_domain_alignment(
            seed,
            allowed_indices,
            domain_sets,
        )

        np.save(
            split_path / f"{split}_{split_name}_domain_split.npy",
            domain_split,  # type: ignore
        )


@click.command("alignment", help="Create a dataset alignment split")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option(
    "--dataset_path",
    "-p",
    default="./",
    type=str,
    help="Path to the dataset",
)
@click.option(
    "--alignment_train_max_idx",
    default=-1,
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
    alignment_train_max_idx: int,
    domain_alignment: list[tuple[str, float]],
) -> None:
    dataset_location = Path(dataset_path)
    assert dataset_location.exists()

    create_domain_split(
        seed, dataset_location, domain_alignment, alignment_train_max_idx
    )


@click.command("unpaired", help="Create an unpaired attribute for each domains")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option(
    "--dataset_path",
    "-d",
    default="./",
    type=str,
    help="Location to the dataset",
)
def unpaired_attributes_command(
    seed: int,
    dataset_path: str,
) -> None:
    create_unpaired_attributes(seed, Path(dataset_path))


@click.command("ood", help="Create Out-Of-Distribution splits for the dataset.")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option(
    "--dataset_path",
    "-d",
    default="./",
    type=str,
    help="Location to the dataset",
)
@click.option(
    "--min_scale",
    default=7,
    type=int,
    help="Minimum size of the shapes (in pixels)",
)
@click.option(
    "--max_scale",
    default=14,
    type=int,
    help="Maximum size of the shapes (in pixels)",
)
def create_ood_split(
    seed: int,
    dataset_path: str,
    min_scale: int,
    max_scale: int,
) -> None:
    dataset_location = Path(dataset_path)
    assert dataset_location.exists()
    split_location = dataset_location / "ood_splits"
    split_location.mkdir(exist_ok=True)

    np.random.seed(seed)

    boundary_infos = ood_split(32, min_scale, max_scale)
    with open(split_location / f"boundaries_{seed}.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["kind", "min", "max"])
        for bound_info in boundary_infos:
            writer.writerow([bound_info.kind.value, *bound_info.boundary.description()])

    train_set = load_labels(dataset_location / "train_labels.npy")
    train_in_dist, train_ood = filter_dataset(train_set, boundary_infos)
    np.save(split_location / f"train_in_dist_{seed}.npy", train_in_dist)
    np.save(split_location / f"train_ood_{seed}.npy", train_ood)

    val_set = load_labels(dataset_location / "val_labels.npy")
    val_in_dist, val_ood = filter_dataset(val_set, boundary_infos)
    np.save(split_location / f"val_in_dist_{seed}.npy", val_in_dist)
    np.save(split_location / f"val_ood_{seed}.npy", val_ood)

    test_set = load_labels(dataset_location / "test_labels.npy")
    test_in_dist, test_ood = filter_dataset(test_set, boundary_infos)
    np.save(split_location / f"test_in_dist_{seed}.npy", test_in_dist)
    np.save(split_location / f"test_ood_{seed}.npy", test_ood)

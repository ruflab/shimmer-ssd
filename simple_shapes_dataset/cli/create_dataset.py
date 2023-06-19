import logging
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from simple_shapes_dataset.text import composer

from .utils import (
    generate_dataset,
    get_deterministic_name,
    get_domain_split,
    save_bert_latents,
    save_dataset,
    save_labels,
)


@click.command("create")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option(
    "--img_size", "--is", default=32, type=int, help="Size of the images"
)
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
    "--domain_alignment",
    "-a",
    multiple=True,
    type=click.Tuple([str, float]),
    help="Domain alignment proportions. Format: 'domain1,domain2,...,domainN prop'.",
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
    domain_alignment: list[tuple[str, float]],
) -> None:
    dataset_location = Path(output_path)
    dataset_location.mkdir(exist_ok=True)

    np.random.seed(seed)

    domain_sets = {}
    for domain, prop in domain_alignment:
        domain_set = frozenset(domain.split(","))
        if domain_set in domain_sets:
            logging.warning(
                f"Domain set {domain_set} is defined multiple times. "
                f"The value will be overwritten by {prop}."
            )
        domain_sets[domain_set] = prop

    split_name = get_deterministic_name(domain_alignment, seed)

    train_labels = generate_dataset(
        num_train_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
    )
    val_labels = generate_dataset(
        num_val_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
    )
    test_labels = generate_dataset(
        num_test_examples,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        img_size,
    )

    print("Save labels...")
    save_labels(dataset_location / "train_labels.npy", train_labels)
    save_labels(dataset_location / "val_labels.npy", val_labels)
    save_labels(dataset_location / "test_labels.npy", test_labels)

    print("Saving training set...")
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, img_size)
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels, img_size)
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels, img_size)

    print("Saving captions...")
    for split in ["train", "val", "test"]:
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
        np.save(
            str(dataset_location / f"{split}_caption_choices.npy"), choices
        )

        save_bert_latents(
            captions,
            bert_path,
            dataset_location / f"{split}_latent.npy",
            torch.device("cuda"),
            compute_statistics=(split == "train"),
        )

        domain_split = get_domain_split(
            labels.shape[0],
            domain_sets,
        )

        np.save(
            dataset_location / f"{split}_{split_name}_domain_split.npy",
            domain_split,  # type: ignore
        )
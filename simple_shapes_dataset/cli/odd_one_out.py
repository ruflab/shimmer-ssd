import random
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm


def closest_shape(ref, labels, keys):
    dists = np.linalg.norm(labels[:, keys] - ref[keys], axis=1)
    return np.argsort(dists)[1]


def select_odd_one_out(ref1, ref2, labels):
    dists = np.minimum(
        np.min(np.abs(labels - ref1), axis=1),
        np.min(np.abs(labels - ref2), axis=1),
    )
    sorted_dists = np.argsort(-dists)
    return np.random.choice(sorted_dists[: sorted_dists.shape[0] // 1000], 1)[0]


def normalize_labels(labels):
    labels -= labels.min(axis=0)
    labels /= labels.max(axis=0)
    return labels


@click.command("ooo", help="Create the odd-one-out dataset.")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option(
    "--train_dataset_path",
    default=None,
    type=str,
    help="Location to the dataset",
)
@click.option(
    "--dataset_path",
    "-d",
    default="./",
    type=str,
    help="Location to the dataset",
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
def create_odd_one_out_dataset(
    seed: int,
    train_dataset_path: str | None,
    dataset_path: str,
    num_train_examples: int,
    num_val_examples: int,
    num_test_examples: int,
) -> None:
    dataset_location = Path(dataset_path)
    assert dataset_location.exists()

    np.random.seed(seed)

    possible_keys = [[0], [1, 2], [3], [4], [5, 6, 7]]

    n_examples = {
        "train": num_train_examples,
        "val": num_val_examples,
        "test": num_test_examples,
    }

    for split in ["train", "val", "test"]:
        if split == "train" and train_dataset_path is not None:
            labels = normalize_labels(
                np.load(str(Path(train_dataset_path) / f"{split}_labels.npy"))[:, :8]
            )
        else:
            labels = normalize_labels(
                np.load(str(dataset_location / f"{split}_labels.npy"))[:, :8]
            )
        dataset = []

        for i in tqdm(range(n_examples[split]), total=n_examples[split]):
            ref = labels[i]
            key = random.choice(possible_keys)
            closest_key = closest_shape(ref, labels, key)
            odd_one_out = select_odd_one_out(ref, labels[closest_key], labels)
            order = np.random.permutation(3)
            idx = [i, closest_key, odd_one_out]
            dataset.append(
                [
                    idx[order[0]],
                    idx[order[1]],
                    idx[order[2]],
                    np.where(order == 2)[0][0],
                ]
            )
        np.save(str(dataset_location / f"{split}_odd_one_out_labels.npy"), dataset)

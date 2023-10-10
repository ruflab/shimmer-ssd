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
    return np.random.choice(sorted_dists[: sorted_dists.shape[0] // 1000], 1)[
        0
    ]


def normalize_labels(labels):
    labels -= labels.min(axis=0)
    labels /= labels.max(axis=0)
    return labels


@click.command("ooo", help="Create the odd-one-out dataset.")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option(
    "--dataset_path",
    "-d",
    default="./",
    type=str,
    help="Location to the dataset",
)
def create_odd_one_out_dataset(
    seed: int,
    dataset_path: str,
) -> None:
    dataset_location = Path(dataset_path)
    assert dataset_location.exists()

    np.random.seed(seed)

    split = "train"

    possible_keys = [[0], [1, 2], [3], [4], [5, 6, 7]]
    all_labels = np.load(str(dataset_location / f"{split}_labels.npy"))[:, :8]
    all_labels = normalize_labels(all_labels)

    for split in ["train", "val", "test"]:
        dataset = []
        if split == "train":
            n_examples = 500_000
            labels = all_labels[:500_000]
        elif split == "val":
            n_examples = 1000
            labels = all_labels[500_000:750_000]
        else:
            n_examples = 1000
            labels = all_labels[750_000:]

        for i in tqdm(range(n_examples), total=n_examples):
            ref = labels[i]
            key = random.choice(possible_keys)
            closest_key = closest_shape(ref, labels, key)
            rd = select_odd_one_out(ref, labels[closest_key], labels)
            order = np.random.permutation(3)
            idx = [i, closest_key, rd]
            dataset.append(
                [
                    idx[order[0]],
                    idx[order[1]],
                    idx[order[2]],
                    np.where(order == 2)[0][0],
                ]
            )
        np.save(
            str(dataset_location / f"{split}_odd_image_labels.npy"), dataset
        )

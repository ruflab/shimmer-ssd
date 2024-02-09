from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Subset

from simple_shapes_dataset.cli.utils import get_deterministic_name
from simple_shapes_dataset.dataset.dataset import SimpleShapesDataset


def get_alignment(
    dataset_path: str | Path,
    split: str,
    domain_proportions: Mapping[frozenset[str], float],
    seed: int,
) -> Mapping[frozenset[str], np.ndarray]:
    assert split in ["train", "val", "test"]

    dataset_path = Path(dataset_path)

    alignment_split_name = get_deterministic_name(domain_proportions, seed)

    alignement_split_path = (
        dataset_path / f"domain_splits/{split}_{alignment_split_name}_domain_split.npy"
    )
    if not alignement_split_path.exists():
        domain_alignment = [
            f'--domain_alignment {",".join(sorted(list(domain)))} {prop}'
            for domain, prop in domain_proportions.items()
        ]
        raise ValueError(
            "Domain split not found. "
            "To create it, use `shapesd alignment "
            f'--dataset_path "{str(dataset_path.resolve())}" '
            f"--seed {seed} {' '.join(domain_alignment)}`"
        )
    domain_split: Mapping[frozenset[str], np.ndarray] = np.load(
        alignement_split_path, allow_pickle=True
    ).item()

    return domain_split


def get_aligned_datasets(
    dataset_path: str | Path,
    split: str,
    domain_proportions: Mapping[frozenset[str], float],
    seed: int,
    transforms: Mapping[str, Callable[[Any], Any]] | None = None,
    domain_args: Mapping[str, Any] | None = None,
) -> dict[frozenset[str], Subset]:
    domain_split = get_alignment(dataset_path, split, domain_proportions, seed)

    datasets: dict[frozenset[str], Subset] = {}
    for domain_group, indices in domain_split.items():
        dataset = SimpleShapesDataset(
            dataset_path,
            split,
            list(domain_group),
            transforms,
            domain_args,
        )
        domains = frozenset(dataset.domains.keys())

        datasets[domains] = Subset(dataset, indices.tolist())

    return datasets

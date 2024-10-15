from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from shimmer import DataDomain, DomainDesc, ShimmerDataset
from torch.utils.data import Subset

from simple_shapes_dataset.cli.utils import get_deterministic_name


def get_alignment(
    dataset_path: str | Path,
    split: str,
    domain_proportions: Mapping[frozenset[str], float],
    seed: int,
    max_size: int | None,
) -> Mapping[frozenset[str], np.ndarray]:
    assert split in ["train", "val", "test"]

    dataset_path = Path(dataset_path)
    if max_size is None:
        max_size = np.load(dataset_path / f"{split}_labels.npy").shape[0]
    assert max_size is not None, "Error loading label file."

    alignment_split_name = get_deterministic_name(domain_proportions, seed, max_size)

    alignment_split_path = (
        dataset_path
        / f"domain_splits_v2/{split}_{alignment_split_name}_domain_split.npy"
    )
    if not alignment_split_path.exists():
        domain_alignment = [
            f'--domain_alignment {",".join(sorted(list(domain)))} {prop}'
            for domain, prop in domain_proportions.items()
        ]
        raise ValueError(
            "Domain split not found. "
            "To create it, use `shapesd alignment add "
            f'--dataset_path "{str(dataset_path.resolve())}" '
            f"--seed {seed} {' '.join(domain_alignment)} "
            f"--ms {max_size}`"
        )
    domain_split: Mapping[frozenset[str], np.ndarray] = np.load(
        alignment_split_path, allow_pickle=True
    ).item()

    return domain_split


def get_aligned_datasets(
    dataset_path: str | Path,
    split: str,
    domain_classes: Mapping[DomainDesc, type[DataDomain]],
    domain_proportions: Mapping[frozenset[str], float],
    seed: int,
    max_size: int | None = None,
    transforms: Mapping[str, Callable[[Any], Any]] | None = None,
    domain_args: Mapping[str, Any] | None = None,
) -> dict[frozenset[str], Subset]:
    domain_split = get_alignment(
        dataset_path, split, domain_proportions, seed, max_size
    )

    datasets: dict[frozenset[str], Subset] = {}
    for domain_group, indices in domain_split.items():
        sub_domain_cls = {
            domain_type: domain_cls
            for domain_type, domain_cls in domain_classes.items()
            if domain_type.base in domain_group
        }
        dataset = ShimmerDataset(
            dataset_path,
            split,
            sub_domain_cls,
            max_size,
            transforms,
            domain_args,
        )
        domains = frozenset(dataset.domains.keys())

        datasets[domains] = Subset(dataset, indices.tolist())

    return datasets

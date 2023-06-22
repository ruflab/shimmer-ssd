from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch.utils.data as torchdata

from simple_shapes_dataset.cli.utils import get_deterministic_name
from simple_shapes_dataset.modules.domain import AVAILABLE_DOMAINS


class SimpleShapesDataset(torchdata.Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        selected_domains: list[str],
        domain_proportions: dict[frozenset[str], float],
        seed: int,
        transforms: dict[str, Callable[[Any], Any]] | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.domain_proportions = domain_proportions

        self.selected_domains = selected_domains
        self.domains: dict[str, Sequence] = {}

        for domain in self.selected_domains:
            transform = None
            if transforms is not None and domain in transforms:
                transform = transforms[domain]
            self.domains[domain] = AVAILABLE_DOMAINS[domain](
                dataset_path, split, transform
            )

        domain_split_name = get_deterministic_name(
            domain_proportions, seed
        )

        domain_split_path = (
            self.dataset_path
            / f"{split}_{domain_split_name}_domain_split.npy"
        )
        if not domain_split_path.exists():
            domain_alignment = [
                f'--domain_alignment {",".join(sorted(list(domain)))} {prop}'
                for domain, prop in domain_proportions.items()
            ]
            raise ValueError(
                "Domain split not found. "
                "To create it, use `shapesd split "
                f'--dataset_path "{str(self.dataset_path.resolve())}" '
                f"--seed {seed} {' '.join(domain_alignment)}`"
            )
        self.domain_split = np.load(
            domain_split_path, allow_pickle=True
        ).item()

    def __len__(self) -> int:
        for domain in self.domains.values():
            return len(domain)
        return 0

    def __getitem__(self, index) -> dict[str, Any]:
        return {
            domain_name: domain[index]
            for domain_name, domain in self.domains.items()
        }

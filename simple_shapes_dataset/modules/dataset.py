from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch.utils.data as torchdata

from simple_shapes_dataset.modules.domain import AVAILABLE_DOMAINS


class SimpleShapesDataset(torchdata.Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        selected_domains: list[str],
        transforms: dict[str, Callable[[Any], Any]] | None = None,
        domain_args: dict[str, Any] | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split

        self.selected_domains = selected_domains
        self.domains: dict[str, Sequence] = {}
        self.domain_args = domain_args or {}

        for domain in self.selected_domains:
            transform = None
            if transforms is not None and domain in transforms:
                transform = transforms[domain]
            self.domains[domain] = AVAILABLE_DOMAINS[domain](
                dataset_path,
                split,
                transform,
                self.domain_args.get(domain, None),
            )

    def __len__(self) -> int:
        for domain in self.domains.values():
            return len(domain)
        return 0

    def __getitem__(self, index) -> dict[str, Any]:
        return {
            domain_name: domain[index]
            for domain_name, domain in self.domains.items()
        }

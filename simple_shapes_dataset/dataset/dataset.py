from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch.utils.data as torchdata

from simple_shapes_dataset.dataset.domain import (
    AVAILABLE_DOMAINS,
    SimpleShapesDomain,
)


class SimpleShapesDataset(torchdata.Dataset):
    """
    Dataset class to obtain a SimpleShapesDataset.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        selected_domains: Iterable[str],
        transforms: dict[str, Callable[[Any], Any]] | None = None,
        domain_args: dict[str, Any] | None = None,
    ):
        """
        Params:
            dataset_path (str | pathlib.Path): Path to the dataset.
            split (str): Split to use. One of 'train', 'val', 'test'.
            selected_domains (Iterable[str]): Domains to include in the dataset.
                If "v" is given and "v_latents" key is in domain_args, then "v" is
                replaced by the "v_latents" domain.
            transforms (dict[str, (Any) -> Any]): Optional transforms to apply
                to the domains. The keys are the domain names,
                the values are the transforms.
            domain_args (dict[str, Any]): Optional additional arguments to pass
                to the domains.
        """
        self.dataset_path = Path(dataset_path)
        self.split = split

        self.domains: dict[str, SimpleShapesDomain] = {}
        self.domain_args = domain_args or {}

        for domain in selected_domains:
            if domain == "v" and "v_latents" in self.domain_args:
                domain = "v_latents"

            transform = None
            if transforms is not None and domain in transforms:
                transform = transforms[domain]

            self.domains[domain] = AVAILABLE_DOMAINS[domain](
                dataset_path,
                split,
                transform,
                self.domain_args.get(domain, None),
            )

        lengths = {len(domain) for domain in self.domains.values()}
        assert len(lengths) == 1, "Domains have different lengths"

    def __len__(self) -> int:
        """
        All domains should be the same length.
        Returns the length of the first domain.
        """
        for domain in self.domains.values():
            return len(domain)
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Params:
            index (int): Index of the item to get.
        Returns:
            dict[str, Any]: Dictionary containing the domains. The keys are the
            domain names, the values are the domains as given by the domain model at
            the given index.
        """
        return {
            domain_name: domain[index]
            for domain_name, domain in self.domains.items()
        }

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch.utils.data as torchdata

from simple_shapes_dataset.dataset.domain import SimpleShapesDomain


class SizedDataset(torchdata.Dataset):
    def __len__(self) -> int:
        raise NotImplementedError


class SimpleShapesDataset(SizedDataset):
    """
    Dataset class to obtain a SimpleShapesDataset.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        domain_classes: Mapping[str, type[SimpleShapesDomain]],
        max_size: int = -1,
        transforms: Mapping[str, Callable[[Any], Any]] | None = None,
        domain_args: Mapping[str, Any] | None = None,
    ):
        """
        Params:
            dataset_path (str | pathlib.Path): Path to the dataset.
            split (str): Split to use. One of 'train', 'val', 'test'.
            domain_classes (Mapping[str, type[SimpleShapesDomain]]): Classes of
                domain loaders to include in the dataset.
            max_size (int): Max size of the dataset.
            transforms (Mapping[str, (Any) -> Any]): Optional transforms to apply
                to the domains. The keys are the domain names,
                the values are the transforms.
            domain_args (Mapping[str, Any]): Optional additional arguments to pass
                to the domains.
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_size = max_size

        self.domains: dict[str, SimpleShapesDomain] = {}
        self.domain_args = domain_args or {}

        for domain, domain_cls in domain_classes.items():
            transform = None
            if transforms is not None and domain in transforms:
                transform = transforms[domain]

            self.domains[domain] = domain_cls(
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
        if self.max_size != -1:
            return self.max_size
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
            domain_name: domain[index] for domain_name, domain in self.domains.items()
        }

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import Compose, ToTensor

from simple_shapes_dataset.dataset.dataset import SimpleShapesDataset
from simple_shapes_dataset.dataset.domain_alignment import get_aligned_datasets
from simple_shapes_dataset.dataset.pre_process import (
    NormalizeAttributes,
    attribute_to_tensor,
)


class SimpleShapesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str | Path,
        domain_proportions: dict[frozenset[str], float],
        batch_size: int,
        num_workers: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.domain_proportions = domain_proportions
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Mapping[frozenset[str], Dataset] | None = None
        self.val_dataset: Mapping[frozenset[str], Dataset] | None = None
        self.test_dataset: Mapping[frozenset[str], Dataset] | None = None

    def _get_transforms(
        self, domains: Iterable[str]
    ) -> dict[str, Callable[[Any], Any]]:
        transforms = {}
        for domain in domains:
            if domain == "attr":
                transforms[domain] = Compose(
                    [NormalizeAttributes(image_size=32), attribute_to_tensor]
                )
            if domain == "v":
                transforms[domain] = ToTensor()
        return transforms

    def _require_aligned_dataset(self) -> bool:
        for domain, prop in self.domain_proportions.items():
            if len(domain) > 1 or prop < 1:
                return True
        return False

    def _get_selected_domains(self) -> set[str]:
        selected_domains: set[str] = set()
        for domain in self.domain_proportions.keys():
            selected_domains.update(domain)
        return selected_domains

    def _get_dataset(self, split: str) -> Mapping[frozenset[str], Dataset]:
        assert split in ("train", "val", "test")

        domains = self._get_selected_domains()

        if split == "train" and self._require_aligned_dataset():
            if self.seed is None:
                raise ValueError(
                    "Seed must be provided when using aligned dataset"
                )

            return get_aligned_datasets(
                self.dataset_path,
                split=split,
                domain_proportions=self.domain_proportions,
                seed=self.seed,
                transforms=self._get_transforms(domains),
            )

        if split in ("val", "test"):
            return {
                frozenset(domains): SimpleShapesDataset(
                    self.dataset_path,
                    split=split,
                    selected_domains=domains,
                    transforms=self._get_transforms(domains),
                )
            }
        return {
            frozenset([domain]): SimpleShapesDataset(
                self.dataset_path,
                split=split,
                selected_domains=[domain],
                transforms=self._get_transforms([domain]),
            )
            for domain in domains
        }

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self._get_dataset("train")

        self.val_dataset = self._get_dataset("val")
        self.test_dataset = self._get_dataset("test")

    def get_samples(
        self, split: str, amount: int
    ) -> dict[frozenset[str], dict[str, Any]]:
        datasets = self._get_dataset(split)

        return {
            domain: default_collate([dataset[k] for k in range(amount)])
            for domain, dataset in datasets.items()
        }

    def train_dataloader(
        self,
    ) -> CombinedLoader:
        assert self.train_dataset is not None

        dataloaders = {}
        for domain, dataset in self.train_dataset.items():
            dataloaders[domain] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def val_dataloader(
        self,
    ) -> CombinedLoader:
        assert self.val_dataset is not None

        dataloaders = {}
        for domain, dataset in self.val_dataset.items():
            dataloaders[domain] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        return CombinedLoader(dataloaders, mode="sequential")

    def test_dataloader(
        self,
    ) -> CombinedLoader:
        assert self.test_dataset is not None

        dataloaders = {}
        for domain, dataset in self.test_dataset.items():
            dataloaders[domain] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        return CombinedLoader(dataloaders, mode="sequential")

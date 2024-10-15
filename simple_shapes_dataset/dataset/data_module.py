from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from shimmer import DataDomain, DomainDesc, RepeatedDataset, ShimmerDataset
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision.transforms import Compose, ToTensor

from simple_shapes_dataset.dataset.domain_alignment import get_aligned_datasets
from simple_shapes_dataset.dataset.pre_process import (
    NormalizeAttributes,
    TextAndAttrs,
    attribute_to_tensor,
)

DatasetT = ShimmerDataset | Subset


class SimpleShapesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str | Path,
        domain_classes: Mapping[DomainDesc, type[DataDomain]],
        domain_proportions: Mapping[frozenset[str], float],
        batch_size: int,
        max_train_size: int | None = None,
        num_workers: int = 0,
        seed: int | None = None,
        ood_seed: int | None = None,
        domain_args: Mapping[str, Any] | None = None,
        additional_transforms: (
            Mapping[str, Sequence[Callable[[Any], Any]]] | None
        ) = None,
        train_transforms: (Mapping[str, Sequence[Callable[[Any], Any]]] | None) = None,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        use_default_transforms: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.domain_classes = domain_classes
        self.domain_proportions = domain_proportions
        self.seed = seed
        self.ood_seed = ood_seed
        self.domain_args = domain_args or {}
        self.additional_transforms = additional_transforms or {}
        self._train_transform = train_transforms or {}
        self._use_default_transforms = use_default_transforms

        self.max_train_size = max_train_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Mapping[frozenset[str], DatasetT] | None = None
        self.val_dataset: Mapping[frozenset[str], DatasetT] | None = None
        self.test_dataset: Mapping[frozenset[str], DatasetT] | None = None

        self.train_dataset_ood: Mapping[frozenset[str], DatasetT] | None = None
        self.val_dataset_ood: Mapping[frozenset[str], DatasetT] | None = None
        self.test_dataset_ood: Mapping[frozenset[str], DatasetT] | None = None

        self._collate_fn = collate_fn

    def _get_transforms(
        self, domains: Iterable[str], mode: str
    ) -> dict[str, Callable[[Any], Any]]:
        transforms: dict[str, Callable[[Any], Any]] = {}
        for domain in domains:
            domain_transforms: list[Callable[[Any], Any]] = []
            if domain == "attr" and self._use_default_transforms:
                domain_transforms.extend(
                    [
                        NormalizeAttributes(image_size=32),
                        attribute_to_tensor,
                    ]
                )

            if domain == "v" and self._use_default_transforms:
                domain_transforms.append(ToTensor())

            if domain == "t" and self._use_default_transforms:
                domain_transforms.append(TextAndAttrs(image_size=32))

            if domain in self.additional_transforms:
                domain_transforms.extend(self.additional_transforms[domain])
            if domain in self._train_transform and mode == "train":
                domain_transforms.extend(self._train_transform[domain])
            transforms[domain] = Compose(domain_transforms)
        return transforms

    def _requires_aligned_dataset(self) -> bool:
        for domain, prop in self.domain_proportions.items():
            if len(domain) > 1 or prop < 1:
                return True
        return False

    def _get_selected_domains(self) -> set[str]:
        return {domain.kind for domain in self.domain_classes}

    def _get_dataset(self, split: str) -> Mapping[frozenset[str], DatasetT]:
        assert split in ("train", "val", "test")

        domains = self._get_selected_domains()

        if split == "train" and self._requires_aligned_dataset():
            if self.seed is None:
                raise ValueError("Seed must be provided when using aligned dataset")

            return get_aligned_datasets(
                self.dataset_path,
                split,
                self.domain_classes,
                self.domain_proportions,
                self.seed,
                self.max_train_size,
                self._get_transforms(domains, split),
                self.domain_args,
            )

        if split in ("val", "test"):
            return {
                frozenset(domains): ShimmerDataset(
                    self.dataset_path,
                    split,
                    self.domain_classes,
                    transforms=self._get_transforms(domains, split),
                    domain_args=self.domain_args,
                )
            }
        return {
            frozenset([domain]): ShimmerDataset(
                self.dataset_path,
                split,
                {
                    domain_type: domain_cls
                    for domain_type, domain_cls in self.domain_classes.items()
                    if domain_type.kind == domain
                },
                self.max_train_size,
                self._get_transforms([domain], split),
                self.domain_args,
            )
            for domain in domains
        }

    def _filter_ood(
        self,
        dataset: Mapping[frozenset[str], DatasetT],
        split: Literal["train", "val", "test"],
    ) -> tuple[
        Mapping[frozenset[str], DatasetT],
        Mapping[frozenset[str], DatasetT] | None,
    ]:
        if self.ood_seed is None:
            return dataset, None
        split_path = self.dataset_path / "ood_splits"
        assert (split_path / f"boundaries_{self.ood_seed}.csv").exists()
        in_dist: list[int] = np.load(
            split_path / f"{split}_in_dist_{self.ood_seed}.npy"
        )
        ood: list[int] = np.load(split_path / f"{split}_ood_{self.ood_seed}.npy")
        dataset_in_dist: dict[frozenset[str], Subset] = {}
        for k, d in dataset.items():
            if isinstance(d, Subset):
                indices = list(set(d.indices).intersection(set(in_dist)))
                dataset_in_dist[k] = Subset(d.dataset, indices)
            else:
                dataset_in_dist[k] = Subset(d, in_dist)
        dataset_ood = {k: Subset(dataset[k], ood) for k in dataset}
        return (dataset_in_dist, dataset_ood)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self._get_dataset("train")
            self.train_dataset, self.train_dataset_ood = self._filter_ood(
                self.train_dataset, "train"
            )

        self.val_dataset = self._get_dataset("val")
        self.test_dataset = self._get_dataset("test")

        self.val_dataset, self.val_dataset_ood = self._filter_ood(
            self.val_dataset, "val"
        )
        self.test_dataset, self.test_dataset_ood = self._filter_ood(
            self.test_dataset, "test"
        )

    def get_samples(
        self,
        split: Literal["train", "val", "test"],
        amount: int,
        ood: bool = False,
    ) -> dict[frozenset[str], dict[str, Any]]:
        datasets = self._get_dataset(split)

        if ood:
            _, ood_datasets = self._filter_ood(datasets, split)
            assert ood_datasets is not None
            datasets = ood_datasets

        collate_fn = self._collate_fn or default_collate

        return {
            domain: collate_fn([dataset[k] for k in range(amount)])
            for domain, dataset in datasets.items()
        }

    def train_dataloader(
        self,
        shuffle=True,
        drop_last=True,
        **kwargs,
    ) -> CombinedLoader:
        assert self.train_dataset is not None

        dataloaders = {}
        max_sized_dataset = max(len(dataset) for dataset in self.train_dataset.values())
        for domain, dataset in self.train_dataset.items():
            dataloaders[domain] = DataLoader(
                RepeatedDataset(dataset, max_sized_dataset, drop_last=False),  # type: ignore
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=self._collate_fn,
                **kwargs,
            )
        return CombinedLoader(dataloaders, mode="min_size")

    def val_dataloader(
        self,
    ) -> CombinedLoader:
        assert self.val_dataset is not None

        dataloaders = {}
        for domain, dataset in self.val_dataset.items():
            dataloaders[domain] = DataLoader(
                dataset,
                pin_memory=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
            )
            if self.val_dataset_ood is not None:
                ood_domains = frozenset({d + "_ood" for d in domain})
                dataloaders[ood_domains] = DataLoader(
                    self.val_dataset_ood[domain],
                    pin_memory=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self._collate_fn,
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
                pin_memory=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
            )
            if self.test_dataset_ood is not None:
                ood_domains = frozenset({d + "_ood" for d in domain})
                dataloaders[ood_domains] = DataLoader(
                    self.test_dataset_ood[domain],
                    pin_memory=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self._collate_fn,
                )
        return CombinedLoader(dataloaders, mode="sequential")

    def predict_dataloader(self):
        assert self.val_dataset is not None

        dataloaders = {}
        for domain, dataset in self.val_dataset.items():
            dataloaders[domain] = DataLoader(
                Subset(dataset, range(self.batch_size)),
                drop_last=False,
                pin_memory=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
            )
        return CombinedLoader(dataloaders, mode="sequential")

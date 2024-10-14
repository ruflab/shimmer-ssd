from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from shimmer import DataDomain, DomainDesc, RepeatedDataset
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision.transforms import Compose, ToTensor

from simple_shapes_dataset.dataset.domain_alignment import get_alignment
from simple_shapes_dataset.dataset.downstream.odd_one_out.dataset import (
    OddOneOutDataset,
)
from simple_shapes_dataset.dataset.pre_process import (
    NormalizeAttributes,
    TextAndAttrs,
    attribute_to_tensor,
)

DatasetT = OddOneOutDataset | Subset[OddOneOutDataset]


def get_aligned_datasets(
    dataset_path: str | Path,
    split: str,
    domain_classes: Mapping[DomainDesc, type[DataDomain]],
    domain_proportions: Mapping[frozenset[str], float],
    seed: int,
    max_size: int = -1,
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
        dataset = OddOneOutDataset(
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


class OddOneOutDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str | Path,
        domain_classes: Mapping[DomainDesc, type[DataDomain]],
        domain_proportions: Mapping[frozenset[str], float],
        batch_size: int,
        max_train_size: int | None = None,
        num_workers: int = 0,
        seed: int | None = None,
        domain_args: Mapping[str, Any] | None = None,
        additional_transforms: (
            Mapping[str, Sequence[Callable[[Any], Any]]] | None
        ) = None,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.domain_classes = domain_classes
        self.domain_proportions = domain_proportions
        self.seed = seed
        self.domain_args = domain_args or {}
        self.additional_transforms = additional_transforms or {}

        self.max_train_size = max_train_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Mapping[frozenset[str], DatasetT] | None = None
        self.val_dataset: Mapping[frozenset[str], DatasetT] | None = None
        self.test_dataset: Mapping[frozenset[str], DatasetT] | None = None

    def _get_transforms(
        self, domains: Iterable[str]
    ) -> dict[str, Callable[[Any], Any]]:
        transforms: dict[str, Callable[[Any], Any]] = {}
        for domain in domains:
            domain_transforms: list[Callable[[Any], Any]] = []
            if domain == "attr":
                domain_transforms.extend(
                    [
                        NormalizeAttributes(image_size=32),
                        attribute_to_tensor,
                    ]
                )

            if domain == "v":
                domain_transforms.append(ToTensor())

            if domain == "t":
                domain_transforms.append(TextAndAttrs(image_size=32))

            if domain in self.additional_transforms:
                domain_transforms.extend(self.additional_transforms[domain])
            transforms[domain] = Compose(domain_transforms)
        return transforms

    def _require_aligned_dataset(self) -> bool:
        for domain, prop in self.domain_proportions.items():
            if len(domain) > 1 or prop < 1:
                return True
        return False

    def _get_selected_domains(self) -> set[str]:
        return {domain.kind for domain in self.domain_classes}

    def _get_dataset(self, split: str) -> Mapping[frozenset[str], DatasetT]:
        assert split in ("train", "val", "test")

        domains = self._get_selected_domains()

        if split == "train" and self._require_aligned_dataset():
            if self.seed is None:
                raise ValueError("Seed must be provided when using aligned dataset")

            return get_aligned_datasets(
                self.dataset_path,
                split,
                self.domain_classes,
                self.domain_proportions,
                self.seed,
                self.max_train_size or -1,
                self._get_transforms(domains),
                self.domain_args,
            )

        if split in ("val", "test"):
            return {
                frozenset(domains): OddOneOutDataset(
                    self.dataset_path,
                    split,
                    self.domain_classes,
                    transforms=self._get_transforms(domains),
                    domain_args=self.domain_args,
                )
            }
        return {
            frozenset([domain]): OddOneOutDataset(
                self.dataset_path,
                split,
                {
                    domain_type: domain_cls
                    for domain_type, domain_cls in self.domain_classes.items()
                    if domain_type.kind == domain
                },
                self.max_train_size or -1,
                self._get_transforms([domain]),
                self.domain_args,
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
            )
        return CombinedLoader(dataloaders, mode="sequential")

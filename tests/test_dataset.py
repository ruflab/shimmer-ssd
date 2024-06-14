from shimmer import ShimmerDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.domain import (
    get_default_domains,
)
from simple_shapes_dataset.dataset.domain_alignment import get_aligned_datasets


def test_dataset():
    dataset = ShimmerDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        domain_classes=get_default_domains(["v", "attr"]),
    )

    assert len(dataset) == 4

    item = dataset[0]
    for domain in ["v", "attr"]:
        assert domain in item


def test_dataset_val():
    dataset = ShimmerDataset(
        PROJECT_DIR / "sample_dataset",
        split="val",
        domain_classes=get_default_domains(["v", "attr"]),
    )

    assert len(dataset) == 2

    item = dataset[0]
    for domain in ["v", "attr"]:
        assert domain in item


def test_dataloader():
    transform = {
        "v": ToTensor(),
    }
    dataset = ShimmerDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        domain_classes=get_default_domains(["v", "attr"]),
        transforms=transform,
    )

    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    for domain in ["v", "attr"]:
        assert domain in item


def test_get_aligned_datasets():
    datasets = get_aligned_datasets(
        PROJECT_DIR / "sample_dataset",
        "train",
        domain_classes=get_default_domains(["v", "t"]),
        domain_proportions={
            frozenset(["v", "t"]): 0.5,
            frozenset("v"): 1.0,
            frozenset("t"): 1.0,
        },
        max_size=4,
        seed=0,
    )

    assert len(datasets) == 3
    for dataset_name, _ in datasets.items():
        assert dataset_name in [
            frozenset(["v", "t"]),
            frozenset(["v"]),
            frozenset(["t"]),
        ]


def test_datamodule():
    datamodule = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        get_default_domains(["attr"]),
        domain_proportions={
            frozenset(["attr"]): 1.0,
        },
        batch_size=2,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    item, _, _ = next(iter(train_dataloader))
    assert isinstance(item, dict)
    assert len(item) == 1
    assert frozenset(["attr"]) in item


def test_datamodule_aligned_dataset():
    datamodule = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        get_default_domains(["v", "attr"]),
        domain_proportions={
            frozenset(["v", "attr"]): 0.5,
            frozenset(["v"]): 1.0,
            frozenset(["attr"]): 1.0,
        },
        batch_size=2,
        seed=0,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    item, _, _ = next(iter(train_dataloader))
    assert isinstance(item, dict)
    for domain in item:
        assert domain in [
            frozenset(["v", "attr"]),
            frozenset(["v"]),
            frozenset(["attr"]),
        ]

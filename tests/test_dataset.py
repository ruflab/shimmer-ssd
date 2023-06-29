import torch.utils.data
import torchvision
from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset.dataset import SimpleShapesDataset
from simple_shapes_dataset.dataset.domain_alignment import get_aligned_datasets


def test_dataset():
    selected_domains = ["v", "attr"]
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_domains=selected_domains,
    )

    assert len(dataset) == 4

    item = dataset[0]
    for domain in ["v", "attr"]:
        assert domain in item


def test_dataset_val():
    selected_domains = ["v", "attr"]
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="val",
        selected_domains=selected_domains,
    )

    assert len(dataset) == 2

    item = dataset[0]
    for domain in ["v", "attr"]:
        assert domain in item


def test_dataloader():
    transform = {
        "v": torchvision.transforms.ToTensor(),
    }
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_domains=["v", "attr"],
        transforms=transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    for domain in ["v", "attr"]:
        assert domain in item


def test_get_aligned_datasets():
    datasets = get_aligned_datasets(
        PROJECT_DIR / "sample_dataset",
        "train",
        domain_proportions={
            frozenset(["v", "t"]): 0.5,
            frozenset("v"): 1.0,
            frozenset("t"): 1.0,
        },
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
        domain_proportions={
            frozenset(["attr"]): 1.0,
        },
        batch_size=2,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    assert isinstance(train_dataloader, dict)
    assert len(train_dataloader) == 1
    assert frozenset(["attr"]) in train_dataloader.keys()


def test_datamodule_aligned_dataset():
    datamodule = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        domain_proportions={
            frozenset(["v", "t"]): 0.5,
            frozenset("v"): 1.0,
            frozenset("t"): 1.0,
        },
        batch_size=2,
        seed=0,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    assert isinstance(train_dataloader, dict)
    for domain in train_dataloader.keys():
        assert domain in [
            frozenset(["v", "t"]),
            frozenset(["v"]),
            frozenset(["t"]),
        ]

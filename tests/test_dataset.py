from pathlib import Path

import torch.utils.data

from simple_shapes_dataset.modules.dataset import SimpleShapesDataset
from simple_shapes_dataset.modules.domain_alignment import get_aligned_datasets

PROJECT_DIR = Path(__file__).resolve().parents[1]


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


def test_dataloader():
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_domains=["v", "attr"],
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

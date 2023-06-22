from pathlib import Path

import torch.utils.data
import torchvision

from simple_shapes_dataset.modules.dataset import SimpleShapesDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_dataset():
    selected_domains = ["v", "attr"]
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_domains=selected_domains,
        domain_proportions={frozenset(["v", "t"]): 0.01},
        seed=0,
    )

    assert len(dataset) == 4

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
        domain_proportions={frozenset(["v", "t"]): 0.01},
        seed=0,
        transforms=transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    for domain in ["v", "attr"]:
        assert domain in item

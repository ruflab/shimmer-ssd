from pathlib import Path

import torch.utils.data
import torchvision

from simple_shapes_dataset.modules.dataset import SimpleShapesDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_dataset():
    selected_modalities = ["v", "attr"]
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_modalities=selected_modalities,
        modality_proportions={frozenset(["v", "t"]): 0.01},
        seed=0,
    )

    assert len(dataset) == 4

    item = dataset[0]
    for modality in ["v", "attr"]:
        assert modality in item


def test_dataloader():
    transform = {
        "v": torchvision.transforms.ToTensor(),
    }
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_modalities=["v", "attr"],
        modality_proportions={frozenset(["v", "t"]): 0.01},
        seed=0,
        transforms=transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    for modality in ["v", "attr"]:
        assert modality in item

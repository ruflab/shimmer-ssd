import torch
import torch.utils.data
from utils import PROJECT_DIR

from simple_shapes_dataset.dataset.dataset import SimpleShapesDataset
from simple_shapes_dataset.dataset.pre_process import attribute_to_tensor


def test_attr_preprocess():
    transform = {
        "attr": attribute_to_tensor,
    }
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        selected_domains=["attr"],
        transforms=transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    assert isinstance(item["attr"], list)
    assert len(item["attr"]) == 2
    assert isinstance(item["attr"][0], torch.Tensor)
    assert item["attr"][0].shape == (2, 3)
    assert isinstance(item["attr"][1], torch.Tensor)
    assert item["attr"][1].shape == (2, 8)
